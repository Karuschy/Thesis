"""
Convolutional VAE for volatility surfaces.

Replaces the MLP baseline with 2-D convolutions that preserve the
spatial (maturity × delta) structure of the implied-volatility grid.

Architecture
------------
Encoder:
    Conv2d (no stride)  → [B, ch0, H, W]        preserve spatial dims
    Conv2d (stride 2)   → [B, ch1, H', W']      downsample
    Conv2d (stride 2)   → [B, ch2, H'', W'']    downsample
    Flatten → FC → (mu, logvar)

Decoder:
    FC → Reshape → [B, ch2, H'', W'']
    Upsample + Conv2d   → [B, ch1, H', W']
    Upsample + Conv2d   → [B, ch0, H, W]
    Conv2d (1×1)        → [B, C, H, W]          back to input channels

Upsample + Conv is used instead of ConvTranspose2d to avoid
checkerboard artifacts and to handle arbitrary spatial sizes cleanly.

Input / Output
--------------
Same interface as ``MLPVAE``:
    forward(x)  →  (recon, mu, logvar)
    encode(x)   →  (mu, logvar)
    decode(z)   →  recon
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _ConvBlock(nn.Module):
    """Conv2d → optional BatchNorm → ReLU."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        batchnorm: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding),
        ]
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _UpsampleConvBlock(nn.Module):
    """Bilinear upsample to explicit target size → Conv2d → BN → ReLU."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        padding: int = 1,
        batchnorm: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding),
        ]
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Convolutional VAE
# ---------------------------------------------------------------------------

class ConvVAE(nn.Module):
    """
    Convolutional VAE for volatility surfaces.

    Parameters
    ----------
    in_shape : (C, H, W)
        Channel count (2 = call/put), maturities, deltas.
    latent_dim : int
        Dimensionality of the latent space.
    channels : tuple of int
        Channel widths for each encoder stage.
        The first stage has stride 1 (preserves spatial dims);
        every subsequent stage has stride 2 (halves spatial dims).
        Default (32, 64, 128) gives two downsampling steps.
    fc_dim : int
        Width of the FC layer between conv features and latent.
    batchnorm : bool
        Whether to use BatchNorm2d after each conv layer.
    """

    def __init__(
        self,
        in_shape: tuple[int, int, int],
        latent_dim: int = 16,
        channels: tuple[int, ...] | Sequence[int] = (32, 64, 128),
        fc_dim: int = 256,
        batchnorm: bool = True,
    ):
        super().__init__()
        C, H, W = in_shape
        self.in_shape = (C, H, W)
        self.latent_dim = latent_dim
        self.channels = tuple(channels)
        self.fc_dim = fc_dim

        if len(channels) < 2:
            raise ValueError("channels must have at least 2 entries")

        # ----- compute spatial sizes at each stage -----
        # sizes[0] = (H, W)  ... sizes[k] = size after encoder stage k
        sizes: list[tuple[int, int]] = [(H, W)]
        h, w = H, W
        for i in range(1, len(channels)):
            h = math.ceil(h / 2)
            w = math.ceil(w / 2)
            sizes.append((h, w))
        self._sizes = sizes          # stored for decoder upsampling targets
        self._flat_dim = channels[-1] * sizes[-1][0] * sizes[-1][1]

        # ===================== Encoder =====================
        enc: list[nn.Module] = []
        # Stage 0: stride-1 conv (preserve spatial dims)
        enc.append(_ConvBlock(C, channels[0], stride=1, batchnorm=batchnorm))
        # Stages 1..N-1: stride-2 conv (downsample)
        for i in range(1, len(channels)):
            enc.append(
                _ConvBlock(channels[i - 1], channels[i], stride=2, batchnorm=batchnorm)
            )
        self.encoder_conv = nn.Sequential(*enc)

        # Flatten → FC → (mu, logvar)
        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._flat_dim, fc_dim),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(fc_dim, latent_dim)
        self.fc_logvar = nn.Linear(fc_dim, latent_dim)

        # ===================== Decoder =====================
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, fc_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dim, self._flat_dim),
            nn.ReLU(inplace=True),
        )

        # Upsample + Conv stages (reverse order of encoder stride-2 stages)
        dec: list[nn.Module] = []
        for i in range(len(channels) - 1, 0, -1):
            dec.append(
                _UpsampleConvBlock(channels[i], channels[i - 1], batchnorm=batchnorm)
            )
        self.decoder_upsample = nn.ModuleList(dec)

        # Final projection back to C channels (no BN, no ReLU)
        self.decoder_head = nn.Conv2d(channels[0], C, kernel_size=1)

    # ------------------------------------------------------------------ #
    #  Forward helpers
    # ------------------------------------------------------------------ #

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x [B,C,H,W] → (mu, logvar)  [B, latent_dim] each."""
        h = self.encoder_conv(x)
        h = self.encoder_fc(h)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z [B, latent_dim] → reconstruction [B, C, H, W]."""
        h = self.decoder_fc(z)
        h = h.view(z.size(0), self.channels[-1], *self._sizes[-1])

        # Walk backwards through sizes to determine target for each up-stage
        # decoder_upsample[0] maps sizes[-1] → sizes[-2], etc.
        for i, up_block in enumerate(self.decoder_upsample):
            target_idx = len(self._sizes) - 2 - i
            h = up_block(h, self._sizes[target_idx])

        return self.decoder_head(h)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode → reparameterise → decode.  Returns (recon, mu, logvar)."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
