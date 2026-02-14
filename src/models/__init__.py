"""Model definitions for volatility surface VAE and Heston benchmark."""
from __future__ import annotations

from src.models.vae_mlp import MLPVAE, PointwiseMLPVAE, vae_loss, masked_vae_loss
from src.models.vae_conv import ConvVAE
from src.models import heston

__all__ = [
    "MLPVAE",
    "PointwiseMLPVAE",
    "ConvVAE",
    "vae_loss",
    "masked_vae_loss",
    "heston",
    "create_model",
]


def create_model(
    model_type: str,
    in_shape: tuple[int, int, int],
    latent_dim: int = 8,
    *,
    # MLP-specific
    hidden_dims: tuple[int, ...] = (256, 128),
    # Conv-specific
    channels: tuple[int, ...] = (32, 64, 128),
    fc_dim: int = 256,
    batchnorm: bool = True,
):
    """Factory that returns the requested VAE variant.

    Parameters
    ----------
    model_type : {"mlp", "conv"}
        Which architecture to use.
    in_shape : (C, H, W)
        Input tensor shape (channels, maturities, deltas).
    latent_dim : int
        Latent dimension.
    hidden_dims : tuple of int
        MLP encoder/decoder hidden widths (only used for model_type="mlp").
    channels : tuple of int
        Conv encoder channel widths (only used for model_type="conv").
    fc_dim : int
        FC bottleneck width after conv flatten (only for model_type="conv").
    batchnorm : bool
        Whether to use BatchNorm in conv blocks (only for model_type="conv").
    """
    if model_type == "mlp":
        return MLPVAE(in_shape=in_shape, latent_dim=latent_dim, hidden_dims=hidden_dims)
    if model_type == "conv":
        return ConvVAE(
            in_shape=in_shape,
            latent_dim=latent_dim,
            channels=channels,
            fc_dim=fc_dim,
            batchnorm=batchnorm,
        )
    raise ValueError(f"Unknown model_type {model_type!r}. Choose 'mlp' or 'conv'.")

