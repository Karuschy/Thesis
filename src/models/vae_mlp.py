from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPVAE(nn.Module):
    """
    Grid-wise VAE over flattened volatility surfaces.
    
    Input:  [B, C, H, W] where C=2 (call/put), H=maturities, W=deltas
    Output: reconstruction same shape
    
    The encoder takes the full flattened grid and produces latent (mu, logvar).
    The decoder takes z and outputs the full reconstructed grid.
    """
    def __init__(
        self,
        in_shape: tuple[int, int, int],  # (C, H, W)
        latent_dim: int = 16,
        hidden_dims: tuple[int, ...] = (256, 128),
    ):
        super().__init__()
        C, H, W = in_shape
        self.in_shape = in_shape
        self.in_dim = C * H * W
        self.latent_dim = latent_dim

        # Encoder: flattened grid -> hidden -> (mu, logvar)
        enc_layers = []
        prev = self.in_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

        # Decoder: z -> hidden -> flattened grid
        dec_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec_layers += [nn.Linear(prev, self.in_dim)]
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input grid to latent distribution parameters."""
        b = x.shape[0]
        h = self.encoder(x.view(b, -1))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z from q(z|x) using reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstructed grid."""
        out = self.decoder(z)
        return out.view(z.shape[0], *self.in_shape)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: encode, sample, decode."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class PointwiseMLPVAE(nn.Module):
    """
    Pointwise VAE for volatility surfaces.
    
    The encoder is the same as MLPVAE (takes full grid).
    The decoder takes (z, delta, maturity) and outputs a single IV value.
    This allows flexible interpolation at any (delta, maturity) coordinate.
    
    Input:  [B, C, H, W] for encoding
    Output: Can decode at any (delta, maturity) points given z
    """
    def __init__(
        self,
        in_shape: tuple[int, int, int],  # (C, H, W)
        latent_dim: int = 16,
        hidden_dims: tuple[int, ...] = (256, 128),
        decoder_hidden_dims: tuple[int, ...] = (64, 32),
    ):
        super().__init__()
        C, H, W = in_shape
        self.in_shape = in_shape
        self.in_dim = C * H * W
        self.latent_dim = latent_dim
        self.n_channels = C  # Call and Put

        # Encoder: same as grid-wise VAE
        enc_layers = []
        prev = self.in_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

        # Pointwise decoder: (z, delta, maturity, cp_flag) -> IV
        # Input: latent_dim + 2 (delta, maturity) + 1 (cp_flag as 0/1)
        decoder_input_dim = latent_dim + 3
        dec_layers = []
        prev = decoder_input_dim
        for h in decoder_hidden_dims:
            dec_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec_layers += [nn.Linear(prev, 1)]  # Output single IV value
        self.pointwise_decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input grid to latent distribution parameters."""
        b = x.shape[0]
        h = self.encoder(x.view(b, -1))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z from q(z|x) using reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_point(
        self, 
        z: torch.Tensor, 
        delta: torch.Tensor, 
        maturity: torch.Tensor, 
        cp_flag: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode at specific (delta, maturity, cp_flag) coordinates.
        
        Args:
            z: [B, latent_dim] latent vectors
            delta: [B, N] or [B] delta values (normalized 0-1)
            maturity: [B, N] or [B] maturity values (normalized 0-1)
            cp_flag: [B, N] or [B] call/put flag (0=call, 1=put)
        
        Returns:
            IV values at the specified points [B, N] or [B]
        """
        # Ensure proper shapes
        if delta.dim() == 1:
            delta = delta.unsqueeze(1)
            maturity = maturity.unsqueeze(1)
            cp_flag = cp_flag.unsqueeze(1)
        
        B, N = delta.shape
        
        # Expand z to match number of points
        z_exp = z.unsqueeze(1).expand(B, N, -1)  # [B, N, latent_dim]
        
        # Stack coordinates
        coords = torch.stack([delta, maturity, cp_flag.float()], dim=-1)  # [B, N, 3]
        
        # Concatenate z with coordinates
        decoder_input = torch.cat([z_exp, coords], dim=-1)  # [B, N, latent_dim + 3]
        
        # Decode
        iv = self.pointwise_decoder(decoder_input).squeeze(-1)  # [B, N]
        return iv

    def decode_grid(self, z: torch.Tensor, delta_grid: torch.Tensor, maturity_grid: torch.Tensor) -> torch.Tensor:
        """
        Decode full grid from latent vector.
        
        Args:
            z: [B, latent_dim] latent vectors
            delta_grid: [W] normalized delta values
            maturity_grid: [H] normalized maturity values
        
        Returns:
            Reconstructed grid [B, C, H, W]
        """
        B = z.shape[0]
        H = len(maturity_grid)
        W = len(delta_grid)
        C = self.n_channels
        
        # Create meshgrid of coordinates
        mat_mesh, delta_mesh = torch.meshgrid(maturity_grid, delta_grid, indexing='ij')
        mat_flat = mat_mesh.flatten()  # [H*W]
        delta_flat = delta_mesh.flatten()  # [H*W]
        
        # Decode for each channel (call=0, put=1)
        recon = torch.zeros(B, C, H, W, device=z.device)
        for c in range(C):
            cp_flat = torch.full_like(delta_flat, c)  # [H*W]
            
            # Expand for batch
            delta_batch = delta_flat.unsqueeze(0).expand(B, -1)  # [B, H*W]
            mat_batch = mat_flat.unsqueeze(0).expand(B, -1)
            cp_batch = cp_flat.unsqueeze(0).expand(B, -1)
            
            iv_flat = self.decode_point(z, delta_batch, mat_batch, cp_batch)  # [B, H*W]
            recon[:, c] = iv_flat.view(B, H, W)
        
        return recon

    def forward(
        self, 
        x: torch.Tensor,
        delta_grid: torch.Tensor | None = None,
        maturity_grid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode, sample, decode full grid.
        
        If grids not provided, creates uniform normalized grids matching input shape.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        
        # Create default normalized grids if not provided
        if delta_grid is None:
            delta_grid = torch.linspace(0, 1, self.in_shape[2], device=x.device)
        if maturity_grid is None:
            maturity_grid = torch.linspace(0, 1, self.in_shape[1], device=x.device)
        
        recon = self.decode_grid(z, delta_grid, maturity_grid)
        return recon, mu, logvar


def vae_loss(
    recon: torch.Tensor, 
    x: torch.Tensor, 
    mu: torch.Tensor, 
    logvar: torch.Tensor, 
    beta: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Standard VAE ELBO loss = reconstruction + beta * KL divergence.
    
    Args:
        recon: Reconstructed output [B, C, H, W]
        x: Original input [B, C, H, W]
        mu: Latent mean [B, latent_dim]
        logvar: Latent log-variance [B, latent_dim]
        beta: KL divergence weight (beta-VAE)
    
    Returns:
        (total_loss, recon_loss, kl_loss)
    """
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss.detach(), kl.detach()


def masked_vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VAE loss computed only on masked (missing) points.
    
    This is used for surface completion evaluation where we measure
    error only on the points that were hidden from the encoder.
    
    Args:
        recon: Reconstructed output [B, C, H, W]
        x: Original input [B, C, H, W]
        mu: Latent mean [B, latent_dim]
        logvar: Latent log-variance [B, latent_dim]
        mask: Binary mask [C, H, W] or [B, C, H, W] where 1 = masked (hidden), 0 = observed
        beta: KL divergence weight
    
    Returns:
        (total_loss, recon_loss_on_masked, kl_loss)
    """
    # Expand mask if needed
    if mask.dim() == 3:
        mask = mask.unsqueeze(0).expand_as(x)
    
    # Compute MSE only on masked points
    diff_sq = (recon - x).pow(2)
    masked_diff_sq = diff_sq * mask
    n_masked = mask.sum()
    
    if n_masked > 0:
        recon_loss = masked_diff_sq.sum() / n_masked
    else:
        recon_loss = torch.tensor(0.0, device=x.device)
    
    # KL is computed normally (on full latent space)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl, recon_loss.detach(), kl.detach()

