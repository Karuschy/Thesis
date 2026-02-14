"""
Evaluation utilities for the volatility surface VAE.

Provides comprehensive metrics computation in both normalized and original space.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.vae_mlp import MLPVAE, vae_loss
from src.utils.scaler import ChannelStandardizer


@dataclass
class EvalMetrics:
    """Container for evaluation metrics."""
    # In normalized space (training objective)
    elbo: float
    recon_loss: float
    kl_loss: float
    mse: float
    mae: float
    rmse: float
    
    # In original space (interpretable; None if no scaler)
    mse_original: Optional[float] = None
    mae_original: Optional[float] = None
    rmse_original: Optional[float] = None
    
    # Additional stats
    n_samples: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self, path: str | Path, indent: int = 2) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=indent)


@dataclass
class DetailedEvalResult:
    """
    Detailed evaluation result with per-sample errors.
    """
    metrics: EvalMetrics
    # Per-sample data (for plotting)
    all_errors: torch.Tensor  # [N, C, H, W] absolute error per sample
    all_recons: torch.Tensor  # [N, C, H, W] reconstructions
    all_targets: torch.Tensor  # [N, C, H, W] original targets
    all_mus: torch.Tensor  # [N, latent_dim] latent means
    dates: Optional[np.ndarray] = None  # [N] dates if available


@torch.no_grad()
def evaluate_vae(
    model: MLPVAE,
    loader: DataLoader,
    device: torch.device,
    beta: float = 1.0,
    scaler: Optional[ChannelStandardizer] = None,
    collect_details: bool = False,
) -> EvalMetrics | DetailedEvalResult:
    """
    Evaluate VAE on a data loader.
    
    Args:
        model: Trained VAE model.
        loader: DataLoader (test or validation).
        device: Torch device.
        beta: KL weight (should match training).
        scaler: If provided, also compute metrics in original IV space.
        collect_details: If True, return DetailedEvalResult with per-sample data.
    
    Returns:
        EvalMetrics (or DetailedEvalResult if collect_details=True).
    """
    model.eval()
    
    total_elbo = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_mse_orig = 0.0
    total_mae_orig = 0.0
    n_samples = 0
    
    # For detailed results
    all_errors_list: List[torch.Tensor] = []
    all_recons_list: List[torch.Tensor] = []
    all_targets_list: List[torch.Tensor] = []
    all_mus_list: List[torch.Tensor] = []
    all_dates_list: List[np.ndarray] = []
    
    has_dates = False
    
    for batch in loader:
        # Handle (x,) or (x, date) tuples
        if isinstance(batch, (list, tuple)):
            x = batch[0]
            if len(batch) > 1:
                has_dates = True
                batch_dates = batch[1]
                if collect_details:
                    # Convert to numpy if needed
                    if isinstance(batch_dates, torch.Tensor):
                        all_dates_list.append(batch_dates.numpy())
                    else:
                        all_dates_list.append(np.array(batch_dates))
        else:
            x = batch
        
        x = x.to(device, non_blocking=True)
        bs = x.size(0)
        
        # Forward pass
        recon, mu, logvar = model(x)
        
        # VAE loss components
        elbo, recon_l, kl_l = vae_loss(recon, x, mu, logvar, beta=beta)
        
        # Per-element errors (normalized space)
        mse_batch = torch.mean((recon - x) ** 2).item()
        mae_batch = torch.mean(torch.abs(recon - x)).item()
        
        total_elbo += elbo.item() * bs
        total_recon += recon_l.item() * bs
        total_kl += kl_l.item() * bs
        total_mse += mse_batch * bs
        total_mae += mae_batch * bs
        
        # Original space metrics
        if scaler is not None:
            recon_orig = scaler.inverse_transform(recon.cpu())
            x_orig = scaler.inverse_transform(x.cpu())
            mse_orig_batch = torch.mean((recon_orig - x_orig) ** 2).item()
            mae_orig_batch = torch.mean(torch.abs(recon_orig - x_orig)).item()
            total_mse_orig += mse_orig_batch * bs
            total_mae_orig += mae_orig_batch * bs
        
        # Collect detailed data
        if collect_details:
            all_errors_list.append(torch.abs(recon - x).cpu())
            all_recons_list.append(recon.cpu())
            all_targets_list.append(x.cpu())
            all_mus_list.append(mu.cpu())
        
        n_samples += bs
    
    # Compute averages
    avg_elbo = total_elbo / n_samples
    avg_recon = total_recon / n_samples
    avg_kl = total_kl / n_samples
    avg_mse = total_mse / n_samples
    avg_mae = total_mae / n_samples
    avg_rmse = np.sqrt(avg_mse)
    
    mse_orig = mae_orig = rmse_orig = None
    if scaler is not None:
        mse_orig = total_mse_orig / n_samples
        mae_orig = total_mae_orig / n_samples
        rmse_orig = np.sqrt(mse_orig)
    
    metrics = EvalMetrics(
        elbo=avg_elbo,
        recon_loss=avg_recon,
        kl_loss=avg_kl,
        mse=avg_mse,
        mae=avg_mae,
        rmse=avg_rmse,
        mse_original=mse_orig,
        mae_original=mae_orig,
        rmse_original=rmse_orig,
        n_samples=n_samples,
    )
    
    if not collect_details:
        return metrics
    
    # Concatenate all collected tensors
    all_errors = torch.cat(all_errors_list, dim=0)
    all_recons = torch.cat(all_recons_list, dim=0)
    all_targets = torch.cat(all_targets_list, dim=0)
    all_mus = torch.cat(all_mus_list, dim=0)
    
    dates_arr = None
    if has_dates and all_dates_list:
        dates_arr = np.concatenate(all_dates_list, axis=0)
    
    return DetailedEvalResult(
        metrics=metrics,
        all_errors=all_errors,
        all_recons=all_recons,
        all_targets=all_targets,
        all_mus=all_mus,
        dates=dates_arr,
    )


def compute_per_cell_error(
    errors: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Aggregate error across samples to get per-cell error heatmap.
    
    Args:
        errors: [N, C, H, W] absolute errors
        reduction: 'mean', 'std', 'max'
    
    Returns:
        [C, H, W] aggregated error per grid cell
    """
    if reduction == "mean":
        return errors.mean(dim=0)
    elif reduction == "std":
        return errors.std(dim=0)
    elif reduction == "max":
        return errors.max(dim=0).values
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def get_worst_reconstructions(
    result: DetailedEvalResult,
    k: int = 5,
) -> List[int]:
    """
    Get indices of the k worst reconstructed samples (by mean absolute error).
    """
    per_sample_mae = result.all_errors.mean(dim=(1, 2, 3))  # [N]
    _, indices = torch.topk(per_sample_mae, k=min(k, len(per_sample_mae)))
    return indices.tolist()


def get_best_reconstructions(
    result: DetailedEvalResult,
    k: int = 5,
) -> List[int]:
    """
    Get indices of the k best reconstructed samples (by mean absolute error).
    """
    per_sample_mae = result.all_errors.mean(dim=(1, 2, 3))  # [N]
    _, indices = torch.topk(per_sample_mae, k=min(k, len(per_sample_mae)), largest=False)
    return indices.tolist()
