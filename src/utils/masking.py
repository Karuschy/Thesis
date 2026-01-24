"""
Masking utilities for surface completion experiments.

Provides functions to create masks and evaluate VAE performance
at different levels of missing data (0%, 25%, 50%, 75%).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.vae_mlp import MLPVAE, vae_loss, masked_vae_loss
from src.utils.scaler import ChannelStandardizer


# ============================================================================
# Mask Generation
# ============================================================================

def create_random_mask(
    shape: tuple[int, ...],
    mask_ratio: float,
    seed: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Create a random binary mask.
    
    Args:
        shape: Shape of the mask (e.g., (C, H, W) for a single surface).
        mask_ratio: Fraction of points to mask (0.0 to 1.0).
        seed: Random seed for reproducibility. If None, uses random state.
        device: Device to create mask on.
    
    Returns:
        Binary mask tensor where 1 = masked (hidden), 0 = observed.
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    total_points = np.prod(shape)
    n_masked = int(total_points * mask_ratio)
    
    # Create flat mask
    mask_flat = np.zeros(total_points, dtype=np.float32)
    if n_masked > 0:
        masked_indices = rng.choice(total_points, size=n_masked, replace=False)
        mask_flat[masked_indices] = 1.0
    
    # Reshape and convert to tensor
    mask = mask_flat.reshape(shape)
    return torch.from_numpy(mask).to(device)


def create_structured_mask(
    shape: tuple[int, int, int],
    mask_type: str,
    mask_ratio: float = 0.5,
    seed: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Create a structured mask that reflects real-world missingness patterns.
    
    Args:
        shape: (C, H, W) where H=maturities, W=deltas.
        mask_type: Type of structured mask:
            - "otm": Mask out-of-the-money options (low/high delta)
            - "long_maturity": Mask long-dated options
            - "corners": Mask corners (OTM + long maturity)
        mask_ratio: Approximate fraction of points to mask.
        seed: Random seed.
        device: Device to create mask on.
    
    Returns:
        Binary mask tensor where 1 = masked (hidden), 0 = observed.
    """
    C, H, W = shape
    mask = torch.zeros(shape, device=device)
    
    if mask_type == "otm":
        # Mask low and high delta (OTM options)
        n_delta_mask = int(W * mask_ratio / 2)
        mask[:, :, :n_delta_mask] = 1.0  # Low delta
        mask[:, :, -n_delta_mask:] = 1.0  # High delta
        
    elif mask_type == "long_maturity":
        # Mask long-dated options
        n_mat_mask = int(H * mask_ratio)
        mask[:, -n_mat_mask:, :] = 1.0
        
    elif mask_type == "corners":
        # Mask corners (OTM + long maturity)
        n_delta = int(W * 0.3)
        n_mat = int(H * 0.3)
        mask[:, -n_mat:, :n_delta] = 1.0  # Long mat, low delta
        mask[:, -n_mat:, -n_delta:] = 1.0  # Long mat, high delta
        
    else:
        raise ValueError(f"Unknown mask_type: {mask_type}")
    
    return mask


# ============================================================================
# Masked Evaluation Metrics
# ============================================================================

@dataclass
class MaskedEvalMetrics:
    """Metrics for masked (surface completion) evaluation."""
    mask_ratio: float
    n_samples: int
    
    # Error on masked points only (in normalized space)
    mse_masked: float
    mae_masked: float
    rmse_masked: float
    
    # Error on observed points only
    mse_observed: float
    mae_observed: float
    rmse_observed: float
    
    # Full surface error (for comparison)
    mse_full: float
    mae_full: float
    rmse_full: float
    
    # In original IV space (if scaler provided)
    mse_masked_original: Optional[float] = None
    mae_masked_original: Optional[float] = None
    rmse_masked_original: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@torch.no_grad()
def evaluate_with_masking(
    model: MLPVAE,
    loader: DataLoader,
    mask: torch.Tensor,
    device: torch.device,
    scaler: Optional[ChannelStandardizer] = None,
) -> MaskedEvalMetrics:
    """
    Evaluate VAE with a fixed mask applied to inputs.
    
    The evaluation process:
    1. Apply mask to input (set masked points to 0)
    2. Encode the masked input
    3. Decode to get full reconstruction
    4. Measure error on masked points (completion accuracy)
    
    Args:
        model: Trained VAE model.
        loader: DataLoader (test set).
        mask: Binary mask [C, H, W] where 1 = masked.
        device: Torch device.
        scaler: If provided, compute errors in original IV space too.
    
    Returns:
        MaskedEvalMetrics with completion errors.
    """
    model.eval()
    mask = mask.to(device)
    mask_ratio = float(mask.mean())
    
    # Accumulators
    total_sq_masked = 0.0
    total_abs_masked = 0.0
    total_sq_observed = 0.0
    total_abs_observed = 0.0
    total_sq_full = 0.0
    total_abs_full = 0.0
    total_sq_masked_orig = 0.0
    total_abs_masked_orig = 0.0
    n_masked_points = 0
    n_observed_points = 0
    n_samples = 0
    
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)
        bs = x.size(0)
        
        # Apply mask to input (set masked points to 0)
        mask_exp = mask.unsqueeze(0).expand_as(x)
        x_masked = x * (1 - mask_exp)  # Zero out masked points
        
        # Forward pass with masked input
        recon, mu, logvar = model(x_masked)
        
        # Compute errors
        diff = recon - x
        diff_sq = diff.pow(2)
        diff_abs = diff.abs()
        
        # Errors on masked points
        masked_sq = (diff_sq * mask_exp).sum()
        masked_abs = (diff_abs * mask_exp).sum()
        n_masked = mask_exp.sum()
        
        # Errors on observed points
        observed_mask = 1 - mask_exp
        observed_sq = (diff_sq * observed_mask).sum()
        observed_abs = (diff_abs * observed_mask).sum()
        n_observed = observed_mask.sum()
        
        # Full surface errors
        full_sq = diff_sq.sum()
        full_abs = diff_abs.sum()
        
        # Accumulate
        total_sq_masked += float(masked_sq)
        total_abs_masked += float(masked_abs)
        total_sq_observed += float(observed_sq)
        total_abs_observed += float(observed_abs)
        total_sq_full += float(full_sq)
        total_abs_full += float(full_abs)
        n_masked_points += int(n_masked)
        n_observed_points += int(n_observed)
        n_samples += bs
        
        # Original space metrics (for masked points)
        if scaler is not None:
            recon_orig = scaler.inverse_transform(recon.cpu())
            x_orig = scaler.inverse_transform(x.cpu())
            diff_orig = recon_orig - x_orig
            mask_cpu = mask_exp.cpu()
            
            masked_sq_orig = (diff_orig.pow(2) * mask_cpu).sum()
            masked_abs_orig = (diff_orig.abs() * mask_cpu).sum()
            total_sq_masked_orig += float(masked_sq_orig)
            total_abs_masked_orig += float(masked_abs_orig)
    
    # Compute averages
    n_total_points = n_masked_points + n_observed_points
    
    mse_masked = total_sq_masked / n_masked_points if n_masked_points > 0 else 0.0
    mae_masked = total_abs_masked / n_masked_points if n_masked_points > 0 else 0.0
    
    mse_observed = total_sq_observed / n_observed_points if n_observed_points > 0 else 0.0
    mae_observed = total_abs_observed / n_observed_points if n_observed_points > 0 else 0.0
    
    mse_full = total_sq_full / n_total_points
    mae_full = total_abs_full / n_total_points
    
    # Original space
    mse_masked_orig = mae_masked_orig = rmse_masked_orig = None
    if scaler is not None and n_masked_points > 0:
        mse_masked_orig = total_sq_masked_orig / n_masked_points
        mae_masked_orig = total_abs_masked_orig / n_masked_points
        rmse_masked_orig = np.sqrt(mse_masked_orig)
    
    return MaskedEvalMetrics(
        mask_ratio=mask_ratio,
        n_samples=n_samples,
        mse_masked=mse_masked,
        mae_masked=mae_masked,
        rmse_masked=np.sqrt(mse_masked),
        mse_observed=mse_observed,
        mae_observed=mae_observed,
        rmse_observed=np.sqrt(mse_observed),
        mse_full=mse_full,
        mae_full=mae_full,
        rmse_full=np.sqrt(mse_full),
        mse_masked_original=mse_masked_orig,
        mae_masked_original=mae_masked_orig,
        rmse_masked_original=rmse_masked_orig,
    )


def evaluate_completion_sweep(
    model: MLPVAE,
    loader: DataLoader,
    grid_shape: tuple[int, int, int],
    device: torch.device,
    mask_ratios: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75),
    seed: int = 42,
    scaler: Optional[ChannelStandardizer] = None,
) -> List[MaskedEvalMetrics]:
    """
    Evaluate surface completion at multiple masking levels.
    
    Uses the same random mask (per ratio) across all samples for reproducibility.
    
    Args:
        model: Trained VAE model.
        loader: Test DataLoader.
        grid_shape: (C, H, W) shape of the volatility surface.
        device: Torch device.
        mask_ratios: Tuple of masking ratios to test.
        seed: Base seed for mask generation.
        scaler: If provided, compute original-space metrics.
    
    Returns:
        List of MaskedEvalMetrics, one per mask ratio.
    """
    results = []
    
    for i, ratio in enumerate(mask_ratios):
        # Create mask with unique seed per ratio
        mask = create_random_mask(
            shape=grid_shape,
            mask_ratio=ratio,
            seed=seed + i * 1000,  # Different seed per ratio
            device=device,
        )
        
        metrics = evaluate_with_masking(
            model=model,
            loader=loader,
            mask=mask,
            device=device,
            scaler=scaler,
        )
        results.append(metrics)
        
        print(f"Mask {ratio*100:5.1f}% | "
              f"MAE masked: {metrics.mae_masked:.6f} | "
              f"MAE observed: {metrics.mae_observed:.6f} | "
              f"MAE full: {metrics.mae_full:.6f}")
    
    return results


def print_completion_summary(results: List[MaskedEvalMetrics]) -> None:
    """Print a formatted summary table of completion results."""
    print("\n" + "=" * 70)
    print("SURFACE COMPLETION EVALUATION")
    print("=" * 70)
    print(f"{'Mask %':>8} | {'MSE Masked':>12} | {'MAE Masked':>12} | {'RMSE Masked':>12}")
    print("-" * 70)
    
    for r in results:
        print(f"{r.mask_ratio*100:>7.1f}% | {r.mse_masked:>12.6f} | {r.mae_masked:>12.6f} | {r.rmse_masked:>12.6f}")
    
    if results[0].mae_masked_original is not None:
        print("-" * 70)
        print("In original IV space (vol points):")
        print(f"{'Mask %':>8} | {'MSE':>12} | {'MAE':>12} | {'RMSE':>12}")
        print("-" * 70)
        for r in results:
            if r.mae_masked_original is not None:
                mae_bps = r.mae_masked_original * 100  # Convert to percentage points
                print(f"{r.mask_ratio*100:>7.1f}% | {r.mse_masked_original:>12.6f} | {r.mae_masked_original:>12.6f} ({mae_bps:.2f}%) | {r.rmse_masked_original:>12.6f}")
    
    print("=" * 70)
