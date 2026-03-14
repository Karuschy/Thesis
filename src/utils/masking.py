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
            - "short_maturity": Mask short-dated options (early maturities)
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

    if not 0.0 <= mask_ratio <= 1.0:
        raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")

    if mask_ratio == 0.0:
        return mask
    
    if mask_type == "otm":
        # Mask low and high delta (OTM options)
        n_delta_mask = int(W * mask_ratio / 2)
        if n_delta_mask > 0:
            mask[:, :, :n_delta_mask] = 1.0  # Low delta
            mask[:, :, W - n_delta_mask:] = 1.0  # High delta
        
    elif mask_type == "long_maturity":
        # Mask long-dated options
        n_mat_mask = int(H * mask_ratio)
        if n_mat_mask > 0:
            mask[:, H - n_mat_mask:, :] = 1.0
        
    elif mask_type == "corners":
        # Mask corners (OTM + long maturity), roughly matching mask_ratio.
        # Approximation target:
        #   ratio ≈ 2 * (n_mat/H) * (n_delta/W)
        # choose n_mat ~ H*sqrt(r), n_delta ~ W*sqrt(r)/2
        sqrt_r = np.sqrt(mask_ratio)
        n_mat = max(1, min(H, int(H * sqrt_r)))
        n_delta = max(1, min(W // 2, int(W * sqrt_r / 2)))
        mask[:, H - n_mat:, :n_delta] = 1.0  # Long mat, low delta
        mask[:, H - n_mat:, W - n_delta:] = 1.0  # Long mat, high delta
        
    elif mask_type == "short_maturity":
        # Mask short-dated options (early maturities)
        n_mat_mask = int(H * mask_ratio)
        if n_mat_mask > 0:
            mask[:, :n_mat_mask, :] = 1.0

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
    model: torch.nn.Module,
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
    model: torch.nn.Module,
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




@torch.no_grad()
def evaluate_completion_multiseed(
    model: torch.nn.Module,
    X_test: torch.Tensor,
    scaler: ChannelStandardizer,
    mask_ratio: float,
    n_seeds: int = 5,
    batch_size: int = 32,
    device: torch.device = torch.device("cpu"),
) -> tuple[float, float, np.ndarray]:
    """
    Evaluate surface completion with per-date, per-seed random masks.

    For each seed, every test date receives its own independently drawn random
    mask. Errors are measured only on the masked (hidden) cells and then
    averaged over all seeds and dates.

    When mask_ratio == 0, no cells are masked, so the function falls back to
    reporting the full-surface reconstruction MAE -- useful as a sanity check
    that 0% masking == normal reconstruction error.

    Args:
        model: Trained VAE model (must already be on ``device``).
        X_test: Normalised test surfaces, shape (N, C, H, W), on CPU.
        scaler: Fitted ChannelStandardizer (must already be on ``device``).
        mask_ratio: Fraction of grid cells to hide (0.0 to 1.0).
        n_seeds: Number of independent random seeds to average over.
        batch_size: GPU batch size for inference.
        device: Torch device.

    Returns:
        Tuple of:
        - mean_mae_vp  : float   -- mean MAE in vol points across dates and seeds
        - mean_rmse_vp : float   -- sqrt of mean MSE in vol points
        - per_date_mae : ndarray -- shape (N,), per-date MAE averaged over seeds
    """
    model.eval()
    N, C, H, W = X_test.shape
    n_cells = C * H * W
    n_masked = int(n_cells * mask_ratio)

    per_date_mae_sum = np.zeros(N, dtype=np.float64)
    per_date_mse_sum = np.zeros(N, dtype=np.float64)

    for seed_idx in range(n_seeds):
        # Build (N, C, H, W) mask -- each date gets an independent random mask
        rng = np.random.RandomState(seed_idx * 10007)
        masks_np = np.zeros((N, C, H, W), dtype=np.float32)
        for i in range(N):
            if n_masked > 0:
                flat = np.zeros(n_cells, dtype=np.float32)
                flat[rng.choice(n_cells, size=n_masked, replace=False)] = 1.0
                masks_np[i] = flat.reshape(C, H, W)
        masks = torch.from_numpy(masks_np)  # stay on CPU

        # Batch inference
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            x_batch = X_test[start:end].to(device)
            m_batch = masks[start:end].to(device)

            # Zero out masked cells before encoding
            x_masked = x_batch * (1.0 - m_batch)
            recon, mu, logvar = model(x_masked)

            # Inverse-transform to original IV space (vol points)
            recon_orig = scaler.inverse_transform(recon).cpu()
            x_orig = scaler.inverse_transform(x_batch).cpu()
            m_cpu = m_batch.cpu()

            diff = recon_orig - x_orig

            bs = end - start
            for j in range(bs):
                n_m = int(m_cpu[j].sum().item())
                if n_m > 0:
                    # Error on masked (hidden) cells only
                    mae_j = float((diff[j].abs() * m_cpu[j]).sum()) / n_m
                    mse_j = float((diff[j].pow(2) * m_cpu[j]).sum()) / n_m
                else:
                    # mask_ratio == 0: full-surface reconstruction error
                    mae_j = float(diff[j].abs().mean())
                    mse_j = float(diff[j].pow(2).mean())
                per_date_mae_sum[start + j] += mae_j
                per_date_mse_sum[start + j] += mse_j

    per_date_mae = per_date_mae_sum / n_seeds
    per_date_mse = per_date_mse_sum / n_seeds
    mean_mae_vp = float(per_date_mae.mean())
    mean_rmse_vp = float(np.sqrt(per_date_mse.mean()))

    return mean_mae_vp, mean_rmse_vp, per_date_mae


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
                mae_bps = r.mae_masked_original * 100
                print(f"{r.mask_ratio*100:>7.1f}% | {r.mse_masked_original:>12.6f} | {r.mae_masked_original:>12.6f} ({mae_bps:.2f}%) | {r.rmse_masked_original:>12.6f}")

    print("=" * 70)
