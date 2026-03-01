"""
No-arbitrage penalty functions for implied volatility surfaces.

Differentiable penalties for training (PyTorch) and counting functions
for evaluation (NumPy).

Calendar penalty:  total variance σ²T must be non-decreasing in T
Butterfly penalty: IV smile must be convex in delta (second diff ≥ 0)

References:
    Ackerer et al. (2020) — "Deep Smoothing of the Implied Volatility Surface"
    Na et al. (2024) — "Computing Volatility Surfaces using GANs with
                         Minimal Arbitrage Violations"
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


# ── Differentiable penalties (for training) ─────────────────────────────

def calendar_penalty(sigma: torch.Tensor, days_grid: torch.Tensor) -> torch.Tensor:
    """
    Penalize violations of calendar-spread no-arbitrage condition.

    Total variance ω(δ, T) = σ²(δ, T) · T must be non-decreasing in T
    for every fixed delta.

        L_cal = mean( ReLU( -(ω_{h+1} - ω_h) ) )

    Args:
        sigma:     [B, C, H, W] implied volatilities in **original** (raw) scale.
        days_grid: [H] maturity grid in calendar days (e.g. [30, 60, …, 730]).
                   Must be a float tensor on the same device as sigma.

    Returns:
        Scalar penalty tensor (0 when no violations).
    """
    # T in years, shape [1, 1, H, 1] for broadcasting
    T = (days_grid / 365.0).reshape(1, 1, -1, 1)
    total_var = sigma.pow(2) * T                                # [B, C, H, W]
    diffs = total_var[:, :, 1:, :] - total_var[:, :, :-1, :]   # [B, C, H-1, W]
    return torch.mean(F.relu(-diffs))


def butterfly_penalty(sigma: torch.Tensor) -> torch.Tensor:
    """
    Penalize violations of butterfly (convexity) no-arbitrage condition.

    The implied volatility smile must be convex in delta for each fixed
    maturity.  We compute the second finite difference along the delta axis:

        d²σ = σ_{w+1} - 2·σ_w + σ_{w-1}

    and penalize negative values:

        L_but = mean( ReLU( -d²σ ) )

    Args:
        sigma: [B, C, H, W] implied volatilities in **original** (raw) scale.

    Returns:
        Scalar penalty tensor (0 when no violations).
    """
    d2 = sigma[:, :, :, 2:] - 2 * sigma[:, :, :, 1:-1] + sigma[:, :, :, :-2]
    return torch.mean(F.relu(-d2))


def compute_arb_penalty(
    sigma: torch.Tensor,
    days_grid: torch.Tensor,
    lambda_cal: float = 0.1,
    lambda_but: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute combined arbitrage penalty.

    Args:
        sigma:      [B, C, H, W] implied volatilities in **original** (raw) scale.
        days_grid:  [H] maturity grid in calendar days, float tensor, same device.
        lambda_cal: Relative weight for calendar penalty.
        lambda_but: Relative weight for butterfly penalty.

    Returns:
        (combined_penalty, cal_penalty, but_penalty) — all scalar tensors.
    """
    cal = calendar_penalty(sigma, days_grid)
    but = butterfly_penalty(sigma)
    combined = lambda_cal * cal + lambda_but * but
    return combined, cal, but


# ── Counting functions for evaluation (NumPy) ──────────────────────────

def count_calendar_violations(
    sigma: np.ndarray, days_grid: np.ndarray
) -> tuple[int, int]:
    """
    Count calendar-spread arbitrage violations.

    Args:
        sigma:     [N, C, H, W] or [C, H, W] implied volatilities (raw scale).
        days_grid: [H] maturity grid in calendar days.

    Returns:
        (violation_count, total_cells) where violation is σ²T decreasing in T.
    """
    if sigma.ndim == 3:
        sigma = sigma[np.newaxis]
    T = days_grid / 365.0  # [H]
    total_var = sigma ** 2 * T[np.newaxis, np.newaxis, :, np.newaxis]  # [N,C,H,W]
    diffs = np.diff(total_var, axis=2)  # [N, C, H-1, W]
    violations = int((diffs < 0).sum())
    total = diffs.size
    return violations, total


def count_butterfly_violations(sigma: np.ndarray) -> tuple[int, int]:
    """
    Count butterfly (convexity) arbitrage violations.

    Args:
        sigma: [N, C, H, W] or [C, H, W] implied volatilities (raw scale).

    Returns:
        (violation_count, total_cells) where violation is d²σ/dδ² < 0.
    """
    if sigma.ndim == 3:
        sigma = sigma[np.newaxis]
    d2 = np.diff(sigma, n=2, axis=3)  # [N, C, H, W-2]
    violations = int((d2 < 0).sum())
    total = d2.size
    return violations, total
