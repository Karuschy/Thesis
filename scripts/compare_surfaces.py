"""
Compare VAE-reconstructed surfaces with Heston-calibrated surfaces.

Loads both surface sets, aligns by date, and computes comparison metrics.

Usage:
    python scripts/compare_surfaces.py \
        --vae_dir artifacts/eval/surfaces \
        --heston_dir data/processed/heston/surfaces \
        --output_dir artifacts/comparison
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd


@dataclass
class ComparisonMetrics:
    """Metrics for comparing two surface sets."""
    mse: float
    mae: float
    rmse: float
    max_error: float
    # Per-cell stats
    mean_per_cell_mae: float
    std_per_cell_mae: float
    # Sample stats
    n_samples: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_metrics(surfaces_a: np.ndarray, surfaces_b: np.ndarray) -> ComparisonMetrics:
    """
    Compute comparison metrics between two surface arrays.
    
    Args:
        surfaces_a: First surface array [N, C, H, W]
        surfaces_b: Second surface array [N, C, H, W]
    
    Returns:
        ComparisonMetrics with error statistics.
    """
    assert surfaces_a.shape == surfaces_b.shape, "Surface shapes must match"
    
    errors = surfaces_a - surfaces_b
    abs_errors = np.abs(errors)
    sq_errors = errors ** 2
    
    mse = float(np.mean(sq_errors))
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(mse))
    max_error = float(np.max(abs_errors))
    
    # Per-cell MAE: [C, H, W] averaged across samples
    per_cell_mae = np.mean(abs_errors, axis=0)  # [C, H, W]
    mean_per_cell_mae = float(np.mean(per_cell_mae))
    std_per_cell_mae = float(np.std(per_cell_mae))
    
    return ComparisonMetrics(
        mse=mse,
        mae=mae,
        rmse=rmse,
        max_error=max_error,
        mean_per_cell_mae=mean_per_cell_mae,
        std_per_cell_mae=std_per_cell_mae,
        n_samples=surfaces_a.shape[0],
    )


def load_vae_surfaces(vae_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load VAE surfaces, market surfaces, dates, and grid spec."""
    vae_surfaces = np.load(vae_dir / "vae_surfaces.npy")
    market_surfaces = np.load(vae_dir / "market_surfaces.npy")
    dates_df = pd.read_csv(vae_dir / "vae_surface_dates.csv")
    dates = pd.to_datetime(dates_df["date"]).values
    
    with open(vae_dir / "grid_spec.json") as f:
        grid_spec = json.load(f)
    
    return vae_surfaces, market_surfaces, dates, grid_spec


def load_heston_surfaces(heston_dir: Path, ticker: str = "AAPL") -> Tuple[np.ndarray, np.ndarray]:
    """Load Heston surfaces and dates."""
    heston_surfaces = np.load(heston_dir / f"{ticker}_heston_surfaces.npy")
    dates_df = pd.read_csv(heston_dir / f"{ticker}_heston_surface_dates.csv")
    dates = pd.to_datetime(dates_df["date"]).values
    
    return heston_surfaces, dates


def align_surfaces_by_date(
    surfaces_a: np.ndarray,
    dates_a: np.ndarray,
    surfaces_b: np.ndarray,
    dates_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align two surface arrays by date (keep only common dates).
    
    Returns:
        (aligned_a, aligned_b, common_dates)
    """
    # Find common dates
    dates_a_set = set(pd.to_datetime(dates_a).date)
    dates_b_set = set(pd.to_datetime(dates_b).date)
    common_dates_set = dates_a_set & dates_b_set
    
    # Create masks
    dates_a_date = [pd.to_datetime(d).date() for d in dates_a]
    dates_b_date = [pd.to_datetime(d).date() for d in dates_b]
    
    mask_a = [d in common_dates_set for d in dates_a_date]
    mask_b = [d in common_dates_set for d in dates_b_date]
    
    aligned_a = surfaces_a[mask_a]
    aligned_b = surfaces_b[mask_b]
    common_dates = dates_a[mask_a]
    
    return aligned_a, aligned_b, common_dates


def save_comparison_plots(
    vae_surfaces: np.ndarray,
    heston_surfaces: np.ndarray,
    market_surfaces: np.ndarray,
    dates: np.ndarray,
    grid_spec: dict,
    output_dir: Path,
    n_samples: int = 5,
):
    """Generate comparison plots."""
    import matplotlib.pyplot as plt
    
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    days_grid = np.array(grid_spec["days_grid"])
    delta_grid = np.array(grid_spec["delta_grid"])
    cp_order = grid_spec["cp_order"]
    
    # 1. Error heatmaps (mean across samples)
    vae_error = np.mean(np.abs(vae_surfaces - market_surfaces), axis=0)
    heston_error = np.mean(np.abs(heston_surfaces - market_surfaces), axis=0)
    
    for c, cp in enumerate(cp_order):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # VAE error
        im1 = axes[0].imshow(vae_error[c], aspect="auto", origin="lower", cmap="Reds")
        axes[0].set_title(f"VAE MAE ({cp})")
        axes[0].set_ylabel("Maturity (days)")
        axes[0].set_xlabel("Delta")
        axes[0].set_yticks(range(len(days_grid)))
        axes[0].set_yticklabels([int(d) for d in days_grid])
        plt.colorbar(im1, ax=axes[0], format="%.3f")
        
        # Heston error
        im2 = axes[1].imshow(heston_error[c], aspect="auto", origin="lower", cmap="Reds")
        axes[1].set_title(f"Heston MAE ({cp})")
        axes[1].set_xlabel("Delta")
        axes[1].set_yticks(range(len(days_grid)))
        axes[1].set_yticklabels([int(d) for d in days_grid])
        plt.colorbar(im2, ax=axes[1], format="%.3f")
        
        # Difference (VAE - Heston), blue=VAE better, red=Heston better
        diff = vae_error[c] - heston_error[c]
        vmax = max(abs(diff.min()), abs(diff.max()))
        im3 = axes[2].imshow(diff, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[2].set_title(f"VAE - Heston ({cp})\n(blue=VAE better)")
        axes[2].set_xlabel("Delta")
        axes[2].set_yticks(range(len(days_grid)))
        axes[2].set_yticklabels([int(d) for d in days_grid])
        plt.colorbar(im3, ax=axes[2], format="%.3f")
        
        plt.tight_layout()
        plt.savefig(plots_dir / f"error_heatmap_{cp}.png", dpi=150, bbox_inches="tight")
        plt.close()
    
    # 2. Sample surface comparisons
    # Pick samples spread across the date range
    n_samples = min(n_samples, len(dates))
    sample_indices = np.linspace(0, len(dates) - 1, n_samples, dtype=int)
    
    for idx in sample_indices:
        date_str = str(dates[idx])[:10]
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        for c, cp in enumerate(cp_order):
            row = c
            
            # Market
            im0 = axes[row, 0].imshow(market_surfaces[idx, c], aspect="auto", origin="lower", cmap="viridis")
            axes[row, 0].set_title(f"Market IV ({cp})")
            if c == 0:
                axes[row, 0].set_ylabel("Maturity")
            plt.colorbar(im0, ax=axes[row, 0], format="%.3f")
            
            # VAE
            im1 = axes[row, 1].imshow(vae_surfaces[idx, c], aspect="auto", origin="lower", cmap="viridis")
            axes[row, 1].set_title(f"VAE ({cp})")
            plt.colorbar(im1, ax=axes[row, 1], format="%.3f")
            
            # Heston
            im2 = axes[row, 2].imshow(heston_surfaces[idx, c], aspect="auto", origin="lower", cmap="viridis")
            axes[row, 2].set_title(f"Heston ({cp})")
            plt.colorbar(im2, ax=axes[row, 2], format="%.3f")
            
            # Error comparison
            vae_err = np.abs(vae_surfaces[idx, c] - market_surfaces[idx, c])
            heston_err = np.abs(heston_surfaces[idx, c] - market_surfaces[idx, c])
            
            axes[row, 3].bar([0, 1], [vae_err.mean(), heston_err.mean()], 
                           tick_label=["VAE", "Heston"], color=["blue", "orange"])
            axes[row, 3].set_title(f"MAE ({cp})")
            axes[row, 3].set_ylabel("Mean Abs Error")
        
        fig.suptitle(f"Surface Comparison - {date_str}", fontsize=14)
        plt.tight_layout()
        plt.savefig(plots_dir / f"sample_{date_str}.png", dpi=150, bbox_inches="tight")
        plt.close()
    
    # 3. Time series of errors
    vae_mae_ts = np.abs(vae_surfaces - market_surfaces).mean(axis=(1, 2, 3))
    heston_mae_ts = np.abs(heston_surfaces - market_surfaces).mean(axis=(1, 2, 3))
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, vae_mae_ts, label="VAE", alpha=0.7)
    ax.plot(dates, heston_mae_ts, label="Heston", alpha=0.7)
    ax.set_xlabel("Date")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title("VAE vs Heston: Error Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "error_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"  Saved plots to: {plots_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare VAE and Heston surfaces")
    
    parser.add_argument("--vae_dir", type=str, default="artifacts/eval/surfaces",
                        help="Directory with VAE surfaces (from eval_vae.py)")
    parser.add_argument("--heston_dir", type=str, default="data/processed/heston/surfaces",
                        help="Directory with Heston surfaces")
    parser.add_argument("--ticker", type=str, default="AAPL",
                        help="Ticker symbol for Heston files")
    parser.add_argument("--output_dir", type=str, default="artifacts/comparison",
                        help="Directory to save comparison outputs")
    parser.add_argument("--n_plot_samples", type=int, default=5,
                        help="Number of sample surfaces to plot")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    vae_dir = Path(args.vae_dir)
    heston_dir = Path(args.heston_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load surfaces
    print("Loading VAE surfaces...")
    vae_surfaces, market_surfaces, vae_dates, grid_spec = load_vae_surfaces(vae_dir)
    print(f"  VAE: {vae_surfaces.shape}, {len(vae_dates)} dates")
    
    print("Loading Heston surfaces...")
    heston_surfaces, heston_dates = load_heston_surfaces(heston_dir, args.ticker)
    print(f"  Heston: {heston_surfaces.shape}, {len(heston_dates)} dates")
    
    # Align by date
    print("\nAligning surfaces by date...")
    vae_aligned, heston_aligned, common_dates = align_surfaces_by_date(
        vae_surfaces, vae_dates, heston_surfaces, heston_dates
    )
    market_aligned, _, _ = align_surfaces_by_date(
        market_surfaces, vae_dates, heston_surfaces, heston_dates
    )
    
    print(f"  Common dates: {len(common_dates)}")
    print(f"  Date range: {str(common_dates[0])[:10]} to {str(common_dates[-1])[:10]}")
    
    if len(common_dates) == 0:
        print("\nERROR: No common dates found!")
        print(f"  VAE dates:    {str(vae_dates[0])[:10]} to {str(vae_dates[-1])[:10]}")
        print(f"  Heston dates: {str(heston_dates[0])[:10]} to {str(heston_dates[-1])[:10]}")
        return
    
    # Compute metrics
    print("\n" + "=" * 60)
    print("COMPARISON METRICS (on test set)")
    print("=" * 60)
    
    print("\n1. VAE vs Market (reconstruction quality):")
    vae_vs_market = compute_metrics(vae_aligned, market_aligned)
    print(f"   MSE:  {vae_vs_market.mse:.6f}")
    print(f"   MAE:  {vae_vs_market.mae:.6f}")
    print(f"   RMSE: {vae_vs_market.rmse:.6f}")
    print(f"   MAE (vol points): {vae_vs_market.mae * 100:.2f}%")
    
    print("\n2. Heston vs Market (calibration quality):")
    heston_vs_market = compute_metrics(heston_aligned, market_aligned)
    print(f"   MSE:  {heston_vs_market.mse:.6f}")
    print(f"   MAE:  {heston_vs_market.mae:.6f}")
    print(f"   RMSE: {heston_vs_market.rmse:.6f}")
    print(f"   MAE (vol points): {heston_vs_market.mae * 100:.2f}%")
    
    print("\n3. VAE vs Heston (direct comparison):")
    vae_vs_heston = compute_metrics(vae_aligned, heston_aligned)
    print(f"   MSE:  {vae_vs_heston.mse:.6f}")
    print(f"   MAE:  {vae_vs_heston.mae:.6f}")
    print(f"   RMSE: {vae_vs_heston.rmse:.6f}")
    
    # Winner summary
    print("\n" + "-" * 60)
    if vae_vs_market.mae < heston_vs_market.mae:
        improvement = (heston_vs_market.mae - vae_vs_market.mae) / heston_vs_market.mae * 100
        print(f"WINNER: VAE (MAE {improvement:.1f}% lower than Heston)")
    else:
        improvement = (vae_vs_market.mae - heston_vs_market.mae) / vae_vs_market.mae * 100
        print(f"WINNER: Heston (MAE {improvement:.1f}% lower than VAE)")
    print("=" * 60)
    
    # Save metrics
    metrics_summary = {
        "n_common_dates": len(common_dates),
        "date_range": [str(common_dates[0])[:10], str(common_dates[-1])[:10]],
        "vae_vs_market": vae_vs_market.to_dict(),
        "heston_vs_market": heston_vs_market.to_dict(),
        "vae_vs_heston": vae_vs_heston.to_dict(),
    }
    
    metrics_path = output_dir / "comparison_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")
    
    # Generate plots
    print("\nGenerating comparison plots...")
    save_comparison_plots(
        vae_surfaces=vae_aligned,
        heston_surfaces=heston_aligned,
        market_surfaces=market_aligned,
        dates=common_dates,
        grid_spec=grid_spec,
        output_dir=output_dir,
        n_samples=args.n_plot_samples,
    )
    
    print(f"\n✓ Comparison complete! Results in: {output_dir}")


if __name__ == "__main__":
    main()
