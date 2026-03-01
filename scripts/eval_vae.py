"""
Evaluate trained VAE on the test set.

Loads the best checkpoint, runs comprehensive evaluation, and generates plots.

Usage:
    python scripts/eval_vae.py --checkpoint artifacts/train/vae_checkpoint.pt --parquet data/processed/vae/parquet/AAPL_vsurf_processed.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.dataloaders import create_dataloaders
from src.models import create_model
from src.utils.eval import (
    evaluate_vae,
    compute_per_cell_error,
    get_worst_reconstructions,
    get_best_reconstructions,
    DetailedEvalResult,
)
from src.utils.scaler import ChannelStandardizer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained VAE on test set")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint (.pt file)")
    parser.add_argument("--parquet", type=str, required=True,
                        help="Path to processed parquet file")
    parser.add_argument("--output_dir", type=str, default="artifacts/eval",
                        help="Directory to save evaluation outputs")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--n_plot_samples", type=int, default=5,
                        help="Number of samples to plot (best + worst)")
    
    return parser.parse_args()


def save_surface_comparison_plot(
    target: np.ndarray,
    recon: np.ndarray,
    error: np.ndarray,
    days_grid: np.ndarray,
    delta_grid: np.ndarray,
    cp_order: list,
    date_str: str,
    output_path: Path,
):
    """
    Save a comparison plot: original vs reconstructed vs error for both calls and puts.
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for c, cp in enumerate(cp_order):
        row = c
        
        # Original
        im1 = axes[row, 0].imshow(target[c], aspect='auto', origin='lower', cmap='viridis')
        axes[row, 0].set_title(f'{cp} - Original IV')
        axes[row, 0].set_ylabel('Maturity (days)')
        axes[row, 0].set_xlabel('Delta')
        axes[row, 0].set_yticks(range(len(days_grid)))
        axes[row, 0].set_yticklabels([int(d) for d in days_grid])
        axes[row, 0].set_xticks(range(0, len(delta_grid), 2))
        axes[row, 0].set_xticklabels([f'{d:.2f}' for d in delta_grid[::2]])
        plt.colorbar(im1, ax=axes[row, 0])
        
        # Reconstructed
        im2 = axes[row, 1].imshow(recon[c], aspect='auto', origin='lower', cmap='viridis')
        axes[row, 1].set_title(f'{cp} - Reconstructed IV')
        axes[row, 1].set_xlabel('Delta')
        axes[row, 1].set_yticks(range(len(days_grid)))
        axes[row, 1].set_yticklabels([int(d) for d in days_grid])
        axes[row, 1].set_xticks(range(0, len(delta_grid), 2))
        axes[row, 1].set_xticklabels([f'{d:.2f}' for d in delta_grid[::2]])
        plt.colorbar(im2, ax=axes[row, 1])
        
        # Error heatmap
        im3 = axes[row, 2].imshow(error[c], aspect='auto', origin='lower', cmap='Reds')
        axes[row, 2].set_title(f'{cp} - Absolute Error')
        axes[row, 2].set_xlabel('Delta')
        axes[row, 2].set_yticks(range(len(days_grid)))
        axes[row, 2].set_yticklabels([int(d) for d in days_grid])
        axes[row, 2].set_xticks(range(0, len(delta_grid), 2))
        axes[row, 2].set_xticklabels([f'{d:.2f}' for d in delta_grid[::2]])
        plt.colorbar(im3, ax=axes[row, 2])
    
    fig.suptitle(f'Volatility Surface Reconstruction - {date_str}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_error_heatmap(
    mean_error: np.ndarray,
    days_grid: np.ndarray,
    delta_grid: np.ndarray,
    cp_order: list,
    output_path: Path,
):
    """
    Save mean error heatmap across all test samples.
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for c, cp in enumerate(cp_order):
        im = axes[c].imshow(mean_error[c], aspect='auto', origin='lower', cmap='Reds')
        axes[c].set_title(f'{cp} - Mean Absolute Error')
        axes[c].set_ylabel('Maturity (days)')
        axes[c].set_xlabel('Delta')
        axes[c].set_yticks(range(len(days_grid)))
        axes[c].set_yticklabels([int(d) for d in days_grid])
        axes[c].set_xticks(range(0, len(delta_grid), 2))
        axes[c].set_xticklabels([f'{d:.2f}' for d in delta_grid[::2]])
        plt.colorbar(im, ax=axes[c])
    
    fig.suptitle('Mean Reconstruction Error Across Test Set', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    train_args = checkpoint["args"]
    input_shape = tuple(checkpoint["input_shape"])
    
    # Recreate dataloaders with same config (enable pin_memory for GPU)
    use_cuda = device.type == "cuda"
    print(f"Loading data from: {args.parquet}")
    bundle = create_dataloaders(
        parquet_path=args.parquet,
        value_col=train_args.get("value_col", "impl_volatility"),
        train_ratio=train_args.get("train_ratio", 0.80),
        val_ratio=train_args.get("val_ratio", 0.10),
        batch_size=train_args.get("batch_size", 32),
        normalize=train_args.get("normalize", True),
        use_log_transform=train_args.get("log_transform", False),
        return_date=True,  # Need dates for plotting
        pin_memory=use_cuda,
        num_workers=0,  # Windows spawn overhead kills GPU perf
    )
    
    print(f"Test set size: {len(bundle.test_dates)} samples")
    
    # Recreate model (supports both MLP and Conv)
    model_type = checkpoint.get("model_type", train_args.get("model_type", "mlp"))
    model = create_model(
        model_type=model_type,
        in_shape=input_shape,
        latent_dim=train_args.get("latent_dim", 8),
        hidden_dims=tuple(train_args.get("hidden_dims", [256, 128])),
        channels=tuple(train_args.get("channels", [32, 64, 128])),
        fc_dim=train_args.get("fc_dim", 256),
        batchnorm=not train_args.get("no_batchnorm", False),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    print(f"Model [{model_type}] loaded: latent_dim={model.latent_dim}")
    
    # Run detailed evaluation
    print("\nRunning detailed evaluation on test set...")
    result: DetailedEvalResult = evaluate_vae(
        model=model,
        loader=bundle.test_loader,
        device=device,
        beta=train_args.get("beta", 1.0),
        scaler=bundle.scaler,
        collect_details=True,
    )
    
    metrics = result.metrics
    print("\n" + "="*50)
    print("TEST SET METRICS")
    print("="*50)
    print(f"ELBO (total loss):     {metrics.elbo:.6f}")
    print(f"Reconstruction loss:   {metrics.recon_loss:.6f}")
    print(f"KL divergence:         {metrics.kl_loss:.6f}")
    print(f"MSE (normalized):      {metrics.mse:.6f}")
    print(f"MAE (normalized):      {metrics.mae:.6f}")
    print(f"RMSE (normalized):     {metrics.rmse:.6f}")
    
    if metrics.mse_original is not None:
        print("-"*50)
        print("Metrics in original IV space:")
        print(f"MSE (original):        {metrics.mse_original:.6f}")
        print(f"MAE (original):        {metrics.mae_original:.6f}")
        print(f"RMSE (original):       {metrics.rmse_original:.6f}")
        # MAE in basis points (IV is typically 0.xx, so *100 for %)
        print(f"MAE (vol points):      {metrics.mae_original * 100:.2f}%")
    print("="*50)
    
    # Save metrics to JSON
    metrics_path = output_dir / "test_metrics.json"
    metrics.to_json(metrics_path)
    print(f"\nMetrics saved to: {metrics_path}")
    
    # Generate plots
    print("\nGenerating plots...")
    
    days_grid = bundle.grid_spec.days_grid
    delta_grid = bundle.grid_spec.delta_grid
    cp_order = bundle.grid_spec.cp_order
    
    # Get reconstructions in original space if scaler exists
    if bundle.scaler is not None:
        recons_orig = bundle.scaler.inverse_transform(result.all_recons)
        targets_orig = bundle.scaler.inverse_transform(result.all_targets)
        errors_orig = torch.abs(recons_orig - targets_orig)
    else:
        recons_orig = result.all_recons
        targets_orig = result.all_targets
        errors_orig = result.all_errors
    
    # 1. Mean error heatmap
    mean_error = compute_per_cell_error(errors_orig, reduction="mean").numpy()
    save_error_heatmap(
        mean_error=mean_error,
        days_grid=days_grid,
        delta_grid=delta_grid,
        cp_order=cp_order,
        output_path=plots_dir / "mean_error_heatmap.png",
    )
    print(f"  Saved: mean_error_heatmap.png")
    
    # 2. Best and worst reconstructions
    best_idx = get_best_reconstructions(result, k=args.n_plot_samples)
    worst_idx = get_worst_reconstructions(result, k=args.n_plot_samples)
    
    for i, idx in enumerate(best_idx):
        date_str = str(result.dates[idx])[:10] if result.dates is not None else f"sample_{idx}"
        save_surface_comparison_plot(
            target=targets_orig[idx].numpy(),
            recon=recons_orig[idx].numpy(),
            error=errors_orig[idx].numpy(),
            days_grid=days_grid,
            delta_grid=delta_grid,
            cp_order=cp_order,
            date_str=f"BEST #{i+1} - {date_str}",
            output_path=plots_dir / f"best_{i+1}_{date_str}.png",
        )
    print(f"  Saved: {len(best_idx)} best reconstruction plots")
    
    for i, idx in enumerate(worst_idx):
        date_str = str(result.dates[idx])[:10] if result.dates is not None else f"sample_{idx}"
        save_surface_comparison_plot(
            target=targets_orig[idx].numpy(),
            recon=recons_orig[idx].numpy(),
            error=errors_orig[idx].numpy(),
            days_grid=days_grid,
            delta_grid=delta_grid,
            cp_order=cp_order,
            date_str=f"WORST #{i+1} - {date_str}",
            output_path=plots_dir / f"worst_{i+1}_{date_str}.png",
        )
    print(f"  Saved: {len(worst_idx)} worst reconstruction plots")
    
    # 3. Save latent representations for analysis
    latent_path = output_dir / "test_latents.pt"
    torch.save({
        "mus": result.all_mus,
        "dates": result.dates,
    }, latent_path)
    print(f"\nLatent representations saved to: {latent_path}")
    
    # 4. Save reconstructed surfaces for comparison with Heston
    # Save in same format as Heston: surfaces.npy + dates.csv
    surfaces_dir = output_dir / "surfaces"
    surfaces_dir.mkdir(exist_ok=True)
    
    # Save surfaces array [N, C, H, W] in original IV space
    import pandas as pd
    vae_surfaces = recons_orig.numpy()
    surfaces_path = surfaces_dir / "vae_surfaces.npy"
    np.save(surfaces_path, vae_surfaces)
    print(f"\nVAE surfaces saved to: {surfaces_path}")
    print(f"  Shape: {vae_surfaces.shape} (N, C, H, W)")
    
    # Save dates for matching with Heston
    dates_df = pd.DataFrame({"date": result.dates})
    dates_path = surfaces_dir / "vae_surface_dates.csv"
    dates_df.to_csv(dates_path, index=False)
    print(f"  Dates saved to: {dates_path}")
    
    # Also save targets (original market IVs) for comparison
    target_surfaces = targets_orig.numpy()
    targets_path = surfaces_dir / "market_surfaces.npy"
    np.save(targets_path, target_surfaces)
    print(f"  Market surfaces saved to: {targets_path}")
    
    # Save grid spec for reference
    grid_spec_info = {
        "days_grid": days_grid.tolist(),
        "delta_grid": delta_grid.tolist(),
        "cp_order": list(cp_order),
        "shape": list(vae_surfaces.shape),
    }
    grid_path = surfaces_dir / "grid_spec.json"
    with open(grid_path, "w") as f:
        json.dump(grid_spec_info, f, indent=2)
    print(f"  Grid spec saved to: {grid_path}")
    
    print(f"\n✓ Evaluation complete! Results in: {output_dir}")


if __name__ == "__main__":
    main()
