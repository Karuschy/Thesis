"""
Train VAE script for volatility surface reconstruction.

Usage:
    python scripts/train_vae.py --parquet Data/processed/parquet/AAPL_vsurf_processed.parquet
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from src.data.dataloaders import create_dataloaders
from src.models.vae_mlp import MLPVAE
from src.utils.training import fit_vae, evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Train VAE on volatility surfaces")
    
    # Data args
    parser.add_argument("--parquet", type=str, required=True,
                        help="Path to processed parquet file")
    parser.add_argument("--value_col", type=str, default="impl_volatility",
                        help="Column to use as target")
    
    # Split args
    parser.add_argument("--train_ratio", type=float, default=0.80)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    
    # Model args
    parser.add_argument("--latent_dim", type=int, default=8,
                        help="Latent dimension (start small: 4-8)")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 128],
                        help="Hidden layer sizes for encoder/decoder")
    
    # Training args
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0,
                        help="KL weight (beta-VAE)")
    parser.add_argument("--patience", type=int, default=None,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--normalize", action="store_true", default=True,
                        help="Normalize data (fit on train)")
    parser.add_argument("--no_normalize", dest="normalize", action="store_false")
    
    # Output args
    parser.add_argument("--output_dir", type=str, default="artifacts/train",
                        help="Directory to save checkpoints and logs")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'cpu', 'cuda', or 'auto'")
    
    return parser.parse_args()


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
    
    # Create dataloaders
    print(f"Loading data from: {args.parquet}")
    bundle = create_dataloaders(
        parquet_path=args.parquet,
        value_col=args.value_col,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        normalize=args.normalize,
        return_date=False,
    )
    
    print(f"Data splits - Train: {len(bundle.train_dates)}, Val: {len(bundle.val_dates)}, Test: {len(bundle.test_dates)}")
    print(f"Input shape: {bundle.input_shape} (C, H, W)")
    print(f"Grid spec: {len(bundle.grid_spec.days_grid)} maturities, {len(bundle.grid_spec.delta_grid)} deltas")
    
    # Create model
    model = MLPVAE(
        in_shape=bundle.input_shape,
        latent_dim=args.latent_dim,
        hidden_dims=tuple(args.hidden_dims),
    )
    print(f"\nModel: latent_dim={args.latent_dim}, hidden_dims={args.hidden_dims}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    history = fit_vae(
        model=model,
        train_loader=bundle.train_loader,
        val_loader=bundle.val_loader,
        epochs=args.epochs,
        lr=args.lr,
        beta=args.beta,
        device=device,
        checkpoint_dir=output_dir,  # Will save best_model.pt here
        patience=args.patience,
    )
    
    # Find best epoch by validation loss (ELBO)
    best_epoch = min(range(len(history)), key=lambda i: history[i].val_loss)
    best_val_loss = history[best_epoch].val_loss
    print(f"\nBest validation ELBO: {best_val_loss:.6f} at epoch {best_epoch + 1}")
    
    # Save final checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "input_shape": bundle.input_shape,
        "grid_spec": {
            "days_grid": bundle.grid_spec.days_grid.tolist(),
            "delta_grid": bundle.grid_spec.delta_grid.tolist(),
            "cp_order": bundle.grid_spec.cp_order,
        },
        "scaler": {
            "mean": bundle.scaler.mean.tolist() if bundle.scaler else None,
            "std": bundle.scaler.std.tolist() if bundle.scaler else None,
        },
        "best_epoch": best_epoch + 1,
        "best_val_loss": best_val_loss,
        "final_epoch": args.epochs,
    }
    
    checkpoint_path = output_dir / "vae_checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to: {checkpoint_path}")
    
    # Save training history
    history_data = [
        {
            "epoch": i + 1,
            "train_loss": h.train_loss,
            "train_recon": h.train_recon,
            "train_kl": h.train_kl,
            "val_loss": h.val_loss,
            "val_recon": h.val_recon,
            "val_kl": h.val_kl,
        }
        for i, h in enumerate(history)
    ]
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history_data, f, indent=2)
    print(f"Training history saved to: {history_path}")
    
    # Final test evaluation (quick preview)
    print("\n--- Test Set Preview ---")
    test_loss, test_recon, test_kl = evaluate(model, bundle.test_loader, device, beta=args.beta)
    print(f"Test ELBO: {test_loss:.6f} (recon: {test_recon:.6f}, KL: {test_kl:.6f})")
    print("\nRun scripts/eval_vae.py for detailed test evaluation with plots.")


if __name__ == "__main__":
    main()
