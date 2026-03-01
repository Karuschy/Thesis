"""
Training utilities for VAE models.

Provides training loop, epoch functions, and checkpoint management.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.vae_mlp import vae_loss
from src.utils.arbitrage import compute_arb_penalty
from src.utils.scaler import ChannelStandardizer


@dataclass
class TrainStats:
    """Statistics for one training epoch."""
    train_loss: float
    train_recon: float
    train_kl: float
    val_loss: float
    val_recon: float
    val_kl: float
    train_arb: float = 0.0
    val_arb: float = 0.0


@torch.no_grad()
def evaluate(
    model, 
    loader: DataLoader, 
    device: torch.device, 
    beta: float = 1.0,
    *,
    scaler: Optional[ChannelStandardizer] = None,
    days_grid: Optional[torch.Tensor] = None,
    arb_weight: float = 0.0,
    lambda_cal: float = 0.1,
    lambda_but: float = 1.0,
) -> tuple[float, float, float, float]:
    """
    Evaluate model on a data loader.
    
    Args:
        model: VAE model.
        loader: DataLoader to evaluate on.
        device: Torch device.
        beta: KL weight.
        scaler: If provided (with arb_weight > 0), used to inverse-transform
                reconstructions for arbitrage penalty computation.
        days_grid: [H] maturity grid as float tensor on device (for calendar penalty).
        arb_weight: Overall weight for arbitrage penalty. 0 = disabled.
        lambda_cal: Relative weight for calendar penalty within arb term.
        lambda_but: Relative weight for butterfly penalty within arb term.
    
    Returns:
        (total_loss, recon_loss, kl_loss, arb_penalty) averaged over all samples.
    """
    model.eval()
    total, rtot, ktot, atot, n = 0.0, 0.0, 0.0, 0.0, 0
    
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device, non_blocking=True)
        recon, mu, logvar = model(x)
        loss, r, k = vae_loss(recon, x, mu, logvar, beta=beta)

        arb_pen = 0.0
        if arb_weight > 0 and scaler is not None and days_grid is not None:
            recon_orig = scaler.inverse_transform(recon)
            combined, _, _ = compute_arb_penalty(
                recon_orig, days_grid, lambda_cal=lambda_cal, lambda_but=lambda_but
            )
            arb_pen = float(combined)
            loss = loss + arb_weight * combined
        
        bs = x.size(0)
        total += float(loss) * bs
        rtot += float(r) * bs
        ktot += float(k) * bs
        atot += float(arb_pen) * bs
        n += bs
    
    return total / n, rtot / n, ktot / n, atot / n


def train_epoch(
    model, 
    loader: DataLoader, 
    opt, 
    device: torch.device, 
    beta: float = 1.0,
    *,
    scaler: Optional[ChannelStandardizer] = None,
    days_grid: Optional[torch.Tensor] = None,
    arb_weight: float = 0.0,
    lambda_cal: float = 0.1,
    lambda_but: float = 1.0,
) -> tuple[float, float, float, float]:
    """
    Train for one epoch.
    
    Args:
        model: VAE model.
        loader: Training DataLoader.
        opt: Optimizer.
        device: Torch device.
        beta: KL weight.
        scaler: If provided (with arb_weight > 0), used to inverse-transform
                reconstructions for arbitrage penalty computation.
        days_grid: [H] maturity grid as float tensor on device (for calendar penalty).
        arb_weight: Overall weight for arbitrage penalty. 0 = disabled.
        lambda_cal: Relative weight for calendar penalty within arb term.
        lambda_but: Relative weight for butterfly penalty within arb term.
    
    Returns:
        (total_loss, recon_loss, kl_loss, arb_penalty) averaged over all samples.
    """
    model.train()
    total, rtot, ktot, atot, n = 0.0, 0.0, 0.0, 0.0, 0
    
    for batch in tqdm(loader, desc="train", leave=False):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device, non_blocking=True)

        recon, mu, logvar = model(x)
        loss, r, k = vae_loss(recon, x, mu, logvar, beta=beta)

        arb_pen = 0.0
        if arb_weight > 0 and scaler is not None and days_grid is not None:
            recon_orig = scaler.inverse_transform(recon)
            combined, _, _ = compute_arb_penalty(
                recon_orig, days_grid, lambda_cal=lambda_cal, lambda_but=lambda_but
            )
            arb_pen = float(combined)
            loss = loss + arb_weight * combined

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        bs = x.size(0)
        total += float(loss) * bs
        rtot += float(r) * bs
        ktot += float(k) * bs
        atot += float(arb_pen) * bs
        n += bs
    
    return total / n, rtot / n, ktot / n, atot / n


def fit_vae(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    beta: float = 1.0,
    device: str | torch.device = "cpu",
    checkpoint_dir: Optional[str | Path] = None,
    patience: Optional[int] = None,
    *,
    scaler: Optional[ChannelStandardizer] = None,
    days_grid: Optional[torch.Tensor] = None,
    arb_weight: float = 0.0,
    lambda_cal: float = 0.1,
    lambda_but: float = 1.0,
) -> list[TrainStats]:
    """
    Train the VAE model with early stopping and checkpointing.
    
    Args:
        model: VAE model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        epochs: Maximum number of training epochs.
        lr: Learning rate for Adam optimizer.
        weight_decay: L2 regularization weight for Adam optimizer.
        beta: KL divergence weight (beta-VAE).
        device: Device to train on ('cpu', 'cuda', or torch.device).
        checkpoint_dir: If provided, save best model checkpoint here.
        patience: Early stopping patience (epochs without improvement). None = disabled.
        scaler: ChannelStandardizer for inverse-transforming to raw IV (arb penalty).
        days_grid: [H] maturity grid as float tensor (for calendar penalty).
        arb_weight: Overall weight for arbitrage penalty. 0.0 = disabled (baseline).
        lambda_cal: Relative weight for calendar penalty.
        lambda_but: Relative weight for butterfly penalty.
    
    Returns:
        List of TrainStats for each completed epoch.
    """
    device = torch.device(device)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Prepare arb penalty kwargs (keyword-only args forwarded to train_epoch / evaluate)
    arb_kwargs = dict(
        scaler=scaler,
        days_grid=days_grid,
        arb_weight=arb_weight,
        lambda_cal=lambda_cal,
        lambda_but=lambda_but,
    )

    history: list[TrainStats] = []
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if arb_weight > 0:
        print(f"  Arbitrage penalty enabled: arb_weight={arb_weight}, lambda_cal={lambda_cal}, lambda_but={lambda_but}")
    
    for ep in range(1, epochs + 1):
        tr_l, tr_r, tr_k, tr_a = train_epoch(model, train_loader, opt, device, beta=beta, **arb_kwargs)
        va_l, va_r, va_k, va_a = evaluate(model, val_loader, device, beta=beta, **arb_kwargs)
        history.append(TrainStats(tr_l, tr_r, tr_k, va_l, va_r, va_k, tr_a, va_a))
        
        # Check for improvement
        improved = va_l < best_val_loss
        if improved:
            best_val_loss = va_l
            best_epoch = ep
            epochs_without_improvement = 0
            
            # Save best checkpoint
            if checkpoint_dir is not None:
                torch.save({
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "val_loss": va_l,
                }, checkpoint_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1
        
        star = " *" if improved else ""
        arb_str = f", arb {tr_a:.6f}" if arb_weight > 0 else ""
        print(f"Epoch {ep:03d} | train: {tr_l:.6f} (recon {tr_r:.6f}, kl {tr_k:.6f}{arb_str}) | val: {va_l:.6f} (recon {va_r:.6f}, kl {va_k:.6f}){star}")
        
        # Early stopping
        if patience is not None and epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {ep}. Best epoch was {best_epoch} with val_loss={best_val_loss:.6f}")
            break
    
    if checkpoint_dir is not None:
        print(f"Best model saved at epoch {best_epoch} with val_loss={best_val_loss:.6f}")
    
    return history
