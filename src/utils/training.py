"""
Training utilities for VAE models.

Provides training loop, epoch functions, and checkpoint management.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.vae_mlp import vae_loss


@dataclass
class TrainStats:
    """Statistics for one training epoch."""
    train_loss: float
    train_recon: float
    train_kl: float
    val_loss: float
    val_recon: float
    val_kl: float


@torch.no_grad()
def evaluate(
    model, 
    loader: DataLoader, 
    device: torch.device, 
    beta: float = 1.0
) -> tuple[float, float, float]:
    """
    Evaluate model on a data loader.
    
    Args:
        model: VAE model.
        loader: DataLoader to evaluate on.
        device: Torch device.
        beta: KL weight.
    
    Returns:
        (total_loss, recon_loss, kl_loss) averaged over all samples.
    """
    model.eval()
    total, rtot, ktot, n = 0.0, 0.0, 0.0, 0
    
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)
        recon, mu, logvar = model(x)
        loss, r, k = vae_loss(recon, x, mu, logvar, beta=beta)
        bs = x.size(0)
        total += float(loss) * bs
        rtot += float(r) * bs
        ktot += float(k) * bs
        n += bs
    
    return total / n, rtot / n, ktot / n


def train_epoch(
    model, 
    loader: DataLoader, 
    opt, 
    device: torch.device, 
    beta: float = 1.0
) -> tuple[float, float, float]:
    """
    Train for one epoch.
    
    Args:
        model: VAE model.
        loader: Training DataLoader.
        opt: Optimizer.
        device: Torch device.
        beta: KL weight.
    
    Returns:
        (total_loss, recon_loss, kl_loss) averaged over all samples.
    """
    model.train()
    total, rtot, ktot, n = 0.0, 0.0, 0.0, 0
    
    for batch in tqdm(loader, desc="train", leave=False):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)

        recon, mu, logvar = model(x)
        loss, r, k = vae_loss(recon, x, mu, logvar, beta=beta)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        bs = x.size(0)
        total += float(loss) * bs
        rtot += float(r) * bs
        ktot += float(k) * bs
        n += bs
    
    return total / n, rtot / n, ktot / n


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
    
    Returns:
        List of TrainStats for each completed epoch.
    """
    device = torch.device(device)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: list[TrainStats] = []
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for ep in range(1, epochs + 1):
        tr_l, tr_r, tr_k = train_epoch(model, train_loader, opt, device, beta=beta)
        va_l, va_r, va_k = evaluate(model, val_loader, device, beta=beta)
        history.append(TrainStats(tr_l, tr_r, tr_k, va_l, va_r, va_k))
        
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
        print(f"Epoch {ep:03d} | train: {tr_l:.6f} (recon {tr_r:.6f}, kl {tr_k:.6f}) | val: {va_l:.6f} (recon {va_r:.6f}, kl {va_k:.6f}){star}")
        
        # Early stopping
        if patience is not None and epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {ep}. Best epoch was {best_epoch} with val_loss={best_val_loss:.6f}")
            break
    
    if checkpoint_dir is not None:
        print(f"Best model saved at epoch {best_epoch} with val_loss={best_val_loss:.6f}")
    
    return history
