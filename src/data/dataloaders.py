"""
DataLoader factory for volatility surface VAE training.

Creates train/val/test DataLoaders with:
- Chronological split (no data leakage)
- Optional normalization (fitted on train only)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.volsurface_grid import (
    GridSpec,
    VolSurfaceGridDataset,
    chronological_split_indices,
    load_parquet_tensorize,
)
from src.utils.scaler import ChannelStandardizer


@dataclass
class DataLoaderBundle:
    """Container for train/val/test loaders and metadata."""
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    scaler: Optional[ChannelStandardizer]
    grid_spec: GridSpec
    # Raw dates for each split (useful for plotting/eval)
    train_dates: np.ndarray
    val_dates: np.ndarray
    test_dates: np.ndarray
    # Input shape for model instantiation
    input_shape: Tuple[int, int, int]  # (C, H, W)


def create_dataloaders(
    parquet_path: str | Path,
    value_col: str = "impl_volatility",
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    batch_size: int = 32,
    normalize: bool = True,
    num_workers: int = 0,
    return_date: bool = False,
    pin_memory: bool = False,
) -> DataLoaderBundle:
    """
    Create train/val/test DataLoaders from a parquet file.

    Args:
        parquet_path: Path to the processed parquet file.
        value_col: Column name for the target variable (e.g., 'impl_volatility').
        train_ratio: Fraction of data for training (chronologically first).
        val_ratio: Fraction of data for validation (chronologically after train).
        batch_size: Batch size for all loaders.
        normalize: If True, fit ChannelStandardizer on train and apply to all splits.
        num_workers: Number of worker processes for data loading.
        return_date: If True, dataset returns (x, date) tuples.
        pin_memory: If True, enables pin_memory for CUDA transfers.

    Returns:
        DataLoaderBundle with loaders, scaler, grid spec, dates, and input shape.
    """
    # Load and tensorize
    X, dates, grid_spec = load_parquet_tensorize(
        parquet_path=parquet_path,
        value_col=value_col,
    )

    # Chronological split
    train_idx, val_idx, test_idx = chronological_split_indices(
        dates=dates,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]

    train_dates = dates[train_idx]
    val_dates = dates[val_idx]
    test_dates = dates[test_idx]

    # Normalization (fit on train only!)
    scaler: Optional[ChannelStandardizer] = None
    if normalize:
        scaler = ChannelStandardizer()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    # Create datasets
    train_ds = VolSurfaceGridDataset(X_train, train_dates, return_date=return_date)
    val_ds = VolSurfaceGridDataset(X_val, val_dates, return_date=return_date)
    test_ds = VolSurfaceGridDataset(X_test, test_dates, return_date=return_date)

    # Create loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,  # shuffle train only
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Input shape for model: (C, H, W)
    input_shape = (X.shape[1], X.shape[2], X.shape[3])

    return DataLoaderBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scaler=scaler,
        grid_spec=grid_spec,
        train_dates=train_dates,
        val_dates=val_dates,
        test_dates=test_dates,
        input_shape=input_shape,
    )


def create_test_loader_from_bundle(
    bundle: DataLoaderBundle,
    batch_size: Optional[int] = None,
    return_date: bool = True,
) -> DataLoader:
    """
    Create a fresh test loader from an existing bundle (e.g., with return_date=True).
    
    Useful when you trained with return_date=False but need dates for evaluation.
    """
    # We need to reload or re-create. For simplicity, just re-wrap the data.
    # This assumes the bundle's test_loader dataset has the raw X.
    test_ds = bundle.test_loader.dataset
    
    # Create new dataset with return_date
    new_ds = VolSurfaceGridDataset(
        X=test_ds.X,
        dates=test_ds.dates,
        return_date=return_date,
    )
    
    bs = batch_size if batch_size else bundle.test_loader.batch_size
    return DataLoader(new_ds, batch_size=bs, shuffle=False)
