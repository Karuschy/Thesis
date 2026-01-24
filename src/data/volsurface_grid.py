from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _infer_cp_order(levels: List[str]) -> List[str]:
    """
    Prefer call then put if possible.
    Accepts levels like ['C','P'] or ['CALL','PUT'] etc.
    """
    up = [str(x).upper() for x in levels]
    call_candidates = {"C", "CALL"}
    put_candidates = {"P", "PUT"}
    call = [levels[i] for i, u in enumerate(up) if u in call_candidates]
    put  = [levels[i] for i, u in enumerate(up) if u in put_candidates]
    if len(call) == 1 and len(put) == 1:
        return [call[0], put[0]]
    # fallback: stable order
    return list(levels)


@dataclass(frozen=True)
class GridSpec:
    days_grid: np.ndarray       # shape [H]
    delta_grid: np.ndarray      # shape [W]  (abs delta)
    cp_order: List[str]         # len=2 


def load_parquet_tensorize(
    parquet_path: str | Path,
    value_col: str = "impl_volatility",
    delta_round: int = 6,
) -> Tuple[torch.Tensor, np.ndarray, GridSpec]:
    """
    Returns:
      X: torch.float32 [N, C, H, W]
      dates: np.datetime64 array [N]
      grid: GridSpec
    """
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    required = {"date", "days", "delta", "cp_flag", value_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in parquet: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    # safety: if someone forgot normalization earlier
    if df["delta"].abs().max() > 2:
        df["delta"] = df["delta"] / 100.0

    df["delta_abs"] = df["delta"].abs().round(delta_round).astype(np.float32)

    # grids
    days_grid = np.sort(df["days"].unique()).astype(np.float32)
    delta_grid = np.sort(df["delta_abs"].unique()).astype(np.float32)

    # cp order
    if str(df["cp_flag"].dtype) == "category":
        cp_levels = list(df["cp_flag"].cat.categories)
    else:
        cp_levels = sorted(df["cp_flag"].unique().tolist())
    cp_order = _infer_cp_order([str(x) for x in cp_levels])

    # dates
    dates = np.sort(df["date"].unique())
    N, C, H, W = len(dates), len(cp_order), len(days_grid), len(delta_grid)

    X = np.empty((N, C, H, W), dtype=np.float32)

    # build tensor per date/cp_flag
    for i, d in enumerate(dates):
        ddf = df[df["date"] == d]
        for j, cp in enumerate(cp_order):
            sdf = ddf[ddf["cp_flag"].astype(str) == str(cp)]
            mat = sdf.pivot(index="days", columns="delta_abs", values=value_col)
            mat = mat.reindex(index=days_grid, columns=delta_grid)

            if mat.isnull().any().any():
                # with your 100% coverage, this should never happen
                bad = mat.isnull().sum().sum()
                raise ValueError(f"Found {bad} missing points on date={d} cp={cp}")

            X[i, j] = mat.values.astype(np.float32)

    X_t = torch.from_numpy(X)  # [N,C,H,W]
    grid = GridSpec(days_grid=days_grid, delta_grid=delta_grid, cp_order=cp_order)
    return X_t, dates, grid


def chronological_split_indices(
    dates: np.ndarray,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split by date order (no leakage).
    Returns indices for train/val/test.
    """
    if not (0 < train_ratio < 1) or not (0 <= val_ratio < 1):
        raise ValueError("Bad ratios.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1.")

    order = np.argsort(dates)
    n = len(dates)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = order[:n_train]
    val_idx = order[n_train:n_train + n_val]
    test_idx = order[n_train + n_val:]
    return train_idx, val_idx, test_idx


class VolSurfaceGridDataset(Dataset):
    """
    Holds grid tensors [N,C,H,W] and returns x only (or x, date if requested).
    """
    def __init__(self, X: torch.Tensor, dates: np.ndarray, return_date: bool = False):
        if X.dim() != 4:
            raise ValueError(f"Expected X [N,C,H,W], got {tuple(X.shape)}")
        if len(dates) != X.shape[0]:
            raise ValueError("dates length must match X.shape[0]")
        self.X = X
        self.dates = dates
        # Convert dates to strings for DataLoader compatibility (datetime64 can't be collated)
        self.dates_str = np.array([str(d)[:10] for d in dates])
        self.return_date = return_date

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = self.X[idx]
        if self.return_date:
            return x, self.dates_str[idx]
        return x
