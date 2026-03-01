"""
Centralized configuration for VAE volatility surface experiments.

All hyperparameters and paths are defined here to avoid magic numbers
scattered across notebooks and scripts.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """
    Set random seeds for reproducibility.

    Covers Python stdlib, NumPy, and PyTorch (CPU + CUDA).
    Call once at the start of any script or notebook.

    Args:
        seed: Random seed.
        deterministic: If True, enforce fully deterministic cuDNN
            (slower).  If False (default), enable cuDNN benchmark
            autotuner for faster training on fixed-size inputs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = deterministic
    # benchmark=True lets cuDNN pick the fastest algorithm for fixed-size
    # inputs (our grids are always the same shape).
    torch.backends.cudnn.benchmark = not deterministic


# ============================================================================
# Path Configuration
# ============================================================================

@dataclass
class PathConfig:
    """Paths for data and artifacts."""
    # Base directories
    project_root: Path = Path(".")
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    artifacts_dir: Path = field(default_factory=lambda: Path("./artifacts"))
    
    # =========================================================================
    # Raw data directories (WRDS IvyDB)
    # =========================================================================
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"
    
    @property
    def ivydb_dir(self) -> Path:
        return self.raw_dir / "ivydb"
    
    @property
    def raw_vol_surface_dir(self) -> Path:
        return self.ivydb_dir / "vol_surface"
    
    @property
    def raw_security_price_dir(self) -> Path:
        return self.ivydb_dir / "security_price"
    
    @property
    def raw_zero_curve_dir(self) -> Path:
        return self.ivydb_dir / "zero_curve"
    
    @property
    def raw_std_option_price_dir(self) -> Path:
        return self.ivydb_dir / "std_option_price"
    
    # =========================================================================
    # Processed data directories
    # =========================================================================
    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"
    
    # VAE data
    @property
    def vae_dir(self) -> Path:
        return self.processed_dir / "vae"
    
    @property
    def vae_parquet_dir(self) -> Path:
        return self.vae_dir / "parquet"
    
    @property
    def vae_meta_dir(self) -> Path:
        return self.vae_dir / "meta"
    
    # Heston data
    @property
    def heston_dir(self) -> Path:
        return self.processed_dir / "heston"
    
    @property
    def heston_inputs_dir(self) -> Path:
        return self.heston_dir / "inputs"
    
    @property
    def heston_surfaces_dir(self) -> Path:
        return self.heston_dir / "surfaces"
    
    # =========================================================================
    # Artifact directories
    # =========================================================================
    @property
    def checkpoints_dir(self) -> Path:
        return self.artifacts_dir / "train"
    
    @property
    def eval_dir(self) -> Path:
        return self.artifacts_dir / "eval"
    
    # =========================================================================
    # Path helpers
    # =========================================================================
    def vae_parquet_path(self, ticker: str) -> Path:
        """Get VAE training parquet path for a specific ticker."""
        return self.vae_parquet_dir / f"{ticker}_vsurf_processed.parquet"
    
    def heston_inputs_path(self, ticker: str) -> Path:
        """Get Heston inputs parquet path for a specific ticker."""
        return self.heston_inputs_dir / f"{ticker}_heston_inputs.parquet"
    
    def heston_surface_path(self, ticker: str) -> Path:
        """Get Heston surface parquet path for a specific ticker."""
        return self.heston_surfaces_dir / f"{ticker}_heston_surface.parquet"
    
    # Legacy alias for backward compatibility
    def parquet_path(self, ticker: str) -> Path:
        """Alias for vae_parquet_path (backward compatibility)."""
        return self.vae_parquet_path(ticker)
    
    def ensure_dirs(self) -> None:
        """Create all directories if they don't exist."""
        dirs = [
            self.raw_vol_surface_dir, self.raw_security_price_dir,
            self.raw_zero_curve_dir, self.raw_std_option_price_dir,
            self.vae_parquet_dir, self.vae_meta_dir,
            self.heston_inputs_dir, self.heston_surfaces_dir,
            self.checkpoints_dir, self.eval_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
