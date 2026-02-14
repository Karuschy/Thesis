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
from typing import Literal

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


# ============================================================================
# Data Configuration
# ============================================================================

@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    # Data source
    ticker: str = "AAPL"
    value_col: str = "impl_volatility"
    
    # Chronological split ratios
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    # test_ratio = 1 - train_ratio - val_ratio
    
    # DataLoader settings
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = False
    
    # Normalization
    normalize: bool = True
    
    @property
    def test_ratio(self) -> float:
        return 1.0 - self.train_ratio - self.val_ratio
    
    def __post_init__(self):
        if self.train_ratio + self.val_ratio >= 1.0:
            raise ValueError("train_ratio + val_ratio must be < 1.0")


# ============================================================================
# Model Configuration
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for VAE model architecture."""
    # Architecture type: "mlp" (baseline) or "conv" (spatial)
    model_type: Literal["mlp", "conv"] = "mlp"
    
    # Latent space
    latent_dim: int = 8
    
    # --- MLP-specific ---
    hidden_dims: tuple[int, ...] = (256, 128)
    
    # --- Conv-specific ---
    channels: tuple[int, ...] = (32, 64, 128)
    fc_dim: int = 256
    batchnorm: bool = True
    
    # Input shape will be set from data
    in_shape: tuple[int, int, int] | None = None


# ============================================================================
# Training Configuration
# ============================================================================

@dataclass
class TrainConfig:
    """Configuration for training loop."""
    # Optimization
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 0.0
    
    # VAE-specific
    beta: float = 1.0  # KL weight (beta-VAE)
    
    # Early stopping
    patience: int | None = 20  # None = no early stopping
    
    # Device
    device: str = "auto"  # "auto", "cpu", or "cuda"
    
    def get_device(self) -> str:
        """Resolve 'auto' to actual device."""
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


# ============================================================================
# Evaluation Configuration
# ============================================================================

@dataclass
class EvalConfig:
    """Configuration for evaluation and masking experiments."""
    # Masking percentages for completion experiments
    mask_ratios: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75)
    
    # Use same mask across all samples (for reproducibility)
    fixed_mask_seed: int = 42
    
    # Number of samples to visualize (best/worst)
    n_viz_samples: int = 5


# ============================================================================
# Experiment Configuration (combines all)
# ============================================================================

@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str = "default"
    seed: int = 42
    
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    
    @property
    def experiment_dir(self) -> Path:
        """Directory for this specific experiment's artifacts."""
        return self.paths.artifacts_dir / self.name
    
    def ensure_dirs(self) -> None:
        """Create all necessary directories."""
        self.paths.ensure_dirs()
        self.experiment_dir.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Default configurations
# ============================================================================

def get_default_config(ticker: str = "AAPL") -> ExperimentConfig:
    """Get default experiment configuration for a ticker."""
    return ExperimentConfig(
        name=f"{ticker}_vae",
        data=DataConfig(ticker=ticker),
    )


def get_quick_test_config(ticker: str = "AAPL") -> ExperimentConfig:
    """Get a quick test configuration with fewer epochs."""
    return ExperimentConfig(
        name=f"{ticker}_test",
        data=DataConfig(ticker=ticker, batch_size=32),
        train=TrainConfig(epochs=10, patience=5),
    )
