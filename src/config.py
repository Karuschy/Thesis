"""
Centralized configuration for VAE volatility surface experiments.

All hyperparameters and paths are defined here to avoid magic numbers
scattered across notebooks and scripts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


# ============================================================================
# Path Configuration
# ============================================================================

@dataclass
class PathConfig:
    """Paths for data and artifacts."""
    # Base directories
    project_root: Path = Path(".")
    data_dir: Path = field(default_factory=lambda: Path("./Data"))
    artifacts_dir: Path = field(default_factory=lambda: Path("./artifacts"))
    
    # Data subdirectories
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"
    
    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"
    
    @property
    def parquet_dir(self) -> Path:
        return self.processed_dir / "parquet"
    
    @property
    def meta_dir(self) -> Path:
        return self.processed_dir / "meta"
    
    # Artifact subdirectories
    @property
    def checkpoints_dir(self) -> Path:
        return self.artifacts_dir / "train"
    
    @property
    def eval_dir(self) -> Path:
        return self.artifacts_dir / "eval"
    
    def parquet_path(self, ticker: str) -> Path:
        """Get parquet path for a specific ticker."""
        return self.parquet_dir / f"{ticker}_vsurf_processed.parquet"
    
    def ensure_dirs(self) -> None:
        """Create all directories if they don't exist."""
        for d in [self.raw_dir, self.parquet_dir, self.meta_dir, 
                  self.checkpoints_dir, self.eval_dir]:
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
    # Architecture type
    model_type: Literal["grid", "pointwise"] = "grid"
    
    # Latent space
    latent_dim: int = 8
    
    # Encoder hidden layers
    hidden_dims: tuple[int, ...] = (256, 128)
    
    # Pointwise decoder hidden layers (only used if model_type="pointwise")
    decoder_hidden_dims: tuple[int, ...] = (64, 32)
    
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
