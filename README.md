# VAE-based Implied Volatility Surface Modeling

A thesis project comparing **Variational Autoencoder (VAE)** reconstruction of implied volatility surfaces against the traditional **Heston stochastic volatility model** benchmark.

## Overview

This project implements:
1. **Data Pipeline**: Pull and process options data from WRDS IvyDB
2. **VAE Model**: MLP-based VAE for volatility surface reconstruction
3. **Heston Benchmark**: QuantLib-based Heston model calibration
4. **Comparison Framework**: Side-by-side evaluation of both approaches

The goal is to evaluate whether a data-driven VAE approach can match or exceed the accuracy of parametric Heston model calibration for implied volatility surface modeling.

## Installation

### Prerequisites
- Python 3.10+
- WRDS account (for data access)

### Setup

```bash
# Clone repository
git clone <repository-url>
cd Thesis

# Install dependencies
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

### Key Dependencies
- `torch>=2.0` - Deep learning framework
- `QuantLib>=1.30` - Quantitative finance library for Heston pricing
- `wrds` - WRDS data access
- `pandas`, `numpy`, `polars` - Data manipulation
- `matplotlib`, `plotly` - Visualization

## Project Structure

```
Thesis/
├── README.md
├── pyproject.toml
├── artifacts/                    # Model outputs
│   ├── train/                   # Training checkpoints
│   │   └── best_model.pt
│   ├── eval/                    # Evaluation outputs
│   │   └── surfaces/            # VAE generated surfaces
│   └── comparison/              # VAE vs Heston comparison
├── data/
│   ├── raw/                     # Raw WRDS data
│   │   └── ivydb/
│   ├── processed/
│   │   ├── heston/             # Heston calibration data
│   │   │   ├── inputs/         # Calibration inputs
│   │   │   └── surfaces/       # Generated surfaces
│   │   └── vae/                # VAE training data
│   │       ├── meta/
│   │       └── parquet/
│   └── logs/
├── notebooks/
│   ├── data_prep/              # Data preparation notebooks
│   ├── experiments/            # Model experiments
│   └── exploration/            # Data exploration
├── scripts/                    # CLI scripts
│   ├── train_vae.py
│   ├── eval_vae.py
│   └── compare_surfaces.py
└── src/                        # Source code
    ├── config.py
    ├── data/                   # Data loading utilities
    ├── models/                 # Model implementations
    │   ├── vae_mlp.py         # VAE model
    │   └── heston/            # Heston model
    │       ├── bs.py          # Black-Scholes utilities
    │       ├── pricer.py      # Heston pricing
    │       └── calibrate.py   # Heston calibration
    └── utils/                  # Training/evaluation utilities
```

## Workflow

The complete pipeline consists of 4 stages. Follow these in order:

### Stage 1: Data Preparation

Pull and process options data from WRDS IvyDB.

```
📁 notebooks/data_prep/
```

| Order | Notebook | Description |
|-------|----------|-------------|
| 1 | `pull_vol_surface.ipynb` | Pull implied volatility surfaces from WRDS |
| 2 | `pull_heston_inputs.ipynb` | Pull spot prices, zero curves, forward prices |
| 3 | `build_heston_inputs.ipynb` | Merge data and convert delta → strike |
| 4 | `validate_data.ipynb` | Validate data quality and coverage |

**Outputs:**
- `data/processed/vae/parquet/AAPL_vsurf_processed.parquet` - VAE training data
- `data/processed/heston/inputs/AAPL_heston_inputs.parquet` - Heston calibration inputs

### Stage 2: VAE Training & Evaluation

Train and evaluate the VAE model.

```bash
# Train VAE
python scripts/train_vae.py \
    --parquet data/processed/vae/parquet/AAPL_vsurf_processed.parquet \
    --epochs 100 \
    --latent_dim 8 \
    --hidden_dims 256 128

# Evaluate and save surfaces
python scripts/eval_vae.py \
    --checkpoint artifacts/train/best_model.pt \
    --parquet data/processed/vae/parquet/AAPL_vsurf_processed.parquet
```

**Outputs:**
- `artifacts/train/best_model.pt` - Trained model checkpoint
- `artifacts/eval/surfaces/vae_surfaces.npy` - VAE reconstructed surfaces
- `artifacts/eval/surfaces/market_surfaces.npy` - Original market surfaces
- `artifacts/eval/test_metrics.json` - Evaluation metrics

### Stage 3: Heston Calibration

Calibrate Heston model and generate benchmark surfaces.

```
📁 notebooks/experiments/
```

| Notebook | Description |
|----------|-------------|
| `heston_calibration.ipynb` | Run Heston calibration for all dates |
| `test_heston_calibration.ipynb` | Unit tests and debugging for calibration |

**Outputs:**
- `data/processed/heston/surfaces/AAPL_heston_surfaces.npy` - Heston IV surfaces
- `data/processed/heston/surfaces/AAPL_heston_params.csv` - Calibrated parameters
- `data/processed/heston/surfaces/AAPL_heston_surface_dates.csv` - Surface dates

### Stage 4: Comparison

Compare VAE and Heston surfaces.

```bash
# Run comparison script
python scripts/compare_surfaces.py \
    --vae_dir artifacts/eval/surfaces \
    --heston_dir data/processed/heston/surfaces
```

Or use the interactive notebook:
```
📁 notebooks/experiments/compare_vae_heston.ipynb
```

**Outputs:**
- `artifacts/comparison/comparison_metrics.json` - Summary metrics
- `artifacts/comparison/plots/` - Comparison visualizations

## Notebooks Reference

### Data Preparation (`notebooks/data_prep/`)

| Notebook | Purpose | Prerequisites |
|----------|---------|---------------|
| `pull_vol_surface.ipynb` | Pull IV surfaces from WRDS IvyDB | WRDS credentials |
| `pull_heston_inputs.ipynb` | Pull spot, rates, dividends | WRDS credentials |
| `build_heston_inputs.ipynb` | Build Heston calibration inputs | Steps 1-2 complete |
| `validate_data.ipynb` | Data quality checks | Any data files |

### Experiments (`notebooks/experiments/`)

| Notebook | Purpose | Prerequisites |
|----------|---------|---------------|
| `experiment.ipynb` | VAE training experiments | VAE training data |
| `heston_calibration.ipynb` | Run full Heston calibration | Heston inputs |
| `test_heston_calibration.ipynb` | Debug/test Heston functions | Heston inputs |
| `compare_vae_heston.ipynb` | Compare VAE vs Heston results | Both surfaces generated |

### Exploration (`notebooks/exploration/`)

| Notebook | Purpose | Prerequisites |
|----------|---------|---------------|
| `visualization.ipynb` | Visualize volatility surfaces | Processed data |

## Scripts Reference

### `train_vae.py`

Train the VAE model on volatility surfaces.

```bash
python scripts/train_vae.py --help

Options:
  --parquet PATH        Path to processed parquet file (required)
  --epochs INT          Number of training epochs (default: 100)
  --batch_size INT      Batch size (default: 32)
  --latent_dim INT      Latent dimension (default: 8)
  --hidden_dims INT...  Hidden layer sizes (default: 256 128)
  --lr FLOAT            Learning rate (default: 1e-3)
  --beta FLOAT          KL weight (default: 1.0)
  --patience INT        Early stopping patience
  --output_dir PATH     Output directory (default: artifacts/train)
```

### `eval_vae.py`

Evaluate trained VAE and save reconstructed surfaces.

```bash
python scripts/eval_vae.py --help

Options:
  --checkpoint PATH     Path to trained checkpoint (required)
  --parquet PATH        Path to processed parquet file (required)
  --output_dir PATH     Output directory (default: artifacts/eval)
  --n_plot_samples INT  Number of samples to plot (default: 5)
```

### `compare_surfaces.py`

Compare VAE and Heston surfaces.

```bash
python scripts/compare_surfaces.py --help

Options:
  --vae_dir PATH        VAE surfaces directory (default: artifacts/eval/surfaces)
  --heston_dir PATH     Heston surfaces directory (default: data/processed/heston/surfaces)
  --ticker STR          Ticker symbol (default: AAPL)
  --output_dir PATH     Output directory (default: artifacts/comparison)
  --n_plot_samples INT  Number of sample surfaces to plot (default: 5)
```

## Grid Specification

Both VAE and Heston use the same standardized IV surface grid:

| Dimension | Values |
|-----------|--------|
| **Channels** | 2 (Calls, Puts) |
| **Maturities** | 11 points: 30, 60, 90, 120, 150, 180, 252, 365, 547, 730, 1095 days |
| **Deltas** | 13 points: 0.20 to 0.80 in 0.05 increments |

**Total grid size:** 2 × 11 × 13 = 286 implied volatilities per surface

## Model Architecture

### VAE (MLPVAE)

```
Encoder: [B, 2, 11, 13] → Flatten(286) → 256 → 128 → (μ, log σ²) → z ∈ ℝ⁸
Decoder: z → 128 → 256 → 286 → Reshape [B, 2, 11, 13]
Loss: ELBO = Reconstruction (MSE) + β × KL divergence
```

### Heston Model

Five parameters calibrated per surface:
- `v0` - Initial variance
- `κ (kappa)` - Mean reversion speed
- `θ (theta)` - Long-term variance
- `σ (sigma)` - Volatility of variance
- `ρ (rho)` - Correlation with spot

Calibrated using QuantLib's `LevenbergMarquardt` optimizer with `HestonModelHelper`.

## Comparison Metrics

| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error (in IV units, e.g., 0.01 = 1%) |
| RMSE | Root Mean Squared Error |
| MSE | Mean Squared Error |
| Per-cell MAE | Error heatmap across grid |
| Time series | Error evolution over time |

## Quick Start

```bash
# 1. Install
pip install -e .

# 2. Prepare data (run notebooks in order)
#    notebooks/data_prep/pull_vol_surface.ipynb
#    notebooks/data_prep/pull_heston_inputs.ipynb
#    notebooks/data_prep/build_heston_inputs.ipynb

# 3. Train VAE
python scripts/train_vae.py --parquet data/processed/vae/parquet/AAPL_vsurf_processed.parquet

# 4. Evaluate VAE
python scripts/eval_vae.py --checkpoint artifacts/train/best_model.pt --parquet data/processed/vae/parquet/AAPL_vsurf_processed.parquet

# 5. Run Heston calibration
#    notebooks/experiments/heston_calibration.ipynb

# 6. Compare results
python scripts/compare_surfaces.py

# Or interactively:
#    notebooks/experiments/compare_vae_heston.ipynb
```

## License

[Add license information]

## Citation

[Add citation information for thesis]
