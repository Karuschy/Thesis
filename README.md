# VAE-based Implied Volatility Surface Modeling

A thesis project comparing **Variational Autoencoder** (MLP and Convolutional) reconstruction of implied volatility surfaces against the traditional **Heston stochastic volatility model**.

## Overview

This project evaluates whether data-driven VAE approaches can match or exceed the accuracy of parametric Heston model calibration for implied volatility surface modeling.

Three modelling approaches are compared on a common grid:

| Model | Type | Parameters | Description |
|-------|------|-----------|-------------|
| **MLP VAE** | Data-driven | ~261 K | Flatten-based encoder/decoder with fully-connected layers |
| **Conv VAE** | Data-driven | ~1.18 M | 2-D convolutional encoder/decoder preserving spatial structure |
| **Heston** | Parametric | 5 per surface | QuantLib LevenbergMarquardt calibration of the Heston SV model |

All models produce surfaces on the **same standardised grid** (2 × 11 × 17) so results are directly comparable.

## Grid Specification

| Dimension | Size | Values |
|-----------|------|--------|
| **Channels** | 2 | Calls (C), Puts (P) |
| **Maturities** | 11 | 10, 30, 60, 91, 122, 152, 182, 273, 365, 547, 730 days |
| **Deltas** | 17 | 0.10 to 0.90 in 0.05 steps |

**Total:** 2 × 11 × 17 = 374 implied volatilities per surface per date.

## Installation

### Prerequisites
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- WRDS account (for raw data access)

### Setup

```bash
git clone <repository-url>
cd Thesis

# With uv (recommended)
uv venv
uv pip install -e .

# Or with pip
python -m venv .venv
pip install -e .
```

### Key Dependencies
- `torch >= 2.0` — Deep learning framework
- `QuantLib >= 1.30` — Heston pricing & calibration
- `wrds` — WRDS IvyDB data access
- `pandas`, `numpy` — Data manipulation
- `matplotlib`, `plotly`, `ipywidgets` — Visualisation

## Project Structure

```
Thesis/
├── pyproject.toml
├── README.md
├── scripts/                         # CLI entry points (canonical pipeline)
│   ├── run_pipeline.py              #   Full end-to-end orchestrator
│   ├── pull_data.py                 #   Stage 1 — pull raw data from WRDS
│   ├── prepare_vae_data.py          #   Stage 2a — build VAE training parquet
│   ├── prepare_heston_data.py       #   Stage 2b — build Heston calibration inputs
│   ├── train_vae.py                 #   Stage 3 — train MLP or Conv VAE
│   ├── eval_vae.py                  #   Stage 4 — evaluate VAE & save surfaces
│   ├── calibrate_heston.py          #   Stage 5 — Heston calibration & surfaces
│   └── compare_surfaces.py          #   Stage 6 — 3-way comparison (tables + plots)
├── src/                             # Importable library
│   ├── config.py                    #   Global config, set_seed, ModelConfig
│   ├── data/
│   │   ├── dataloaders.py           #   VolSurfaceDataset, make_dataloaders
│   │   └── volsurface_grid.py       #   Grid construction utilities
│   ├── models/
│   │   ├── __init__.py              #   create_model() factory
│   │   ├── vae_mlp.py               #   MLPVAE architecture
│   │   ├── vae_conv.py              #   ConvVAE architecture
│   │   └── heston/                  #   Heston model
│   │       ├── bs.py                #   Black-Scholes utilities
│   │       ├── pricer.py            #   Heston pricing engine
│   │       └── calibrate.py         #   QuantLib calibration wrapper
│   └── utils/
│       ├── training.py              #   fit_vae, train_epoch, evaluate
│       ├── eval.py                  #   evaluate_vae, surface reconstruction
│       ├── scaler.py                #   ChannelStandardizer
│       └── masking.py               #   Surface masking utilities
├── notebooks/                       # Interactive exploration (optional)
│   ├── data_prep/                   #   Data pulling & validation
│   ├── experiments/                 #   Training, eval, comparison, interactive viz
│   └── exploration/                 #   Ad-hoc visualisation
├── data/                            # Raw & processed data (git-ignored)
│   ├── raw/ivydb/                   #   WRDS IvyDB downloads
│   └── processed/
│       ├── vae/                     #   VAE training data
│       └── heston/                  #   Heston inputs & surfaces
└── artifacts/                       # Outputs (git-ignored)
    ├── train/{mlp,conv}/            #   Checkpoints & training history
    ├── eval/{mlp,conv}/             #   Metrics, plots, surfaces
    └── comparison/                  #   3-way comparison results
        ├── comparison_metrics.json
        ├── plots/                   #   Heatmaps, time series, box plots, samples
        └── tables/                  #   CSV summaries, per-maturity/delta/date MAE
```

## Pipeline

The end-to-end pipeline can be run step-by-step or via the orchestrator script.

### Full Pipeline (one command)

```bash
python scripts/run_pipeline.py --ticker AAPL
```

### Step-by-Step

#### Stage 1 — Pull Raw Data

```bash
python scripts/pull_data.py --ticker AAPL
```

Downloads IV surfaces, spot prices, zero-rate curves, and forward data from WRDS IvyDB.

#### Stage 2 — Prepare Training Data

```bash
# VAE training parquet
python scripts/prepare_vae_data.py --ticker AAPL

# Heston calibration inputs (rate/100 correction applied automatically)
python scripts/prepare_heston_data.py --ticker AAPL
```

#### Stage 3 — Train VAE

```bash
# MLP VAE (default)
python scripts/train_vae.py \
    --parquet data/processed/vae/parquet/AAPL_vsurf_processed.parquet \
    --model_type mlp \
    --latent_dim 8 --hidden_dims 256 128 \
    --epochs 100 --batch_size 64 --patience 20 \
    --output_dir artifacts/train/mlp

# Convolutional VAE
python scripts/train_vae.py \
    --parquet data/processed/vae/parquet/AAPL_vsurf_processed.parquet \
    --model_type conv \
    --latent_dim 8 --channels 32 64 128 --fc_dim 256 \
    --epochs 100 --batch_size 64 --patience 20 \
    --output_dir artifacts/train/conv
```

<details>
<summary><b>All train_vae.py options</b></summary>

| Flag | Default | Description |
|------|---------|-------------|
| `--parquet` | *(required)* | Path to processed parquet file |
| `--model_type` | `mlp` | `mlp` or `conv` |
| `--latent_dim` | `8` | Latent space dimensionality |
| `--hidden_dims` | `256 128` | MLP hidden layer sizes |
| `--channels` | `32 64 128` | Conv encoder channel sizes |
| `--fc_dim` | `256` | Conv FC bottleneck size |
| `--no_batchnorm` | off | Disable batch normalisation (conv) |
| `--epochs` | `100` | Maximum training epochs |
| `--batch_size` | `32` | Mini-batch size |
| `--lr` | `1e-3` | Learning rate |
| `--beta` | `1.0` | KL divergence weight |
| `--patience` | `None` | Early stopping patience |
| `--weight_decay` | `0.0` | AdamW weight decay |
| `--normalize` | `True` | Channel-wise standardisation |
| `--output_dir` | `artifacts/train` | Checkpoint output directory |
| `--device` | `auto` | `cpu`, `cuda`, or `auto` |
| `--seed` | `42` | Random seed |

</details>

#### Stage 4 — Evaluate VAE

```bash
# MLP
python scripts/eval_vae.py \
    --checkpoint artifacts/train/mlp/best_model.pt \
    --parquet data/processed/vae/parquet/AAPL_vsurf_processed.parquet \
    --output_dir artifacts/eval/mlp

# Conv
python scripts/eval_vae.py \
    --checkpoint artifacts/train/conv/best_model.pt \
    --parquet data/processed/vae/parquet/AAPL_vsurf_processed.parquet \
    --output_dir artifacts/eval/conv
```

Outputs per model: test metrics JSON, latent embeddings, surface `.npy` arrays, grid spec, sample plots.

#### Stage 5 — Heston Calibration

```bash
python scripts/calibrate_heston.py \
    --ticker AAPL \
    --dates_from artifacts/eval/mlp/surfaces/vae_surface_dates.csv \
    --grid_spec artifacts/eval/mlp/surfaces/grid_spec.json \
    --output_dir data/processed/heston/surfaces
```

Calibrates Heston's 5 parameters (v₀, κ, θ, σ, ρ) per date using QuantLib. Uses `--dates_from` and `--grid_spec` to ensure surfaces match the VAE test set exactly.

<details>
<summary><b>All calibrate_heston.py options</b></summary>

| Flag | Default | Description |
|------|---------|-------------|
| `--ticker` | *(required)* | Ticker symbol |
| `--input_dir` | `data/processed/heston` | Heston inputs directory |
| `--output_dir` | `None` | Output dir (default: `{input_dir}/surfaces`) |
| `--dates_from` | `None` | CSV with `date` column to restrict calibration dates |
| `--grid_spec` | `None` | JSON grid spec to use for surface generation |
| `--min_fill` | `0.50` | Minimum cell fill rate to keep a surface |

</details>

#### Stage 6 — Three-Way Comparison

```bash
python scripts/compare_surfaces.py \
    --mlp_dir  artifacts/eval/mlp/surfaces \
    --conv_dir artifacts/eval/conv/surfaces \
    --heston_dir data/processed/heston/surfaces
```

Aligns all models to common dates, computes metrics, and saves:

**Tables** (CSV, in `artifacts/comparison/tables/`):
- `summary.csv` — MSE, MAE, RMSE per model vs market
- `pairwise.csv` — Model-to-model distances
- `mae_by_maturity_{C,P}.csv` — MAE by each of the 11 maturities
- `mae_by_delta_{C,P}.csv` — MAE by each of the 17 deltas
- `mae_timeseries.csv` — Per-date MAE for all common dates
- `cell_mae_{model}_{C,P}.csv` — Full 11×17 per-cell heatmap
- `report.txt` — Plain-text summary report

**Plots** (PNG, in `artifacts/comparison/plots/`):
- Error heatmaps, pairwise difference maps, time series, box plots, maturity/delta bar charts, sample-date surface comparisons

## Model Architectures

### MLP VAE

```
Encoder: [B, 2, 11, 17] → Flatten(374) → 256 → 128 → (μ, log σ²) → z ∈ ℝ⁸
Decoder: z → 128 → 256 → 374 → Reshape [B, 2, 11, 17]
Loss:    ELBO = Reconstruction (MSE) + β · KL(q(z|x) ‖ p(z))
```

### Conv VAE

```
Encoder: [B, 2, 11, 17] → Conv(32) → Conv(64) → Conv(128) → Flatten → FC(256) → (μ, log σ²) → z ∈ ℝ⁸
Decoder: z → FC(256) → Unflatten → Upsample+Conv(64) → Upsample+Conv(32) → Conv(2) → [B, 2, 11, 17]
Loss:    ELBO = Reconstruction (MSE) + β · KL(q(z|x) ‖ p(z))
```

Uses stride-2 down-sampling, bilinear up-sampling, and optional batch normalisation.

### Heston Stochastic Volatility

Five parameters calibrated per surface date:

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| Initial variance | v₀ | Starting variance level |
| Mean reversion speed | κ | Speed of reversion to θ |
| Long-term variance | θ | Long-run variance level |
| Vol-of-vol | σ | Volatility of the variance process |
| Correlation | ρ | Correlation between spot and variance |

Calibrated via QuantLib's `HestonModelHelper` + `LevenbergMarquardt` optimiser.

## Interactive Visualisation

An interactive notebook lets you explore all surfaces side-by-side with a date slider:

```
notebooks/experiments/interactive_surfaces.ipynb
```

Features 3-D plotly surfaces for Market, MLP VAE, Conv VAE, and Heston, plus a VAE-advantage overlay (blue = VAE closer, red = Heston closer) with per-date statistics.

## Notebooks Reference

Notebooks are provided for interactive exploration but are **not required** — all pipeline steps have equivalent CLI scripts.

| Notebook | Purpose | Script equivalent |
|----------|---------|-------------------|
| `data_prep/pull_vol_surface.ipynb` | Pull IV surfaces from WRDS | `pull_data.py` |
| `data_prep/pull_heston_inputs.ipynb` | Pull spot, rates, dividends | `pull_data.py` |
| `data_prep/build_heston_inputs.ipynb` | Build Heston calibration inputs | `prepare_heston_data.py` |
| `data_prep/validate_data.ipynb` | Data quality checks | — |
| `experiments/training_vae.ipynb` | Train VAE interactively | `train_vae.py` |
| `experiments/eval_vae.ipynb` | Evaluate VAE | `eval_vae.py` |
| `experiments/heston_calibration.ipynb` | Heston calibration | `calibrate_heston.py` |
| `experiments/compare_vae_heston.ipynb` | 2-way comparison (legacy) | `compare_surfaces.py` |
| `experiments/interactive_surfaces.ipynb` | Interactive 3-D surface explorer | — |
| `experiments/test_heston_calibration.ipynb` | Debug Heston functions | — |
| `exploration/visualization.ipynb` | Ad-hoc data visualisation | — |

## Data

- **Source:** WRDS IvyDB (AAPL options, 2016–2025)
- **Train/Val/Test split:** Chronological 80/10/10 (no leakage)
  - Train: ~1943 dates
  - Val: ~242 dates
  - Test: ~244 dates
- **Test period:** September 2024 — August 2025

## License

[Add license information]

## Citation

[Add citation information for thesis]
