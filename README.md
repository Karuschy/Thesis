# Variational Autoencoder vs Heston Model for Implied Volatility Surface Construction

> **CS4490 Thesis** — Comparing data-driven VAE approaches against the parametric Heston stochastic volatility model for implied volatility surface reconstruction across 8 equity and index tickers.

**[Read the full paper (PDF)](TODO)**

## Overview

Implied volatility surfaces (IVS) are central to options pricing, hedging, and risk management. This thesis systematically compares **six VAE model variants** (MLP and Convolutional, with raw/log/arb-penalized configurations) against a **robust Heston benchmark** across 8 tickers spanning equities, energy, and volatility indices.

**Key contributions:**

1. Multi-architecture VAE comparison on high-resolution IVS (2 x 11 x 17 = 374 cells)
2. Soft no-arbitrage penalties reducing butterfly violations by 91-99% with minimal accuracy cost
3. Multi-ticker generalization study (8 assets with diverse dynamics)
4. Surface completion under random and structured missingness
5. Latent space explainability mapping VAE factors to financial quantities

## Data

- **Source:** WRDS OptionMetrics IvyDB (2016-2025)
- **Tickers:** AAPL, GOOGL, NVDA, TSLA, F, XOM, CVX, VIX
- **Grid:** 2 channels (call/put) x 11 maturities (10-730 days) x 17 deltas (0.10-0.90)
- **Split:** Chronological 80/10/10 (no leakage), 244 test dates per ticker

## Results

### Single-Ticker Accuracy (AAPL, test set)

| Model | MAE (vol pts) | RMSE (vol pts) | Butterfly % | Calendar % |
|-------|:---:|:---:|:---:|:---:|
| MLP-log | **1.09** | 1.72 | 23.5% | 0% |
| Conv | 1.12 | **1.62** | 23.7% | 0% |
| MLP | 1.14 | 1.64 | 27.5% | 0% |
| Conv-log | 1.15 | 1.76 | 24.3% | 0% |
| MLP-log-arb | 1.12 | 1.65 | **5.7%** | 0% |
| Conv-log-arb | 1.24 | 1.76 | **1.1%** | 0% |
| Heston (robust) | 1.28 | 2.19 | **0%** | 0% |
| *Market (ref)* | *--* | *--* | *2.5%* | *0.03%* |

### Cross-Ticker Comparison (best model per ticker)

| Ticker | Best VAE | VAE MAE | Heston MAE | Winner |
|--------|----------|:---:|:---:|---------|
| AAPL | MLP-log | **1.09** | 1.28 | VAE |
| GOOGL | MLP-log | **1.38** | 1.43 | VAE |
| NVDA | MLP-log | 1.80 | **1.64** | Heston |
| TSLA | MLP | 2.61 | **1.58** | Heston |
| F | MLP-log | **7.32** | 8.16 | VAE |
| XOM | MLP-log-arb | 1.75 | **1.57** | Heston |
| CVX | Conv-log-arb | **1.67** | 1.71 | Tie |
| VIX | MLP | **7.12** | 13.34 | VAE |

**Tally: VAE best 4/8, Heston best 3/8, tie 1/8.** The result is ticker-dependent.

### No-Arbitrage Penalty Effectiveness

Butterfly violation reduction with Conv-log-arb vs best baseline:

| Ticker | Conv-log-arb | Best Baseline | Market | Reduction |
|--------|:---:|:---:|:---:|:---:|
| AAPL | **1.1%** | 23.6% | 2.5% | -95% |
| GOOGL | **0.5%** | 19.9% | 3.5% | -97% |
| TSLA | **0.2%** | 19.1% | 5.6% | -99% |
| NVDA | **1.1%** | 25.9% | 6.7% | -96% |
| VIX | **3.5%** | 41.0% | 26.2% | -91% |

Conv-log-arb achieves sub-market butterfly rates on 5/8 tickers.

### Latent Space Dimensionality

Despite a 24-dimensional bottleneck, 80% of total KL divergence concentrates in just 3 latent dimensions across all tickers. The learned factors correspond to classical financial quantities (level, skew, term structure) and map to Heston parameters at |r| ~ 0.8.

## Installation

```bash
git clone <repository-url>
cd Thesis

# With uv (recommended)
uv venv && uv pip install -e .

# Or with pip
python -m venv .venv && pip install -e .
```

**Requirements:** Python 3.10+, PyTorch 2.0+, QuantLib 1.30+, WRDS account (for data access).

## Usage

### Full pipeline (single ticker)

```bash
python scripts/run_pipeline.py --ticker AAPL
```

### Step-by-step

```bash
# 1. Pull data from WRDS
python scripts/pull_data.py --ticker AAPL

# 2. Prepare training data
python scripts/prepare_vae_data.py --ticker AAPL
python scripts/prepare_heston_data.py --ticker AAPL

# 3. Train VAE (example: MLP with log transform + arb penalty)
python scripts/train_vae.py \
    --parquet data/processed/vae/parquet/AAPL_vsurf_processed.parquet \
    --model_type mlp --latent_dim 24 --hidden_dims 384 192 \
    --lr 7e-4 --beta 0.25 --batch_size 64 --epochs 500 --patience 75 \
    --log_transform --arb_weight 100 --lambda_but 1.0 --lambda_cal 0.1 \
    --output_dir artifacts/train/AAPL/mlp_log_arb

# 4. Evaluate
python scripts/eval_vae.py \
    --checkpoint artifacts/train/AAPL/mlp_log_arb/vae_checkpoint.pt \
    --parquet data/processed/vae/parquet/AAPL_vsurf_processed.parquet \
    --output_dir artifacts/eval/AAPL/mlp_log_arb

# 5. Heston calibration
python scripts/calibrate_heston.py --ticker AAPL \
    --dates_from artifacts/eval/AAPL/mlp_log_arb/surfaces/vae_surface_dates.csv

# 6. Compare all models
python scripts/compare_surfaces.py --ticker AAPL
```

## Project Structure

```
Thesis/
├── src/                        # Core library
│   ├── models/                 #   VAE architectures + Heston pricing
│   ├── data/                   #   Data loading and grid construction
│   └── utils/                  #   Training, evaluation, scaling, arbitrage
├── scripts/                    # CLI pipeline scripts
├── notebooks/
│   ├── tickers/{TICKER}/       #   Per-ticker comparison, validation, latent analysis
│   └── experiments/            #   Cross-ticker analysis, surface completion
├── data/                       # Raw & processed data (git-ignored)
└── artifacts/                  # Model outputs & results (git-ignored)
    ├── train/{TICKER}/{variant}/
    ├── eval/{TICKER}/{variant}/
    ├── comparison/{TICKER}/
    ├── validation/{TICKER}/
    └── completion/{TICKER}/
```

## Citation

```bibtex
TODO
```

## License

[Add license information]
