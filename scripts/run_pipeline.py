"""
Master pipeline: run the full data → train → eval → compare workflow.

Orchestrates all scripts in the correct order for one or more tickers.

Usage:
    # Full pipeline for one ticker (skip WRDS pull if data already exists)
    python scripts/run_pipeline.py --ticker AAPL

    # Pull fresh data and run everything
    python scripts/run_pipeline.py --ticker AAPL --pull

    # Multiple tickers
    python scripts/run_pipeline.py --ticker AAPL MSFT GOOG --pull

    # Only data prep (no training)
    python scripts/run_pipeline.py --ticker AAPL --stages data

    # Only training + eval
    python scripts/run_pipeline.py --ticker AAPL --stages train eval
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


VALID_STAGES = ("pull", "data", "train", "eval", "heston", "compare")


def run(cmd: list[str], label: str) -> None:
    """Run a subprocess and abort on failure."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n*** FAILED: {label} (exit code {result.returncode}) ***")
        sys.exit(result.returncode)


def parse_args():
    p = argparse.ArgumentParser(description="Run the full thesis pipeline")
    p.add_argument("--ticker", type=str, nargs="+", required=True,
                   help="One or more ticker symbols")
    p.add_argument("--start", type=str, default="2016-01-01")
    p.add_argument("--end", type=str, default="2025-12-31")
    p.add_argument("--stages", type=str, nargs="+",
                   default=list(VALID_STAGES),
                   help=f"Pipeline stages to run. Choices: {VALID_STAGES}")
    # Training knobs
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--latent_dim", type=int, default=8)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    py = sys.executable  # path to current python interpreter

    for stage in args.stages:
        if stage not in VALID_STAGES:
            print(f"Unknown stage: {stage}. Valid: {VALID_STAGES}")
            sys.exit(1)

    for ticker in args.ticker:
        ticker = ticker.upper()
        print(f"\n{'#'*60}")
        print(f"#  PIPELINE: {ticker}")
        print(f"{'#'*60}")

        raw_dir = "data/raw/ivydb"
        start, end = args.start, args.end

        # --- 1. Pull raw data from WRDS ---
        if "pull" in args.stages:
            run(
                [py, "scripts/pull_data.py",
                 "--ticker", ticker,
                 "--start", start, "--end", end,
                 "--output_dir", raw_dir],
                f"[{ticker}] Pull raw WRDS data",
            )

        # --- 2. Prepare processed data ---
        if "data" in args.stages:
            # VAE parquet
            run(
                [py, "scripts/prepare_vae_data.py",
                 "--ticker", ticker,
                 "--start", start, "--end", end,
                 "--raw_dir", raw_dir],
                f"[{ticker}] Prepare VAE parquet",
            )
            # Heston inputs (with rate/100 fix)
            run(
                [py, "scripts/prepare_heston_data.py",
                 "--ticker", ticker,
                 "--start", start, "--end", end,
                 "--raw_dir", raw_dir],
                f"[{ticker}] Prepare Heston inputs",
            )

        # --- 3. Train VAE ---
        if "train" in args.stages:
            parquet = f"data/processed/vae/parquet/{ticker}_vsurf_processed.parquet"
            out_dir = f"artifacts/train/{ticker}"
            run(
                [py, "scripts/train_vae.py",
                 "--parquet", parquet,
                 "--output_dir", out_dir,
                 "--epochs", str(args.epochs),
                 "--batch_size", str(args.batch_size),
                 "--latent_dim", str(args.latent_dim),
                 "--patience", str(args.patience),
                 "--seed", str(args.seed),
                 "--device", "auto"],
                f"[{ticker}] Train VAE",
            )

        # --- 4. Evaluate VAE ---
        if "eval" in args.stages:
            parquet = f"data/processed/vae/parquet/{ticker}_vsurf_processed.parquet"
            checkpoint = f"artifacts/train/{ticker}/vae_checkpoint.pt"
            eval_dir = f"artifacts/eval/{ticker}"
            run(
                [py, "scripts/eval_vae.py",
                 "--parquet", parquet,
                 "--checkpoint", checkpoint,
                 "--output_dir", eval_dir],
                f"[{ticker}] Evaluate VAE",
            )

        # --- 5. Calibrate Heston ---
        if "heston" in args.stages:
            heston_dir = "data/processed/heston"
            surface_dir = f"{heston_dir}/surfaces/{ticker}"
            # Use VAE test dates if available
            vae_dates = f"artifacts/eval/{ticker}/surfaces/vae_surface_dates.csv"
            dates_arg = ["--dates_from", vae_dates] if Path(vae_dates).exists() else []
            run(
                [py, "scripts/calibrate_heston.py",
                 "--ticker", ticker,
                 "--input_dir", heston_dir,
                 "--output_dir", surface_dir,
                 *dates_arg],
                f"[{ticker}] Calibrate Heston",
            )

        # --- 6. Compare VAE vs Heston ---
        if "compare" in args.stages:
            vae_dir = f"artifacts/eval/{ticker}/surfaces"
            heston_dir = f"data/processed/heston/surfaces/{ticker}"
            comp_dir = f"artifacts/comparison/{ticker}"
            run(
                [py, "scripts/compare_surfaces.py",
                 "--vae_dir", vae_dir,
                 "--heston_dir", heston_dir,
                 "--output_dir", comp_dir],
                f"[{ticker}] Compare VAE vs Heston",
            )

        print(f"\n{'#'*60}")
        print(f"#  DONE: {ticker}")
        print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
