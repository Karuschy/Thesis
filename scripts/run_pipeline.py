"""
Master pipeline: run the full data → train → eval → compare workflow.

Orchestrates all scripts in the correct order for one or more tickers.
Trains all 6 VAE variants per ticker by default.

Usage:
    # Full pipeline for one ticker (skip WRDS pull if data already exists)
    python scripts/run_pipeline.py --ticker AAPL

    # Pull fresh data and run everything
    python scripts/run_pipeline.py --ticker AAPL --stages pull data train eval heston compare

    # Multiple tickers
    python scripts/run_pipeline.py --ticker AAPL GOOGL NVDA TSLA F CNQ ENB

    # Only data prep (no training)
    python scripts/run_pipeline.py --ticker AAPL --stages data

    # Only training + eval
    python scripts/run_pipeline.py --ticker AAPL --stages train eval

    # Train only specific variants
    python scripts/run_pipeline.py --ticker AAPL --stages train eval --variants mlp mlp_log
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


VALID_STAGES = ("pull", "data", "train", "eval", "heston", "compare")

# All 6 VAE variants with their training configuration.
# Uses AAPL-tuned hyperparams (latent=24, lr=7e-4, beta=0.25, etc.)
VARIANTS = [
    {"name": "mlp",          "model_type": "mlp",  "log_transform": False, "arb_weight": 0.0},
    {"name": "mlp_log",      "model_type": "mlp",  "log_transform": True,  "arb_weight": 0.0},
    {"name": "mlp_log_arb",  "model_type": "mlp",  "log_transform": True,  "arb_weight": 100.0,
     "lambda_cal": 0.1, "lambda_but": 1.0},
    {"name": "conv",         "model_type": "conv", "log_transform": False, "arb_weight": 0.0},
    {"name": "conv_log",     "model_type": "conv", "log_transform": True,  "arb_weight": 0.0},
    {"name": "conv_log_arb", "model_type": "conv", "log_transform": True,  "arb_weight": 100.0,
     "lambda_cal": 0.1, "lambda_but": 1.0},
]

VALID_VARIANT_NAMES = [v["name"] for v in VARIANTS]

# Display name for compare_surfaces.py (matches MODEL_COLOURS keys)
VARIANT_DISPLAY = {
    "mlp": "MLP",
    "mlp_log": "MLP-log",
    "mlp_log_arb": "MLP-log-arb",
    "conv": "Conv",
    "conv_log": "Conv-log",
    "conv_log_arb": "Conv-log-arb",
}


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
    p.add_argument("--variants", type=str, nargs="+",
                   default=VALID_VARIANT_NAMES,
                   help=f"VAE variants to train/eval. Choices: {VALID_VARIANT_NAMES}")
    # Training knobs (AAPL-tuned defaults)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--latent_dim", type=int, default=24)
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--beta", type=float, default=0.25)
    p.add_argument("--patience", type=int, default=75)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--seed", type=int, default=42)
    # MLP-specific
    p.add_argument("--hidden_dims", type=int, nargs="+", default=[384, 192])
    # Conv-specific
    p.add_argument("--channels", type=int, nargs="+", default=[32, 64, 128])
    p.add_argument("--fc_dim", type=int, default=128)
    return p.parse_args()


def main():
    args = parse_args()
    py = sys.executable  # path to current python interpreter

    for stage in args.stages:
        if stage not in VALID_STAGES:
            print(f"Unknown stage: {stage}. Valid: {VALID_STAGES}")
            sys.exit(1)

    # Filter to requested variants
    selected_variants = [v for v in VARIANTS if v["name"] in args.variants]
    if not selected_variants:
        print(f"No valid variants selected. Choices: {VALID_VARIANT_NAMES}")
        sys.exit(1)

    for ticker in args.ticker:
        ticker = ticker.upper()
        print(f"\n{'#'*60}")
        print(f"#  PIPELINE: {ticker}")
        print(f"#  Variants: {[v['name'] for v in selected_variants]}")
        print(f"{'#'*60}")

        raw_dir = "data/raw/ivydb"
        start, end = args.start, args.end
        parquet = f"data/processed/vae/parquet/{ticker}_vsurf_processed.parquet"

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

        # --- 3. Train all VAE variants ---
        if "train" in args.stages:
            for variant in selected_variants:
                vname = variant["name"]
                train_dir = f"artifacts/train/{ticker}/{vname}"

                # Build train command
                cmd = [
                    py, "scripts/train_vae.py",
                    "--parquet", parquet,
                    "--output_dir", train_dir,
                    "--model_type", variant["model_type"],
                    "--latent_dim", str(args.latent_dim),
                    "--epochs", str(args.epochs),
                    "--batch_size", str(args.batch_size),
                    "--lr", str(args.lr),
                    "--beta", str(args.beta),
                    "--patience", str(args.patience),
                    "--weight_decay", str(args.weight_decay),
                    "--seed", str(args.seed),
                    "--device", "auto",
                ]

                # MLP vs Conv architecture args
                if variant["model_type"] == "mlp":
                    cmd += ["--hidden_dims"] + [str(d) for d in args.hidden_dims]
                else:
                    cmd += ["--channels"] + [str(c) for c in args.channels]
                    cmd += ["--fc_dim", str(args.fc_dim)]

                # Log transform
                if variant["log_transform"]:
                    cmd.append("--log_transform")

                # Arb penalty
                if variant["arb_weight"] > 0:
                    cmd += [
                        "--arb_weight", str(variant["arb_weight"]),
                        "--lambda_cal", str(variant.get("lambda_cal", 0.1)),
                        "--lambda_but", str(variant.get("lambda_but", 1.0)),
                    ]

                run(cmd, f"[{ticker}] Train VAE — {vname}")

        # --- 4. Evaluate all VAE variants ---
        if "eval" in args.stages:
            for variant in selected_variants:
                vname = variant["name"]
                checkpoint = f"artifacts/train/{ticker}/{vname}/vae_checkpoint.pt"
                eval_dir = f"artifacts/eval/{ticker}/{vname}"

                if not Path(checkpoint).exists():
                    print(f"  SKIP eval for {vname}: no checkpoint at {checkpoint}")
                    continue

                run(
                    [py, "scripts/eval_vae.py",
                     "--parquet", parquet,
                     "--checkpoint", checkpoint,
                     "--output_dir", eval_dir],
                    f"[{ticker}] Evaluate VAE — {vname}",
                )

        # --- 5. Calibrate Heston ---
        if "heston" in args.stages:
            heston_input = "data/processed/heston"
            heston_out = f"data/processed/heston/surfaces"

            # Use VAE test dates from first available variant for date alignment
            vae_dates_file = None
            for variant in selected_variants:
                candidate = (
                    f"artifacts/eval/{ticker}/{variant['name']}"
                    f"/surfaces/vae_surface_dates.csv"
                )
                if Path(candidate).exists():
                    vae_dates_file = candidate
                    break

            dates_arg = ["--dates_from", vae_dates_file] if vae_dates_file else []

            # Also pass grid_spec from first available variant
            grid_arg = []
            for variant in selected_variants:
                candidate = (
                    f"artifacts/eval/{ticker}/{variant['name']}"
                    f"/surfaces/grid_spec.json"
                )
                if Path(candidate).exists():
                    grid_arg = ["--grid_spec", candidate]
                    break

            run(
                [py, "scripts/calibrate_heston.py",
                 "--ticker", ticker,
                 "--input_dir", heston_input,
                 "--output_dir", heston_out,
                 *dates_arg,
                 *grid_arg],
                f"[{ticker}] Calibrate Heston",
            )

        # --- 6. Compare all variants vs Heston ---
        if "compare" in args.stages:
            heston_dir = "data/processed/heston/surfaces"
            comp_dir = f"artifacts/comparison/{ticker}"

            # Build --vae_dir entries for each evaluated variant
            vae_dir_args = []
            for variant in selected_variants:
                surf_dir = f"artifacts/eval/{ticker}/{variant['name']}/surfaces"
                if Path(surf_dir).exists():
                    display = VARIANT_DISPLAY.get(variant["name"], variant["name"])
                    vae_dir_args += ["--vae_dir", f"{display}={surf_dir}"]

            if not vae_dir_args:
                print(f"  SKIP compare: no evaluated variants for {ticker}")
            else:
                run(
                    [py, "scripts/compare_surfaces.py",
                     *vae_dir_args,
                     "--heston_dir", heston_dir,
                     "--ticker", ticker,
                     "--output_dir", comp_dir],
                    f"[{ticker}] Compare all variants vs Heston",
                )

        print(f"\n{'#'*60}")
        print(f"#  DONE: {ticker}")
        print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
