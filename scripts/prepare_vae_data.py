"""
Prepare VAE training data from raw vol-surface CSV.

Reads the raw IvyDB vol-surface CSV, applies sanity filters,
normalises types, and writes a clean parquet + metadata JSON
ready for the VAE dataloader.

Usage:
    python scripts/prepare_vae_data.py --ticker AAPL
    python scripts/prepare_vae_data.py --ticker MSFT --start 2016-01-01 --end 2025-12-31
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Prepare VAE training parquet")
    p.add_argument("--ticker", type=str, required=True)
    p.add_argument("--start", type=str, default="2016-01-01")
    p.add_argument("--end", type=str, default="2025-12-31")
    p.add_argument("--raw_dir", type=str, default="data/raw/ivydb",
                   help="Root for raw IvyDB files")
    p.add_argument("--output_dir", type=str, default="data/processed/vae",
                   help="Output directory for parquet + meta")
    return p.parse_args()


def main():
    args = parse_args()
    ticker = args.ticker.upper()

    # --- paths ---
    raw_dir = Path(args.raw_dir)
    vs_path = raw_dir / "vol_surface" / f"{ticker}_vsurfd_{args.start}_{args.end}.csv.gz"

    out_dir = Path(args.output_dir)
    parquet_dir = out_dir / "parquet"
    meta_dir = out_dir / "meta"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    out_parquet = parquet_dir / f"{ticker}_vsurf_processed.parquet"
    out_meta = meta_dir / f"{ticker}_vsurf_processed_meta.json"

    print(f"=== Preparing VAE data for {ticker} ===")
    print(f"  Input:  {vs_path}")
    print(f"  Output: {out_parquet}")

    if not vs_path.exists():
        raise FileNotFoundError(
            f"Raw vol-surface file not found: {vs_path}\n"
            f"Run  python scripts/pull_data.py --ticker {ticker}  first."
        )

    # --- load ---
    vs = pd.read_csv(vs_path, parse_dates=["date"])
    print(f"  Loaded {len(vs):,} rows")

    # --- required columns ---
    required = [
        "secid", "date", "days", "delta", "cp_flag",
        "impl_volatility", "impl_strike", "impl_premium", "dispersion",
    ]
    missing = [c for c in required if c not in vs.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # --- types ---
    vs["date"] = pd.to_datetime(vs["date"], errors="raise").dt.normalize()
    vs["secid"] = vs["secid"].astype("int32")
    vs["days"] = pd.to_numeric(vs["days"], errors="raise").astype("float32")
    vs["delta"] = pd.to_numeric(vs["delta"], errors="raise").astype("float32")
    vs["cp_flag"] = vs["cp_flag"].astype("category")
    for c in ["impl_volatility", "impl_strike", "impl_premium", "dispersion"]:
        vs[c] = pd.to_numeric(vs[c], errors="raise").astype("float32")

    # --- sanity filters ---
    mask = (
        (vs["days"] > 0)
        & np.isfinite(vs["impl_volatility"]) & (vs["impl_volatility"] > 0)
        & np.isfinite(vs["impl_strike"]) & (vs["impl_strike"] > 0)
        & np.isfinite(vs["impl_premium"]) & (vs["impl_premium"] >= 0)
        & np.isfinite(vs["dispersion"])
    )
    n_before = len(vs)
    vs = vs.loc[mask].copy()
    n_dropped = n_before - len(vs)
    print(f"  Dropped {n_dropped:,} rows with invalid values ({100 * n_dropped / n_before:.2f}%)")

    # --- deduplicate ---
    key = ["date", "days", "delta", "cp_flag"]
    dup_count = int(vs.duplicated(subset=key).sum())
    if dup_count > 0:
        vs = vs.drop_duplicates(subset=key, keep="first").copy()
        print(f"  Dropped {dup_count:,} duplicate rows")

    # --- sort ---
    vs = vs.sort_values(["date", "days", "delta", "cp_flag"]).reset_index(drop=True)

    # --- write parquet ---
    vs.to_parquet(out_parquet, index=False, engine="pyarrow", compression="zstd")
    print(f"  Wrote parquet: {out_parquet}  ({out_parquet.stat().st_size / 1e6:.1f} MB)")

    # --- write metadata ---
    meta = {
        "ticker": ticker,
        "secid_min": int(vs["secid"].min()),
        "secid_max": int(vs["secid"].max()),
        "date_min": str(vs["date"].min().date()),
        "date_max": str(vs["date"].max().date()),
        "n_rows": len(vs),
        "n_dates": int(vs["date"].nunique()),
        "n_days": int(vs["days"].nunique()),
        "n_delta": int(vs["delta"].nunique()),
        "cp_flag_levels": list(map(str, vs["cp_flag"].cat.categories)),
        "duplicate_rows_dropped": dup_count,
        "invalid_rows_dropped": n_dropped,
        "columns": list(vs.columns),
    }
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Wrote meta: {out_meta}")

    # --- summary ---
    print(f"\n  Dates: {meta['n_dates']:,}  ({meta['date_min']} → {meta['date_max']})")
    print(f"  Grid:  {meta['n_days']} maturities × {meta['n_delta']} deltas × {len(meta['cp_flag_levels'])} cp")
    print(f"  Rows:  {meta['n_rows']:,}")
    print("  Done.\n")


if __name__ == "__main__":
    main()
