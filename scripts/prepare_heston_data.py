"""
Build Heston calibration inputs from raw IvyDB data.

Merges vol surface, spot prices, zero curve, and forward prices,
then converts delta → strike using Black-Scholes.

**CRITICAL**: IvyDB zero-curve rates are in PERCENT (e.g. 5.42 = 5.42%).
This script divides by 100 before any computation.

Usage:
    python scripts/prepare_heston_data.py --ticker AAPL
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Delta → Strike conversion
# ---------------------------------------------------------------------------

def delta_to_strike(
    delta: float,
    S0: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    cp_flag: str,
) -> float:
    """
    Convert Black-Scholes delta (absolute, 0-1) to strike price K.

    Args:
        delta:   Absolute delta in (0, 1).
        S0:      Spot price.
        T:       Time to maturity in years.
        r:       Risk-free rate (DECIMAL, e.g. 0.05 for 5%).
        q:       Dividend / carry yield (decimal).
        sigma:   Implied volatility (decimal).
        cp_flag: "C" or "P".

    Returns:
        Strike price K, or NaN on failure.
    """
    if T <= 0 or sigma <= 0:
        return np.nan

    # IvyDB stores signed deltas: positive for calls, negative for puts.
    # The inversion formula needs absolute delta in (0, 1).
    delta = abs(delta)

    sqrt_T = np.sqrt(T)

    if cp_flag == "C":
        adj = delta * np.exp(q * T)
        if adj <= 0 or adj >= 1:
            return np.nan
        d1 = norm.ppf(adj)
    else:  # P
        adj = delta * np.exp(q * T)
        if adj <= 0 or adj >= 1:
            return np.nan
        d1 = -norm.ppf(adj)

    K = S0 * np.exp(-(d1 * sigma * sqrt_T - (r - q + 0.5 * sigma**2) * T))
    return K


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Build Heston calibration inputs")
    p.add_argument("--ticker", type=str, required=True)
    p.add_argument("--start", type=str, default="2016-01-01")
    p.add_argument("--end", type=str, default="2025-12-31")
    p.add_argument("--raw_dir", type=str, default="data/raw/ivydb")
    p.add_argument("--output_dir", type=str, default="data/processed/heston")
    # Carry-rate clipping bounds (decimal)
    p.add_argument("--q_cap", type=float, default=0.10,
                   help="Max dividend yield (decimal). Default 0.10 = 10%%.")
    p.add_argument("--q_floor", type=float, default=-0.05,
                   help="Min carry rate (decimal). Default -0.05.")
    return p.parse_args()


def main():
    args = parse_args()
    ticker = args.ticker.upper()

    raw = Path(args.raw_dir)
    vs_path = raw / "vol_surface" / f"{ticker}_vsurfd_{args.start}_{args.end}.csv.gz"
    px_path = raw / "security_price" / f"{ticker}_underlying_{args.start}_{args.end}.csv.gz"
    zc_path = raw / "zero_curve" / f"zero_curve_{args.start}_{args.end}.csv.gz"
    stdop_path = raw / "std_option_price" / f"{ticker}_stdopd_{args.start}_{args.end}.csv.gz"

    for p in [vs_path, px_path, zc_path, stdop_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing: {p}\nRun  python scripts/pull_data.py --ticker {ticker}  first."
            )

    out_dir = Path(args.output_dir)
    inputs_dir = out_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Building Heston inputs for {ticker} ===\n")

    # ------------------------------------------------------------------
    # 1. Load raw data
    # ------------------------------------------------------------------
    vs = pd.read_csv(vs_path, parse_dates=["date"])
    px = pd.read_csv(px_path, parse_dates=["date"])
    zc = pd.read_csv(zc_path, parse_dates=["date"])
    stdop = pd.read_csv(stdop_path, parse_dates=["date"])
    print(f"  Vol surface:   {vs.shape}")
    print(f"  Prices:        {px.shape}")
    print(f"  Zero curve:    {zc.shape}")
    print(f"  Std options:   {stdop.shape}")

    # ------------------------------------------------------------------
    # 2. Normalise vol surface
    # ------------------------------------------------------------------
    if vs["delta"].abs().max() > 2:
        vs["delta"] = vs["delta"] / 100.0
    vs["cp_flag"] = vs["cp_flag"].str.upper()
    vs["T"] = vs["days"] / 365.0

    # ------------------------------------------------------------------
    # 3. Market data components
    # ------------------------------------------------------------------
    # Spot (one per date)
    spot_df = px[["date", "close"]].drop_duplicates(subset=["date"]).copy()
    spot_df.columns = ["date", "S0"]

    # *** CRITICAL FIX: IvyDB rates are in PERCENT → divide by 100 ***
    rate_df = zc[["date", "days", "rate"]].drop_duplicates(subset=["date", "days"]).copy()
    rate_df["r"] = rate_df["rate"] / 100.0       # <── THE FIX
    rate_df = rate_df[["date", "days", "r"]]
    print(f"\n  Rate range (decimal): [{rate_df['r'].min():.6f}, {rate_df['r'].max():.6f}]")

    # Forward prices (one per date × days)
    fwd_df = (
        stdop.groupby(["date", "days"])["forward_price"]
        .first()
        .reset_index()
        .rename(columns={"forward_price": "F"})
    )

    # ------------------------------------------------------------------
    # 4. Compute carry rate  q(T) = r(T) - (1/T) ln(F/S)
    # ------------------------------------------------------------------
    carry = fwd_df.merge(rate_df, on=["date", "days"], how="inner")
    carry = carry.merge(spot_df, on="date", how="inner")
    carry["T"] = carry["days"] / 365.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        carry["q"] = carry["r"] - (1.0 / carry["T"]) * np.log(carry["F"] / carry["S0"])

    # Clip extreme q
    q_out = (carry["q"] < args.q_floor) | (carry["q"] > args.q_cap)
    print(f"  Raw q range: [{carry['q'].min():.6f}, {carry['q'].max():.6f}]")
    print(f"  Outlier q rows clipped: {q_out.sum()} ({100 * q_out.mean():.2f}%)")
    carry["q"] = carry["q"].clip(args.q_floor, args.q_cap)

    carry = carry[["date", "days", "S0", "r", "q", "T"]]
    print(f"  Carry rows: {len(carry):,}")

    # ------------------------------------------------------------------
    # 5. Build Heston input table
    # ------------------------------------------------------------------
    hdf = vs[["date", "days", "delta", "cp_flag", "impl_volatility"]].copy()
    hdf.columns = ["date", "days", "delta", "cp_flag", "iv_market"]

    # Filter invalid IVs
    valid = (hdf["iv_market"] > 0) & (hdf["iv_market"] < 2.0)
    hdf = hdf[valid].copy()

    # Merge market data
    hdf = hdf.merge(carry, on=["date", "days"], how="inner")
    print(f"\n  Merged rows: {len(hdf):,}")

    # ------------------------------------------------------------------
    # 6. Delta → Strike
    # ------------------------------------------------------------------
    print("  Converting delta → strike...")
    hdf["K"] = hdf.apply(
        lambda row: delta_to_strike(
            row["delta"], row["S0"], row["T"],
            row["r"], row["q"], row["iv_market"], row["cp_flag"],
        ),
        axis=1,
    )

    failed = hdf["K"].isna()
    print(f"  Strike conversion failures: {failed.sum()} ({100 * failed.mean():.2f}%)")
    hdf = hdf[~failed].copy()

    hdf["moneyness"] = hdf["K"] / hdf["S0"]

    # ------------------------------------------------------------------
    # 7. Sanity checks
    # ------------------------------------------------------------------
    print(f"\n  Final dataset: {len(hdf):,} rows")
    print(f"  Dates:       {hdf['date'].nunique():,}")
    print(f"  Moneyness:   [{hdf['moneyness'].min():.4f}, {hdf['moneyness'].max():.4f}]")
    print(f"  r (decimal): [{hdf['r'].min():.6f}, {hdf['r'].max():.6f}]")
    print(f"  q (decimal): [{hdf['q'].min():.6f}, {hdf['q'].max():.6f}]")

    extreme_K = (hdf["moneyness"] < 0.3) | (hdf["moneyness"] > 3.0)
    if extreme_K.any():
        print(f"  WARNING: {extreme_K.sum()} rows with moneyness outside [0.3, 3.0]")

    # ------------------------------------------------------------------
    # 8. Save
    # ------------------------------------------------------------------
    output_cols = [
        "date", "days", "T", "delta", "cp_flag",
        "S0", "K", "r", "q", "iv_market", "moneyness",
    ]
    out = hdf[output_cols].copy()

    out_path = inputs_dir / f"{ticker}_heston_inputs.parquet"
    out.to_parquet(out_path, index=False)
    print(f"\n  Saved: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")

    # Per-date summary
    summary = (
        hdf.groupby("date")
        .agg(
            S0=("S0", "first"),
            r_mean=("r", "mean"),
            q_mean=("q", "mean"),
            iv_mean=("iv_market", "mean"),
            iv_std=("iv_market", "std"),
            n_points=("iv_market", "count"),
        )
        .round(6)
        .reset_index()
    )
    summary_path = inputs_dir / f"{ticker}_heston_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}")

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
