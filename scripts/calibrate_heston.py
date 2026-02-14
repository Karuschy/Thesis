"""
Calibrate the Heston model for each date and generate IV surfaces.

Reads the prepared Heston inputs (from prepare_heston_data.py),
calibrates (v0, kappa, theta, sigma, rho) per date, then
regenerates delta-grid IV surfaces for comparison with the VAE.

Usage:
    python scripts/calibrate_heston.py --ticker AAPL
    python scripts/calibrate_heston.py --ticker AAPL --dates_from artifacts/eval/surfaces/vae_surface_dates.csv
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.data.volsurface_grid import GridSpec
from src.models.heston import (
    CalibrationResult,
    HestonParams,
    calibrate_heston,
    heston_iv,
    strike_from_delta,
)


# ---------------------------------------------------------------------------
# Surface generation
# ---------------------------------------------------------------------------

def generate_heston_surface(
    params: HestonParams,
    S0: float,
    r: float,
    q: float,
    grid: GridSpec,
) -> np.ndarray:
    """
    Build an IV surface on the standard (cp × days × delta) grid.

    Returns:
        np.ndarray  shape (C, H, W), NaN where pricing fails.
    """
    n_cp = len(grid.cp_order)
    n_days = len(grid.days_grid)
    n_delta = len(grid.delta_grid)
    surface = np.full((n_cp, n_days, n_delta), np.nan, dtype=np.float32)

    sigma_atm = max(np.sqrt(abs(params.v0)), 0.01)

    for i_cp, cp in enumerate(grid.cp_order):
        for i_d, days in enumerate(grid.days_grid):
            T = float(days) / 365.0
            if T <= 0:
                continue
            for i_del, delta in enumerate(grid.delta_grid):
                try:
                    K = strike_from_delta(S0, T, r, q, sigma_atm, float(delta), cp)
                    if K is None or np.isnan(K) or K <= 0:
                        continue
                    iv = heston_iv(S0, K, T, r, q, params, cp)
                    if iv is not None and 0 < iv < 2.0:
                        surface[i_cp, i_d, i_del] = iv
                except Exception:
                    pass
    return surface


# ---------------------------------------------------------------------------
# Single-date calibration
# ---------------------------------------------------------------------------

def calibrate_single_date(date_df: pd.DataFrame) -> tuple[CalibrationResult, dict]:
    S0 = float(date_df["S0"].iloc[0])
    r = float(date_df["r"].mean())
    q = float(date_df["q"].mean())
    result = calibrate_heston(
        S0=S0, r=r, q=q,
        maturities=date_df["T"].values,
        strikes=date_df["K"].values,
        market_ivs=date_df["iv_market"].values,
        cp_flags=date_df["cp_flag"].values,
        max_iterations=500,
    )
    return result, {"S0": S0, "r": r, "q": q}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Calibrate Heston & generate IV surfaces")
    p.add_argument("--ticker", type=str, required=True)
    p.add_argument("--input_dir", type=str, default="data/processed/heston")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output dir for surfaces & params (default: <input_dir>/surfaces)")
    p.add_argument("--dates_from", type=str, default=None,
                   help="CSV with a 'date' column to restrict calibration dates "
                        "(e.g. VAE test dates for fair comparison)")
    p.add_argument("--grid_spec", type=str, default=None,
                   help="JSON grid spec from VAE eval (ensures identical grids). "
                        "Default: artifacts/eval/surfaces/grid_spec.json")
    p.add_argument("--min_fill", type=float, default=0.50,
                   help="Minimum fraction of non-NaN cells to keep a surface (default 0.50)")
    return p.parse_args()


def _load_grid(path: Optional[str]) -> GridSpec:
    """Load GridSpec from JSON or fall back to defaults."""
    candidates = [
        path,
        "artifacts/eval/surfaces/grid_spec.json",
    ]
    for c in candidates:
        if c is not None and Path(c).exists():
            with open(c) as f:
                g = json.load(f)
            print(f"  Grid loaded from {c}")
            return GridSpec(
                days_grid=np.array(g["days_grid"], dtype=np.float32),
                delta_grid=np.array(g["delta_grid"], dtype=np.float32),
                cp_order=list(g["cp_order"]),
            )

    print("  Grid: using hardcoded defaults")
    return GridSpec(
        days_grid=np.array([10, 30, 60, 91, 122, 152, 182, 273, 365, 547, 730], dtype=np.float32),
        delta_grid=np.arange(0.10, 0.91, 0.05).round(2).astype(np.float32),
        cp_order=["C", "P"],
    )


def main():
    args = parse_args()
    ticker = args.ticker.upper()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir) if args.output_dir else input_dir / "surfaces"
    out_dir.mkdir(parents=True, exist_ok=True)

    input_path = input_dir / "inputs" / f"{ticker}_heston_inputs.parquet"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing: {input_path}\n"
            f"Run  python scripts/prepare_heston_data.py --ticker {ticker}  first."
        )

    print(f"=== Heston calibration for {ticker} ===\n")

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    df = pd.read_parquet(input_path)
    df["date"] = pd.to_datetime(df["date"])
    all_dates = sorted(df["date"].unique())
    print(f"  Input: {len(df):,} rows, {len(all_dates):,} dates")

    # Optionally restrict to specific dates (e.g. VAE test dates)
    if args.dates_from and Path(args.dates_from).exists():
        ref = pd.read_csv(args.dates_from, parse_dates=["date"])
        ref_set = set(ref["date"].dt.normalize())
        dates = [d for d in all_dates if pd.Timestamp(d).normalize() in ref_set]
        print(f"  Restricted to {len(dates)} dates from {args.dates_from}")
    else:
        dates = all_dates
        print(f"  Calibrating all {len(dates)} dates")

    grid = _load_grid(args.grid_spec)
    expected_cells = len(grid.cp_order) * len(grid.days_grid) * len(grid.delta_grid)
    print(f"  Grid: {len(grid.cp_order)}×{len(grid.days_grid)}×{len(grid.delta_grid)} "
          f"= {expected_cells} cells\n")

    # ------------------------------------------------------------------
    # Calibrate
    # ------------------------------------------------------------------
    calib_rows = []
    mkt_by_date = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for date in tqdm(dates, desc="Calibrating"):
            ddf = df[df["date"] == date]
            try:
                result, mkt = calibrate_single_date(ddf)
                calib_rows.append({
                    "date": date,
                    "v0": result.params.v0,
                    "kappa": result.params.kappa,
                    "theta": result.params.theta,
                    "sigma": result.params.sigma,
                    "rho": result.params.rho,
                    "error": result.error,
                    "success": result.success,
                    "feller": result.params.feller_condition,
                })
                mkt_by_date[date] = mkt
            except Exception as e:
                calib_rows.append({
                    "date": date,
                    "v0": np.nan, "kappa": np.nan, "theta": np.nan,
                    "sigma": np.nan, "rho": np.nan,
                    "error": np.inf, "success": False, "feller": False,
                })

    calib_df = pd.DataFrame(calib_rows)
    n_ok = calib_df["success"].sum()
    print(f"\n  Calibrated: {n_ok}/{len(calib_df)} succeeded "
          f"({100 * n_ok / len(calib_df):.1f}%)")
    print(f"  Feller satisfied: {calib_df['feller'].sum()}")

    calib_path = out_dir / f"{ticker}_heston_params.csv"
    calib_df.to_csv(calib_path, index=False)
    print(f"  Saved params: {calib_path}")

    # ------------------------------------------------------------------
    # Generate surfaces
    # ------------------------------------------------------------------
    surfaces, surface_dates = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _, row in tqdm(calib_df.iterrows(), total=len(calib_df), desc="Generating surfaces"):
            date = row["date"]
            if not row["success"] or np.isnan(row["v0"]):
                continue
            mkt = mkt_by_date.get(date)
            if mkt is None:
                continue
            params = HestonParams(
                v0=row["v0"], kappa=row["kappa"],
                theta=row["theta"], sigma=row["sigma"], rho=row["rho"],
            )
            try:
                surf = generate_heston_surface(params, mkt["S0"], mkt["r"], mkt["q"], grid)
                fill = np.sum(~np.isnan(surf)) / surf.size
                if fill >= args.min_fill:
                    surfaces.append(surf)
                    surface_dates.append(date)
            except Exception:
                pass

    print(f"\n  Valid surfaces: {len(surfaces)}/{n_ok}")

    if not surfaces:
        print("  WARNING: No valid surfaces generated. Check data / calibration.")
        return

    stacked = np.stack(surfaces, axis=0)  # (N, C, H, W)
    print(f"  Tensor shape: {stacked.shape}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    np.save(out_dir / f"{ticker}_heston_surfaces.npy", stacked)
    pd.DataFrame({"date": surface_dates}).to_csv(
        out_dir / f"{ticker}_heston_surface_dates.csv", index=False
    )

    # Grid spec (so downstream comparison can verify grids match)
    with open(out_dir / "grid_spec.json", "w") as f:
        json.dump({
            "days_grid": grid.days_grid.tolist(),
            "delta_grid": grid.delta_grid.tolist(),
            "cp_order": grid.cp_order,
        }, f, indent=2)

    print(f"\n  Saved to {out_dir}:")
    print(f"    {ticker}_heston_surfaces.npy  ({stacked.nbytes / 1e6:.1f} MB)")
    print(f"    {ticker}_heston_surface_dates.csv")
    print(f"    {ticker}_heston_params.csv")
    print(f"    grid_spec.json")
    print("\n  Done.\n")


if __name__ == "__main__":
    main()
