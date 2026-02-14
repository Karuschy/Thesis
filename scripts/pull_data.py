"""
Pull raw IvyDB data from WRDS for a given ticker.

Downloads 4 datasets:
  1. Vol surface    (vsurfdYYYY)
  2. Underlying px  (security_price / secprd*)
  3. Zero curve     (zerocdYYYY)
  4. Std option px  (stdopdYYYY)

Usage:
    python scripts/pull_data.py --ticker AAPL
    python scripts/pull_data.py --ticker MSFT --start 2016-01-01 --end 2025-12-31
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import wrds
from sqlalchemy import text


# ---------------------------------------------------------------------------
# WRDS helpers
# ---------------------------------------------------------------------------

class WRDSPuller:
    """Thin wrapper around a WRDS connection with IvyDB convenience methods."""

    LIB = "optionm"

    def __init__(self, username: str):
        self.db = wrds.Connection(wrds_username=username)
        self.engine = self.db.engine
        self._tables_cache: set[str] | None = None

    # -- low-level --------------------------------------------------------
    def close(self) -> None:
        self.db.close()

    def query(self, sql: str, date_cols: list[str] | None = None) -> pd.DataFrame:
        with self.engine.connect() as conn:
            result = conn.execute(text(sql))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
        for col in date_cols or []:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        return df

    def tables(self) -> set[str]:
        if self._tables_cache is None:
            self._tables_cache = set(self.db.list_tables(library=self.LIB))
        return self._tables_cache

    def has_table(self, name: str) -> bool:
        return name in self.tables()

    def describe(self, table: str) -> list[str]:
        return self.db.describe_table(library=self.LIB, table=table)["name"].tolist()

    @staticmethod
    def _pick(cols: list[str], candidates: list[str], required: bool = True) -> str | None:
        low = [c.lower() for c in cols]
        for c in candidates:
            if c.lower() in low:
                return cols[low.index(c.lower())]
        if required:
            raise KeyError(f"None of {candidates} found in {cols}")
        return None

    # -- SECID resolution --------------------------------------------------
    def resolve_secid(
        self, ticker: str, start: str, end: str, years: list[int]
    ) -> int:
        ticker = ticker.upper()
        q = f"""
        SELECT DISTINCT secid
        FROM {self.LIB}.secnmd
        WHERE ticker = '{ticker}'
        """
        secids = self.query(q)
        if secids.empty:
            raise RuntimeError(f"No SECID for {ticker}")
        print(f"  SECID candidates: {secids['secid'].tolist()}")

        if len(secids) == 1:
            return int(secids.iloc[0]["secid"])

        # Pick the one with greatest surface coverage
        id_list = ",".join(str(int(x)) for x in secids["secid"])
        frames = []
        for y in years:
            t = f"vsurfd{y}"
            if not self.has_table(t):
                continue
            frames.append(self.query(f"""
                SELECT secid, COUNT(DISTINCT date) AS n
                FROM {self.LIB}.{t}
                WHERE secid IN ({id_list})
                  AND date BETWEEN '{start}' AND '{end}'
                GROUP BY secid
            """))
        if not frames:
            return int(secids.iloc[0]["secid"])

        cov = (
            pd.concat(frames, ignore_index=True)
            .groupby("secid", as_index=False)["n"]
            .sum()
            .sort_values("n", ascending=False)
        )
        return int(cov.iloc[0]["secid"])

    # -- 1. Vol surface ----------------------------------------------------
    def pull_vol_surface(
        self, secid: int, start: str, end: str, years: list[int]
    ) -> pd.DataFrame:
        frames = []
        for y in years:
            t = f"vsurfd{y}"
            if not self.has_table(t):
                continue
            print(f"  {t}...", end=" ", flush=True)
            df = self.query(f"""
                SELECT secid, date, days, delta, cp_flag,
                       impl_volatility, impl_strike, impl_premium, dispersion
                FROM {self.LIB}.{t}
                WHERE secid = {secid}
                  AND date BETWEEN '{start}' AND '{end}'
            """, date_cols=["date"])
            print(f"{len(df):,} rows")
            frames.append(df)
        if not frames:
            raise RuntimeError("No vol-surface data found")
        vs = pd.concat(frames, ignore_index=True)
        # normalise
        if vs["delta"].max() > 2:
            vs["delta"] = vs["delta"] / 100.0
        vs["days"] = vs["days"].astype(int)
        vs["cp_flag"] = vs["cp_flag"].str.upper()
        return vs

    # -- 2. Underlying prices -----------------------------------------------
    def pull_underlying_prices(
        self, secid: int, start: str, end: str, years: list[int]
    ) -> pd.DataFrame:
        # try security_price first
        if self.has_table("security_price"):
            cols = self.describe("security_price")
            id_col = self._pick(cols, ["secid"], required=False)
            dt_col = self._pick(cols, ["date", "dt"], required=False)
            if id_col and dt_col:
                df = self.query(f"""
                    SELECT * FROM {self.LIB}.security_price
                    WHERE {id_col} = {secid}
                      AND {dt_col} BETWEEN '{start}' AND '{end}'
                """, date_cols=["date"])
                if not df.empty:
                    return df

        # try secprd
        if self.has_table("secprd"):
            df = self.query(f"""
                SELECT * FROM {self.LIB}.secprd
                WHERE secid = {secid}
                  AND date BETWEEN '{start}' AND '{end}'
            """, date_cols=["date"])
            if not df.empty:
                return df

        # yearly tables
        frames = []
        for y in years:
            t = f"secprd{y}"
            if not self.has_table(t):
                continue
            frames.append(self.query(f"""
                SELECT * FROM {self.LIB}.{t}
                WHERE secid = {secid}
                  AND date BETWEEN '{start}' AND '{end}'
            """, date_cols=["date"]))
        if frames:
            return pd.concat(frames, ignore_index=True)

        raise RuntimeError("Could not pull underlying prices")

    # -- 3. Zero curve -----------------------------------------------------
    def pull_zero_curve(
        self, start: str, end: str, years: list[int]
    ) -> pd.DataFrame:
        frames = []
        for y in years:
            t = f"zerocd{y}"
            if not self.has_table(t):
                continue
            print(f"  {t}...", end=" ", flush=True)
            df = self.query(f"""
                SELECT date, days, rate
                FROM {self.LIB}.{t}
                WHERE date BETWEEN '{start}' AND '{end}'
            """, date_cols=["date"])
            print(f"{len(df):,} rows")
            frames.append(df)

        if not frames and self.has_table("zerocd"):
            df = self.query(f"""
                SELECT date, days, rate
                FROM {self.LIB}.zerocd
                WHERE date BETWEEN '{start}' AND '{end}'
            """, date_cols=["date"])
            frames.append(df)

        if not frames:
            raise RuntimeError("No zero-curve data found")
        zc = pd.concat(frames, ignore_index=True)
        return zc.drop_duplicates(subset=["date", "days"]).sort_values(["date", "days"])

    # -- 4. Standard option prices (forward prices) -------------------------
    def pull_std_option_price(
        self, secid: int, start: str, end: str, years: list[int]
    ) -> pd.DataFrame:
        frames = []
        for y in years:
            t = f"stdopd{y}"
            if not self.has_table(t):
                continue
            cols = self.describe(t)
            cols_l = [c.lower() for c in cols]

            core = ["secid", "date", "days", "cp_flag", "forward_price", "strike_price"]
            optional = ["impl_volatility", "impl_premium", "bs_price", "delta"]
            select = [c for c in core if c.lower() in cols_l]
            select += [c for c in optional if c.lower() in cols_l]

            print(f"  {t}...", end=" ", flush=True)
            df = self.query(f"""
                SELECT {', '.join(select)}
                FROM {self.LIB}.{t}
                WHERE secid = {secid}
                  AND date BETWEEN '{start}' AND '{end}'
            """, date_cols=["date"])
            print(f"{len(df):,} rows")
            frames.append(df)

        if not frames and self.has_table("stdopd"):
            cols = self.describe("stdopd")
            cols_l = [c.lower() for c in cols]
            core = ["secid", "date", "days", "cp_flag", "forward_price", "strike_price"]
            optional = ["impl_volatility", "impl_premium", "bs_price", "delta"]
            select = [c for c in core if c.lower() in cols_l]
            select += [c for c in optional if c.lower() in cols_l]
            df = self.query(f"""
                SELECT {', '.join(select)}
                FROM {self.LIB}.stdopd
                WHERE secid = {secid}
                  AND date BETWEEN '{start}' AND '{end}'
            """, date_cols=["date"])
            frames.append(df)

        if not frames:
            raise RuntimeError("No std-option-price data found")
        return (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates()
            .sort_values(["date", "days", "cp_flag"])
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Pull raw IvyDB data from WRDS")
    p.add_argument("--ticker", type=str, required=True)
    p.add_argument("--start", type=str, default="2016-01-01")
    p.add_argument("--end", type=str, default="2025-12-31")
    p.add_argument("--username", type=str, default="acaraman",
                   help="WRDS username (default: acaraman)")
    p.add_argument("--output_dir", type=str, default="data/raw/ivydb",
                   help="Root directory for raw data output")
    return p.parse_args()


def main():
    args = parse_args()
    ticker = args.ticker.upper()
    start, end = args.start, args.end
    years = list(range(int(start[:4]), int(end[:4]) + 1))

    out = Path(args.output_dir)
    for sub in ["vol_surface", "security_price", "zero_curve", "std_option_price"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    print(f"=== Pulling data for {ticker}  [{start} → {end}] ===\n")

    puller = WRDSPuller(args.username)
    try:
        # 0. Resolve SECID
        print("[1/5] Resolving SECID...")
        secid = puller.resolve_secid(ticker, start, end, years)
        print(f"  → SECID = {secid}\n")

        # 1. Vol surface
        print("[2/5] Pulling vol surface...")
        vs = puller.pull_vol_surface(secid, start, end, years)
        vs_path = out / "vol_surface" / f"{ticker}_vsurfd_{start}_{end}.csv.gz"
        vs.to_csv(vs_path, index=False, compression="gzip")
        print(f"  → {len(vs):,} rows  →  {vs_path}\n")

        # 2. Underlying prices
        print("[3/5] Pulling underlying prices...")
        px = puller.pull_underlying_prices(secid, start, end, years)
        px_path = out / "security_price" / f"{ticker}_underlying_{start}_{end}.csv.gz"
        px.to_csv(px_path, index=False, compression="gzip")
        print(f"  → {len(px):,} rows  →  {px_path}\n")

        # 3. Zero curve (shared across tickers)
        zc_path = out / "zero_curve" / f"zero_curve_{start}_{end}.csv.gz"
        if zc_path.exists():
            print("[4/5] Zero curve already exists — skipping.")
            zc = pd.read_csv(zc_path, parse_dates=["date"])
        else:
            print("[4/5] Pulling zero curve...")
            zc = puller.pull_zero_curve(start, end, years)
            zc.to_csv(zc_path, index=False, compression="gzip")
        print(f"  → {len(zc):,} rows  →  {zc_path}\n")

        # 4. Std option prices (forward prices)
        print("[5/5] Pulling std option prices...")
        stdop = puller.pull_std_option_price(secid, start, end, years)
        stdop_path = out / "std_option_price" / f"{ticker}_stdopd_{start}_{end}.csv.gz"
        stdop.to_csv(stdop_path, index=False, compression="gzip")
        print(f"  → {len(stdop):,} rows  →  {stdop_path}\n")

        # Summary
        print("=" * 60)
        print(f"DONE: {ticker} (SECID {secid})")
        print(f"  Vol surface:    {len(vs):,} rows")
        print(f"  Underlying:     {len(px):,} rows")
        print(f"  Zero curve:     {len(zc):,} rows")
        print(f"  Std option px:  {len(stdop):,} rows")
        print("=" * 60)

    finally:
        puller.close()


if __name__ == "__main__":
    main()
