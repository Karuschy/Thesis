"""
Three-way comparison: MLP VAE vs Conv VAE vs Heston vs Market.

Loads all surface sets, aligns by common dates, and computes comparison
metrics + plots.

Usage:
    python scripts/compare_surfaces.py \
        --mlp_dir  artifacts/eval/mlp/surfaces \
        --conv_dir artifacts/eval/conv/surfaces \
        --heston_dir data/processed/heston/surfaces \
        --output_dir artifacts/comparison
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd


# ── Metrics ──────────────────────────────────────────────────────────────
@dataclass
class ComparisonMetrics:
    """Metrics for comparing two surface sets."""
    mse: float
    mae: float
    rmse: float
    max_error: float
    mean_per_cell_mae: float
    std_per_cell_mae: float
    n_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_metrics(a: np.ndarray, b: np.ndarray) -> ComparisonMetrics:
    """Compute comparison metrics between two [N,C,H,W] surface arrays."""
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
    err = a - b
    ae = np.abs(err)
    se = err ** 2
    per_cell = np.mean(ae, axis=0)
    return ComparisonMetrics(
        mse=float(np.mean(se)),
        mae=float(np.mean(ae)),
        rmse=float(np.sqrt(np.mean(se))),
        max_error=float(np.max(ae)),
        mean_per_cell_mae=float(np.mean(per_cell)),
        std_per_cell_mae=float(np.std(per_cell)),
        n_samples=a.shape[0],
    )


# ── Loaders ──────────────────────────────────────────────────────────────
def _load_dates(path: Path) -> np.ndarray:
    return pd.to_datetime(pd.read_csv(path)["date"]).values


def load_vae_surfaces(vae_dir: Path):
    """Return (model_surfaces, market_surfaces, dates, grid_spec)."""
    model = np.load(vae_dir / "vae_surfaces.npy")
    market = np.load(vae_dir / "market_surfaces.npy")
    dates = _load_dates(vae_dir / "vae_surface_dates.csv")
    with open(vae_dir / "grid_spec.json") as f:
        gs = json.load(f)
    return model, market, dates, gs


def load_heston_surfaces(heston_dir: Path, ticker: str = "AAPL"):
    """Return (heston_surfaces, dates)."""
    heston = np.load(heston_dir / f"{ticker}_heston_surfaces.npy")
    dates = _load_dates(heston_dir / f"{ticker}_heston_surface_dates.csv")
    return heston, dates


# ── Alignment ────────────────────────────────────────────────────────────
def _dates_to_date_list(arr):
    return [pd.to_datetime(d).date() for d in arr]


def align_multiple(
    *arrays_dates: Tuple[np.ndarray, np.ndarray],
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Align N (surfaces, dates) pairs to the intersection of all date sets.

    Returns:
        (list_of_aligned_surfaces, common_dates_array)
    """
    date_lists = [_dates_to_date_list(d) for _, d in arrays_dates]
    common = set(date_lists[0])
    for dl in date_lists[1:]:
        common &= set(dl)

    aligned = []
    for i, (surf, _) in enumerate(arrays_dates):
        mask = [d in common for d in date_lists[i]]
        aligned.append(surf[mask])

    # build common dates from first array
    mask0 = [d in common for d in date_lists[0]]
    common_dates = arrays_dates[0][1][mask0]
    return aligned, common_dates


# ── Table / CSV export ────────────────────────────────────────────────────
def save_tables(
    model_surfaces: Dict[str, np.ndarray],
    market: np.ndarray,
    dates: np.ndarray,
    grid_spec: dict,
    output_dir: Path,
):
    """Save CSV tables that can be inspected without a notebook."""
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    days_grid = np.array(grid_spec["days_grid"])
    delta_grid = np.array(grid_spec["delta_grid"])
    cp_order = grid_spec["cp_order"]
    names = list(model_surfaces.keys())

    # ── 1. Summary table ─────────────────────────────────────────────────
    rows = []
    for name in names:
        m = compute_metrics(model_surfaces[name], market)
        rows.append({
            "Model": name,
            "MSE": f"{m.mse:.6f}",
            "MAE": f"{m.mae:.6f}",
            "MAE (vol pts)": f"{m.mae * 100:.2f}",
            "RMSE": f"{m.rmse:.6f}",
            "RMSE (vol pts)": f"{m.rmse * 100:.2f}",
            "Max Error": f"{m.max_error:.6f}",
            "Mean Per-Cell MAE": f"{m.mean_per_cell_mae:.6f}",
            "Std Per-Cell MAE": f"{m.std_per_cell_mae:.6f}",
            "N Samples": m.n_samples,
        })
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(tables_dir / "summary.csv", index=False)

    # ── 2. Pairwise model-model table ────────────────────────────────────
    pw_rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            m = compute_metrics(model_surfaces[names[i]],
                                model_surfaces[names[j]])
            pw_rows.append({
                "Pair": f"{names[i]} vs {names[j]}",
                "MSE": f"{m.mse:.6f}",
                "MAE": f"{m.mae:.6f}",
                "RMSE": f"{m.rmse:.6f}",
                "Max Error": f"{m.max_error:.6f}",
            })
    pd.DataFrame(pw_rows).to_csv(tables_dir / "pairwise.csv", index=False)

    # ── 3. Per-maturity MAE table ────────────────────────────────────────
    for c, cp in enumerate(cp_order):
        rows_mat = []
        for mi, day in enumerate(days_grid):
            row = {"Maturity (days)": int(day)}
            for name in names:
                mae_val = np.abs(
                    model_surfaces[name][:, c, mi, :] - market[:, c, mi, :]
                ).mean()
                row[name] = f"{mae_val:.6f}"
            rows_mat.append(row)
        pd.DataFrame(rows_mat).to_csv(
            tables_dir / f"mae_by_maturity_{cp}.csv", index=False
        )

    # ── 4. Per-delta MAE table ───────────────────────────────────────────
    for c, cp in enumerate(cp_order):
        rows_d = []
        for di, delta in enumerate(delta_grid):
            row = {"Delta": f"{delta:.2f}"}
            for name in names:
                mae_val = np.abs(
                    model_surfaces[name][:, c, :, di] - market[:, c, :, di]
                ).mean()
                row[name] = f"{mae_val:.6f}"
            rows_d.append(row)
        pd.DataFrame(rows_d).to_csv(
            tables_dir / f"mae_by_delta_{cp}.csv", index=False
        )

    # ── 5. Per-date time series MAE ──────────────────────────────────────
    ts_rows = []
    date_strs = [str(d)[:10] for d in dates]
    for t, ds in enumerate(date_strs):
        row = {"Date": ds}
        for name in names:
            mae_val = np.abs(
                model_surfaces[name][t] - market[t]
            ).mean()
            row[f"{name} MAE"] = f"{mae_val:.6f}"
        ts_rows.append(row)
    pd.DataFrame(ts_rows).to_csv(
        tables_dir / "mae_timeseries.csv", index=False
    )

    # ── 6. Per-cell MAE heatmaps (one CSV per model × cp) ───────────────
    for name in names:
        mae_map = np.mean(np.abs(model_surfaces[name] - market), axis=0)
        for c, cp in enumerate(cp_order):
            df_heat = pd.DataFrame(
                mae_map[c],
                index=[f"{int(d)}d" for d in days_grid],
                columns=[f"Δ{d:.2f}" for d in delta_grid],
            )
            safe_name = name.lower().replace(" ", "_")
            df_heat.to_csv(
                tables_dir / f"cell_mae_{safe_name}_{cp}.csv"
            )

    # ── 7. Plain-text report ─────────────────────────────────────────────
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("  3-WAY COMPARISON REPORT")
    report_lines.append(f"  Models: {', '.join(names)}")
    report_lines.append(f"  Common dates: {len(dates)}")
    report_lines.append(f"  Date range: {date_strs[0]} → {date_strs[-1]}")
    report_lines.append(f"  Grid: {cp_order} × {len(days_grid)} mat × {len(delta_grid)} delta")
    report_lines.append("=" * 70)
    report_lines.append("")
    report_lines.append("  MODEL VS MARKET")
    report_lines.append("  " + "-" * 66)
    report_lines.append(f"  {'Model':<12} {'MSE':>10} {'MAE':>10} {'MAE(vp)':>10} "
                        f"{'RMSE':>10} {'RMSE(vp)':>10} {'Max':>10}")
    report_lines.append("  " + "-" * 66)
    for name in names:
        m = compute_metrics(model_surfaces[name], market)
        report_lines.append(
            f"  {name:<12} {m.mse:>10.6f} {m.mae:>10.6f} {m.mae*100:>9.2f}% "
            f"{m.rmse:>10.6f} {m.rmse*100:>9.2f}% {m.max_error:>10.6f}"
        )
    report_lines.append("  " + "-" * 66)
    report_lines.append("")

    report_lines.append("  PAIRWISE MODEL-MODEL")
    report_lines.append("  " + "-" * 50)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            m = compute_metrics(model_surfaces[names[i]],
                                model_surfaces[names[j]])
            report_lines.append(
                f"  {names[i]} vs {names[j]}:  "
                f"MAE={m.mae:.6f}  RMSE={m.rmse:.6f}"
            )
    report_lines.append("")

    mae_map = {n: compute_metrics(model_surfaces[n], market).mae for n in names}
    winner = min(mae_map, key=mae_map.get)
    report_lines.append(f"  WINNER: {winner} (MAE = {mae_map[winner]:.6f})")
    for n in names:
        if n != winner:
            pct = (mae_map[n] - mae_map[winner]) / mae_map[n] * 100
            report_lines.append(f"    vs {n}: {pct:.1f}% lower MAE")
    report_lines.append("=" * 70)

    report_text = "\n".join(report_lines)
    (tables_dir / "report.txt").write_text(report_text, encoding="utf-8")
    print(report_text)

    print(f"\n  Saved tables to: {tables_dir}")


# ── Plotting ─────────────────────────────────────────────────────────────
MODEL_COLOURS = {
    "MLP VAE": "#1f77b4",
    "Conv VAE": "#2ca02c",
    "Heston": "#ff7f0e",
}


def _set_grid_ticks(ax, days_grid, delta_grid, xlabel=True, ylabel=True):
    ax.set_yticks(range(len(days_grid)))
    ax.set_yticklabels([int(d) for d in days_grid], fontsize=7)
    ax.set_xticks(range(0, len(delta_grid), 2))
    ax.set_xticklabels([f"{delta_grid[i]:.2f}" for i in range(0, len(delta_grid), 2)],
                       fontsize=7, rotation=45)
    if xlabel:
        ax.set_xlabel("Delta", fontsize=8)
    if ylabel:
        ax.set_ylabel("Maturity (days)", fontsize=8)


def save_comparison_plots(
    model_surfaces: Dict[str, np.ndarray],
    market: np.ndarray,
    dates: np.ndarray,
    grid_spec: dict,
    output_dir: Path,
    n_samples: int = 5,
):
    """Generate comparison plots for N models vs market."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    days_grid = np.array(grid_spec["days_grid"])
    delta_grid = np.array(grid_spec["delta_grid"])
    cp_order = grid_spec["cp_order"]
    names = list(model_surfaces.keys())
    n_models = len(names)

    # ── 1. Error heatmaps (per-cp, one row per model + diff panel) ───────
    model_errors = {
        name: np.mean(np.abs(surf - market), axis=0)  # [C, H, W]
        for name, surf in model_surfaces.items()
    }

    for c, cp in enumerate(cp_order):
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
        if n_models == 1:
            axes = [axes]
        vmax = max(model_errors[n][c].max() for n in names)

        for i, name in enumerate(names):
            im = axes[i].imshow(
                model_errors[name][c], aspect="auto", origin="lower",
                cmap="Reds", vmin=0, vmax=vmax,
            )
            axes[i].set_title(f"{name} MAE ({cp})", fontsize=9)
            _set_grid_ticks(axes[i], days_grid, delta_grid,
                            ylabel=(i == 0))
            plt.colorbar(im, ax=axes[i], format="%.3f", shrink=0.85)

        plt.tight_layout()
        plt.savefig(plots_dir / f"error_heatmap_{cp}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

    # ── 2. Pairwise difference heatmaps ──────────────────────────────────
    pairs = [(names[i], names[j])
             for i in range(n_models) for j in range(i + 1, n_models)]
    for c, cp in enumerate(cp_order):
        fig, axes = plt.subplots(1, len(pairs),
                                 figsize=(5 * len(pairs), 4.5))
        if len(pairs) == 1:
            axes = [axes]
        for k, (n1, n2) in enumerate(pairs):
            diff = model_errors[n1][c] - model_errors[n2][c]
            vabs = max(abs(diff.min()), abs(diff.max())) or 1e-6
            im = axes[k].imshow(
                diff, aspect="auto", origin="lower",
                cmap="RdBu_r", vmin=-vabs, vmax=vabs,
            )
            axes[k].set_title(
                f"{n1} − {n2} ({cp})\nblue = {n1} better",
                fontsize=8,
            )
            _set_grid_ticks(axes[k], days_grid, delta_grid,
                            ylabel=(k == 0))
            plt.colorbar(im, ax=axes[k], format="%.3f", shrink=0.85)
        plt.tight_layout()
        plt.savefig(plots_dir / f"diff_heatmap_{cp}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

    # ── 3. Sample surface comparisons ────────────────────────────────────
    n_show = min(n_samples, len(dates))
    sample_idx = np.linspace(0, len(dates) - 1, n_show, dtype=int)
    ncols = 1 + n_models + 1  # market + models + bar chart

    for idx in sample_idx:
        date_str = str(dates[idx])[:10]
        fig, axes = plt.subplots(
            len(cp_order), ncols,
            figsize=(4.5 * ncols, 4.5 * len(cp_order)),
        )
        if len(cp_order) == 1:
            axes = axes[np.newaxis, :]

        for c, cp in enumerate(cp_order):
            # Market
            im = axes[c, 0].imshow(
                market[idx, c], aspect="auto", origin="lower", cmap="viridis",
            )
            axes[c, 0].set_title(f"Market ({cp})", fontsize=9)
            _set_grid_ticks(axes[c, 0], days_grid, delta_grid)
            plt.colorbar(im, ax=axes[c, 0], format="%.3f", shrink=0.85)

            # Each model
            for j, name in enumerate(names, start=1):
                im = axes[c, j].imshow(
                    model_surfaces[name][idx, c],
                    aspect="auto", origin="lower", cmap="viridis",
                )
                axes[c, j].set_title(f"{name} ({cp})", fontsize=9)
                _set_grid_ticks(axes[c, j], days_grid, delta_grid,
                                ylabel=False)
                plt.colorbar(im, ax=axes[c, j], format="%.3f", shrink=0.85)

            # Bar chart of per-model MAE for this sample
            maes = [
                np.abs(model_surfaces[n][idx, c] - market[idx, c]).mean()
                for n in names
            ]
            colours = [MODEL_COLOURS.get(n, "gray") for n in names]
            axes[c, -1].barh(names, maes, color=colours)
            axes[c, -1].set_xlabel("MAE", fontsize=8)
            axes[c, -1].set_title(f"MAE ({cp})", fontsize=9)
            axes[c, -1].tick_params(labelsize=7)

        fig.suptitle(f"Surface Comparison — {date_str}", fontsize=13)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(plots_dir / f"sample_{date_str}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

    # ── 4. Error time-series ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 5))
    for name, surf in model_surfaces.items():
        ts = np.abs(surf - market).mean(axis=(1, 2, 3))
        ax.plot(dates, ts, label=name, alpha=0.75,
                color=MODEL_COLOURS.get(name, None))
    ax.set_xlabel("Date")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title("Model vs Market: MAE Over Time")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "error_timeseries.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── 5. Box-plot of per-date MAE ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    box_data, box_labels, box_colours = [], [], []
    for name, surf in model_surfaces.items():
        per_date_mae = np.abs(surf - market).mean(axis=(1, 2, 3))
        box_data.append(per_date_mae)
        box_labels.append(name)
        box_colours.append(MODEL_COLOURS.get(name, "gray"))

    bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True,
                    showmeans=True)
    for patch, col in zip(bp["boxes"], box_colours):
        patch.set_facecolor(col)
        patch.set_alpha(0.4)
    ax.set_ylabel("Per-Date MAE")
    ax.set_title("Distribution of Daily MAE")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(plots_dir / "mae_boxplot.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── 6. Per-maturity MAE bar chart ────────────────────────────────────
    fig, axes = plt.subplots(1, len(cp_order), figsize=(7 * len(cp_order), 5))
    if len(cp_order) == 1:
        axes = [axes]
    x = np.arange(len(days_grid))
    width = 0.8 / n_models
    for c, cp in enumerate(cp_order):
        for j, name in enumerate(names):
            per_mat = np.abs(model_surfaces[name][:, c] - market[:, c]).mean(axis=(0, 2))
            axes[c].bar(x + j * width, per_mat, width,
                        label=name, color=MODEL_COLOURS.get(name, None),
                        alpha=0.8)
        axes[c].set_xticks(x + width * (n_models - 1) / 2)
        axes[c].set_xticklabels([int(d) for d in days_grid], fontsize=8)
        axes[c].set_xlabel("Maturity (days)")
        axes[c].set_ylabel("MAE")
        axes[c].set_title(f"MAE by Maturity ({cp})")
        axes[c].legend(fontsize=8)
        axes[c].grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(plots_dir / "mae_by_maturity.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── 7. Per-delta MAE bar chart ───────────────────────────────────────
    fig, axes = plt.subplots(1, len(cp_order), figsize=(7 * len(cp_order), 5))
    if len(cp_order) == 1:
        axes = [axes]
    x = np.arange(len(delta_grid))
    for c, cp in enumerate(cp_order):
        for j, name in enumerate(names):
            per_delta = np.abs(model_surfaces[name][:, c] - market[:, c]).mean(axis=(0, 1))
            axes[c].bar(x + j * width, per_delta, width,
                        label=name, color=MODEL_COLOURS.get(name, None),
                        alpha=0.8)
        axes[c].set_xticks(x + width * (n_models - 1) / 2)
        axes[c].set_xticklabels([f"{d:.2f}" for d in delta_grid],
                                fontsize=7, rotation=45)
        axes[c].set_xlabel("Delta")
        axes[c].set_ylabel("MAE")
        axes[c].set_title(f"MAE by Delta ({cp})")
        axes[c].legend(fontsize=8)
        axes[c].grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(plots_dir / "mae_by_delta.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved plots to: {plots_dir}")


# ── CLI ──────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="3-way comparison: MLP VAE / Conv VAE / Heston vs Market",
    )
    p.add_argument("--mlp_dir", type=str,
                   default="artifacts/eval/mlp/surfaces",
                   help="MLP VAE eval surfaces directory")
    p.add_argument("--conv_dir", type=str,
                   default="artifacts/eval/conv/surfaces",
                   help="Conv VAE eval surfaces directory")
    p.add_argument("--heston_dir", type=str,
                   default="data/processed/heston/surfaces",
                   help="Heston calibrated surfaces directory")
    p.add_argument("--ticker", type=str, default="AAPL")
    p.add_argument("--output_dir", type=str,
                   default="artifacts/comparison",
                   help="Where to write metrics + plots")
    p.add_argument("--n_plot_samples", type=int, default=5,
                   help="Number of sample date slices to plot")
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    mlp_dir = Path(args.mlp_dir)
    conv_dir = Path(args.conv_dir)
    heston_dir = Path(args.heston_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ─────────────────────────────────────────────────────────────
    print("Loading MLP VAE surfaces...")
    mlp_surf, mlp_market, mlp_dates, grid_spec = load_vae_surfaces(mlp_dir)
    print(f"  MLP VAE:  {mlp_surf.shape}  ({len(mlp_dates)} dates)")

    print("Loading Conv VAE surfaces...")
    conv_surf, conv_market, conv_dates, _ = load_vae_surfaces(conv_dir)
    print(f"  Conv VAE: {conv_surf.shape}  ({len(conv_dates)} dates)")

    print("Loading Heston surfaces...")
    heston_surf, heston_dates = load_heston_surfaces(heston_dir, args.ticker)
    print(f"  Heston:   {heston_surf.shape}  ({len(heston_dates)} dates)")

    # ── Align to common dates ────────────────────────────────────────────
    print("\nAligning to common dates...")
    aligned, common_dates = align_multiple(
        (mlp_surf, mlp_dates),
        (conv_surf, conv_dates),
        (heston_surf, heston_dates),
        (mlp_market, mlp_dates),
    )
    mlp_a, conv_a, heston_a, market_a = aligned
    n = len(common_dates)
    print(f"  Common dates: {n}")
    if n == 0:
        print("ERROR: no common dates – nothing to compare.")
        return
    print(f"  Range: {str(common_dates[0])[:10]}  →  {str(common_dates[-1])[:10]}")

    # ── Metrics ──────────────────────────────────────────────────────────
    model_surfaces = {
        "MLP VAE": mlp_a,
        "Conv VAE": conv_a,
        "Heston": heston_a,
    }
    names = list(model_surfaces.keys())

    print("\n" + "=" * 65)
    print("  COMPARISON METRICS  (on common test dates)")
    print("=" * 65)

    metrics_dict: Dict[str, Any] = {
        "n_common_dates": n,
        "date_range": [str(common_dates[0])[:10], str(common_dates[-1])[:10]],
    }

    # Each model vs market
    for name in names:
        m = compute_metrics(model_surfaces[name], market_a)
        tag = f"{name} vs Market"
        print(f"\n  {tag}:")
        print(f"    MSE  = {m.mse:.6f}")
        print(f"    MAE  = {m.mae:.6f}  ({m.mae * 100:.2f} vol pts)")
        print(f"    RMSE = {m.rmse:.6f}  ({m.rmse * 100:.2f} vol pts)")
        print(f"    Max  = {m.max_error:.6f}")
        metrics_dict[tag] = m.to_dict()

    # Pairwise model-model
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            m = compute_metrics(model_surfaces[names[i]],
                                model_surfaces[names[j]])
            tag = f"{names[i]} vs {names[j]}"
            print(f"\n  {tag}:")
            print(f"    MSE  = {m.mse:.6f}")
            print(f"    MAE  = {m.mae:.6f}")
            print(f"    RMSE = {m.rmse:.6f}")
            metrics_dict[tag] = m.to_dict()

    # Winner
    mae_vs_market = {
        n: compute_metrics(model_surfaces[n], market_a).mae for n in names
    }
    winner = min(mae_vs_market, key=mae_vs_market.get)
    print("\n" + "-" * 65)
    print(f"  WINNER (lowest MAE vs Market): {winner}  "
          f"(MAE = {mae_vs_market[winner]:.6f})")
    for n in names:
        if n != winner:
            gap = (mae_vs_market[n] - mae_vs_market[winner]) / mae_vs_market[n] * 100
            print(f"    vs {n}: {gap:.1f}% lower MAE")
    print("=" * 65)

    metrics_dict["winner"] = winner
    metrics_dict["mae_vs_market"] = {k: float(v) for k, v in mae_vs_market.items()}

    metrics_path = output_dir / "comparison_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")

    # ── Tables ───────────────────────────────────────────────────────────
    print("\nGenerating CSV tables & report...")
    save_tables(
        model_surfaces=model_surfaces,
        market=market_a,
        dates=common_dates,
        grid_spec=grid_spec,
        output_dir=output_dir,
    )

    # ── Plots ────────────────────────────────────────────────────────────
    print("\nGenerating comparison plots...")
    save_comparison_plots(
        model_surfaces=model_surfaces,
        market=market_a,
        dates=common_dates,
        grid_spec=grid_spec,
        output_dir=output_dir,
        n_samples=args.n_plot_samples,
    )

    print(f"\nComparison complete! Results in: {output_dir}")


if __name__ == "__main__":
    main()
