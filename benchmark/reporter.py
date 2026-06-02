"""
Benchmark reporter — reads outputs/benchmark/summary.csv and generates plots.

Plots produced (saved to outputs/benchmark/plots/):
  1. timings_by_workers.png  — t_load, t_sbert, t_graph, t_train vs n_workers
  2. timings_by_volume.png   — same breakdown vs dataset size
  3. model_perf.png          — NDCG@10, Precision@10, MAP@10 vs dataset size
  4. resource_heatmap.png    — GPU util & mem by (model, size)
  5. speedup.png             — training speedup factor vs 2-worker baseline

Usage:
    python benchmark/reporter.py [--csv outputs/benchmark/summary.csv]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
PLOT_DIR = Path("outputs/benchmark/plots")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _size_order(df: pd.DataFrame) -> list[str]:
    """Return dataset sizes sorted by numeric value."""
    def _num(s: str) -> int:
        s = s.lower().replace("k", "000").replace("m", "000000")
        try:
            return int(s)
        except ValueError:
            return 0
    sizes = df["size_tag"].unique().tolist()
    return sorted(sizes, key=_num)


def _save(fig: plt.Figure, name: str) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ── Plot 1: timings breakdown by n_workers (averaged over sizes & models) ──────

def plot_timings_by_workers(df: pd.DataFrame) -> None:
    phases = ["t_load", "t_sbert", "t_graph", "t_train", "t_eval"]
    phases = [p for p in phases if p in df.columns]
    agg = df.groupby("n_workers")[phases].mean().reset_index()

    fig, ax = plt.subplots(figsize=(9, 5))
    agg.set_index("n_workers")[phases].plot(kind="bar", ax=ax, width=0.7)
    ax.set_xlabel("Spark Workers")
    ax.set_ylabel("Time (s)")
    ax.set_title("Average Phase Timings vs Number of Workers")
    ax.legend(title="Phase", loc="upper right")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0fs"))
    _save(fig, "timings_by_workers.png")


# ── Plot 2: timings breakdown by dataset volume (averaged over workers & models)

def plot_timings_by_volume(df: pd.DataFrame) -> None:
    phases  = ["t_load", "t_sbert", "t_graph", "t_train", "t_eval"]
    phases  = [p for p in phases if p in df.columns]
    sizes   = _size_order(df)
    agg     = df.groupby("size_tag")[phases].mean().reindex(sizes)

    fig, ax = plt.subplots(figsize=(10, 5))
    agg.plot(kind="bar", ax=ax, width=0.7)
    ax.set_xlabel("Dataset Size")
    ax.set_ylabel("Time (s)")
    ax.set_title("Average Phase Timings vs Dataset Volume")
    ax.legend(title="Phase", loc="upper left")
    _save(fig, "timings_by_volume.png")


# ── Plot 3: model quality vs dataset size ─────────────────────────────────────

def plot_model_perf(df: pd.DataFrame) -> None:
    metrics = ["ndcg@10", "precision@10", "map@10"]
    metrics = [m for m in metrics if m in df.columns]
    sizes   = _size_order(df)
    models  = sorted(df["model"].unique())

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4),
                             sharey=False)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        for model in models:
            sub = df[df["model"] == model].groupby("size_tag")[metric].mean()
            sub = sub.reindex(sizes)
            ax.plot(sizes, sub.values, marker="o", label=model)
        ax.set_title(metric.upper())
        ax.set_xlabel("Dataset Size")
        ax.set_ylabel("Score")
        ax.legend()

    fig.suptitle("Model Quality vs Dataset Volume", y=1.02)
    fig.tight_layout()
    _save(fig, "model_perf.png")


# ── Plot 4: GPU resource heatmaps ─────────────────────────────────────────────

def plot_resource_heatmap(df: pd.DataFrame) -> None:
    if "gpu_util_mean" not in df.columns:
        return
    sizes  = _size_order(df)
    models = sorted(df["model"].unique())

    for col, title, fmt in [
        ("gpu_util_mean",  "GPU Utilisation (%) — training phase", ".0f"),
        ("gpu_mem_max_mb", "GPU Memory Peak (MB) — training phase", ".0f"),
    ]:
        if col not in df.columns:
            continue
        pivot = (df.groupby(["model", "size_tag"])[col]
                   .mean()
                   .unstack("size_tag")
                   .reindex(columns=sizes, index=models))
        fig, ax = plt.subplots(figsize=(7, 3 + len(models)))
        sns.heatmap(pivot, annot=True, fmt=fmt, cmap="YlOrRd", ax=ax,
                    linewidths=0.5)
        ax.set_title(title)
        ax.set_xlabel("Dataset Size")
        ax.set_ylabel("Model")
        _save(fig, f"resource_{'util' if 'util' in col else 'mem'}.png")


# ── Plot 5: speedup vs 2-worker baseline ──────────────────────────────────────

def plot_speedup(df: pd.DataFrame) -> None:
    if "t_train" not in df.columns or df["n_workers"].nunique() < 2:
        return
    baseline = (df[df["n_workers"] == df["n_workers"].min()]
                .groupby(["size_tag", "model"])["t_train"]
                .mean()
                .rename("t_baseline"))
    merged = df.merge(baseline.reset_index(), on=["size_tag", "model"])
    merged["speedup"] = merged["t_baseline"] / merged["t_train"]
    agg = merged.groupby(["n_workers", "model"])["speedup"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 4))
    for model in sorted(agg["model"].unique()):
        sub = agg[agg["model"] == model]
        ax.plot(sub["n_workers"], sub["speedup"], marker="s", label=model)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Spark Workers")
    ax.set_ylabel("Speedup (×)")
    ax.set_title("Training Speedup vs 2-Worker Baseline")
    ax.legend(title="Model")
    _save(fig, "speedup.png")


# ── Summary table ──────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    cols = ["size_tag", "model", "n_workers", "t_total",
            "ndcg@10", "gpu_util_mean"]
    cols = [c for c in cols if c in df.columns]
    print("\n── Benchmark Summary ───────────────────────────────────────────")
    print(df[cols].sort_values(["size_tag", "model", "n_workers"]).to_string(
        index=False, float_format=lambda x: f"{x:.3f}"
    ))


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="outputs/benchmark/summary.csv")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[reporter] {csv_path} not found — run benchmark/runner.py first.")
        return

    df = pd.read_csv(csv_path)
    print(f"[reporter] Loaded {len(df)} runs from {csv_path}")

    print_summary(df)
    plot_timings_by_workers(df)
    plot_timings_by_volume(df)
    plot_model_perf(df)
    plot_resource_heatmap(df)
    plot_speedup(df)

    print(f"\n[reporter] All plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
