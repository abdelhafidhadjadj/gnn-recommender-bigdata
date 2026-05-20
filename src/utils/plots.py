"""
Training curve and model comparison plots/reports.
Saves to outputs/plots/ and outputs/reports/.
"""
from __future__ import annotations
import json, os
from pathlib import Path


def save_training_curve(
    history: list[float],
    model_name: str,
    output_dir: str = "outputs/plots",
) -> str | None:
    """Save BPR loss curve as PNG. Returns path or None if matplotlib unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(history) + 1))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, history, linewidth=1.5, color="#2563eb")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BPR Loss")
    ax.set_title(f"{model_name.upper()} — Training Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, f"{model_name}_training_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_metrics_json(
    results: dict,
    model_name: str,
    output_dir: str = "outputs/metrics",
) -> str:
    """Save evaluation results dict as JSON."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(output_dir, f"{model_name}_metrics.json")

    def _make_serialisable(obj):
        if isinstance(obj, dict):
            return {k: _make_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_make_serialisable(v) for v in obj]
        try:
            return float(obj)
        except (TypeError, ValueError):
            return str(obj)

    with open(path, "w") as f:
        json.dump(_make_serialisable(results), f, indent=2)
    return path


def generate_comparison_report(
    metrics_dir: str = "outputs/metrics",
    report_path: str = "outputs/reports/model_comparison.md",
) -> str | None:
    """
    Read all *_metrics.json files and write a Markdown comparison table.
    Returns the report path, or None if no metric files found.
    """
    metrics_dir_p = Path(metrics_dir)
    files = sorted(metrics_dir_p.glob("*_metrics.json"))
    if not files:
        return None

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        model = f.stem.replace("_metrics", "")
        row = {"Model": model.upper()}
        row["RMSE"]        = f"{data.get('rmse', 0):.4f}"
        row["MAE"]         = f"{data.get('mae',  0):.4f}"
        for k in [5, 10, 20]:
            rk = data.get("ranking", {}).get(str(k), {})
            row[f"P@{k}"]    = f"{rk.get('P',    0):.4f}"
            row[f"R@{k}"]    = f"{rk.get('R',    0):.4f}"
            row[f"NDCG@{k}"] = f"{rk.get('NDCG', 0):.4f}"
        rows.append(row)

    if not rows:
        return None

    headers = list(rows[0].keys())
    sep     = ["---"] * len(headers)

    lines = [
        "# Model Comparison\n",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep)     + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[h] for h in headers) + " |")

    lines += [
        "",
        "_Generated automatically. Higher is better for all ranking metrics._",
        "_Lower is better for RMSE and MAE._",
    ]

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return report_path
