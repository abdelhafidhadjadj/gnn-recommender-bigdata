"""
Single benchmark training run.

Reads preprocessed parquet from /processed/{size_tag}/,
runs SBERT + GNN training, writes results to:
  /workspace/outputs/benchmark/{size_tag}_{model}_{run_id}.json

Called by runner.py via docker exec or directly inside the trainer container.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Make src importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from benchmark.resource_monitor import ResourceMonitor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-tag",    required=True,
                        help="e.g. 1k, 5k, 10k")
    parser.add_argument("--model",       default="sage",
                        choices=["sage", "gat", "lightgcn"])
    parser.add_argument("--n-epochs",    type=int, default=50)
    parser.add_argument("--n-workers",   type=int, default=2,
                        help="Number of Spark workers used (metadata only)")
    parser.add_argument("--processed-dir", default="/workspace/processed",
                        help="Root dir with parquet outputs from Spark")
    parser.add_argument("--output-dir",  default="/workspace/outputs/benchmark")
    parser.add_argument("--run-id",      default="0")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    processed_root = Path(args.processed_dir) / args.size_tag
    meta_path = processed_root / "meta.json"
    spark_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_one] device={device}  model={args.model}  "
          f"size={args.size_tag}  epochs={args.n_epochs}")

    monitor = ResourceMonitor(interval=2.0)
    monitor.start()
    timings: dict[str, float] = {}

    # ── Load preprocessed data ─────────────────────────────────────────────────
    monitor.mark("t_load_start")
    t0 = time.perf_counter()
    from src.data.loader_spark import load_from_parquet
    data = load_from_parquet(str(processed_root))
    timings["t_load"] = round(time.perf_counter() - t0, 3)
    monitor.mark("t_load_end")

    # ── SBERT encoding ─────────────────────────────────────────────────────────
    monitor.mark("t_sbert_start")
    t1 = time.perf_counter()
    from src.data.graph_builder import build_sbert_item_projections
    from src.config import GraphConfig
    graph_cfg = GraphConfig()
    item_projections = build_sbert_item_projections(
        data["business_df"], emb_dim=64,
        embed_device=device, cfg=graph_cfg
    )
    timings["t_sbert"] = round(time.perf_counter() - t1, 3)
    monitor.mark("t_sbert_end")

    # ── Graph construction ─────────────────────────────────────────────────────
    monitor.mark("t_graph_start")
    t2 = time.perf_counter()
    from src.data.graph_builder import build_graph, build_ui_edges
    from src.config import DataConfig
    data_cfg = DataConfig()
    ui_edge_index, ui_edge_values = build_ui_edges(data["train_df"], data_cfg)
    edge_index, edge_weight = build_graph(
        train_edge_index=ui_edge_index,
        train_edge_values=ui_edge_values,
        train_review_df_full=data["train_df"],
        business_df_ordered=data["business_df"],
        n_users=data["n_users"],
        embed_device=device,
        graph_device=device,
        graph_cfg=graph_cfg,
    )
    timings["t_graph"] = round(time.perf_counter() - t2, 3)
    monitor.mark("t_graph_end")

    # ── GNN Training ───────────────────────────────────────────────────────────
    monitor.mark("t_train_start")
    t3 = time.perf_counter()

    from src.config import Config
    from src.models import build_model
    from src.training.trainer import train_model, build_optimizer
    from src.training.amp_utils import AMPContext
    from src.data.preprocessing import build_train_test
    from src.data.graph_builder import warm_start_item_embeddings
    from src.evaluation.metrics import evaluate_model

    cfg = Config()
    cfg.model.emb_dim    = 64
    cfg.train.num_epochs = args.n_epochs
    cfg.train.eval_every = max(1, args.n_epochs // 10)
    cfg.model_type       = args.model

    n_nodes = data["n_users"] + data["n_items"]
    model = build_model(
        args.model,
        num_nodes=n_nodes,
        emb_dim=cfg.model.emb_dim,
        dropout=cfg.model.dropout,
    ).to(device)

    warm_start_item_embeddings(model, item_projections, data["n_users"])

    # Build train pairs from edge_index
    train_u   = data["train_df"]["user_id"].values
    train_pos = (data["train_df"]["item_id"].values - data["n_users"])
    train_u   = torch.tensor(train_u,   dtype=torch.long, device=device)
    train_pos = torch.tensor(train_pos, dtype=torch.long, device=device)

    optimizer = build_optimizer(model, cfg.train)
    amp_ctx   = AMPContext(enabled=(device.type == "cuda"))

    train_model(
        model, optimizer, edge_index,
        train_u, train_pos,
        n_users=data["n_users"],
        n_items=data["n_items"],
        cfg=cfg.train,
        amp_ctx=amp_ctx,
        verbose=True,
    )
    timings["t_train"] = round(time.perf_counter() - t3, 3)
    monitor.mark("t_train_end")

    # ── Evaluation ─────────────────────────────────────────────────────────────
    monitor.mark("t_eval_start")
    t4 = time.perf_counter()
    test_metrics = evaluate_model(
        model, edge_index,
        data["test_df"],
        n_users=data["n_users"],
        cfg=cfg.eval,
    )
    timings["t_eval"] = round(time.perf_counter() - t4, 3)
    monitor.mark("t_eval_end")
    best_metrics = {}

    monitor.stop()
    timings["t_total"] = round(sum(timings.values()), 3)

    # ── Collect resource stats ─────────────────────────────────────────────────
    resources = {
        "sbert":  monitor.phase_stats("t_sbert_start",  "t_sbert_end"),
        "graph":  monitor.phase_stats("t_graph_start",  "t_graph_end"),
        "train":  monitor.phase_stats("t_train_start",  "t_train_end"),
        "eval":   monitor.phase_stats("t_eval_start",   "t_eval_end"),
    }

    # ── Write result ───────────────────────────────────────────────────────────
    result = {
        "run_id":       args.run_id,
        "size_tag":     args.size_tag,
        "model":        args.model,
        "n_workers":    args.n_workers,
        "n_epochs":     args.n_epochs,
        "device":       str(device),
        "n_users":      data["n_users"],
        "n_items":      data["n_items"],
        "n_edges":      data["n_edges"],
        "timings":      timings,
        "spark_timings": spark_meta.get("timings", {}),
        "resources":    resources,
        "val_metrics":  best_metrics,
        "test_metrics": test_metrics,
    }

    out_file = out_dir / f"{args.size_tag}_{args.model}_w{args.n_workers}_{args.run_id}.json"
    out_file.write_text(json.dumps(result, indent=2, default=str))
    print(f"\n[train_one] Done → {out_file}")
    print(f"  timings:      {timings}")
    print(f"  test_metrics: {test_metrics}")


if __name__ == "__main__":
    main()
