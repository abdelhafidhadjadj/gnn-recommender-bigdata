"""
Distributed GNN training with N logical GPU partitions.

Each partition is fixed to 2 GB of GPU memory via set_per_process_memory_fraction.
Uses PyTorch DDP with gloo backend — works on a single physical GPU without CUDA MPS.
On a multi-GPU server (4× A100 etc.) the same code uses NCCL automatically.

Architecture:
  Main process:
    1. Load parquet data
    2. SBERT encoding (GPU, monitored)
    3. Build PyG graph (GPU, monitored)
    4. Save tensors to tmpdir
    5. Launch N worker processes via mp.spawn

  Each worker (rank r of N):
    - Limits GPU memory to 2 GB
    - Inits gloo process group
    - Loads edge_index + pair shard r::N
    - Wraps model in DDP
    - Trains (BPR loss, DDP allreduce averages gradients each step)

  Rank 0:
    - Evaluates on test set
    - Writes worker_result.json to tmpdir

  Main process:
    - Reads worker result
    - Assembles final JSON → outputs/benchmark/

Usage:
  python benchmark/train_distributed.py \\
    --size-tag 1k --model sage \\
    --n-partitions 2 --n-epochs 200 \\
    --n-workers 2 --run-id 0001
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from benchmark.resource_monitor import ResourceMonitor

GPU_MEM_PER_PARTITION_GB = 2.0


# ── DDP worker ────────────────────────────────────────────────────────────────

def _worker(rank: int, world_size: int, args, tmp_dir: str,
            data_meta: dict, result_path: str) -> None:
    """
    Standard worker (world_size=1): plain single-GPU training, no DDP.
    Distributed worker (world_size>1): gloo DDP across N GPU partitions.
    """
    distributed = world_size > 1
    device = torch.device("cuda:0")

    # ── Fix 2 GB GPU memory per partition (standard=1×2GB, bigdata=N×2GB) ──────
    total_bytes = torch.cuda.get_device_properties(0).total_memory
    total_gb    = total_bytes / (1024 ** 3)
    fraction    = min(GPU_MEM_PER_PARTITION_GB / total_gb, 1.0)
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)

    # ── Init DDP process group (bigdata only) ─────────────────────────────────
    if distributed:
        store_path = f"/tmp/gloo_store_{args.run_id}"
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{store_path}",
            rank=rank,
            world_size=world_size,
        )

    # ── Load shared tensors written by main process ────────────────────────────
    edge_index = torch.load(f"{tmp_dir}/edge_index.pt").to(device)
    item_proj  = torch.load(f"{tmp_dir}/item_proj.pt").to(device)
    train_u    = torch.load(f"{tmp_dir}/train_u.pt").to(device)
    train_pos  = torch.load(f"{tmp_dir}/train_pos.pt").to(device)

    # ── Shard BPR pairs across partitions (bigdata) or use all pairs (standard)
    train_u_s   = train_u[rank::world_size]
    train_pos_s = train_pos[rank::world_size]

    n_users = data_meta["n_users"]
    n_items = data_meta["n_items"]

    # ── Build model ────────────────────────────────────────────────────────────
    from src.config import Config
    from src.models import build_model
    from src.training.trainer import train_model, build_optimizer
    from src.training.amp_utils import AMPContext
    from src.data.graph_builder import warm_start_item_embeddings

    cfg = Config()
    cfg.model.emb_dim    = 64
    cfg.train.num_epochs = args.n_epochs
    cfg.train.eval_every = max(1, args.n_epochs // 10)
    cfg.model_type       = args.model

    model = build_model(
        args.model,
        num_nodes=n_users + n_items,
        emb_dim=cfg.model.emb_dim,
        dropout=cfg.model.dropout,
    ).to(device)

    warm_start_item_embeddings(model, item_proj, n_users)

    # ── Wrap in DDP only for bigdata distributed mode ─────────────────────────
    if distributed:
        # gloo: no device_ids (CPU-based gradient allreduce)
        model_ddp = DDP(model, find_unused_parameters=False)
        # AMP disabled: gloo allreduce + GradScaler is unstable on single GPU
        amp_ctx = AMPContext(enabled=False)
    else:
        model_ddp = model
        amp_ctx = AMPContext(enabled=True)

    optimizer = build_optimizer(model_ddp, cfg.train)

    # ── Train (ResourceMonitor on rank 0 only) ─────────────────────────────────
    monitor = ResourceMonitor(interval=2.0) if rank == 0 else None
    if monitor:
        monitor.start()
        monitor.mark("t_train_start")

    t_train = time.perf_counter()
    train_model(
        model_ddp, optimizer, edge_index,
        train_u_s, train_pos_s,
        n_users=n_users,
        n_items=n_items,
        cfg=cfg.train,
        amp_ctx=amp_ctx,
        rank=rank,
        verbose=(rank == 0),
    )
    t_train = round(time.perf_counter() - t_train, 3)

    if monitor:
        monitor.mark("t_train_end")

    # ── Rank 0: evaluate + write result ───────────────────────────────────────
    if rank == 0:
        from src.evaluation.metrics import evaluate_model

        with open(f"{tmp_dir}/test_df.pkl", "rb") as fh:
            test_df = pickle.load(fh)

        # Unwrap DDP module for evaluation (standard mode: model is already bare)
        eval_model = model_ddp.module if distributed else model_ddp

        t_eval = time.perf_counter()
        metrics = evaluate_model(
            eval_model, edge_index, test_df,
            n_users=n_users, cfg=cfg.eval,
        )
        t_eval = round(time.perf_counter() - t_eval, 3)

        resources = {}
        if monitor:
            monitor.stop()
            resources = monitor.phase_stats("t_train_start", "t_train_end")

        worker_result = {
            "t_train":   t_train,
            "t_eval":    t_eval,
            "metrics":   metrics,
            "resources": resources,
        }
        with open(result_path, "w") as fh:
            json.dump(worker_result, fh)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-tag",      required=True,
                        help="e.g. 1k, 5k, 10k, full")
    parser.add_argument("--model",         default="sage",
                        choices=["sage", "gat", "lightgcn"])
    parser.add_argument("--n-partitions",  type=int, default=1,
                        help="GPU partitions (1=standard, 2-4=distributed). "
                             "Each gets 2 GB fixed.")
    parser.add_argument("--n-epochs",      type=int, default=200)
    parser.add_argument("--n-workers",     type=int, default=1,
                        help="Spark workers used (metadata only)")
    parser.add_argument("--processed-dir", default="/workspace/processed")
    parser.add_argument("--output-dir",    default="/workspace/outputs/benchmark")
    parser.add_argument("--run-id",        default="0")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    processed_root = Path(args.processed_dir) / args.size_tag
    meta_path = processed_root / "meta.json"
    spark_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_dist] device={device}  model={args.model}  "
          f"size={args.size_tag}  partitions={args.n_partitions}  "
          f"epochs={args.n_epochs}")

    monitor_main = ResourceMonitor(interval=2.0)
    monitor_main.start()
    timings: dict[str, float] = {}

    # ── Load parquet data ──────────────────────────────────────────────────────
    monitor_main.mark("t_load_start")
    t0 = time.perf_counter()
    from src.data.loader_spark import load_from_parquet
    data = load_from_parquet(str(processed_root))
    timings["t_load"] = round(time.perf_counter() - t0, 3)
    monitor_main.mark("t_load_end")

    # ── SBERT encoding ─────────────────────────────────────────────────────────
    monitor_main.mark("t_sbert_start")
    t1 = time.perf_counter()
    from src.data.graph_builder import build_sbert_item_projections
    from src.config import GraphConfig
    graph_cfg = GraphConfig()
    item_projections = build_sbert_item_projections(
        data["business_df"], emb_dim=64,
        embed_device=device, cfg=graph_cfg,
    )
    timings["t_sbert"] = round(time.perf_counter() - t1, 3)
    monitor_main.mark("t_sbert_end")

    # ── Graph construction ─────────────────────────────────────────────────────
    monitor_main.mark("t_graph_start")
    t2 = time.perf_counter()
    from src.data.graph_builder import build_graph, build_ui_edges
    from src.config import DataConfig
    data_cfg = DataConfig()
    ui_edge_index, ui_edge_values = build_ui_edges(data["train_df"], data_cfg)
    edge_index, _ = build_graph(
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
    monitor_main.mark("t_graph_end")

    monitor_main.stop()

    # ── BPR pairs (CPU — workers will move to GPU) ─────────────────────────────
    train_u   = torch.tensor(data["train_df"]["user_id"].values,
                             dtype=torch.long)
    train_pos = torch.tensor(
        data["train_df"]["item_id"].values - data["n_users"],
        dtype=torch.long,
    )

    # ── Save shared data for worker processes ──────────────────────────────────
    tmp_dir     = f"/tmp/ddp_bench_{args.run_id}"
    result_path = f"{tmp_dir}/worker_result.json"
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)

    torch.save(edge_index.cpu(),        f"{tmp_dir}/edge_index.pt")
    torch.save(item_projections.cpu(),  f"{tmp_dir}/item_proj.pt")
    torch.save(train_u,                 f"{tmp_dir}/train_u.pt")
    torch.save(train_pos,               f"{tmp_dir}/train_pos.pt")
    with open(f"{tmp_dir}/test_df.pkl", "wb") as fh:
        pickle.dump(data["test_df"], fh)

    # ── Cleanup any leftover gloo store file from previous run ─────────────────
    store_path = f"/tmp/gloo_store_{args.run_id}"
    Path(store_path).unlink(missing_ok=True)

    # ── Launch workers ─────────────────────────────────────────────────────────
    data_meta = {"n_users": data["n_users"], "n_items": data["n_items"]}
    t3 = time.perf_counter()

    if args.n_partitions == 1:
        # Standard path: single process, no DDP overhead
        _worker(0, 1, args, tmp_dir, data_meta, result_path)
    else:
        # Distributed path: N processes share the GPU (gloo allreduce)
        mp.spawn(
            _worker,
            args=(args.n_partitions, args, tmp_dir, data_meta, result_path),
            nprocs=args.n_partitions,
            join=True,
        )

    timings["t_train_total"] = round(time.perf_counter() - t3, 3)

    # ── Read worker result ─────────────────────────────────────────────────────
    if Path(result_path).exists():
        with open(result_path) as fh:
            worker_res = json.load(fh)
    else:
        worker_res = {}

    timings["t_train"] = worker_res.get("t_train", 0)
    timings["t_eval"]  = worker_res.get("t_eval",  0)
    timings["t_total"] = round(
        timings["t_load"] + timings["t_sbert"] +
        timings["t_graph"] + timings["t_train_total"],
        3,
    )

    # ── Resource stats ─────────────────────────────────────────────────────────
    resources = {
        "sbert": monitor_main.phase_stats("t_sbert_start", "t_sbert_end"),
        "graph": monitor_main.phase_stats("t_graph_start", "t_graph_end"),
        "train": worker_res.get("resources", {}),
    }

    test_metrics = worker_res.get("metrics", {})

    # ── Write final JSON result ────────────────────────────────────────────────
    result = {
        "run_id":                    args.run_id,
        "size_tag":                  args.size_tag,
        "model":                     args.model,
        "n_partitions":              args.n_partitions,
        "n_workers":                 args.n_workers,
        "n_epochs":                  args.n_epochs,
        "device":                    str(device),
        "gpu_mem_per_partition_gb":  GPU_MEM_PER_PARTITION_GB,
        "n_users":                   data["n_users"],
        "n_items":                   data["n_items"],
        "n_edges":                   data["n_edges"],
        "timings":                   timings,
        "spark_timings":             spark_meta.get("timings", {}),
        "resources":                 resources,
        "test_metrics":              test_metrics,
    }

    out_file = (out_dir /
                f"{args.size_tag}_{args.model}"
                f"_w{args.n_workers}_p{args.n_partitions}_{args.run_id}.json")
    out_file.write_text(json.dumps(result, indent=2, default=str))
    print(f"\n[train_dist] Done → {out_file}")
    print(f"  timings:      {timings}")
    print(f"  test_metrics: {test_metrics}")

    # Cleanup tmpdir
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
