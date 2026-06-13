"""
GNN Recommender — production CLI entry point.

Phase 3: hardware auto-detection + adaptive config.

Config precedence (highest to lowest):
  1. CLI flags      --epochs, --emb-dim, --lr  (always win)
  2. HardwareProfile   build_adaptive_config()  (auto-scales to GPU/CPU/debug)
  3. Config defaults   Config()

Usage examples:
  # CPU debug (emb_dim=16, 3 epochs, auto-detected)
  python main.py --model sage --mode scratch --data-dir data/test --debug

  # CPU full run
  python main.py --model sage --mode scratch --data-dir data/raw

  # Single GPU
  python main.py --model sage --mode scratch

  # 4-GPU via torchrun (DDP wired in Phase 5)
  torchrun --nproc_per_node=4 main.py --model sage --mode scratch

  # Evaluate saved checkpoint
  python main.py --model sage --mode evaluate --ckpt checkpoints/sage_best.pt
"""
from __future__ import annotations
import argparse
import gc
import os
import sys

# Add src/ directory to path so all imports resolve correctly when running as
# `python src/main.py` from the project root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

# ── project imports ────────────────────────────────────────────────────────────
from config import Config
from data.loader import load_raw_data, load_via_spark
from data.preprocessing import preprocess, build_train_test
from data.graph_builder import build_ui_edges, build_graph
from models import build_model
from training.trainer import train_model, build_optimizer
from training.amp_utils import AMPContext
from evaluation.metrics import (
    evaluate_model, popularity_baseline, random_baseline, print_evaluation
)
from utils.device import detect_devices
from utils.seed import set_seed
from utils.checkpoint import CheckpointManager
from utils.hardware import (
    detect_hardware, build_adaptive_config,
    resolve_graph_mode, print_hardware_report,
)
from data.samplers import build_pyg_data, make_train_loader, neighbor_sampler_available
from data.graph_builder import build_sbert_item_projections, warm_start_item_embeddings
from training.distributed import (
    init_distributed, cleanup_distributed,
    wrap_ddp, shard_bpr_pairs, barrier, is_main_process,
)
from tuning.optuna_tuner import run_optuna_tuning
from utils.plots import save_training_curve, save_metrics_json, generate_comparison_report


# ── argument parsing ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GNN Recommender — production CLI")

    # Core
    p.add_argument("--model",    default="sage", choices=["sage", "gat", "lightgcn"])
    p.add_argument("--mode",     default="scratch",
                   choices=["scratch", "evaluate", "incremental", "tune", "recommend"])
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--ckpt",     default=None,
                   help="Checkpoint path (evaluate / incremental modes)")
    p.add_argument("--new-data", default=None,
                   help="CSV with new interactions (required for --mode incremental)")
    p.add_argument("--ckpt-dir", default="checkpoints")

    # Explicit overrides (always beat hardware-adaptive defaults)
    p.add_argument("--epochs",   type=int,   default=None)
    p.add_argument("--emb-dim",  type=int,   default=None)
    p.add_argument("--lr",       type=float, default=None)
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--no-amp",   action="store_true",
                   help="Disable AMP even when CUDA is available")

    # Debug / quick-test — triggers HardwareProfile.tier == 'debug'
    p.add_argument("--debug",    action="store_true",
                   help="Debug mode: emb_dim=16, 3 epochs, batch=32 (auto via hardware)")

    # Incremental options
    p.add_argument("--new-data-only", action="store_true",
                   help="Incremental mode: train on new CSV only, skip old interactions and replay buffer")
    p.add_argument("--finetune-epochs",   type=int,   default=None, help="Override incremental finetune epochs")
    p.add_argument("--finetune-lr-scale", type=float, default=None, help="Override incremental LR scale")
    p.add_argument("--replay-ratio",      type=float, default=None, help="Override incremental replay ratio")

    # Tuning options
    p.add_argument("--trials",   type=int, default=None,
                   help="Number of Optuna trials (--mode tune)")

    # Recommend options
    p.add_argument("--user-id",  default=None,
                   help="Raw user_id string for --mode recommend")
    p.add_argument("--top-k",    type=int, default=10,
                   help="Number of recommendations to return (--mode recommend)")

    # Dataset size limits (override config defaults)
    p.add_argument("--max-reviews", type=int, default=None,
                   help="Max number of reviews to load (e.g. 1000, 5000, 50000, None=all)")
    p.add_argument("--max-users",   type=int, default=None,
                   help="Max number of users to load (None=all)")

    # Config file (YAML)
    p.add_argument("--config", default=None,
                   help="Path to YAML config file (applied after hardware defaults, "
                        "before CLI flags — CLI always wins)")

    # Output options
    p.add_argument("--output-dir", default="outputs",
                   help="Root for metrics/plots/tuning output (default: outputs)")

    # Graph mode override (full_batch | neighbor_loader | auto)
    p.add_argument("--graph-mode", default=None,
                   choices=["auto", "full_batch", "neighbor_loader"],
                   help="Override graph training mode (default: auto — selected by hardware)")

    # SBERT warm-start toggle
    p.add_argument("--no-sbert-init", action="store_true",
                   help="Disable SBERT item embedding warm-start (Xavier random init instead)")

    # HTML report
    p.add_argument("--open-report", action="store_true",
                   help="Générer et ouvrir le rapport HTML à la fin du training")

    return p.parse_args()


# ── config assembly ────────────────────────────────────────────────────────────

def _apply_cli_overrides(cfg: Config, args: argparse.Namespace) -> None:
    """
    Apply explicit CLI flags on top of the hardware-adaptive config.
    These always take the highest precedence.
    """
    cfg.model_type    = args.model
    cfg.training_mode = args.mode
    cfg.seed          = args.seed
    cfg.data.data_dir = args.data_dir
    cfg.ckpt.dir      = args.ckpt_dir

    # Optional per-run overrides
    if args.epochs:       cfg.train.num_epochs   = args.epochs
    if args.emb_dim:      cfg.model.emb_dim      = args.emb_dim
    if args.lr:           cfg.train.lr           = args.lr
    if args.max_reviews is not None: cfg.data.max_reviews = args.max_reviews
    if args.max_users   is not None: cfg.data.max_users   = args.max_users
    if args.graph_mode  is not None: cfg.graph_mode            = args.graph_mode
    if args.no_sbert_init:           cfg.graph.use_sbert_item_init = False

    # Ensure min_epochs never exceeds num_epochs so validation always fires.
    cfg.train.min_epochs = min(cfg.train.min_epochs,
                               max(0, cfg.train.num_epochs - cfg.train.eval_every))
    if args.ckpt:               cfg.incremental.ckpt_path        = args.ckpt
    if args.new_data:           cfg.incremental.new_data_csv     = args.new_data
    if args.new_data_only:      cfg.incremental.new_data_only    = True
    if args.trials:             cfg.tune.n_trials                = args.trials
    if args.finetune_epochs:    cfg.incremental.finetune_epochs  = args.finetune_epochs
    if args.finetune_lr_scale:  cfg.incremental.finetune_lr_scale = args.finetune_lr_scale
    if args.replay_ratio is not None: cfg.incremental.replay_ratio = args.replay_ratio


# ── YAML config loading ────────────────────────────────────────────────────────

def _apply_yaml(cfg: Config, path: str) -> None:
    """
    Load a YAML file and apply it to cfg.

    Precedence (highest to lowest):
      CLI flags > YAML values > hardware-adaptive defaults > Config() defaults

    Nested keys map directly to sub-config dataclasses:
      model.emb_dim  ->  cfg.model.emb_dim
      train.lr       ->  cfg.train.lr
      checkpoint.dir ->  cfg.ckpt.dir   (alias handled below)
    """
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f) or {}

    # Sub-config name → attribute on Config
    sub_map = {
        "data":        cfg.data,
        "graph":       cfg.graph,
        "model":       cfg.model,
        "train":       cfg.train,
        "tune":        cfg.tune,
        "eval":        cfg.eval,
        "checkpoint":  cfg.ckpt,   # YAML uses "checkpoint", cfg uses "ckpt"
        "ckpt":        cfg.ckpt,
        "incremental": cfg.incremental,
    }

    for key, val in data.items():
        if key in sub_map and isinstance(val, dict):
            sub = sub_map[key]
            for k, v in val.items():
                if hasattr(sub, k):
                    setattr(sub, k, v)
                else:
                    print(f"[Config] YAML warning: unknown key '{key}.{k}' — ignored")
        elif hasattr(cfg, key):
            setattr(cfg, key, val)
        else:
            print(f"[Config] YAML warning: unknown top-level key '{key}' — ignored")

    print(f"[Config] Loaded {path}")


# ── data pipeline ──────────────────────────────────────────────────────────────

def prepare_data(
    cfg: Config,
    embed_device: torch.device,
    world_size: int = 1,
    rank: int = 0,
):
    """
    Load raw data and build the training graph.

    Loading backend is selected automatically:
      world_size == 1  →  pandas  (standard mode, fast for small data)
      world_size  > 1  →  PySpark (bigdata mode, distributed I/O)

    t_load reflects actual I/O time for the active backend so that the
    compare_distributed.py report shows a meaningful pandas vs Spark delta.
    """
    import time as _time
    timings = {}

    if world_size > 1:
        # ── bigdata mode : load via Spark ─────────────────────────────────────
        if rank == 0:
            print(f"\n[Data] Loading CSVs via Spark  (world_size={world_size}) ...")
        t0 = _time.perf_counter()
        bdf, udf, rdf = load_via_spark(cfg.data, rank=rank)
        timings["t_load"] = round(_time.perf_counter() - t0, 2)
        timings["loader"] = "spark"
        if rank == 0:
            print(f"       t_load = {timings['t_load']}s  [Spark local[1] per rank]")
    else:
        # ── standard mode : load via pandas ──────────────────────────────────
        print("\n[Data] Loading CSVs via pandas  (world_size=1) ...")
        t0 = _time.perf_counter()
        bdf, udf, rdf = load_raw_data(cfg.data)
        timings["t_load"] = round(_time.perf_counter() - t0, 2)
        timings["loader"] = "pandas"
        print(f"       t_load = {timings['t_load']}s  [pandas]")

    print("[Data] Preprocessing ...")
    review_df, review_df_full, user_enc, item_enc, n_users, n_items = preprocess(
        bdf, udf, rdf
    )
    print(f"       n_users={n_users:,}  n_items={n_items:,}")

    print("[Data] Building UI edges and 70/15/15 split ...")
    uei, uev = build_ui_edges(review_df, cfg.data)
    train_u, train_pos, df_val, df_test, train_idx = build_train_test(
        uei, uev, n_users, cfg.data
    )
    print(
        f"       train={len(train_u):,}"
        f"  val={len(df_val):,}"
        f"  test={len(df_test):,}"
    )

    print("[Graph] Aligning business_df to LabelEncoder order ...")
    bdf_ordered = (
        bdf.set_index("business_id")
           .loc[item_enc.classes_]
           .reset_index()
    )

    print("[Graph] Building training graph (train edges only, no leakage) ...")
    t0 = _time.perf_counter()
    train_ei = uei[:, train_idx]
    train_ev = uev[train_idx]
    train_rf = review_df_full.iloc[train_idx].reset_index(drop=True)

    full_edge_index, _ = build_graph(
        train_ei, train_ev, train_rf,
        bdf_ordered if cfg.graph.use_item_item_edges else None,
        n_users,
        embed_device,
        torch.device("cpu"),
        cfg.graph,
    )
    timings["t_graph"] = round(_time.perf_counter() - t0, 2)
    print(f"       t_graph = {timings['t_graph']}s")
    gc.collect()

    df_train  = review_df.iloc[train_idx].copy()
    num_nodes = n_users + n_items

    return (
        full_edge_index, train_ei,
        train_u, train_pos,
        df_val, df_test, df_train,
        n_users, n_items, num_nodes,
        user_enc, item_enc,
        bdf_ordered,        # returned for SBERT warm-start in run_scratch
        timings,
    )


# ── scratch training ───────────────────────────────────────────────────────────

def run_scratch(
    cfg: Config,
    dev,
    full_edge_index: torch.Tensor,
    train_u: torch.Tensor,
    train_pos: torch.Tensor,
    df_val,
    df_test,
    df_train,
    n_users: int,
    n_items: int,
    num_nodes: int,
    user_enc,
    item_enc,
    bdf_ordered=None,
    use_amp: bool = True,
    loader=None,
    # DDP params (Phase 5)
    is_ddp: bool = False,
    rank: int = 0,
    local_rank: int = 0,
    world_size: int = 1,
) -> tuple:
    """
    Train from scratch.  Works in all three modes:
      - CPU / single GPU  (is_ddp=False)
      - 4-GPU DDP         (is_ddp=True, rank=0..3, local_rank=0..3)
    """
    eff_lr = cfg.effective_lr()

    # ── device selection ──────────────────────────────────────────────────────
    if is_ddp:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = dev.sage_device if cfg.model_type != "gat" else dev.gat_device

    # ── move graph to device ──────────────────────────────────────────────────
    edge_idx = full_edge_index.to(device)

    # ── BPR pair sharding (each rank processes a different slice) ─────────────
    t_u, t_pos = shard_bpr_pairs(train_u, train_pos, rank, world_size)
    t_u   = t_u.to(device)
    t_pos = t_pos.to(device)

    # ── build model ───────────────────────────────────────────────────────────
    model = build_model(
        cfg.model_type, num_nodes,
        cfg.model.emb_dim, cfg.model.dropout, cfg.model.gat_heads,
        cfg.model.n_layers, cfg.model.use_residual,
    ).to(device)

    # ── SBERT warm-start (all ranks run identically → same seed → same result)
    import time as _time
    t_sbert = 0.0
    if cfg.graph.use_sbert_item_init and bdf_ordered is not None:
        if is_main_process(rank):
            print("[SBERT] Warm-starting item embeddings from category text ...")
        _t0 = _time.perf_counter()
        item_proj = build_sbert_item_projections(
            bdf_ordered, cfg.model.emb_dim, dev.embed_device, cfg.graph, seed=cfg.seed
        )
        t_sbert = round(_time.perf_counter() - _t0, 2)
        warm_start_item_embeddings(model, item_proj, n_users)
        barrier()    # all ranks sync after warm-start before DDP wrap
        if is_main_process(rank):
            print(f"[SBERT] Item rows initialised  ({n_items} items, emb_dim={cfg.model.emb_dim})  t_sbert={t_sbert}s")

    # ── DDP wrap (must happen AFTER SBERT warm-start) ─────────────────────────
    if is_ddp:
        model = wrap_ddp(model, local_rank)

    # ── optimizer / AMP / checkpoint ──────────────────────────────────────────
    optimizer = build_optimizer(model, cfg.train, lr_override=eff_lr)
    amp_ctx   = AMPContext(enabled=use_amp)
    ckpt_mgr  = CheckpointManager(cfg.ckpt.dir, cfg.model_type, cfg.ckpt.keep_last_n)

    # ── validation closure (all ranks eval; same weights → same result) ───────
    def val_eval_fn(m: torch.nn.Module) -> float:
        from evaluation.metrics import compute_ranking_metrics
        m.eval()
        with torch.no_grad():
            # In DDP m is a DDP-wrapped model; forward still works normally
            ranking = compute_ranking_metrics(m, edge_idx, df_val, n_users, cfg.eval)
        # Composite criterion (eq. 3.21): 0.4×NDCG@K + 0.3×P@K + 0.3×R@K
        k   = cfg.eval.k_list[0]
        met = ranking.get(k, {})
        score = (
            cfg.tune.ndcg_w * met.get("NDCG", 0.0)
            + cfg.tune.prec_w * met.get("P",    0.0)
            + cfg.tune.rec_w  * met.get("R",    0.0)
        )
        barrier()    # sync all ranks before training resumes
        return score

    if is_main_process(rank):
        print(
            f"\n[Train] {cfg.model_type.upper()}"
            f"  emb_dim={cfg.model.emb_dim}  lr={eff_lr}"
            f"  epochs={cfg.train.num_epochs}  AMP={amp_ctx.enabled}"
            f"  world_size={world_size}"
        )

    # ── build train_interactions for checkpoint (needed by incremental mode) ────
    train_interactions_extra = {
        "n_users": n_users,
        "n_items": n_items,
        "train_interactions": {
            "user_ids":       df_train["user_id"].values.astype("int32"),
            "item_ids_local": (df_train["item_id"].values - n_users).astype("int32"),
            "ratings":        df_train["rating"].values.astype("float32"),
        },
    }

    # ── training ──────────────────────────────────────────────────────────────
    _t0_train = _time.perf_counter()
    try:
        history = train_model(
            model, optimizer, edge_idx, t_u, t_pos,
            n_users, n_items, cfg.train,
            loader    = loader,
            amp_ctx   = amp_ctx,
            ckpt_manager = ckpt_mgr,
            ckpt_cfg  = cfg.ckpt,
            full_cfg  = cfg,
            num_nodes = num_nodes,
            user_encoder = user_enc,
            item_encoder = item_enc,
            ckpt_extra   = train_interactions_extra,
            rank      = rank,
            eval_fn   = val_eval_fn,
        )
        ckpt_mgr._last_history = history   # used by save_training_curve
    except torch.cuda.OutOfMemoryError:
        if is_ddp:
            print(f"[OOM] Rank {rank}: reduce batch_size in config and relaunch.")
            cleanup_distributed()
            sys.exit(1)
        print("[OOM] Retrying with batch_size // 2 ...")
        cfg.train.batch_size  //= 2
        cfg.train.use_all_pairs = False
        optimizer = build_optimizer(model, cfg.train, lr_override=eff_lr)
        train_model(model, optimizer, edge_idx, t_u, t_pos,
                    n_users, n_items, cfg.train, amp_ctx=amp_ctx,
                    ckpt_extra=train_interactions_extra)

    # ── final checkpoint (rank 0 only via CheckpointManager) ──────────────────
    # Use 0.0 as fallback score so sage_best.pt is always written even when
    # validation never ran (e.g. --epochs < min_epochs before this fix).
    final_score = ckpt_mgr.best_score if ckpt_mgr.best_score > -float("inf") else 0.0
    ckpt_mgr.save(
        model=model, optimizer=optimizer,
        epoch=cfg.train.num_epochs, val_score=final_score,
        cfg=cfg, num_nodes=num_nodes,
        user_encoder=user_enc, item_encoder=item_enc,
        extra=train_interactions_extra,
        rank=rank,
    )

    t_train = round(_time.perf_counter() - _t0_train, 2)

    # ── final evaluation on test set — rank 0 only ────────────────────────────
    results = None
    if is_main_process(rank):
        print("\n[Eval] Final evaluation on held-out test set ...")
        _t0_eval = _time.perf_counter()
        results = evaluate_model(model, edge_idx, df_test, n_users, cfg.eval,
                                 df_train=df_train)
        t_eval = round(_time.perf_counter() - _t0_eval, 2)
        # Affichage terminal supprimé — résultats disponibles dans le rapport HTML
        # print_evaluation(results)

        print("\n[Baselines]")
        pop  = popularity_baseline(df_train, df_test, n_users, cfg.eval)
        rand = random_baseline(df_test, n_users, n_items, cfg.eval)
        for k in sorted(cfg.eval.k_list):
            print(f"  Popularity @{k}: P={pop[k]['P']:.4f}  "
                  f"HR={pop[k].get('HR',0):.4f}  NDCG={pop[k]['NDCG']:.4f}")
            print(f"  Random     @{k}: P={rand[k]['P']:.4f}  "
                  f"HR={rand[k].get('HR',0):.4f}  NDCG={rand[k]['NDCG']:.4f}")

    # ── save training curve + metrics JSON (rank 0 only) ─────────────────────
    if is_main_process(rank) and results is not None:
        output_dir = getattr(cfg, "_output_dir", "outputs")
        curve_path = save_training_curve(
            history    = getattr(ckpt_mgr, "_last_history", []),
            model_name = cfg.model_type,
            output_dir = os.path.join(output_dir, "plots"),
        )
        if curve_path:
            print(f"[Output] Training curve  -> {curve_path}")

        # Ajouter les timings système aux métriques
        results["timings"] = {
            "t_sbert":    t_sbert,
            "t_train":    t_train,
            "t_eval":     t_eval,
            "world_size": world_size,
            "loader":     "spark" if world_size > 1 else "pandas",
        }
        # Ajouter les timings data/graph (t_load, t_graph, loader) si disponibles
        if hasattr(cfg, "_data_timings"):
            results["timings"].update(cfg._data_timings)

        metrics_path = save_metrics_json(
            results    = results,
            model_name = cfg.model_type,
            output_dir = os.path.join(output_dir, "metrics"),
        )
        print(f"[Output] Metrics JSON    -> {metrics_path}")

        # ── Post-processing metrics ───────────────────────────────────────────
        try:
            import importlib.util as _ilu, os as _os
            _sp = _ilu.spec_from_file_location(
                "_compat",
                _os.path.join(_os.path.dirname(__file__), "utils", "_compat.py")
            )
            _cm = _ilu.module_from_spec(_sp)
            _sp.loader.exec_module(_cm)
            _run  = _os.path.basename(output_dir)
            # ROOT = répertoire contenant outputs/ et results_final/
            _root = _os.path.abspath(_os.path.join(
                _os.path.dirname(_os.path.abspath(metrics_path)), "..", ".."
            ))
            _cm._sync_ranking_cache(
                _os.path.abspath(metrics_path), _run, _root
            )
        except Exception as _ex:
            print(f"[PostProc] {_ex}")

        report = generate_comparison_report(
            metrics_dir = os.path.join(output_dir, "metrics"),
            report_path = os.path.join(output_dir, "reports", "model_comparison.md"),
        )
        if report:
            print(f"[Output] Comparison report -> {report}")

    return model, results, ckpt_mgr


# ── evaluate-only mode ────────────────────────────────────────────────────────

def run_evaluate(cfg: Config, dev, ckpt_path: str,
                 full_edge_index: torch.Tensor,
                 df_test, n_users: int) -> dict:
    model_device = dev.sage_device if cfg.model_type != "gat" else dev.gat_device
    ckpt  = CheckpointManager.load(ckpt_path, model_device)
    model = CheckpointManager.build_model_from_ckpt(ckpt, model_device, build_model)
    edge_idx = full_edge_index.to(model_device)

    print(f"\n[Eval] Loaded v{ckpt['version']} epoch {ckpt['epoch']}")
    results = evaluate_model(model, edge_idx, df_test, n_users, cfg.eval)
    print_evaluation(results)
    return results


# ── hyperparameter tuning mode ───────────────────────────────────────────────

def run_tune(
    cfg: Config,
    dev,
    full_edge_index: torch.Tensor,
    train_u: torch.Tensor,
    train_pos: torch.Tensor,
    df_val,
    n_users: int,
    n_items: int,
    num_nodes: int,
    n_trials: int | None = None,
    output_dir: str = "outputs",
) -> dict:
    device = dev.sage_device if cfg.model_type != "gat" else dev.gat_device
    best = run_optuna_tuning(
        cfg         = cfg,
        num_nodes   = num_nodes,
        n_users     = n_users,
        n_items     = n_items,
        full_edge_index = full_edge_index,
        train_u     = train_u,
        train_pos   = train_pos,
        df_val      = df_val,
        device      = device,
        n_trials    = n_trials,
        output_dir  = os.path.join(output_dir, "tuning"),
    )
    return best


# ── recommend mode ────────────────────────────────────────────────────────────

def run_recommend(
    cfg: Config,
    dev,
    ckpt_path: str,
    raw_user_id: str,
    top_k: int,
    bdf=None,
) -> None:
    from utils.checkpoint import CheckpointManager

    device = dev.sage_device if cfg.model_type != "gat" else dev.gat_device
    ckpt   = CheckpointManager.load(ckpt_path, device)
    model  = CheckpointManager.build_model_from_ckpt(ckpt, device, build_model)
    model.eval()

    user_enc = ckpt.get("user_encoder")
    item_enc = ckpt.get("item_encoder")

    if user_enc is None or not user_enc.is_known(raw_user_id):
        print(f"[Recommend] Unknown user_id: '{raw_user_id}'")
        print("[Recommend] Known user count:", len(user_enc.classes_) if user_enc else 0)
        return

    uid   = int(user_enc.transform([raw_user_id])[0])
    mc    = ckpt["model_config"]
    n_u   = mc["num_nodes"] - len(item_enc.classes_)

    # Build a minimal edge_index (no graph needed for embedding lookup)
    dummy_ei = torch.zeros(2, 0, dtype=torch.long, device=device)

    with torch.no_grad():
        all_emb = model(dummy_ei)
        u_emb   = all_emb[uid]                        # (emb_dim,)
        i_emb   = all_emb[n_u:]                       # (n_items, emb_dim)
        scores  = (i_emb @ u_emb).cpu().numpy()

    top_indices = scores.argsort()[::-1][:top_k]

    print(f"\n[Recommend] Top-{top_k} for user '{raw_user_id}':")
    print(f"{'Rank':<5} {'Item index':<12} {'Business ID':<26} {'Score':>8}")
    print("-" * 55)
    for rank, idx in enumerate(top_indices, 1):
        biz_id = item_enc.classes_[idx] if item_enc else str(idx)
        name   = ""
        if bdf is not None and "business_id" in bdf.columns:
            row = bdf[bdf["business_id"] == biz_id]
            if not row.empty:
                name = row.iloc[0].get("name", "")
        print(f"{rank:<5} {idx:<12} {biz_id:<26} {scores[idx]:>8.4f}  {name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── 1. Detect hardware FIRST (Phase 3) ────────────────────────────────────
    profile = detect_hardware(force_debug=args.debug)

    # ── 2. Build config: defaults → hardware → YAML → CLI (highest priority) ────
    cfg = Config()
    build_adaptive_config(cfg, profile)         # adapt to CPU/GPU/debug tier
    if args.config:
        _apply_yaml(cfg, args.config)           # YAML overrides hardware defaults
    _apply_cli_overrides(cfg, args)             # CLI flags always win

    # ── 3. DDP initialization (Phase 5) ──────────────────────────────────────
    is_ddp, rank, local_rank, world_size = init_distributed()

    # ── 4. Reproducible seeding ───────────────────────────────────────────────
    set_seed(cfg.seed, rank=rank)

    # ── 5. Device assignment (from existing utils/device.py) ──────────────────
    dev = detect_devices()

    # ── 6. Print hardware report (rank 0 only) ────────────────────────────────
    if rank == 0:
        print_hardware_report(profile, cfg)

    # ── 7. Incremental mode (Phase 6) ─────────────────────────────────────────
    if cfg.training_mode == "incremental":
        if not cfg.incremental.ckpt_path:
            print("[Error] --ckpt <path> is required for incremental mode.")
            sys.exit(1)
        if not cfg.incremental.new_data_csv:
            print("[Error] --new-data <path> is required for incremental mode.")
            sys.exit(1)
        from training.incremental import run_incremental
        run_incremental(cfg, dev, rank=rank)
        cleanup_distributed()
        return

    # ── 8. Data pipeline ──────────────────────────────────────────────────────
    (
        full_edge_index, train_ei,
        train_u, train_pos,
        df_val, df_test, df_train,
        n_users, n_items, num_nodes,
        user_enc, item_enc,
        bdf_ordered,
        data_timings,
    ) = prepare_data(cfg, dev.embed_device, world_size=world_size, rank=rank)
    cfg._data_timings = data_timings   # transmis à run_scratch pour le JSON

    # ── 9. Resolve graph mode + build loader if needed (Phase 4) ─────────────
    resolved_graph_mode = resolve_graph_mode(profile, cfg, num_nodes, cfg.model.emb_dim)
    if rank == 0:
        print(f"[Graph mode] {cfg.graph_mode!r} -> resolved: '{resolved_graph_mode}'")

    train_loader = None
    if resolved_graph_mode == "neighbor_loader":
        if not neighbor_sampler_available():
            # pyg-lib / torch-sparse not installed — graceful fallback to full-batch.
            # This is expected on the CPU dev laptop; the GPU server has both libs.
            if rank == 0:
                print(
                    "[Loader] WARNING: neighbor_loader requested but pyg-lib / "
                    "torch-sparse not installed. Falling back to full_batch.\n"
                    "         Install pyg-lib or torch-sparse on the GPU server."
                )
        else:
            pyg_data     = build_pyg_data(full_edge_index, num_nodes)
            train_loader = make_train_loader(
                pyg_data         = pyg_data,
                train_edge_index = train_ei,
                batch_size       = cfg.train.batch_size,
                num_neighbors    = cfg.graph.num_neighbors,
                profile          = profile,
                # rank / world_size added in Phase 5 (DDP)
            )
            if rank == 0:
                print(
                    f"[Loader] LinkNeighborLoader  "
                    f"batch={cfg.train.batch_size}  "
                    f"num_neighbors={cfg.graph.num_neighbors}"
                )

    if train_loader is None and rank == 0:
        print("[Loader] Full-batch (graph fits in memory)")

    # ── 10. AMP flag ──────────────────────────────────────────────────────────
    use_amp = (not args.no_amp) and (profile.num_gpus > 0)

    # ── 11. Mode dispatch ─────────────────────────────────────────────────────
    output_dir = args.output_dir
    cfg._output_dir = output_dir          # carried into run_scratch for saving

    if cfg.training_mode == "scratch":
        run_scratch(
            cfg, dev, full_edge_index, train_u, train_pos,
            df_val, df_test, df_train,
            n_users, n_items, num_nodes, user_enc, item_enc,
            bdf_ordered = bdf_ordered,
            use_amp     = use_amp,
            loader      = train_loader,
            is_ddp      = is_ddp,
            rank        = rank,
            local_rank  = local_rank,
            world_size  = world_size,
        )

        # ── Génération + ouverture rapport HTML ───────────────────────────────
        if getattr(args, "open_report", False) and rank == 0:
            try:
                import subprocess as _sp, sys as _sys, os as _os
                _root = _os.path.abspath(_os.path.join(output_dir, "..", ".."))
                _script = _os.path.join(_root, "generate_report.py")
                if _os.path.exists(_script):
                    print("[Report] Génération du rapport HTML...")
                    _sp.run([_sys.executable, _script], cwd=_root, check=True)
                    _html = _os.path.join(_root, "results_final", "report.html")
                    if _os.path.exists(_html):
                        import webbrowser as _wb
                        _wb.open(f"file:///{_html.replace(chr(92), '/')}")
                        print(f"[Report] Rapport ouvert -> {_html}")
            except Exception as _e:
                print(f"[Report] Erreur génération : {_e}")

    elif cfg.training_mode == "evaluate":
        if args.ckpt is None:
            print("[Error] --ckpt <path> is required for --mode evaluate")
            sys.exit(1)
        results = run_evaluate(cfg, dev, args.ckpt, full_edge_index, df_test, n_users)
        if rank == 0 and results:
            mp = save_metrics_json(results, cfg.model_type,
                                   os.path.join(output_dir, "metrics"))
            print(f"[Output] Metrics JSON -> {mp}")
            rp = generate_comparison_report(
                os.path.join(output_dir, "metrics"),
                os.path.join(output_dir, "reports", "model_comparison.md"),
            )
            if rp:
                print(f"[Output] Comparison report -> {rp}")

    elif cfg.training_mode == "tune":
        run_tune(
            cfg, dev, full_edge_index, train_u, train_pos, df_val,
            n_users, n_items, num_nodes,
            n_trials   = args.trials,
            output_dir = output_dir,
        )

    elif cfg.training_mode == "recommend":
        if args.ckpt is None:
            print("[Error] --ckpt <path> is required for --mode recommend")
            sys.exit(1)
        if args.user_id is None:
            print("[Error] --user-id <id> is required for --mode recommend")
            sys.exit(1)
        from data.loader import load_raw_data
        bdf_raw, _, _ = load_raw_data(cfg.data)
        run_recommend(
            cfg, dev, args.ckpt,
            raw_user_id = args.user_id,
            top_k       = args.top_k,
            bdf         = bdf_raw,
        )

    # ── 12. Teardown DDP process group ────────────────────────────────────────
    cleanup_distributed()


if __name__ == "__main__":
    main()
