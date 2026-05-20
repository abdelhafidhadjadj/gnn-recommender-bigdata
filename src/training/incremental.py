"""
Incremental fine-tuning (Phase 6) — training_mode = "incremental".

Flow:
  1. Load checkpoint  → model + DynamicLabelEncoders + train history
  2. Load new CSV     → detect new users / items
  3. Extend encoders  → add_new() preserves existing mappings
  4. Extend model     → embedding table grows; old weights preserved; conv unchanged
  5. Rebuild graph    → old interactions (offset-adjusted) + new interactions
  6. Mix data         → new_data (70 %) + replay_buffer sample (30 %)
  7. Fine-tune        → lower LR, fewer epochs, cosine schedule
  8. Save checkpoint  → versioned, includes updated encoders + replay buffer

Embedding table layout after extension:
  [0 .. n_old_users-1]                 existing user rows   (UNCHANGED)
  [n_old_users .. n_users-1]           NEW user rows        (Xavier init)
  [n_users .. n_users+n_old_items-1]   existing item rows   (SHIFTED by n_new_users)
  [n_users+n_old_items .. end]         NEW item rows        (Xavier init)

The shift of existing item rows is automatic because the item global index
is always (item_encoded_id + current_n_users). As n_users grows, the item
rows move to the correct new positions in the extended table.
"""
from __future__ import annotations
import copy
import pandas as pd
import torch
import torch.nn as nn

from config import Config
from models import build_model
from data.preprocessing import DynamicLabelEncoder
from data.graph_builder import build_ui_edges
from data.replay_buffer import ReplayBuffer
from training.trainer import train_model, build_optimizer
from training.amp_utils import AMPContext
from utils.checkpoint import CheckpointManager


# ── Embedding table extension ─────────────────────────────────────────────────

def extend_model_embeddings(
    model: nn.Module,
    n_new_users: int,
    n_new_items: int,
    n_old_users: int,
    cfg: Config,
    seed: int = 42,
) -> tuple[nn.Module, int, int]:
    """
    Rebuild the model with an extended embedding table.

    Existing user/item rows are copied to their correct new positions.
    New rows are Xavier-initialised.  Conv layer weights are copied unchanged.

    Args:
        model:       trained model (may be DDP-wrapped)
        n_new_users: number of new users to add
        n_new_items: number of new items to add
        n_old_users: number of users BEFORE this extension
        cfg:         Config (model architecture params)
        seed:        RNG seed for new-row initialisation

    Returns:
        (extended_model, n_users_new, n_items_new)
    """
    # Unwrap DDP if present
    base = model.module if hasattr(model, "module") else model
    old_w = base.embeddings.weight.data.cpu()
    n_old_items = old_w.shape[0] - n_old_users

    n_users_new = n_old_users + n_new_users
    n_items_new = n_old_items + n_new_items
    num_nodes_new = n_users_new + n_items_new

    if n_new_users == 0 and n_new_items == 0:
        return model, n_old_users, n_old_items

    # Build fresh model with new table size
    torch.manual_seed(seed)
    new_model = build_model(
        cfg.model_type, num_nodes_new,
        cfg.model.emb_dim, cfg.model.dropout, cfg.model.gat_heads,
        cfg.model.n_layers, cfg.model.use_residual,
    )

    with torch.no_grad():
        nw = new_model.embeddings.weight.data
        # 1. Old users → same positions
        nw[:n_old_users].copy_(old_w[:n_old_users])
        # 2. New users → Xavier (already initialised by build_model)
        # 3. Old items → shifted by n_new_users
        nw[n_users_new: n_users_new + n_old_items].copy_(
            old_w[n_old_users: n_old_users + n_old_items]
        )
        # 4. New items → Xavier (already initialised)

    # Copy conv / dropout / other param groups unchanged
    old_state = base.state_dict()
    new_state = new_model.state_dict()
    for key, val in old_state.items():
        if key != "embeddings.weight":
            new_state[key] = val.clone()
    new_model.load_state_dict(new_state)

    return new_model, n_users_new, n_items_new


# ── Incremental training orchestration ────────────────────────────────────────

def run_incremental(cfg: Config, dev, rank: int = 0) -> None:
    """
    Incremental fine-tuning from a saved checkpoint.

    Requires:
        cfg.incremental.ckpt_path    — path to a v2 checkpoint
        cfg.incremental.new_data_csv — path to new interactions CSV
    """
    inc = cfg.incremental

    # ── 1. Load checkpoint ────────────────────────────────────────────────────
    print(f"\n[Incremental] Loading checkpoint: {inc.ckpt_path}")
    ckpt = CheckpointManager.load(inc.ckpt_path, dev.device)

    user_enc = ckpt.get("user_encoder")
    item_enc = ckpt.get("item_encoder")

    if not isinstance(user_enc, DynamicLabelEncoder):
        raise RuntimeError(
            "The checkpoint uses a standard sklearn LabelEncoder (not DynamicLabelEncoder).\n"
            "Re-train from scratch with training_mode=scratch to generate a v2 checkpoint."
        )

    n_old_users = len(user_enc.classes_)
    n_old_items = len(item_enc.classes_)

    print(f"[Incremental] Checkpoint v{ckpt['version']}  epoch {ckpt['epoch']}")
    print(f"[Incremental] Known: {n_old_users:,} users  {n_old_items:,} items")

    # Override cfg.model from checkpoint so extend_model_embeddings uses the
    # same architecture that was trained (hardware-adaptive config may differ).
    mc = ckpt["model_config"]
    cfg.model.emb_dim      = mc["emb_dim"]
    cfg.model.dropout      = mc["dropout"]
    cfg.model.gat_heads    = mc.get("gat_heads", 4)
    cfg.model.n_layers     = mc.get("n_layers", 1)
    cfg.model.use_residual = mc.get("use_residual", True)
    cfg.model_type         = mc["model_type"]

    # ── 2. Load new interactions ──────────────────────────────────────────────
    print(f"[Incremental] Loading new data: {inc.new_data_csv}")
    new_raw = pd.read_csv(inc.new_data_csv)
    rating_col = "stars" if "stars" in new_raw.columns else "rating"

    # ── 3. Detect and extend encoders ─────────────────────────────────────────
    new_user_ids = [u for u in new_raw["user_id"].unique()
                    if not user_enc.is_known(u)]
    new_item_ids = [b for b in new_raw["business_id"].unique()
                    if not item_enc.is_known(b)]

    n_added_users = user_enc.add_new(new_user_ids)
    n_added_items = item_enc.add_new(new_item_ids)
    n_users = len(user_enc.classes_)
    n_items = len(item_enc.classes_)

    print(f"[Incremental] New entities: +{n_added_users} users  +{n_added_items} items")
    print(f"[Incremental] Total now:    {n_users:,} users  {n_items:,} items")

    # ── 4. Encode new interactions ────────────────────────────────────────────
    new_df = new_raw[
        new_raw["user_id"].isin(user_enc.classes_) &
        new_raw["business_id"].isin(item_enc.classes_)
    ].copy()
    new_df["user_id"]  = user_enc.transform(new_df["user_id"].values)
    new_df["item_id"]  = item_enc.transform(new_df["business_id"].values) + n_users
    new_df["rating"]   = new_df[rating_col].astype(float)
    new_df = new_df[["user_id", "item_id", "rating"]].reset_index(drop=True)

    print(f"[Incremental] New interactions (encoded): {len(new_df):,}")

    # ── 5. Extend model embedding table ───────────────────────────────────────
    old_model = CheckpointManager.build_model_from_ckpt(ckpt, dev.device, build_model)
    new_model, n_users_ext, n_items_ext = extend_model_embeddings(
        old_model, n_added_users, n_added_items, n_old_users, cfg, seed=cfg.seed
    )
    new_model = new_model.to(dev.device)
    num_nodes = n_users_ext + n_items_ext

    # ── 6. Rebuild training graph ─────────────────────────────────────────────
    extra = ckpt.get("extra", {})

    if inc.new_data_only:
        # Strict new-data-only: graph and BPR pairs come from the new CSV only.
        # Old user/item embeddings that don't appear in new_df receive no gradient
        # (they just don't show up in any positive/negative BPR pair).
        print("[Incremental] new-data-only mode: skipping old interactions and replay buffer.")
        graph_df = new_df.copy()
    else:
        old_train = extra.get("train_interactions")
        if old_train is not None:
            old_df = pd.DataFrame({
                "user_id": old_train["user_ids"],
                # Old item global IDs had offset n_old_users; now offset is n_users_ext
                "item_id": old_train["item_ids_local"].astype(int) + n_users_ext,
                "rating":  old_train["ratings"],
            })
            graph_df = pd.concat([old_df, new_df], ignore_index=True)
        else:
            print("[Incremental] WARNING: no train_interactions in checkpoint — using new data only.")
            graph_df = new_df.copy()

    graph_df = graph_df.drop_duplicates(
        subset=["user_id", "item_id"], keep="last"
    ).reset_index(drop=True)

    print(f"[Incremental] Graph interactions: {len(graph_df):,}")

    from config import DataConfig
    data_cfg = DataConfig(rating_thresh=cfg.data.rating_thresh)
    full_edge_index, _ = build_ui_edges(graph_df, data_cfg)
    full_edge_index = full_edge_index.to(dev.device)

    # ── 7. Fine-tune data: new + replay (30 %) — skipped in new_data_only mode ─
    replay_buf: ReplayBuffer = extra.get(
        "replay_buffer", ReplayBuffer(inc.replay_capacity)
    )

    if inc.new_data_only:
        mixed_df = new_df.copy()
        mixed_df["user_id"] = mixed_df["user_id"].astype("int64")
        mixed_df["item_id"] = mixed_df["item_id"].astype("int64")
        print(f"[Incremental] Fine-tune set: {len(mixed_df):,} new interactions (no replay)")
    else:
        n_replay = int(
            len(new_df) * inc.replay_ratio / max(1 - inc.replay_ratio, 1e-6)
        )
        replay_df = replay_buf.sample(n_replay, current_n_users=n_users_ext)

        if len(replay_df) > 0:
            mixed_df = pd.concat([new_df, replay_df], ignore_index=True)
        else:
            mixed_df = new_df.copy()

        # Ensure numeric types (concat with empty df can produce object dtype)
        mixed_df["user_id"] = mixed_df["user_id"].astype("int64")
        mixed_df["item_id"] = mixed_df["item_id"].astype("int64")
        print(
            f"[Incremental] Fine-tune set: {len(new_df):,} new  "
            f"+ {len(replay_df):,} replay  = {len(mixed_df):,} total"
        )

    mixed_df = mixed_df.sample(frac=1, random_state=cfg.seed).reset_index(drop=True)

    train_u   = torch.tensor(mixed_df["user_id"].values, dtype=torch.long).to(dev.device)
    train_pos = (torch.tensor(mixed_df["item_id"].values, dtype=torch.long)
                 - n_users_ext).to(dev.device)

    # ── 8. Fine-tune ──────────────────────────────────────────────────────────
    fine_cfg                = copy.deepcopy(cfg.train)
    fine_cfg.lr             = fine_cfg.lr * inc.finetune_lr_scale
    fine_cfg.gat_lr         = fine_cfg.gat_lr * inc.finetune_lr_scale
    fine_cfg.num_epochs     = inc.finetune_epochs
    fine_cfg.warmup_epochs  = inc.warmup_epochs
    fine_cfg.min_epochs     = 0      # allow early stopping from first check
    fine_cfg.use_all_pairs  = True   # fine-tune set is small; always use all

    eff_lr    = fine_cfg.gat_lr if cfg.model_type == "gat" else fine_cfg.lr
    optimizer = build_optimizer(new_model, fine_cfg, lr_override=eff_lr)
    amp_ctx   = AMPContext(enabled=dev.num_gpus > 0)
    ckpt_mgr  = CheckpointManager(cfg.ckpt.dir, cfg.model_type, cfg.ckpt.keep_last_n)

    print(
        f"[Incremental] Fine-tuning  epochs={fine_cfg.num_epochs}"
        f"  lr={eff_lr:.5f}  AMP={amp_ctx.enabled}"
    )

    train_model(
        new_model, optimizer, full_edge_index, train_u, train_pos,
        n_users_ext, n_items_ext, fine_cfg,
        amp_ctx=amp_ctx, rank=rank,
    )

    # ── 9. Update replay buffer ───────────────────────────────────────────────
    replay_buf.add(new_df, n_users=n_users_ext)

    # ── 10. Save versioned checkpoint ─────────────────────────────────────────
    # The incremental model is always the most up-to-date — ensure it overwrites
    # sage_best.pt by using a val_score that strictly increases with each run.
    incremental_version_score = float(ckpt.get("version", 0) + 1) * 1e6

    ckpt_mgr.save(
        model       = new_model,
        optimizer   = optimizer,
        epoch       = ckpt["epoch"] + fine_cfg.num_epochs,
        val_score   = incremental_version_score,
        cfg         = cfg,
        num_nodes   = num_nodes,
        user_encoder= user_enc,
        item_encoder= item_enc,
        extra = {
            "n_users": n_users_ext,
            "n_items": n_items_ext,
            "prev_version": ckpt.get("version", 0),
            "train_interactions": {
                "user_ids":      graph_df["user_id"].values.astype("int32"),
                "item_ids_local": (graph_df["item_id"].values - n_users_ext).astype("int32"),
                "ratings":        graph_df["rating"].values.astype("float32"),
            },
            "replay_buffer": replay_buf,
        },
        rank=rank,
    )

    print(
        f"\n[Incremental] Done."
        f"  Model: {n_users_ext:,} users  {n_items_ext:,} items"
        f"  Replay buffer: {len(replay_buf):,} interactions"
    )
