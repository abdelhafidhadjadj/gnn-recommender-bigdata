"""
Hyperparameter optimisation for incremental fine-tuning using Optuna.

Searched params:
  - finetune_epochs   : how many epochs to fine-tune
  - finetune_lr_scale : LR multiplier applied to the base learning rate
  - replay_ratio      : fraction of replay buffer mixed into the fine-tune set

Objective (maximised) — score composite eq. 3.21 + anti-forgetting :
  composite(data) = ndcg_w × NDCG@K + prec_w × P@K + rec_w × R@K
  Score = alpha × composite(new_data) + (1-alpha) × composite(old_data)

  Default weights : ndcg_w=0.4, prec_w=0.3, rec_w=0.3  (from TuneConfig)
  Default alpha   : 0.6  (favour new interactions, penalise forgetting)
"""
from __future__ import annotations

import copy
import warnings
from typing import Any

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")


# ── Composite score helper (eq. 3.21) ────────────────────────────────────────

def _composite_at_k(
    model: torch.nn.Module,
    edge_index: torch.Tensor,
    eval_df: pd.DataFrame,
    n_users: int,
    n_items: int,
    k: int = 10,
    ndcg_w: float = 0.4,
    prec_w: float = 0.3,
    rec_w:  float = 0.3,
) -> float:
    """
    Composite ranking score at K (eq. 3.21):
        Score = ndcg_w × NDCG@K + prec_w × P@K + rec_w × R@K

    Computed as the mean over all users in eval_df with at least one
    positive item (item_id present in eval_df for that user).
    """
    model.eval()
    with torch.no_grad():
        embs = model(edge_index).cpu().numpy()

    user_embs = embs[:n_users]
    item_embs = embs[n_users: n_users + n_items]

    scores_list = []
    for uid, grp in eval_df.groupby("user_id"):
        if uid >= n_users:
            continue
        pos_local = (grp["item_id"].values - n_users)
        pos_local = pos_local[(pos_local >= 0) & (pos_local < n_items)]
        if len(pos_local) == 0:
            continue

        raw_scores = item_embs @ user_embs[uid]
        top_k  = np.argsort(raw_scores)[::-1][:k]
        hits   = np.isin(top_k, pos_local).astype(float)

        # NDCG@K
        dcg  = sum(h / np.log2(r + 2) for r, h in enumerate(hits))
        idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(pos_local), k)))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        # P@K
        prec = hits.sum() / k

        # R@K
        rec  = hits.sum() / len(pos_local) if len(pos_local) > 0 else 0.0

        # Composite (eq. 3.21)
        composite = ndcg_w * ndcg + prec_w * prec + rec_w * rec
        scores_list.append(composite)

    return float(np.mean(scores_list)) if scores_list else 0.0


# ── Single trial ─────────────────────────────────────────────────────────────

def _run_trial(
    ckpt: dict,
    new_df: pd.DataFrame,
    old_val_df: pd.DataFrame,
    cfg_base,
    finetune_epochs: int,
    finetune_lr_scale: float,
    replay_ratio: float,
    device: torch.device,
    alpha: float = 0.6,
    ndcg_w: float = 0.4,
    prec_w: float = 0.3,
    rec_w:  float = 0.3,
) -> float:
    """Train with given params, return combined composite score (eq. 3.21)."""
    import copy
    from models import build_model
    from utils.checkpoint import CheckpointManager
    from data.graph_builder import build_ui_edges
    from data.replay_buffer import ReplayBuffer
    from training.trainer import train_model, build_optimizer
    from training.amp_utils import AMPContext
    from training.incremental import extend_model_embeddings
    from config import DataConfig

    user_enc = ckpt.get("user_encoder")
    item_enc = ckpt.get("item_encoder")
    n_users  = len(user_enc.classes_)
    n_items  = len(item_enc.classes_)

    # Rebuild model from checkpoint weights (fresh copy each trial)
    model = CheckpointManager.build_model_from_ckpt(ckpt, device, build_model)

    # Rebuild graph: old + new
    extra     = ckpt.get("extra", {})
    old_train = extra.get("train_interactions")
    if old_train is not None:
        old_df = pd.DataFrame({
            "user_id": old_train["user_ids"].astype(int),
            "item_id": old_train["item_ids_local"].astype(int) + n_users,
            "rating":  old_train["ratings"].astype(float),
        })
        graph_df = pd.concat([old_df, new_df], ignore_index=True).drop_duplicates(
            subset=["user_id", "item_id"], keep="last"
        )
    else:
        graph_df = new_df.copy()

    data_cfg = DataConfig(rating_thresh=cfg_base.data.rating_thresh)
    edge_index, _ = build_ui_edges(graph_df, data_cfg)
    edge_index = edge_index.to(device)

    # Fine-tune set: new + replay
    replay_buf = extra.get("replay_buffer", ReplayBuffer(10_000))
    n_replay   = int(len(new_df) * replay_ratio / max(1 - replay_ratio, 1e-6))
    replay_df  = replay_buf.sample(n_replay, current_n_users=n_users)
    mixed_df   = pd.concat([new_df, replay_df], ignore_index=True) if len(replay_df) > 0 else new_df.copy()
    mixed_df["user_id"] = mixed_df["user_id"].astype("int64")
    mixed_df["item_id"] = mixed_df["item_id"].astype("int64")
    mixed_df   = mixed_df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_u   = torch.tensor(mixed_df["user_id"].values, dtype=torch.long).to(device)
    train_pos = (torch.tensor(mixed_df["item_id"].values, dtype=torch.long) - n_users).to(device)

    # Fine-tune config
    fine_cfg               = copy.deepcopy(cfg_base.train)
    fine_cfg.lr            = fine_cfg.lr * finetune_lr_scale
    fine_cfg.gat_lr        = fine_cfg.gat_lr * finetune_lr_scale
    fine_cfg.num_epochs    = finetune_epochs
    fine_cfg.warmup_epochs = max(1, finetune_epochs // 10)
    fine_cfg.min_epochs    = 0
    fine_cfg.use_all_pairs = True

    eff_lr    = fine_cfg.gat_lr if cfg_base.model_type == "gat" else fine_cfg.lr
    optimizer = build_optimizer(model, fine_cfg, lr_override=eff_lr)
    amp_ctx   = AMPContext(enabled=False)

    train_model(
        model, optimizer, edge_index, train_u, train_pos,
        n_users, n_items, fine_cfg, amp_ctx=amp_ctx, rank=0,
    )

    # Composite score (eq. 3.21): new data + old data anti-forgetting
    comp_new = _composite_at_k(model, edge_index, new_df, n_users, n_items, k=10,
                                ndcg_w=ndcg_w, prec_w=prec_w, rec_w=rec_w)
    comp_old = (_composite_at_k(model, edge_index, old_val_df, n_users, n_items, k=10,
                                 ndcg_w=ndcg_w, prec_w=prec_w, rec_w=rec_w)
                if len(old_val_df) > 0 else 0.0)

    return alpha * comp_new + (1 - alpha) * comp_old


# ── Public API ────────────────────────────────────────────────────────────────

def tune_incremental_hparams(
    ckpt: dict,
    new_df: pd.DataFrame,
    cfg_base,
    device: torch.device = torch.device("cpu"),
    n_trials: int = 20,
    alpha: float = 0.6,
    ndcg_w: float = 0.4,
    prec_w: float = 0.3,
    rec_w:  float = 0.3,
    callback=None,
) -> dict[str, Any]:
    """
    Run Optuna HPO for incremental fine-tuning.

    Args:
        ckpt      : loaded checkpoint dict
        new_df    : encoded new interactions (user_id, item_id, rating columns)
        cfg_base  : Config object (for architecture + base LR)
        device    : torch device
        n_trials  : number of Optuna trials
        alpha     : weight for new-data composite vs old-data composite (0.6 = favour new)
        ndcg_w    : NDCG weight in composite score (eq. 3.21, default 0.4)
        prec_w    : Precision weight in composite score (eq. 3.21, default 0.3)
        rec_w     : Recall weight in composite score (eq. 3.21, default 0.3)
        callback  : optional callable(trial_number, params, score) for progress

    Returns:
        dict with keys: finetune_epochs, finetune_lr_scale, replay_ratio, best_score
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Build old validation set from checkpoint train_interactions (20% sample)
    extra     = ckpt.get("extra", {})
    old_train = extra.get("train_interactions")
    if old_train is not None and len(old_train["user_ids"]) > 0:
        n_users = len(ckpt["user_encoder"].classes_)
        old_full = pd.DataFrame({
            "user_id": old_train["user_ids"].astype(int),
            "item_id": old_train["item_ids_local"].astype(int) + n_users,
            "rating":  old_train["ratings"].astype(float),
        })
        # Sample 20% as validation (max 500 interactions for speed)
        n_val    = min(500, max(10, int(len(old_full) * 0.20)))
        old_val  = old_full.sample(n=n_val, random_state=42).reset_index(drop=True)
    else:
        old_val = pd.DataFrame(columns=["user_id", "item_id", "rating"])

    def objective(trial: optuna.Trial) -> float:
        params = {
            "finetune_epochs":    trial.suggest_categorical("finetune_epochs", [5, 10, 20, 30, 50]),
            "finetune_lr_scale":  trial.suggest_categorical("finetune_lr_scale", [0.01, 0.05, 0.1, 0.2]),
            "replay_ratio":       trial.suggest_categorical("replay_ratio", [0.0, 0.1, 0.2, 0.3, 0.5]),
        }
        score = _run_trial(
            ckpt=ckpt,
            new_df=new_df,
            old_val_df=old_val,
            cfg_base=cfg_base,
            device=device,
            alpha=alpha,
            ndcg_w=ndcg_w,
            prec_w=prec_w,
            rec_w=rec_w,
            **params,
        )
        if callback:
            callback(trial.number + 1, params, score)
        return score

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best["best_score"] = study.best_value

    return best
