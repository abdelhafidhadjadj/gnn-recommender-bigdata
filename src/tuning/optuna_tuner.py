"""
Optuna hyperparameter tuning.

Objective: maximise validation NDCG@K. Never touches the test set.

Outputs (outputs/tuning/):
  best_{model}.yaml    — best hyperparameters (load with --config)
  study_{model}.db     — Optuna study (resumable)
  trials_{model}.csv   — per-trial history
"""
from __future__ import annotations
import csv, os, time
from pathlib import Path
from typing import Any

import optuna
import torch
import pandas as pd
import yaml

from models import build_model
from training.trainer import train_model, build_optimizer
from training.amp_utils import AMPContext
from evaluation.metrics import compute_ranking_metrics
from config import Config, TrainConfig

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── per-model search spaces ───────────────────────────────────────────────────

def _suggest_sage(trial: optuna.Trial, cfg: Config) -> dict:
    return {
        "emb_dim":    trial.suggest_categorical("emb_dim",    [32, 64, 128]),
        "n_layers":   trial.suggest_categorical("n_layers",   [1, 2]),
        "dropout":    trial.suggest_float("dropout",          0.0, 0.4),
        "lr":         trial.suggest_float("lr",               1e-4, 1e-2, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda",       1e-6, 1e-3, log=True),
    }


def _suggest_gat(trial: optuna.Trial, cfg: Config) -> dict:
    return {
        "emb_dim":    trial.suggest_categorical("emb_dim",    [64, 128]),
        "gat_heads":  trial.suggest_categorical("gat_heads",  [2, 4]),
        "dropout":    trial.suggest_float("dropout",          0.0, 0.4),
        "lr":         trial.suggest_float("lr",               1e-4, 5e-3, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda",       1e-6, 1e-3, log=True),
    }


def _suggest_lightgcn(trial: optuna.Trial, cfg: Config) -> dict:
    return {
        "emb_dim":    trial.suggest_categorical("emb_dim",    [64, 128, 256]),
        "n_layers":   trial.suggest_categorical("n_layers",   [1, 2, 3]),
        "lr":         trial.suggest_float("lr",               1e-4, 1e-2, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda",       1e-6, 1e-3, log=True),
    }


_SUGGESTERS = {
    "sage":     _suggest_sage,
    "gat":      _suggest_gat,
    "lightgcn": _suggest_lightgcn,
}


# ── objective ─────────────────────────────────────────────────────────────────

def _make_objective(
    cfg: Config,
    num_nodes: int, n_users: int, n_items: int,
    full_edge_index: torch.Tensor,
    train_u: torch.Tensor, train_pos: torch.Tensor,
    df_val: pd.DataFrame,
    device: torch.device,
):
    suggest_fn = _SUGGESTERS[cfg.model_type]
    eval_k     = cfg.eval.k_list[0]
    n_epochs   = cfg.tune.optuna_epochs

    def objective(trial: optuna.Trial) -> float:
        p       = suggest_fn(trial, cfg)
        model   = build_model(
            cfg.model_type, num_nodes,
            p["emb_dim"],
            p.get("dropout", 0.0),
            p.get("gat_heads", cfg.model.gat_heads),
            p.get("n_layers",  cfg.model.n_layers),
            use_residual=cfg.model.use_residual,
        ).to(device)

        trial_cfg = TrainConfig(
            num_epochs    = n_epochs,
            lr            = p["lr"], gat_lr = p["lr"],
            reg_lambda    = p["reg_lambda"],
            warmup_epochs = max(1, n_epochs // 10),
            min_epochs    = 0,
            patience      = n_epochs,
            eval_every    = max(1, n_epochs // 3),
            use_scheduler = cfg.train.use_scheduler,
            n_neg         = cfg.train.n_neg,
            use_all_pairs = True,
        )
        optimizer = build_optimizer(model, trial_cfg, lr_override=p["lr"])
        amp_ctx   = AMPContext(enabled=device.type == "cuda")

        train_model(
            model, optimizer,
            full_edge_index.to(device),
            train_u.to(device), train_pos.to(device),
            n_users, n_items, trial_cfg,
            amp_ctx=amp_ctx, verbose=False,
        )

        model.eval()
        with torch.no_grad():
            ranking = compute_ranking_metrics(
                model, full_edge_index, df_val, n_users, cfg.eval
            )
        m = ranking.get(eval_k, {})

        # ── Composite score (eq. 3.21) ─────────────────────────────────────
        # Score = ndcg_w × NDCG@K + prec_w × P@K + rec_w × R@K
        # Weights come from TuneConfig (default: 0.4 / 0.3 / 0.3)
        score = (
            cfg.tune.ndcg_w * m.get("NDCG", 0.0)
            + cfg.tune.prec_w * m.get("P",    0.0)
            + cfg.tune.rec_w  * m.get("R",    0.0)
        )

        trial.report(score, step=n_epochs)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return score

    return objective


# ── main entry ────────────────────────────────────────────────────────────────

def run_optuna_tuning(
    cfg: Config,
    num_nodes: int, n_users: int, n_items: int,
    full_edge_index: torch.Tensor,
    train_u: torch.Tensor, train_pos: torch.Tensor,
    df_val: pd.DataFrame,
    device: torch.device,
    n_trials: int | None = None,
    output_dir: str = "outputs/tuning",
) -> dict:
    """Run Optuna study, save results, return best params dict."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    name     = cfg.model_type
    n_trials = n_trials or cfg.tune.n_trials

    db_path = os.path.join(output_dir, f"study_{name}.db")
    study   = optuna.create_study(
        study_name     = f"gnn_{name}",
        direction      = "maximize",
        storage        = f"sqlite:///{db_path}",
        load_if_exists = True,
        pruner         = optuna.pruners.MedianPruner(n_startup_trials=3),
        sampler        = optuna.samplers.TPESampler(seed=cfg.seed),
    )

    print(f"\n[Tune] {n_trials} trials  model={name}"
          f"  objective=NDCG@{cfg.eval.k_list[0]}  db={db_path}")

    t0 = time.time()
    study.optimize(
        _make_objective(cfg, num_nodes, n_users, n_items,
                        full_edge_index, train_u, train_pos, df_val, device),
        n_trials=n_trials,
    )
    elapsed = time.time() - t0

    best = study.best_trial
    print(f"[Tune] Done  {len(study.trials)} trials  {elapsed:.0f}s")
    print(f"[Tune] Best NDCG@{cfg.eval.k_list[0]}: {best.value:.4f}")
    print(f"[Tune] Best params: {best.params}")

    # ── save YAML ─────────────────────────────────────────────────────────────
    yaml_path = os.path.join(output_dir, f"best_{name}.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump({"model_type": name, **best.params}, f, default_flow_style=False)
    print(f"[Tune] Best params -> {yaml_path}")

    # ── save CSV trial history ─────────────────────────────────────────────────
    csv_path = os.path.join(output_dir, f"trials_{name}.csv")
    rows: list[dict[str, Any]] = []
    for t in study.trials:
        row: dict[str, Any] = {
            "trial": t.number, "value": t.value,
            "state": t.state.name,
            "duration_s": t.duration.total_seconds() if t.duration else None,
        }
        row.update(t.params)
        rows.append(row)
    if rows:
        keys = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    print(f"[Tune] Trial history -> {csv_path}")

    return best.params
