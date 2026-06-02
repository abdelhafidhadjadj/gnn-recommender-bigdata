"""
Central configuration.

Phase 1 additions:
  CheckpointConfig  — versioned checkpoint settings
  Config.seed       — master random seed
  Config.training_mode — "scratch" | "incremental"
  Config.graph_mode    — "auto" | "full_batch" | "neighbor_loader"
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


# ── Data ──────────────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    data_dir: str = "data/raw"
    business_file: str = "yelp_academic_dataset_business_healthandmedical.csv"
    user_file: str = "yelp_academic_dataset_user_healthandmedical.csv"
    review_file: str = "yelp_academic_dataset_review_healthandmedical.csv"
    max_users: int = 0        # 0 = no limit (load all)
    max_reviews: int = 0      # 0 = no limit (load all)
    val_size: float = 0.15    # 70 / 15 / 15 split
    test_size: float = 0.15
    random_state: int = 1
    rating_thresh: int = 1
    relevance_thresh: float = 4.0


# ── Graph ─────────────────────────────────────────────────────────────────────

@dataclass
class GraphConfig:
    sbert_model: str = "all-mpnet-base-v2"
    sbert_batch_size: int = 256
    time_decay_days: float = 365.0
    # Item-item SBERT similarity edges (disabled — causes over-smoothing)
    use_item_item_edges: bool = False
    k_neighbors: int = 5
    # SBERT warm-start: initialise item embedding rows with projected SBERT vectors.
    # Items start content-aware (categories text); BPR fine-tunes from there.
    # Users keep Xavier random init (no text features available).
    use_sbert_item_init: bool = True   # warm-start items avec SBERT (catégories text → emb_dim)
    # Neighbor sampling depth for LinkNeighborLoader (Phase 4)
    num_neighbors: List[int] = field(default_factory=lambda: [15, 10])


# ── Model ─────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    emb_dim: int = 64
    dropout: float = 0.1
    gat_heads: int = 4
    n_layers: int = 1        # 1 layer avoids over-smoothing on sparse UI graph
    use_residual: bool = True


# ── Training ──────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    num_epochs: int = 200
    final_epochs: int = 250          # used after Optuna tuning
    batch_size: int = 1024           # used only when use_all_pairs=False
    lr: float = 0.005                # GraphSAGE
    gat_lr: float = 0.001            # GAT attention is LR-sensitive
    warmup_epochs: int = 10
    grad_clip: float = 1.0
    n_neg: int = 10                  # BPR negatives per positive
    optimizer: str = "adam"          # "adam" | "adamw"
    reg_lambda: float = 1e-5         # L2 weight in BPR loss
    use_scheduler: bool = True       # cosine annealing after warmup
    use_all_pairs: bool = True       # use all training pairs each epoch
    min_epochs: int = 80             # early stopping cannot fire before this
    patience: int = 15               # early-stop patience (× eval_every epochs)
    eval_every: int = 10             # evaluate val NDCG every N epochs


# ── Hyperparameter tuning ──────────────────────────────────────────────────────

@dataclass
class TuneConfig:
    n_trials: int = 25
    optuna_epochs: int = 30
    emb_dim_choices: List[int] = field(default_factory=lambda: [32, 64, 128])
    lr_low: float = 1e-4
    lr_high: float = 1e-2
    dropout_low: float = 0.05
    dropout_high: float = 0.50
    ndcg_w: float = 0.4
    prec_w: float = 0.3
    rec_w: float = 0.3
    eval_k: int = 10


# ── Evaluation ────────────────────────────────────────────────────────────────

@dataclass
class EvalConfig:
    k_list: List[int] = field(default_factory=lambda: [5, 10, 20])
    max_eval_users: int = 5000
    relevance_thresh: float = 3.0   # 3.0 = tout item interagi est pertinent (dataset médical sparse)
    # Phase 2 additions (metrics overhaul) — kept as config now, wired in P2
    use_global_precision: bool = True
    use_k_filter: bool = True        # only evaluate users with ≥K test interactions
    report_all_users: bool = True    # also report over users with ≥1 relevant item


# ── Checkpoint ────────────────────────────────────────────────────────────────

@dataclass
class CheckpointConfig:
    dir: str = "checkpoints"
    keep_last_n: int = 3             # keep last N periodic checkpoints + best
    save_every_n_epochs: int = 20    # periodic save interval
    save_best: bool = True           # always overwrite best.pt when val improves


# ── Incremental training ───────────────────────────────────────────────────────

@dataclass
class IncrementalConfig:
    """Settings for training_mode = 'incremental'."""
    ckpt_path:         str   = ""      # checkpoint to resume from (required)
    new_data_csv:      str   = ""      # CSV with new interactions (required)
    finetune_epochs:   int   = 20      # fine-tune epochs (much less than scratch)
    finetune_lr_scale: float = 0.1     # LR = original_lr × scale (lower LR)
    warmup_epochs:     int   = 2       # short warmup for fine-tuning
    replay_ratio:      float = 0.3     # fraction of each batch from replay buffer
    replay_capacity:   int   = 10_000  # max interactions stored in replay buffer
    new_data_only:     bool  = False   # if True: train on new CSV only, skip old interactions + replay


# ── Debug preset ──────────────────────────────────────────────────────────────

@dataclass
class DebugConfig:
    """
    Minimal settings for CPU smoke testing and fast iteration.
    Applied automatically when --debug flag is passed or by HardwareProfile.
    Mirrors the 'debug' tier in utils/hardware.py::build_adaptive_config().
    """
    emb_dim:         int  = 16
    num_epochs:      int  = 3
    warmup_epochs:   int  = 1
    batch_size:      int  = 32
    n_neg:           int  = 2
    min_epochs:      int  = 1
    eval_every:      int  = 1
    patience:        int  = 1
    num_workers:     int  = 0
    use_scheduler:   bool = False


# ── Top-level config ──────────────────────────────────────────────────────────

@dataclass
class Config:
    data:  DataConfig       = field(default_factory=DataConfig)
    graph: GraphConfig      = field(default_factory=GraphConfig)
    model: ModelConfig      = field(default_factory=ModelConfig)
    train: TrainConfig      = field(default_factory=TrainConfig)
    tune:  TuneConfig       = field(default_factory=TuneConfig)
    eval:  EvalConfig       = field(default_factory=EvalConfig)
    ckpt:        CheckpointConfig  = field(default_factory=CheckpointConfig)
    incremental: IncrementalConfig = field(default_factory=IncrementalConfig)

    model_type: str = "sage"         # "sage" | "gat" | "lightgcn"

    # Phase 1 additions
    seed: int = 42
    training_mode: str = "scratch"   # "scratch" | "incremental"
    graph_mode: str = "auto"         # "auto" | "full_batch" | "neighbor_loader"

    def effective_lr(self) -> float:
        """Return the correct learning rate for the chosen model type."""
        return self.train.gat_lr if self.model_type == "gat" else self.train.lr
