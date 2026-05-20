"""
Streamlit interface for GNN Recommender.

Run:
    cd gnn_recommender
    streamlit run app.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import gc
import tempfile
import contextlib
from io import StringIO
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

from config import Config
from data.loader import load_raw_data
from data.preprocessing import preprocess, build_train_test
from data.graph_builder import build_ui_edges, build_graph
from models import build_model
from training.trainer import train_model, build_optimizer
from training.distributed import run_ddp_training
from evaluation.metrics import (
    evaluate_model, compute_ranking_metrics,
    normalize_metrics, popularity_baseline, random_baseline,
)
from tuning.optuna_tuner import run_optuna_tuning
from utils.device import detect_devices

# ── page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GNN Recommender",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── session-state defaults ─────────────────────────────────────────────────────
_DEFAULTS = dict(
    data_loaded=False, graph_built=False, trained=False, evaluated=False,
    logs=[], train_history=[],
    results=None, baseline_pop=None, baseline_rand=None, norm_metrics=None,
    business_df=None, user_df=None,
    review_df=None, review_df_full=None,
    item_enc=None,
    n_users=0, n_items=0, num_nodes=0,
    full_edge_index=None,
    train_u=None, train_pos=None,
    df_val=None, df_test=None, df_train=None,
    # raw train edges kept for fast experiment graph rebuilds
    train_ei=None, train_ev=None, train_rf=None, bdf_ordered=None,
    model=None, tmp_files=[],
    # experiment state
    exp_adopted={},           # currently adopted {use_ii_edges, n_layers, use_residual, model_type}
    exp_summary=[],           # list of result dicts for comparison table
    exp_step_status={},       # {step_key: 'pending'|'ran'|'adopted'|'skipped'}
    exp_step_metrics={},      # {step_key: results dict}
)
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

ss = st.session_state

# ── helpers ────────────────────────────────────────────────────────────────────

def log(msg: str):
    ss.logs.append(msg)


@contextlib.contextmanager
def capture(label: str = ""):
    buf = StringIO()
    with contextlib.redirect_stdout(buf):
        yield
    for line in buf.getvalue().splitlines():
        log(line)


def show_logs():
    if ss.logs:
        st.code("\n".join(ss.logs[-200:]), language="")


def save_upload(f) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(f.name).suffix)
    tmp.write(f.read()); tmp.close()
    ss.tmp_files.append(tmp.name)
    return tmp.name


def save_checkpoint(model, cfg: Config, num_nodes: int, path: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save({
        "state_dict":   model.state_dict(),
        "model_type":   cfg.model_type,
        "emb_dim":      cfg.model.emb_dim,
        "dropout":      cfg.model.dropout,
        "gat_heads":    cfg.model.gat_heads,
        "n_layers":     cfg.model.n_layers,
        "use_residual": cfg.model.use_residual,
        "num_nodes":    num_nodes,
    }, path)


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise RuntimeError("Outdated checkpoint — re-train once to regenerate.")
    model = build_model(
        ckpt["model_type"], ckpt["num_nodes"],
        ckpt["emb_dim"], ckpt["dropout"],
        ckpt.get("gat_heads",    4),
        ckpt.get("n_layers",     1),
        ckpt.get("use_residual", True),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model


def build_cfg() -> Config:
    cfg = Config()
    cfg.model_type            = ss.get("sb_model", "sage")
    cfg.model.emb_dim         = ss.get("sb_emb_dim", 64)
    cfg.model.dropout         = ss.get("sb_dropout", 0.1)
    cfg.model.gat_heads       = ss.get("sb_gat_heads", 4)
    cfg.train.lr              = ss.get("sb_lr", 0.01)
    cfg.train.gat_lr          = ss.get("sb_gat_lr", 0.001)
    cfg.train.num_epochs      = ss.get("sb_epochs", 65)
    cfg.train.batch_size      = ss.get("sb_batch", 1024)
    cfg.train.n_neg           = ss.get("sb_n_neg", 4)
    cfg.train.optimizer       = ss.get("sb_optim", "adam")
    cfg.train.grad_clip       = ss.get("sb_grad_clip", 1.0)
    cfg.train.warmup_epochs      = ss.get("sb_warmup", 10)
    cfg.train.reg_lambda         = ss.get("sb_reg_lambda", 1e-5)
    cfg.train.use_scheduler      = ss.get("sb_scheduler", True)
    cfg.train.use_all_pairs      = ss.get("sb_all_pairs", True)
    cfg.train.min_epochs         = ss.get("sb_min_epochs", 80)
    cfg.train.patience           = ss.get("sb_patience", 15)
    cfg.train.eval_every         = ss.get("sb_eval_every", 10)
    cfg.model.n_layers           = ss.get("sb_n_layers", 1)
    cfg.model.use_residual       = ss.get("sb_use_residual", True)
    cfg.graph.use_item_item_edges = ss.get("sb_use_ii_edges", False)
    cfg.tune.n_trials         = ss.get("sb_trials", 25)
    cfg.tune.optuna_epochs    = ss.get("sb_optuna_ep", 30)
    cfg.graph.k_neighbors     = ss.get("sb_k_nbr", 10)
    cfg.graph.sbert_model     = ss.get("sb_sbert", "all-mpnet-base-v2")
    cfg.data.max_users        = ss.get("sb_max_users", 50_000)
    cfg.data.max_reviews      = ss.get("sb_max_reviews", 80_000)
    cfg.data.val_size         = ss.get("sb_val_size", 0.15)
    cfg.data.test_size        = ss.get("sb_test_size", 0.15)
    cfg.eval.max_eval_users   = ss.get("sb_max_eval_users", 500)
    cfg.checkpoint_dir        = ss.get("sb_ckpt_dir", "checkpoints")
    return cfg


# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 GNN Recommender")
    dev = detect_devices()
    gpu_label = ("🖥 CPU" if dev.num_gpus == 0
                 else f"⚡ {dev.num_gpus} GPU{'s' if dev.num_gpus > 1 else ''}"
                      + (" — DDP" if dev.use_ddp else ""))
    st.info(gpu_label)

    st.divider()
    st.subheader("Model")
    st.selectbox("Backbone", ["sage", "gat", "lightgcn"], key="sb_model")
    st.selectbox("Optimizer", ["adam", "adamw"], key="sb_optim")
    st.slider("Embedding dim", 16, 256, 64, step=16, key="sb_emb_dim")
    st.slider("Dropout", 0.0, 0.5, 0.1, step=0.05, key="sb_dropout")
    st.slider("GNN layers", 1, 4, 1, key="sb_n_layers",
              help="Step 2: 1 layer avoids over-smoothing on sparse UI graph")
    st.checkbox("Residual connections", value=True, key="sb_use_residual",
                help="Step 3: skip connections preserve item identity after aggregation")
    if ss.get("sb_model") == "gat":
        st.slider("GAT heads", 1, 8, 4, key="sb_gat_heads")

    st.divider()
    st.subheader("Training")

    # One-click preset for CPU training
    if st.button("⚡ Apply CPU-optimised settings", use_container_width=True,
                 help="Sets epochs=200, n_neg=10, all_pairs=True, min_epochs=80, reg=1e-5"):
        for k, v in [("sb_epochs", 200), ("sb_n_neg", 10), ("sb_lr", 0.005),
                     ("sb_gat_lr", 0.001), ("sb_warmup", 10), ("sb_all_pairs", True),
                     ("sb_min_epochs", 80), ("sb_patience", 15), ("sb_eval_every", 10),
                     ("sb_reg_lambda", 1e-5), ("sb_scheduler", True)]:
            st.session_state[k] = v
        st.rerun()

    st.number_input("Epochs", 5, 500, 200, step=10, key="sb_epochs")
    st.number_input("Batch size", 128, 4096, 1024, step=128, key="sb_batch")
    st.number_input("LR (SAGE)", 1e-5, 0.1, 0.005, format="%.5f", step=1e-3, key="sb_lr")
    st.number_input("LR (GAT)",  1e-5, 0.1, 0.001, format="%.5f", step=1e-4, key="sb_gat_lr")
    st.slider("BPR negatives / positive", 1, 20, 10, key="sb_n_neg")
    st.slider("Warmup epochs", 0, 30, 10, key="sb_warmup")
    st.slider("Grad clip", 0.1, 5.0, 1.0, step=0.1, key="sb_grad_clip")
    st.number_input("L2 reg lambda", 0.0, 0.01, 1e-5, format="%.6f", step=1e-5, key="sb_reg_lambda")
    st.checkbox("Cosine LR scheduler", value=True, key="sb_scheduler")
    st.checkbox("Use all training pairs / epoch", value=True, key="sb_all_pairs")
    st.slider("Min epochs before early stopping", 20, 200, 80, step=10, key="sb_min_epochs")
    st.slider("Early-stop patience (checks)", 3, 50, 15, key="sb_patience")
    st.slider("Eval every N epochs", 1, 30, 10, key="sb_eval_every")

    st.divider()
    st.subheader("Optuna tuning")
    st.number_input("Trials", 5, 100, 25, step=5, key="sb_trials")
    st.number_input("Optuna epochs", 5, 100, 30, step=5, key="sb_optuna_ep")

    st.divider()
    st.subheader("Graph")
    st.checkbox("Item-item SBERT edges", value=False, key="sb_use_ii_edges",
                help="OFF by default — pure user-item collaborative filtering. "
                     "Enable only for ablation experiments (see Experiments tab).")
    st.slider("FAISS K-neighbours", 3, 50, 10, key="sb_k_nbr")
    st.selectbox("SBERT model",
                 ["all-mpnet-base-v2", "all-MiniLM-L6-v2", "all-MiniLM-L12-v2"],
                 key="sb_sbert")

    st.divider()
    st.subheader("Data / split")
    st.number_input("Max users",   1_000, 500_000, 50_000, step=1_000, key="sb_max_users")
    st.number_input("Max reviews", 1_000, 1_000_000, 80_000, step=1_000, key="sb_max_reviews")
    st.slider("Val split",  0.05, 0.30, 0.15, step=0.05, key="sb_val_size")
    st.slider("Test split", 0.05, 0.30, 0.15, step=0.05, key="sb_test_size")
    st.number_input("Max eval users", 50, 5_000, 500, step=50, key="sb_max_eval_users")

    st.divider()
    st.text_input("Checkpoint dir", "checkpoints", key="sb_ckpt_dir")

    if st.button("🗑 Reset session", use_container_width=True):
        for k in list(_DEFAULTS.keys()):
            st.session_state.pop(k, None)
        st.rerun()


# ── tabs ───────────────────────────────────────────────────────────────────────
tab_data, tab_graph, tab_train, tab_results, tab_exp = st.tabs(
    ["📁 Dataset", "🔗 Build Graph", "🚀 Train / Tune", "📊 Results", "🧪 Experiments"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATASET
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.header("Load Dataset")
    mode = st.radio("Input method", ["📂 Local file paths", "⬆ Upload files"],
                    horizontal=True)
    st.divider()
    col_b, col_u, col_r = st.columns(3)

    if mode == "📂 Local file paths":
        with col_b:
            st.markdown("**Business CSV**")
            biz_path = st.text_input("Path", key="biz_path",
                                     placeholder="/data/yelp_business.csv")
        with col_u:
            st.markdown("**User CSV**")
            usr_path = st.text_input("Path", key="usr_path",
                                     placeholder="/data/yelp_user.csv")
        with col_r:
            st.markdown("**Review CSV**")
            rev_path = st.text_input("Path", key="rev_path",
                                     placeholder="/data/yelp_review.csv")
        paths_ok = all([biz_path, usr_path, rev_path])
    else:
        with col_b:
            st.markdown("**Business CSV**")
            biz_file = st.file_uploader("", type="csv", key="biz_up")
        with col_u:
            st.markdown("**User CSV**")
            usr_file = st.file_uploader("", type="csv", key="usr_up")
        with col_r:
            st.markdown("**Review CSV**")
            rev_file = st.file_uploader("", type="csv", key="rev_up")
        paths_ok = all([ss.get("biz_up"), ss.get("usr_up"), ss.get("rev_up")])

    st.divider()
    if st.button("Load & Preview", disabled=not paths_ok,
                 use_container_width=True, type="primary"):
        cfg = build_cfg()
        ss.logs = []
        with st.spinner("Loading and preprocessing…"):
            try:
                if mode == "📂 Local file paths":
                    b_p, u_p, r_p = biz_path.strip(), usr_path.strip(), rev_path.strip()
                else:
                    b_p = save_upload(ss.biz_up)
                    u_p = save_upload(ss.usr_up)
                    r_p = save_upload(ss.rev_up)

                with capture():
                    bdf, udf, rdf = load_raw_data(cfg.data, b_p, u_p, r_p)
                    rdf_proc, rdf_full, _, item_enc, n_u, n_i = preprocess(
                        bdf, udf, rdf
                    )

                ss.business_df    = bdf
                ss.user_df        = udf
                ss.review_df      = rdf_proc
                ss.review_df_full = rdf_full
                ss.item_enc       = item_enc
                ss.n_users        = n_u
                ss.n_items        = n_i
                ss.num_nodes      = n_u + n_i
                ss.data_loaded    = True
                ss.graph_built    = False
                ss.trained        = False
                ss.evaluated      = False
                log(f"Loaded: {n_u:,} users | {n_i:,} items | {len(rdf_proc):,} interactions")
            except Exception as exc:
                st.error(f"Load failed: {exc}")

    if ss.data_loaded:
        st.success(
            f"Dataset ready — **{ss.n_users:,} users** · **{ss.n_items:,} items** · "
            f"**{len(ss.review_df):,} interactions**"
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            with st.expander("Business preview"):
                st.dataframe(ss.business_df.head(5), use_container_width=True)
        with c2:
            with st.expander("User preview"):
                st.dataframe(ss.user_df.head(5), use_container_width=True)
        with c3:
            with st.expander("Review preview"):
                st.dataframe(ss.review_df.head(5), use_container_width=True)

        with st.expander("Rating distribution"):
            fig, ax = plt.subplots(figsize=(5, 2.5))
            ss.review_df['rating'].value_counts().sort_index().plot(
                kind='bar', ax=ax, color='steelblue', edgecolor='white')
            ax.set_xlabel("Stars"); ax.set_ylabel("Count")
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    show_logs()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BUILD GRAPH  (F2: train edges only | F5: SBERT order fixed)
# ══════════════════════════════════════════════════════════════════════════════
with tab_graph:
    st.header("Build Graph")

    if not ss.data_loaded:
        st.info("Load a dataset first (Dataset tab).")
    else:
        cfg = build_cfg()
        train_pct = int((1 - cfg.data.val_size - cfg.data.test_size) * 100)
        st.markdown(
            f"Split: **{train_pct}% train / "
            f"{int(cfg.data.val_size*100)}% val / "
            f"{int(cfg.data.test_size*100)}% test**  |  "
            f"Nodes: **{ss.num_nodes:,}**  |  "
            f"FAISS K={cfg.graph.k_neighbors}"
        )
        st.warning(
            "Graph is built from **training edges only** (no leakage). "
            "SBERT encoding may take a few minutes on CPU."
        )

        if st.button("Build Graph", type="primary", use_container_width=True):
            with st.spinner("Building UI edges → splitting → SBERT → FAISS → graph…"):
                try:
                    with capture():
                        # Step 1: all UI edges
                        uei, uev = build_ui_edges(ss.review_df, cfg.data)

                        # Step 2: 70/15/15 split (F1)
                        train_u, train_pos, df_val, df_test, train_idx = build_train_test(
                            uei, uev, ss.n_users, cfg.data
                        )

                        # Step 3: align business_df to LabelEncoder order (F5)
                        bdf_ordered = (
                            ss.business_df
                            .set_index('business_id')
                            .loc[ss.item_enc.classes_]
                            .reset_index()
                        )

                        # Step 4: train-only graph (F2)
                        train_ei = uei[:, train_idx]
                        train_ev = uev[train_idx]
                        train_rf = ss.review_df_full.iloc[train_idx].reset_index(drop=True)

                        fei, _ = build_graph(
                            train_ei, train_ev, train_rf,
                            bdf_ordered, ss.n_users,
                            dev.embed_device,
                            torch.device("cpu"),
                            cfg.graph,
                        )

                    ss.full_edge_index = fei
                    ss.train_u         = train_u
                    ss.train_pos       = train_pos
                    ss.df_val          = df_val
                    ss.df_test         = df_test
                    ss.df_train        = ss.review_df.iloc[train_idx].copy()
                    # store raw train edges for fast experiment graph rebuilds
                    ss.train_ei        = train_ei
                    ss.train_ev        = train_ev
                    ss.train_rf        = train_rf
                    ss.bdf_ordered     = bdf_ordered
                    ss.graph_built     = True
                    ss.trained         = False
                    # reset experiment state when graph is rebuilt
                    ss.exp_adopted     = {}
                    ss.exp_summary     = []
                    ss.exp_step_status = {}
                    ss.exp_step_metrics = {}
                    ss.evaluated       = False
                    gc.collect()

                    n_train = len(train_u)
                    log(f"Graph built — {fei.shape[1]:,} edges")
                    log(f"Train: {n_train:,} | Val: {len(df_val):,} | Test: {len(df_test):,}")
                except Exception as exc:
                    st.error(f"Graph build failed: {exc}")

        if ss.graph_built:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total edges",     f"{ss.full_edge_index.shape[1]:,}")
            m2.metric("Train pairs",     f"{len(ss.train_u):,}")
            m3.metric("Val pairs",       f"{len(ss.df_val):,}")
            m4.metric("Test pairs",      f"{len(ss.df_test):,}")

    show_logs()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TRAIN / TUNE  (F3: Optuna uses df_val | F4,F6,F8,F9,F10 in trainer)
# ══════════════════════════════════════════════════════════════════════════════
with tab_train:
    st.header("Train / Tune")

    if not ss.graph_built:
        st.info("Build the graph first (Build Graph tab).")
    else:
        cfg = build_cfg()
        run_mode = st.radio("Mode",
                            ["train", "tune", "tune+train", "evaluate"],
                            horizontal=True, key="run_mode")

        use_ddp = dev.use_ddp
        if use_ddp:
            st.info(f"Multi-GPU detected ({dev.num_gpus} GPUs) — DDP enabled.")
            if st.checkbox("Disable DDP", value=False):
                use_ddp = False

        model_device = dev.sage_device if cfg.model_type == "sage" else dev.gat_device
        eff_lr       = cfg.effective_lr()
        ckpt_path    = os.path.join(cfg.checkpoint_dir, f"{cfg.model_type}_best.pt")

        prog_bar  = st.empty()
        prog_text = st.empty()
        run_btn   = st.button("▶ Run", type="primary", use_container_width=True)

        if run_btn:
            ss.logs = []
            log(f"[{cfg.model_type.upper()}] mode={run_mode}  lr={eff_lr}  "
                f"emb={cfg.model.emb_dim}  device={model_device}")

            # ── TUNE ──────────────────────────────────────────────────────────
            if run_mode in ("tune", "tune+train"):
                log(f"[Optuna] {cfg.tune.n_trials} trials on val set…")
                with st.spinner("Optuna search…"):
                    try:
                        best = run_optuna_tuning(
                            cfg, ss.num_nodes, ss.n_users, ss.n_items,
                            ss.full_edge_index.to(model_device),
                            ss.train_u.to(model_device),
                            ss.train_pos.to(model_device),
                            ss.df_val,          # ← val set, never test (F3)
                            model_device,
                        )
                        cfg.model.emb_dim = best.get("emb_dim", cfg.model.emb_dim)
                        cfg.train.lr      = best.get("lr",      cfg.train.lr)
                        cfg.model.dropout = best.get("dropout", cfg.model.dropout)
                        cfg.train.num_epochs = cfg.train.final_epochs
                        log(f"[Optuna] best → {best}")
                    except Exception as exc:
                        st.error(f"Optuna failed: {exc}")

            # ── TRAIN ─────────────────────────────────────────────────────────
            if run_mode in ("train", "tune+train"):
                ss.train_history = []
                total_ep = cfg.train.num_epochs

                def on_epoch(epoch: int, loss: float):
                    ss.train_history.append(loss)
                    prog_bar.progress(epoch / total_ep)
                    prog_text.text(f"Epoch {epoch}/{total_ep} | BPR Loss: {loss:.4f}")

                with st.spinner("Training…"):
                    if use_ddp and dev.num_gpus >= 2:
                        try:
                            run_ddp_training(
                                cfg, ss.num_nodes, ss.n_users, ss.n_items,
                                ss.full_edge_index, ss.train_u, ss.train_pos,
                                save_path=ckpt_path, world_size=dev.num_gpus,
                            )
                            model = load_checkpoint(ckpt_path, model_device)
                        except Exception as exc:
                            st.error(f"DDP failed: {exc}"); model = None
                    else:
                        model     = build_model(
                            cfg.model_type, ss.num_nodes,
                            cfg.model.emb_dim, cfg.model.dropout, cfg.model.gat_heads,
                            cfg.model.n_layers, cfg.model.use_residual,
                        ).to(model_device)
                        optimizer = build_optimizer(model, cfg.train, lr_override=eff_lr)
                        edge_idx  = ss.full_edge_index.to(model_device)
                        t_u       = ss.train_u.to(model_device)
                        t_pos     = ss.train_pos.to(model_device)

                        # Early-stopping eval function (F10) on val set
                        def val_eval_fn(m):
                            m.eval()
                            with torch.no_grad():
                                rank = compute_ranking_metrics(
                                    m, edge_idx, ss.df_val, ss.n_users, cfg.eval
                                )
                            return rank.get(cfg.eval.k_list[0], {}).get('NDCG', 0.0)

                        try:
                            with capture():
                                train_model(
                                    model, optimizer, edge_idx, t_u, t_pos,
                                    ss.n_users, ss.n_items, cfg.train,
                                    epoch_callback=on_epoch,
                                    eval_fn=val_eval_fn,
                                )
                        except torch.cuda.OutOfMemoryError:
                            log("[OOM] Retrying with half batch size…")
                            cfg.train.batch_size //= 2
                            optimizer = build_optimizer(model, cfg.train, lr_override=eff_lr)
                            with capture():
                                train_model(
                                    model, optimizer, edge_idx, t_u, t_pos,
                                    ss.n_users, ss.n_items, cfg.train,
                                    epoch_callback=on_epoch,
                                )

                    if model is not None:
                        save_checkpoint(model, cfg, ss.num_nodes, ckpt_path)
                        ss.model   = model
                        ss.trained = True
                        log(f"Checkpoint saved -> {ckpt_path}")

                prog_bar.empty(); prog_text.empty()

            # ── EVALUATE-ONLY ─────────────────────────────────────────────────
            if run_mode == "evaluate":
                if not os.path.exists(ckpt_path):
                    st.error(f"No checkpoint at `{ckpt_path}`")
                else:
                    with st.spinner("Loading checkpoint…"):
                        model      = load_checkpoint(ckpt_path, model_device)
                        ss.model   = model
                        ss.trained = True
                        log(f"Loaded {ckpt_path}")

            # ── FINAL EVALUATION on TEST set (touched only here) ──────────────
            if ss.trained and ss.model is not None:
                with st.spinner("Evaluating on held-out test set…"):
                    try:
                        edge_idx = ss.full_edge_index.to(model_device)
                        with capture():
                            results = evaluate_model(
                                ss.model, edge_idx,
                                ss.df_test,          # ← test set, first and only touch
                                ss.n_users, cfg.eval,
                            )
                            pop  = popularity_baseline(
                                ss.df_train, ss.df_test, ss.n_users, cfg.eval
                            )
                            rand = random_baseline(
                                ss.df_test, ss.n_users, ss.n_items, cfg.eval
                            )
                        ss.results       = results
                        ss.baseline_pop  = pop
                        ss.baseline_rand = rand
                        ss.norm_metrics  = normalize_metrics(
                            results["ranking"], rand,
                            ss.df_test, cfg.eval,
                        )
                        ss.evaluated     = True
                        log("Evaluation done — see Results tab")
                    except Exception as exc:
                        st.error(f"Evaluation failed: {exc}")

        if ss.train_history:
            st.subheader("Loss history")
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.plot(ss.train_history, color="royalblue", linewidth=1.5)
            ax.set_xlabel("Epoch"); ax.set_ylabel("BPR Loss")
            ax.grid(True, alpha=0.3); plt.tight_layout()
            st.pyplot(fig); plt.close(fig)

    show_logs()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_results:
    st.header("Evaluation Results")

    if not ss.evaluated or ss.results is None:
        st.info("Run training or evaluation first (Train / Tune tab).")
    else:
        res = ss.results

        # ── regression metrics ────────────────────────────────────────────────
        st.subheader("Regression metrics  (predictions scaled to [1, 5])")
        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", f"{res['rmse']:.4f}")
        c2.metric("MAE",  f"{res['mae']:.4f}")
        c3.metric("Accuracy (sigmoid ≥ 0.5)", f"{res['accuracy']:.4f}")

        st.divider()

        # ── normalised metrics ────────────────────────────────────────────────
        nm = ss.norm_metrics or {}
        if nm:
            st.subheader("Normalized Metrics")
            st.caption(
                "**_pct** = raw value × 100 (as %).  "
                "**Lift** = GNN / Random (>1 better, <1 worse than random).  "
                "**Norm%** = (GNN − Random) / (Ideal − Random) × 100  "
                "— 0 % means same as random, 100 % means perfect."
            )

            norm_rows = []
            for k in sorted(nm.keys()):
                nk = nm[k]
                norm_rows.append({
                    "K":          k,
                    "P %":        nk.get("P_pct"),
                    "P Lift":     nk.get("P_lift"),
                    "P Norm %":   nk.get("P_norm"),
                    "HR %":       nk.get("HR_pct"),
                    "HR Lift":    nk.get("HR_lift"),
                    "HR Norm %":  nk.get("HR_norm"),
                    "NDCG %":     nk.get("NDCG_pct"),
                    "NDCG Lift":  nk.get("NDCG_lift"),
                    "NDCG Norm %":nk.get("NDCG_norm"),
                    "MAP %":      nk.get("MAP_pct"),
                    "MRR %":      nk.get("MRR_pct"),
                })

            st.dataframe(
                pd.DataFrame(norm_rows).set_index("K"),
                use_container_width=True,
            )

            # Lift bar chart — quick visual: above 1.0 = model beats random
            k_vals_n = sorted(nm.keys())
            x_n      = np.arange(len(k_vals_n))
            w_n      = 0.22
            fig, ax  = plt.subplots(figsize=(9, 4))
            ax.bar(x_n - w_n,  [nm[k].get("P_lift")    or 0 for k in k_vals_n], w_n, label="P@K Lift",    color="steelblue")
            ax.bar(x_n,        [nm[k].get("HR_lift")   or 0 for k in k_vals_n], w_n, label="HR@K Lift",   color="mediumpurple")
            ax.bar(x_n + w_n,  [nm[k].get("NDCG_lift") or 0 for k in k_vals_n], w_n, label="NDCG@K Lift", color="seagreen")
            ax.axhline(1.0, color="red", linestyle="--", linewidth=1.2, label="Random baseline (1.0)")
            ax.set_xticks(x_n)
            ax.set_xticklabels([f"K={k}" for k in k_vals_n])
            ax.set_ylabel("Lift vs Random")
            ax.set_title("Metric Lift over Random Baseline  (>1.0 = model wins)")
            ax.legend(); ax.grid(axis="y", alpha=0.3)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        st.divider()

        # ── ranking metrics + baselines ───────────────────────────────────────
        st.subheader("Ranking metrics vs. baselines")

        ranking  = res.get("ranking", {})
        pop      = ss.baseline_pop  or {}
        rand_b   = ss.baseline_rand or {}

        # ── General Precision summary cards ───────────────────────────────────
        st.subheader("General Precision (Hit Rate@K)")
        st.caption(
            "HR@K = fraction of users who received at least 1 relevant item in their top-K. "
            "More interpretable than P@K when relevant items are sparse."
        )
        hr_cols = st.columns(len(ranking))
        for col, k in zip(hr_cols, sorted(ranking.keys())):
            hr_gnn = ranking[k].get('HR', 0)
            hr_pop = pop.get(k, {}).get('HR', 0)
            delta  = hr_gnn - hr_pop
            col.metric(f"HR@{k}  (GNN)", f"{hr_gnn:.3f}",
                       delta=f"{delta:+.3f} vs Popularity")

        st.divider()

        rows = []
        for k in sorted(ranking.keys()):
            m = ranking[k]
            rows.append({
                "K": k, "Source": "GNN",
                "HR@K":        round(m.get("HR",  0), 4),
                "Precision@K": round(m["P"],       4),
                "Recall@K":    round(m["R"],       4),
                "NDCG@K":      round(m["NDCG"],    4),
                "MAP@K":       round(m.get("MAP",  0), 4),
                "MRR@K":       round(m.get("MRR",  0), 4),
            })
            if k in pop:
                rows.append({
                    "K": k, "Source": "Popularity",
                    "HR@K":        round(pop[k].get("HR",   0), 4),
                    "Precision@K": round(pop[k]["P"],        4),
                    "Recall@K":    round(pop[k]["R"],        4),
                    "NDCG@K":      round(pop[k]["NDCG"],     4),
                    "MAP@K": 0, "MRR@K": 0,
                })
            if k in rand_b:
                rows.append({
                    "K": k, "Source": "Random",
                    "HR@K":        round(rand_b[k].get("HR",  0), 4),
                    "Precision@K": round(rand_b[k]["P"],       4),
                    "Recall@K":    round(rand_b[k]["R"],       4),
                    "NDCG@K":      round(rand_b[k]["NDCG"],    4),
                    "MAP@K": 0, "MRR@K": 0,
                })

        df_disp = pd.DataFrame(rows)
        st.dataframe(df_disp, use_container_width=True)

        # ── grouped bar chart ─────────────────────────────────────────────────
        st.subheader("NDCG@K comparison")
        k_vals = sorted(ranking.keys())
        x      = np.arange(len(k_vals))
        w      = 0.25
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x - w, [ranking[k]["NDCG"] for k in k_vals], w,
               label="GNN", color="steelblue")
        ax.bar(x,     [pop.get(k, {}).get("NDCG", 0) for k in k_vals], w,
               label="Popularity", color="darkorange")
        ax.bar(x + w, [rand_b.get(k, {}).get("NDCG", 0) for k in k_vals], w,
               label="Random", color="lightgray")
        ax.set_xticks(x); ax.set_xticklabels([f"K={k}" for k in k_vals])
        ax.set_ylabel("NDCG@K"); ax.set_title("NDCG@K — GNN vs Baselines")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        ymax = max(
            max((ranking[k]["NDCG"] for k in k_vals), default=0),
            max((pop.get(k, {}).get("NDCG", 0) for k in k_vals), default=0),
        )
        ax.set_ylim(0, max(ymax * 1.4, 0.05))
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        # ── Precision@K ───────────────────────────────────────────────────────
        st.subheader("Precision@K comparison")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x - w, [ranking[k]["P"] for k in k_vals], w,
               label="GNN", color="seagreen")
        ax.bar(x,     [pop.get(k, {}).get("P", 0) for k in k_vals], w,
               label="Popularity", color="darkorange")
        ax.bar(x + w, [rand_b.get(k, {}).get("P", 0) for k in k_vals], w,
               label="Random", color="lightgray")
        ax.set_xticks(x); ax.set_xticklabels([f"K={k}" for k in k_vals])
        ax.set_ylabel("Precision@K"); ax.set_title("Precision@K — GNN vs Baselines")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        # ── Hit Rate@K (General Precision) ────────────────────────────────────
        st.subheader("Hit Rate@K — General Precision")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x - w, [ranking[k].get("HR", 0) for k in k_vals], w,
               label="GNN", color="mediumpurple")
        ax.bar(x,     [pop.get(k, {}).get("HR", 0) for k in k_vals], w,
               label="Popularity", color="darkorange")
        ax.bar(x + w, [rand_b.get(k, {}).get("HR", 0) for k in k_vals], w,
               label="Random", color="lightgray")
        ax.set_xticks(x); ax.set_xticklabels([f"K={k}" for k in k_vals])
        ax.set_ylabel("Hit Rate@K"); ax.set_title("HR@K — fraction of users with ≥1 relevant hit")
        ax.set_ylim(0, 1.05); ax.legend(); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        # ── loss history ──────────────────────────────────────────────────────
        if ss.train_history:
            st.divider()
            st.subheader("Training loss")
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(ss.train_history, color="royalblue", linewidth=1.5)
            ax.axhline(y=0.693, color="red", linestyle="--", alpha=0.5,
                       label="Random init floor (ln 2 ≈ 0.693)")
            ax.set_xlabel("Epoch"); ax.set_ylabel("BPR Loss")
            ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
            st.pyplot(fig); plt.close(fig)

        # ── raw JSON ──────────────────────────────────────────────────────────
        with st.expander("Raw JSON"):
            def _ser(o):
                if isinstance(o, dict):  return {k: _ser(v) for k, v in o.items()}
                if isinstance(o, (float, np.floating)): return round(float(o), 6)
                return o
            st.json(_ser(res))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — EXPERIMENTS  (temporal, gated, validate-then-adopt)
# ══════════════════════════════════════════════════════════════════════════════
with tab_exp:
    st.header("🧪 Ablation Experiments")
    st.caption(
        "Each step proposes ONE architectural change. Run it, inspect the comparison, "
        "then **Adopt** (carries the change forward) or **Skip** (keeps current config). "
        "LightGCN (Step 4) is an independent comparison — it is never adopted into the "
        "SAGE/GAT pipeline."
    )

    if not (ss.graph_built and ss.evaluated):
        st.info(
            "Complete **Dataset → Build Graph → Train** first so there is a "
            "baseline to compare against."
        )
        st.stop()

    cfg = build_cfg()
    model_device = dev.sage_device if cfg.model_type != "gat" else dev.gat_device

    # ── baseline snapshot ────────────────────────────────────────────────────
    if not ss.exp_adopted:
        ss.exp_adopted = {
            "use_ii_edges": ss.get("sb_use_ii_edges", True),
            "n_layers":     ss.get("sb_n_layers", 1),
            "use_residual": ss.get("sb_use_residual", True),
            "model_type":   cfg.model_type,
        }

    baseline_ndcg = ss.results["ranking"].get(10, {}).get("NDCG", 0) if ss.results else 0
    baseline_hr   = ss.results["ranking"].get(10, {}).get("HR",   0) if ss.results else 0
    baseline_p    = ss.results["ranking"].get(10, {}).get("P",    0) if ss.results else 0

    if not ss.exp_summary:
        ss.exp_summary = [{
            "Name":        "Baseline",
            "II Edges":    "✓" if ss.exp_adopted["use_ii_edges"] else "✗",
            "Layers":      ss.exp_adopted["n_layers"],
            "Residual":    "✓" if ss.exp_adopted["use_residual"] else "✗",
            "Model":       ss.exp_adopted["model_type"].upper(),
            "P@10 %":      round(baseline_p    * 100, 3),
            "HR@10 %":     round(baseline_hr   * 100, 3),
            "NDCG@10 %":   round(baseline_ndcg * 100, 3),
            "vs Prev":     "—",
        }]

    # ── experiment helper ─────────────────────────────────────────────────────
    import copy as _copy

    def _run_exp(override: dict, logs_placeholder) -> dict | None:
        """Build graph + train + evaluate with override applied to adopted config."""
        exp_cfg = {**ss.exp_adopted, **override}
        exp_config = _copy.deepcopy(cfg)
        exp_config.graph.use_item_item_edges = exp_cfg["use_ii_edges"]
        exp_config.model.n_layers     = exp_cfg["n_layers"]
        exp_config.model.use_residual = exp_cfg["use_residual"]
        exp_config.model_type         = exp_cfg["model_type"]

        exp_logs = []
        def _log(m): exp_logs.append(m); logs_placeholder.code("\n".join(exp_logs[-60:]))

        try:
            _log(f"Building graph  (II edges: {exp_cfg['use_ii_edges']}) ...")
            bdf_arg = ss.bdf_ordered if exp_cfg["use_ii_edges"] else None
            fei, _ = build_graph(
                ss.train_ei, ss.train_ev, ss.train_rf,
                bdf_arg, ss.n_users,
                dev.embed_device, torch.device("cpu"), exp_config.graph,
            )
            _log(f"Graph: {fei.shape[1]:,} edges")

            _log(f"Building model  ({exp_cfg['model_type'].upper()}, "
                 f"layers={exp_cfg['n_layers']}, residual={exp_cfg['use_residual']}) ...")
            m_dev = dev.sage_device if exp_cfg["model_type"] != "gat" else dev.gat_device
            eff_lr = (exp_config.train.gat_lr
                      if exp_cfg["model_type"] == "gat" else exp_config.train.lr)
            model  = build_model(
                exp_cfg["model_type"], ss.num_nodes,
                exp_config.model.emb_dim, exp_config.model.dropout,
                exp_config.model.gat_heads,
                exp_cfg["n_layers"], exp_cfg["use_residual"],
            ).to(m_dev)
            optimizer = build_optimizer(model, exp_config.train, lr_override=eff_lr)
            edge_idx  = fei.to(m_dev)

            _log(f"Training {exp_config.train.num_epochs} epochs ...")
            with torch.no_grad():
                pass  # force graph to m_dev
            train_model(
                model, optimizer, edge_idx,
                ss.train_u.to(m_dev), ss.train_pos.to(m_dev),
                ss.n_users, ss.n_items, exp_config.train,
                verbose=False,
            )

            _log("Evaluating on test set ...")
            results = evaluate_model(model, edge_idx, ss.df_test, ss.n_users, exp_config.eval)
            _log("Done.")
            return results
        except Exception as e:
            _log(f"ERROR: {e}")
            return None

    # ── step definitions ──────────────────────────────────────────────────────
    STEPS = [
        {
            "key":    "step1",
            "title":  "Step 1 — Remove item-item SBERT edges",
            "desc":   "Dense II edges connect all similar-specialty items, causing "
                      "over-smoothing. Setting use_item_item_edges=False rebuilds the "
                      "graph with UI edges only — faster AND better for CF.",
            "override": {"use_ii_edges": False},
            "label":  "II Edges",
            "value":  "✗",
        },
        {
            "key":    "step2",
            "title":  "Step 2 — Reduce to 1 GNN layer",
            "desc":   "With sparse UI-only graph, 2 layers propagate noise. "
                      "1 layer = user sees only their directly-reviewed items — "
                      "exactly the CF signal we need.",
            "override": {"n_layers": 1},
            "label":  "Layers",
            "value":  1,
        },
        {
            "key":    "step3",
            "title":  "Step 3 — Add residual connections",
            "desc":   "Skip connection h = conv(x) + x preserves each item's "
                      "individual embedding identity after aggregation, preventing "
                      "full representation collapse.",
            "override": {"use_residual": True},
            "label":  "Residual",
            "value":  "✓",
        },
    ]

    # ── render each step ──────────────────────────────────────────────────────
    prev_done = True   # Step 1 always available
    for step in STEPS:
        key    = step["key"]
        status = ss.exp_step_status.get(key, "pending")
        done   = status in ("ran", "adopted", "skipped")

        with st.container(border=True):
            col_info, col_btns = st.columns([3, 1])
            with col_info:
                st.subheader(step["title"])
                st.caption(step["desc"])
                adopted_val = ss.exp_adopted.get(
                    list(step["override"].keys())[0], "?"
                )
                st.markdown(f"**Current:** `{list(step['override'].keys())[0]}` = `{adopted_val}`  →  "
                            f"**Proposed:** `{step['value']}`")

            with col_btns:
                disabled = not prev_done
                run_btn  = st.button(f"▶ Run", key=f"run_{key}", disabled=disabled,
                                     use_container_width=True)

            if run_btn:
                log_box = st.empty()
                with st.spinner(f"Running {step['title']} ..."):
                    res = _run_exp(step["override"], log_box)
                if res is not None:
                    ss.exp_step_metrics[key] = res
                    ss.exp_step_status[key]  = "ran"
                    st.rerun()

            if status in ("ran", "adopted", "skipped"):
                if status == "ran":
                    res = ss.exp_step_metrics.get(key, {})
                    exp_ndcg = res.get("ranking", {}).get(10, {}).get("NDCG", 0) if res else 0
                    exp_hr   = res.get("ranking", {}).get(10, {}).get("HR",   0) if res else 0
                    exp_p    = res.get("ranking", {}).get(10, {}).get("P",    0) if res else 0

                    prev_ndcg = (ss.exp_step_metrics.get(
                                    STEPS[STEPS.index(step)-1]["key"], {})
                                 .get("ranking", {}).get(10, {}).get("NDCG", 0)
                                 if STEPS.index(step) > 0
                                 else baseline_ndcg)
                    delta = exp_ndcg - prev_ndcg

                    c1, c2, c3 = st.columns(3)
                    c1.metric("P@10 %",    f"{exp_p*100:.3f}",
                              f"{(exp_p - baseline_p)*100:+.3f} vs baseline")
                    c2.metric("HR@10 %",   f"{exp_hr*100:.3f}",
                              f"{(exp_hr - baseline_hr)*100:+.3f} vs baseline")
                    c3.metric("NDCG@10 %", f"{exp_ndcg*100:.3f}",
                              f"{delta*100:+.3f} vs prev")

                    col_a, col_s = st.columns(2)
                    if col_a.button("✅ Adopt", key=f"adopt_{key}", use_container_width=True):
                        ss.exp_adopted.update(step["override"])
                        ss.exp_step_status[key] = "adopted"
                        ss.exp_summary.append({
                            "Name":       step["title"].split("—")[1].strip(),
                            "II Edges":   "✓" if ss.exp_adopted["use_ii_edges"] else "✗",
                            "Layers":     ss.exp_adopted["n_layers"],
                            "Residual":   "✓" if ss.exp_adopted["use_residual"] else "✗",
                            "Model":      ss.exp_adopted["model_type"].upper(),
                            "P@10 %":     round(exp_p    * 100, 3),
                            "HR@10 %":    round(exp_hr   * 100, 3),
                            "NDCG@10 %":  round(exp_ndcg * 100, 3),
                            "vs Prev":    f"{delta*100:+.3f} %",
                        })
                        st.rerun()
                    if col_s.button("⏩ Skip", key=f"skip_{key}", use_container_width=True):
                        ss.exp_step_status[key] = "skipped"
                        st.rerun()

                elif status == "adopted":
                    st.success("✅ Adopted")
                elif status == "skipped":
                    st.info("⏩ Skipped")

        prev_done = done

    # ── Step 4 — LightGCN comparison ─────────────────────────────────────────
    lgcn_ready = prev_done
    with st.container(border=True):
        st.subheader("Step 4 — LightGCN comparison (independent model)")
        st.caption(
            "LightGCN uses parameter-free degree-normalised propagation + "
            "mean of all layer embeddings. No feature transform, no activation — "
            "designed specifically for BPR-based collaborative filtering."
        )
        st.markdown(
            f"**Config:** model_type=`lightgcn`  ·  "
            f"n_layers=`3`  ·  II edges=`{ss.exp_adopted.get('use_ii_edges', False)}`"
        )
        lgcn_status = ss.exp_step_status.get("step4", "pending")
        run_lgcn = st.button("▶ Run LightGCN",
                             disabled=not lgcn_ready,
                             key="run_lgcn", use_container_width=True)
        if run_lgcn:
            log_box4 = st.empty()
            with st.spinner("Running LightGCN ..."):
                res4 = _run_exp(
                    {"model_type": "lightgcn", "n_layers": 3, "use_residual": False},
                    log_box4,
                )
            if res4 is not None:
                ss.exp_step_metrics["step4"] = res4
                ss.exp_step_status["step4"]  = "ran"
                lgcn_ndcg = res4["ranking"].get(10, {}).get("NDCG", 0)
                lgcn_hr   = res4["ranking"].get(10, {}).get("HR",   0)
                lgcn_p    = res4["ranking"].get(10, {}).get("P",    0)
                ss.exp_summary.append({
                    "Name":       "LightGCN (3L)",
                    "II Edges":   "✓" if ss.exp_adopted.get("use_ii_edges") else "✗",
                    "Layers":     3,
                    "Residual":   "N/A",
                    "Model":      "LIGHTGCN",
                    "P@10 %":     round(lgcn_p    * 100, 3),
                    "HR@10 %":    round(lgcn_hr   * 100, 3),
                    "NDCG@10 %":  round(lgcn_ndcg * 100, 3),
                    "vs Prev":    f"{(lgcn_ndcg - baseline_ndcg)*100:+.3f} % vs baseline",
                })
                st.rerun()

        if lgcn_status == "ran":
            r4 = ss.exp_step_metrics.get("step4", {})
            c1, c2, c3 = st.columns(3)
            c1.metric("P@10 %",    f"{r4['ranking'][10]['P']*100:.3f}")
            c2.metric("HR@10 %",   f"{r4['ranking'][10]['HR']*100:.3f}")
            c3.metric("NDCG@10 %", f"{r4['ranking'][10]['NDCG']*100:.3f}")

    # ── Summary comparison table ───────────────────────────────────────────────
    if len(ss.exp_summary) > 1:
        st.divider()
        st.subheader("Experiment Summary")
        df_sum = pd.DataFrame(ss.exp_summary)
        st.dataframe(df_sum.set_index("Name"), use_container_width=True)

        # NDCG bar chart across experiments
        fig, ax = plt.subplots(figsize=(10, 4))
        names  = [r["Name"] for r in ss.exp_summary]
        ndcgs  = [r["NDCG@10 %"] for r in ss.exp_summary]
        hrs    = [r["HR@10 %"]   for r in ss.exp_summary]
        x_e    = np.arange(len(names))
        w_e    = 0.35
        ax.bar(x_e - w_e/2, ndcgs, w_e, label="NDCG@10 %", color="steelblue")
        ax.bar(x_e + w_e/2, hrs,   w_e, label="HR@10 %",   color="mediumpurple")
        ax.set_xticks(x_e); ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_ylabel("Score (%)"); ax.set_title("Experiment progression")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ── Apply best config to main Training tab ────────────────────────────────
    st.divider()
    st.subheader("Apply Adopted Config to Training Tab")
    st.caption(
        "Sets the sidebar values to the currently adopted experiment configuration "
        "so the next training run uses the validated settings."
    )
    if st.button("⚙️ Apply adopted config to sidebar", use_container_width=True):
        adp = ss.exp_adopted
        if "use_ii_edges" in adp:
            st.session_state["sb_use_ii_edges"] = adp["use_ii_edges"]
        if "n_layers"     in adp:
            st.session_state["sb_n_layers"]     = adp["n_layers"]
        if "use_residual" in adp:
            st.session_state["sb_use_residual"] = adp["use_residual"]
        if "model_type"   in adp:
            st.session_state["sb_model"]        = adp["model_type"]
        st.success("Sidebar updated — go to Train / Tune tab and click ▶ Run.")
