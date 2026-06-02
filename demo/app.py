"""
GNN Recommender — Interface de démonstration interactive.

Onglets :
  1. Recommandations  : sélectionner un utilisateur existant, voir ses recs
  2. Nouvel utilisateur : simuler un profil froid (cold-start)
  3. Apprentissage incrémental : ajouter des interactions et réentraîner

Lancement :
  cd gnn_recommender
  streamlit run demo/app.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from demo.inference import (
    build_edge_index_from_ckpt,
    build_model_from_ckpt,
    compute_embeddings,
    enrich_recommendations,
    find_checkpoints,
    get_user_history,
    load_business_df,
    load_checkpoint,
    recommend,
    recommend_cold_start,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GNN Recommender Demo",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.metric-box {
    background:#1e1e2e; border-radius:8px; padding:16px; text-align:center;
}
.rec-card {
    background:#2a2a3e; border-radius:6px; padding:10px; margin:4px 0;
    border-left: 3px solid #7c3aed;
}
</style>
""", unsafe_allow_html=True)


# ── Cached resources ───────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Chargement du checkpoint…")
def _load_ckpt(ckpt_path: str) -> dict:
    return load_checkpoint(ckpt_path, torch.device("cpu"))


@st.cache_resource(show_spinner="Construction du graphe…")
def _load_model_and_embs(ckpt_path: str):
    ckpt = _load_ckpt(ckpt_path)
    model = build_model_from_ckpt(ckpt, torch.device("cpu"))
    edge_index, n_users, n_items = build_edge_index_from_ckpt(ckpt)
    if edge_index is None:
        return None, None, n_users, n_items
    embs = compute_embeddings(model, edge_index)
    return model, embs, n_users, n_items


@st.cache_data(show_spinner="Chargement des données business…")
def _load_biz(ckpt_path: str, data_dir: str) -> pd.DataFrame:
    ckpt = _load_ckpt(ckpt_path)
    return load_business_df(ckpt, data_dir)


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔮 GNN Recommender")
    st.markdown("---")

    # Checkpoint selection
    st.subheader("📦 Checkpoint")
    ckpt_files = find_checkpoints(ROOT / "checkpoints")
    if not ckpt_files:
        st.error("Aucun checkpoint trouvé dans `checkpoints/`.\n\n"
                 "Lancez d'abord l'entraînement :\n"
                 "```\npython src/main.py --model sage --mode scratch "
                 "--data-dir data/raw/full --epochs 200 --no-amp\n```")
        st.stop()

    ckpt_labels = [f.relative_to(ROOT).as_posix() for f in ckpt_files]
    ckpt_choice = st.selectbox("Fichier checkpoint", ckpt_labels)
    ckpt_path   = str(ROOT / ckpt_choice)

    # Data directory
    st.subheader("📁 Données")
    data_dir = st.text_input("Répertoire données", value=str(ROOT / "data" / "raw" / "full"))

    st.markdown("---")

    # Load everything
    ckpt  = _load_ckpt(ckpt_path)
    mc    = ckpt["model_config"]
    extra = ckpt.get("extra", {})
    _ue = ckpt.get("user_encoder")
    _ie = ckpt.get("item_encoder")
    n_users_ckpt = extra.get("n_users") or (len(_ue.classes_) if _ue and hasattr(_ue, "classes_") else mc["num_nodes"] // 2)
    n_items_ckpt = extra.get("n_items") or (len(_ie.classes_) if _ie and hasattr(_ie, "classes_") else mc["num_nodes"] - n_users_ckpt)
    item_enc = ckpt.get("item_encoder")
    user_enc = ckpt.get("user_encoder")

    _, embs, n_users, n_items = _load_model_and_embs(ckpt_path)
    biz_df  = _load_biz(ckpt_path, data_dir)

    # Info panel
    st.subheader("ℹ️ Informations modèle")
    st.markdown(f"**Modèle :** `{mc['model_type'].upper()}`")
    st.markdown(f"**Utilisateurs :** {n_users_ckpt:,}")
    st.markdown(f"**Items :** {n_items_ckpt:,}")
    st.markdown(f"**Epoch :** {ckpt.get('epoch', '?')}")
    st.markdown(f"**Mode :** `{ckpt.get('training_mode', 'scratch')}`")

    # val_score display: incremental runs store a sentinel (version × 1e6) to
    # force overwrite of gat_best.pt — show real NDCG when available, else label
    _raw_val = ckpt.get("val_score", 0)
    _mode    = ckpt.get("training_mode", "scratch")
    if _raw_val is not None and _raw_val < 2.0:
        # Real composite score in [0, 1] from scratch/tune
        st.markdown(f"**Val score :** {_raw_val:.4f}")
    elif _mode == "incremental":
        _incr_run = int(_raw_val // 1_000_000)
        st.markdown(f"**Run incrémental :** #{_incr_run}")
    else:
        # Scratch training where validation never ran (very short run)
        st.markdown(f"**Val score :** n/a")

    if embs is None:
        st.warning("Impossible de reconstruire le graphe depuis ce checkpoint.")


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔍 Recommandations",
    "👤 Nouvel Utilisateur",
    "🔄 Apprentissage Incrémental",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Recommandations pour un utilisateur existant
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Recommandations pour un utilisateur existant")

    if embs is None:
        st.error("Embeddings non disponibles — impossible de reconstruire le graphe.")
        st.stop()

    col_left, col_right = st.columns([1, 2], gap="large")

    with col_left:
        st.subheader("Sélection utilisateur")

        # User selection
        if user_enc and hasattr(user_enc, "classes_"):
            user_ids_display = list(user_enc.classes_[:200])
            selected_user_id = st.selectbox(
                "Utilisateur (user_id)", user_ids_display,
                help="Utilisateurs connus du modèle (200 premiers affichés)"
            )
            user_idx = user_enc.transform([selected_user_id])[0]
        else:
            user_idx = st.slider(
                "Index utilisateur", 0, n_users - 1, 0,
                help="Index interne de l'utilisateur"
            )
            selected_user_id = f"user_{user_idx}"

        k = st.slider("Nombre de recommandations (K)", 5, 50, 10)

        st.markdown("---")
        st.subheader("Historique d'interactions")

        history = get_user_history(user_idx, ckpt)
        seen_items = set(history["item_idx"].tolist()) if len(history) > 0 else set()

        if len(history) == 0:
            st.info("Aucun historique trouvé pour cet utilisateur.")
        else:
            # Enrich history with business names
            hist_rows = []
            for _, row in history.iterrows():
                item_idx = int(row["item_idx"])
                biz_id = (
                    item_enc.classes_[item_idx]
                    if item_enc and item_idx < len(item_enc.classes_)
                    else f"item_{item_idx}"
                )
                name = cats = "N/A"
                if not biz_df.empty and "business_id" in biz_df.columns:
                    match = biz_df[biz_df["business_id"].astype(str).str.strip() == str(biz_id).strip()]
                    if len(match) > 0:
                        name = str(match.iloc[0].get("name", biz_id) or biz_id).strip()
                        cats = str(match.iloc[0].get("categories", "N/A") or "N/A")[:70]
                hist_rows.append({
                    "⭐ Rating": f"{'⭐' * int(row['rating'])}",
                    "Nom":       name,
                    "Catégories": cats,
                })

            st.dataframe(
                pd.DataFrame(hist_rows),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(f"{len(history)} interactions dans l'historique")

    with col_right:
        st.subheader(f"Top-{k} recommandations")

        recs = recommend(user_idx, embs, n_users, n_items, seen_items, k=k)
        recs_df = enrich_recommendations(recs, item_enc, biz_df)

        if recs_df.empty:
            st.warning("Aucune recommandation générée.")
        else:
            # Score chart
            fig = px.bar(
                recs_df, x="Score", y="Nom" if "Nom" in recs_df.columns else "business_id",
                orientation="h",
                color="Score",
                color_continuous_scale="Viridis",
                title=f"Scores de recommandation — {selected_user_id}",
            )
            fig.update_layout(
                height=400,
                yaxis={"categoryorder": "total ascending"},
                showlegend=False,
                plot_bgcolor="#0e1117",
                paper_bgcolor="#0e1117",
                font_color="white",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Table
            display_cols = [c for c in ["Rang", "Nom", "Catégories", "Score"] if c in recs_df.columns]
            st.dataframe(
                recs_df[display_cols],
                use_container_width=True,
                hide_index=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Simuler un nouvel utilisateur (cold-start)
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Simuler un nouvel utilisateur — Cold Start")
    st.markdown(
        "_Construisez un profil multi-catégories en ajoutant des items au panier. "
        "Le modèle utilisera la **moyenne des embeddings** comme proxy de profil._"
    )

    if embs is None:
        st.error("Embeddings non disponibles.")
        st.stop()

    # Session state basket
    if "cold_basket" not in st.session_state:
        st.session_state.cold_basket = {}  # {name: biz_id}

    # Build category list
    all_cats = []
    if not biz_df.empty and "categories" in biz_df.columns:
        for cats_str in biz_df["categories"].dropna().unique():
            for cat in str(cats_str).split(","):
                c = cat.strip()
                if c and c != "N/A":
                    all_cats.append(c)
    all_cats = sorted(set(all_cats))[:100]

    if not all_cats:
        st.warning(
            "⚠️ Aucune catégorie trouvée — vérifiez le **Répertoire données** dans la barre latérale. "
            "Le fichier `yelp_academic_dataset_business_healthandmedical.csv` doit s'y trouver."
        )

    # Build full name→biz_id mapping once
    name_col = "name" if "name" in biz_df.columns else "business_id"
    full_name_to_biz = {}
    if not biz_df.empty and "business_id" in biz_df.columns:
        for _, row in biz_df.iterrows():
            full_name_to_biz[row[name_col]] = row["business_id"]

    col_a, col_b = st.columns([1, 2], gap="large")

    with col_a:
        st.subheader("Construire le profil")

        cat_filter = st.selectbox(
            "Filtrer par catégorie", ["(toutes)"] + all_cats,
            help="Changer la catégorie ne vide pas le panier"
        )

        # Filter items by category
        filtered_biz = biz_df.copy()
        if cat_filter != "(toutes)" and "categories" in filtered_biz.columns:
            filtered_biz = filtered_biz[
                filtered_biz["categories"].str.contains(cat_filter, na=False, case=False)
            ]

        display_names = [n for n in filtered_biz[name_col].tolist() if n not in st.session_state.cold_basket]

        selected = st.multiselect(
            f"Items à ajouter ({cat_filter})",
            display_names[:100],
            help="Sélectionnez puis cliquez Ajouter au panier"
        )

        col_add, col_clear = st.columns(2)
        with col_add:
            if st.button("➕ Ajouter au panier", use_container_width=True, disabled=not selected):
                for name in selected:
                    biz_id = full_name_to_biz.get(name, name)
                    st.session_state.cold_basket[name] = biz_id
                st.rerun()
        with col_clear:
            if st.button("🗑️ Vider le panier", use_container_width=True, disabled=not st.session_state.cold_basket):
                st.session_state.cold_basket = {}
                st.rerun()

        st.markdown("---")
        st.markdown(f"**🛒 Panier ({len(st.session_state.cold_basket)}/10 items) :**")
        if st.session_state.cold_basket:
            for name in list(st.session_state.cold_basket):
                c1, c2 = st.columns([4, 1])
                c1.markdown(f"• {name}")
                if c2.button("✕", key=f"rm_{name}", help="Retirer"):
                    del st.session_state.cold_basket[name]
                    st.rerun()
        else:
            st.caption("_Vide — ajoutez des items depuis n'importe quelle catégorie_")

        k2 = st.slider("Recommandations (K)", 5, 30, 10, key="k_cold")

    with col_b:
        st.subheader("Recommandations cold-start")

        basket = st.session_state.cold_basket
        if not basket:
            st.info("Ajoutez au moins un item dans le panier pour voir les recommandations.")
        else:
            # Map basket names → item indices
            liked_item_indices = []
            for name, biz_id in basket.items():
                if item_enc and item_enc.is_known(biz_id):
                    liked_item_indices.append(int(item_enc.transform([biz_id])[0]))

            if not liked_item_indices:
                st.warning("Impossible de résoudre les indices items.")
            else:
                recs_cold = recommend_cold_start(liked_item_indices, embs, n_users, n_items, k=k2)
                recs_cold_df = enrich_recommendations(recs_cold, item_enc, biz_df)

                st.markdown("**Profil simulé :**")
                # Show basket grouped by category
                cats_in_basket = []
                for name in basket:
                    row = biz_df[biz_df[name_col] == name]
                    if not row.empty and "categories" in row.columns:
                        cats_in_basket.append(str(row.iloc[0]["categories"]).split(",")[0].strip())
                if cats_in_basket:
                    from collections import Counter
                    cat_counts = Counter(cats_in_basket)
                    st.markdown(" · ".join([f"`{c}` ×{n}" for c, n in cat_counts.most_common()]))
                st.markdown(" · ".join([f"`{n}`" for n in basket]))

                st.markdown("---")

                if recs_cold_df.empty:
                    st.warning("Aucune recommandation générée.")
                else:
                    fig = px.bar(
                        recs_cold_df,
                        x="Score",
                        y="Nom" if "Nom" in recs_cold_df.columns else "business_id",
                        orientation="h",
                        color="Score",
                        color_continuous_scale="Plasma",
                        title=f"Recommandations — Profil cold-start ({len(basket)} items)",
                    )
                    fig.update_layout(
                        height=max(300, len(recs_cold_df) * 35),
                        yaxis={"categoryorder": "total ascending"},
                        showlegend=False,
                        plot_bgcolor="#0e1117",
                        paper_bgcolor="#0e1117",
                        font_color="white",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    display_cols = [c for c in ["Rang", "Nom", "Catégories", "Score"] if c in recs_cold_df.columns]
                    st.dataframe(recs_cold_df[display_cols], use_container_width=True, hide_index=True)

                    st.caption(
                        "⚠️ Cold-start via moyenne d'embeddings. "
                        "Après apprentissage incrémental, la précision augmente."
                    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Apprentissage incrémental
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Apprentissage Incrémental")
    st.markdown(
        "Le modèle se met à jour sans réentraînement complet : "
        "**fine-tuning** sur les nouvelles données + **replay buffer** pour éviter l'oubli catastrophique."
    )

    # ── Info box explaining the 3 components ──────────────────────────────────
    with st.expander("ℹ️ Comment fonctionne l'apprentissage incrémental ?", expanded=False):
        st.markdown("""
Un apprentissage incrémental correct sur un GNN nécessite **3 composants** :

| Composant | Rôle dans le graphe | Ce qui change dans le modèle |
|-----------|--------------------|-----------------------------|
| **Nouvel utilisateur** | Nouveau nœud user | Ligne ajoutée dans la matrice d'embeddings (Xavier init) |
| **Nouvel item** | Nouveau nœud item | Ligne ajoutée dans la matrice d'embeddings (Xavier init) |
| **Nouvelles interactions** | Nouvelles arêtes user↔item | Poids affinés par fine-tuning (BPR loss + replay) |

Le graphe bipartite **s'étend** à chaque run incrémental.
Les anciens embeddings sont **préservés** — seuls les nouveaux nœuds sont initialisés aléatoirement.
        """)

    # ── Mode selector ─────────────────────────────────────────────────────────
    input_mode = st.radio(
        "Mode d'entrée",
        ["📁 Upload fichier (recommandé — 1k+ interactions)", "✏️ Saisie manuelle (demo rapide)"],
        horizontal=True,
        key="incr_input_mode",
    )
    upload_mode = input_mode.startswith("📁")

    st.markdown("---")
    col1, col2 = st.columns([1, 1], gap="large")

    # ── Left column ───────────────────────────────────────────────────────────
    with col1:

        new_interactions = []   # filled by either branch
        user_is_new = False
        incr_user_id = ""

        # ══════════════════════════════════════════════════════════════════════
        # MODE A : Upload 3 fichiers Yelp
        # ══════════════════════════════════════════════════════════════════════
        if upload_mode:
            st.subheader("📁 Upload des 3 fichiers")

            with st.expander("ℹ️ Format attendu (identique à data/1k/, data/5k/…)"):
                st.markdown("""
| Fichier | Colonnes requises | Colonnes optionnelles |
|---------|------------------|-----------------------|
| **Reviews** *(obligatoire)* | `user_id`, `business_id`, `stars` | `date`, `review_id`, `text`, `useful`, `funny`, `cool` |
| **Users** *(optionnel)* | `user_id` | `name`, `review_count`, `yelping_since`, `average_stars`, `fans` |
| **Business** *(optionnel)* | `business_id` | `name`, `address`, `city`, `state`, `stars`, `categories` |

Les `user_id` / `business_id` inconnus du modèle sont **automatiquement ajoutés** comme nouveaux nœuds.
                """)

            # ── 3 uploaders côte à côte ───────────────────────────────────────
            up_col1, up_col2, up_col3 = st.columns(3)
            with up_col1:
                st.markdown("**① Reviews** *(obligatoire)*")
                f_reviews = st.file_uploader("", type=["csv", "json"], key="up_reviews",
                                             label_visibility="collapsed")
            with up_col2:
                st.markdown("**② Users** *(optionnel)*")
                f_users = st.file_uploader("", type=["csv", "json"], key="up_users",
                                           label_visibility="collapsed")
            with up_col3:
                st.markdown("**③ Business** *(optionnel)*")
                f_business = st.file_uploader("", type=["csv", "json"], key="up_business",
                                              label_visibility="collapsed")

            # ── Helper: lire CSV ou JSON ──────────────────────────────────────
            def _read_upload(f):
                if f is None:
                    return None
                try:
                    if f.name.endswith(".json"):
                        import json as _json
                        df = pd.DataFrame(_json.load(f))
                    else:
                        df = pd.read_csv(f)
                    df.columns = [c.strip().lower() for c in df.columns]
                    return df
                except Exception as e:
                    st.error(f"Erreur lecture `{f.name}` : {e}")
                    return None

            upload_df      = None   # reviews
            upload_users   = None
            upload_biz_new = None

            # ── Lire reviews ─────────────────────────────────────────────────
            if f_reviews is not None:
                _rv = _read_upload(f_reviews)
                if _rv is not None:
                    if "rating" in _rv.columns and "stars" not in _rv.columns:
                        _rv = _rv.rename(columns={"rating": "stars"})
                    missing = {"user_id", "business_id", "stars"} - set(_rv.columns)
                    if missing:
                        st.error(f"Reviews — colonnes manquantes : {missing}")
                    else:
                        _rv["stars"] = pd.to_numeric(_rv["stars"], errors="coerce").fillna(3.0)
                        if "date" not in _rv.columns:
                            _rv["date"] = "2024-01-01"
                        _rv = _rv.dropna(subset=["user_id", "business_id"])
                        upload_df = _rv

            # ── Lire users ───────────────────────────────────────────────────
            if f_users is not None:
                _uu = _read_upload(f_users)
                if _uu is not None and "user_id" in _uu.columns:
                    upload_users = _uu
                elif _uu is not None:
                    st.warning("Users — colonne `user_id` manquante, fichier ignoré.")

            # ── Lire business ─────────────────────────────────────────────────
            if f_business is not None:
                _bb = _read_upload(f_business)
                if _bb is not None and "business_id" in _bb.columns:
                    upload_biz_new = _bb
                elif _bb is not None:
                    st.warning("Business — colonne `business_id` manquante, fichier ignoré.")

            # ── Analyse croisée ───────────────────────────────────────────────
            if upload_df is not None and not upload_df.empty:
                all_users_up = upload_df["user_id"].astype(str).unique()
                all_items_up = upload_df["business_id"].astype(str).unique()
                new_users_up = [u for u in all_users_up if user_enc is None or not user_enc.is_known(u)]
                new_items_up = [b for b in all_items_up if item_enc is None or not item_enc.is_known(b)]

                st.markdown("---")
                st.markdown("**📊 Analyse croisée**")

                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Reviews",        f"{len(upload_df):,}")
                m2.metric("Users uniques",  len(all_users_up))
                m3.metric("Nouveaux users", len(new_users_up),
                          delta=f"+{len(new_users_up)} nœuds" if new_users_up else None)
                m4.metric("Items uniques",  len(all_items_up))
                m5.metric("Nouveaux items", len(new_items_up),
                          delta=f"+{len(new_items_up)} nœuds" if new_items_up else None)
                m6.metric("Fichiers",
                          f"{1 + (upload_users is not None) + (upload_biz_new is not None)}/3")

                # Info nouveaux nœuds
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    if new_users_up:
                        names_known = []
                        if upload_users is not None and "name" in upload_users.columns:
                            for uid in new_users_up[:5]:
                                row = upload_users[upload_users["user_id"] == uid]
                                if len(row) > 0:
                                    names_known.append(row.iloc[0].get("name", uid))
                                else:
                                    names_known.append(uid[:12] + "…")
                        else:
                            names_known = [u[:12] + "…" for u in new_users_up[:5]]
                        extra = f" (+{len(new_users_up)-5} autres)" if len(new_users_up) > 5 else ""
                        st.info(f"🆕 **{len(new_users_up)} nouveaux users** → "
                                f"{', '.join(names_known)}{extra}")
                with col_info2:
                    if new_items_up:
                        names_known = []
                        if upload_biz_new is not None and "name" in upload_biz_new.columns:
                            for bid in new_items_up[:5]:
                                row = upload_biz_new[upload_biz_new["business_id"] == bid]
                                if len(row) > 0:
                                    names_known.append(row.iloc[0].get("name", bid))
                                else:
                                    names_known.append(bid[:12] + "…")
                        else:
                            names_known = [b[:12] + "…" for b in new_items_up[:5]]
                        extra = f" (+{len(new_items_up)-5} autres)" if len(new_items_up) > 5 else ""
                        st.info(f"🆕 **{len(new_items_up)} nouveaux items** → "
                                f"{', '.join(names_known)}{extra}")

                # Tabs d'aperçu
                tab_rv, tab_uu, tab_bb = st.tabs(["Reviews", "Users", "Business"])
                with tab_rv:
                    fig_stars = px.histogram(
                        upload_df, x="stars", nbins=5,
                        title="Distribution des notes",
                        color_discrete_sequence=["#4f8ef7"],
                    )
                    fig_stars.update_layout(height=180, margin=dict(t=30, b=0, l=0, r=0))
                    st.plotly_chart(fig_stars, use_container_width=True)
                    # Enrich preview with names if business file provided
                    preview_rv = upload_df.head(5).copy()
                    if upload_biz_new is not None and "name" in upload_biz_new.columns:
                        preview_rv = preview_rv.merge(
                            upload_biz_new[["business_id", "name"]].rename(columns={"name": "biz_name"}),
                            on="business_id", how="left"
                        )
                    if upload_users is not None and "name" in upload_users.columns:
                        preview_rv = preview_rv.merge(
                            upload_users[["user_id", "name"]].rename(columns={"name": "user_name"}),
                            on="user_id", how="left"
                        )
                    st.dataframe(preview_rv, use_container_width=True, hide_index=True)
                with tab_uu:
                    if upload_users is not None:
                        st.caption(f"{len(upload_users)} utilisateurs — "
                                   f"{sum(upload_users['user_id'].isin(new_users_up))} nouveaux")
                        st.dataframe(upload_users.head(8), use_container_width=True, hide_index=True)
                    else:
                        st.info("Aucun fichier users chargé (optionnel).")
                with tab_bb:
                    if upload_biz_new is not None:
                        n_new_biz = sum(upload_biz_new["business_id"].isin(new_items_up))
                        st.caption(f"{len(upload_biz_new)} businesses — {n_new_biz} nouveaux")
                        st.dataframe(upload_biz_new.head(8), use_container_width=True, hide_index=True)
                    else:
                        st.info("Aucun fichier business chargé (optionnel).")

                # Merge new business metadata into biz_df for post-training display
                if upload_biz_new is not None and not upload_biz_new.empty:
                    st.session_state["upload_biz_new"] = upload_biz_new
                else:
                    st.session_state.pop("upload_biz_new", None)

                # Pour recs BEFORE
                incr_user_id = str(all_users_up[0])
                user_is_new  = incr_user_id in new_users_up
                new_interactions = upload_df[["user_id", "business_id", "stars", "date"]].to_dict("records")

            else:
                if f_reviews is None:
                    st.info("👆 Charge au minimum le fichier **Reviews** pour continuer.")
                # already showed error above

        # ══════════════════════════════════════════════════════════════════════
        # MODE B : Saisie manuelle
        # ══════════════════════════════════════════════════════════════════════
        else:
            # ── SECTION 1 : Utilisateur ───────────────────────────────────────
            st.subheader("① Utilisateur")
            if user_enc and hasattr(user_enc, "classes_"):
                incr_mode = st.radio(
                    "Type",
                    ["Utilisateur existant", "Nouvel utilisateur"],
                    horizontal=True,
                    key="incr_user_mode",
                )
                if incr_mode == "Utilisateur existant":
                    incr_user_id = st.selectbox(
                        "Sélectionner", list(user_enc.classes_[:200]), key="incr_user"
                    )
                    user_is_new = False
                else:
                    incr_user_id = st.text_input(
                        "ID du nouvel utilisateur",
                        value="new_user_demo_001",
                        help="Cet ID sera ajouté comme nouveau nœud dans le graphe",
                    )
                    user_is_new = True
            else:
                incr_mode = "Utilisateur existant"
                incr_user_id = f"user_{st.number_input('Index utilisateur', 0, n_users - 1, 0)}"
                user_is_new = False

            if user_enc and not user_is_new:
                user_is_new = not user_enc.is_known(incr_user_id)
            if user_is_new:
                st.info(f"🆕 Nouveau nœud user : `{incr_user_id}`")
            else:
                idx_str = str(user_enc.transform([incr_user_id])[0]) if (user_enc and user_enc.is_known(incr_user_id)) else "?"
                st.caption(f"Utilisateur connu — index {idx_str}")

            st.markdown("---")

            # ── SECTION 2 : Nouveaux items ────────────────────────────────────
            st.subheader("② Nouveaux Items (optionnel)")
            st.caption("Items qui n'existent pas encore dans le catalogue.")
            n_new_items_form = st.number_input("Nouveaux items à créer", 0, 5, 0, key="n_new_items")

            new_item_label_to_biz = {}
            for ni in range(n_new_items_form):
                with st.container(border=True):
                    c_name, c_cat = st.columns([2, 2])
                    with c_name:
                        ni_name = st.text_input(f"Nom {ni+1}", value=f"Nouvelle Clinique {ni+1}", key=f"ni_name_{ni}")
                    with c_cat:
                        ni_cat  = st.text_input(f"Catégorie {ni+1}", value="Health & Medical", key=f"ni_cat_{ni}")
                    ni_biz_id = f"new_item_{ni+1:03d}_{ni_name[:10].replace(' ','_').lower()}"
                    st.caption(f"ID généré : `{ni_biz_id}`")
                    new_item_label_to_biz[f"[NOUVEAU] {ni_name}  —  {ni_cat}"] = ni_biz_id

            st.markdown("---")

            # ── SECTION 3 : Interactions ──────────────────────────────────────
            st.subheader("③ Interactions (arêtes)")
            n_new_interactions = st.number_input("Nombre d'interactions", 1, 20, 5)

            all_cats_incr = ["(toutes)"]
            if not biz_df.empty and "categories" in biz_df.columns:
                for cats_str in biz_df["categories"].dropna().unique():
                    for cat in str(cats_str).split(","):
                        c = cat.strip()
                        if c and c != "N/A":
                            all_cats_incr.append(c)
                all_cats_incr = ["(toutes)"] + sorted(set(all_cats_incr[1:]))[:100]

            cat_filter_incr = st.selectbox("Filtrer par catégorie", all_cats_incr, key="cat_filter_incr")

            if not biz_df.empty:
                filtered_incr = biz_df.copy()
                if cat_filter_incr != "(toutes)" and "categories" in filtered_incr.columns:
                    filtered_incr = filtered_incr[
                        filtered_incr["categories"].str.contains(cat_filter_incr, na=False, case=False)
                    ]
                def _make_label(row):
                    name = row.get("name", row.get("business_id", "?"))
                    cats = str(row.get("categories", "")).split(",")[0].strip()
                    return f"{name}  —  {cats}" if cats and cats != "N/A" else name
                filtered_incr = filtered_incr.head(200).copy()
                filtered_incr["_label"] = filtered_incr.apply(_make_label, axis=1)
                existing_label_to_biz = dict(zip(filtered_incr["_label"], filtered_incr["business_id"]))
                existing_labels = filtered_incr["_label"].tolist()
            else:
                existing_labels = [
                    item_enc.classes_[i] if item_enc and i < len(item_enc.classes_) else f"item_{i}"
                    for i in range(min(200, n_items))
                ]
                existing_label_to_biz = {l: l for l in existing_labels}

            all_labels_incr  = list(new_item_label_to_biz.keys()) + existing_labels
            all_label_to_biz = {**new_item_label_to_biz, **existing_label_to_biz}

            for i in range(n_new_interactions):
                c_item, c_rating = st.columns([3, 1])
                with c_item:
                    chosen_label = st.selectbox(
                        f"Item {i+1}", all_labels_incr,
                        index=min(i * 7, len(all_labels_incr) - 1),
                        key=f"item_{i}",
                    )
                with c_rating:
                    rating = st.select_slider("⭐", options=[1, 2, 3, 4, 5], value=4, key=f"rating_{i}")
                biz_id = all_label_to_biz.get(chosen_label, chosen_label)
                new_interactions.append({
                    "user_id": incr_user_id, "business_id": biz_id,
                    "stars": rating, "date": "2024-01-01",
                })

            st.markdown("---")
            # Summary
            interacted_ids = set(r["business_id"] for r in new_interactions)
            n_new_it = sum(1 for b in interacted_ids if item_enc is None or not item_enc.is_known(b))
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Nouveaux users", 1 if user_is_new else 0)
            sc2.metric("Nouveaux items", n_new_it)
            sc3.metric("Interactions",   len(new_interactions))

        # ── Training settings (commun aux 2 modes) ────────────────────────────
        st.markdown("---")
        with st.expander("⚙️ Paramètres d'entraînement"):
            auto_hpo = st.checkbox("🔍 Optimiser automatiquement (Optuna)", value=False,
                                   help="Cherche les meilleurs hyperparamètres via Optuna")
            if auto_hpo:
                n_trials      = st.slider("Trials Optuna", 5, 30, 10)
                st.caption("Les paramètres seront choisis automatiquement.")
                incr_epochs   = 20
                incr_lr_scale = 0.1
                incr_replay   = 0.3
            else:
                n_trials      = 0
                incr_epochs   = st.slider("Epochs", 5, 100, 20)
                incr_lr_scale = st.select_slider("Facteur LR", options=[0.01, 0.05, 0.1, 0.2], value=0.1)
                incr_replay   = st.select_slider("Replay ratio", options=[0.0, 0.1, 0.2, 0.3, 0.5], value=0.3)

        # ── Recs BEFORE (pour avoir la comparaison avant/après) ───────────────
        if embs is not None and new_interactions:
            if not user_is_new and user_enc and user_enc.is_known(incr_user_id):
                user_idx_incr  = int(user_enc.transform([incr_user_id])[0])
                history_incr   = get_user_history(user_idx_incr, ckpt)
                seen_incr      = set(history_incr["item_idx"].tolist())
                recs_before    = recommend(user_idx_incr, embs, n_users, n_items, seen_incr, k=10)
            else:
                liked_idx = [
                    int(item_enc.transform([r["business_id"]])[0])
                    for r in new_interactions[:20]
                    if item_enc and item_enc.is_known(r["business_id"])
                ]
                recs_before = recommend_cold_start(liked_idx, embs, n_users, n_items, k=10)
            recs_before_df = enrich_recommendations(recs_before, item_enc, biz_df)
            st.session_state["recs_before"] = recs_before_df

        # ── Launch button ─────────────────────────────────────────────────────
        can_launch = bool(new_interactions)
        launch_btn = st.button(
            "🚀 Lancer l'apprentissage incrémental",
            type="primary",
            use_container_width=True,
            disabled=not can_launch,
        )
        if not can_launch:
            st.caption("Charge un fichier ou saisis des interactions pour activer.")

    # ── Right column: results ──────────────────────────────────────────────────
    with col2:
        st.subheader("📊 Résultats")

        if "recs_before" in st.session_state:
            st.markdown("**Recommandations AVANT :**")
            before_df = st.session_state["recs_before"]
            display_cols = [c for c in ["Rang", "Nom", "Catégories", "Score"] if c in before_df.columns]
            st.dataframe(before_df[display_cols].head(10), use_container_width=True, hide_index=True)

        if launch_btn:
            # Write new interactions to temp CSV
            new_df = pd.DataFrame(new_interactions)

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False, prefix="incr_demo_"
            ) as f:
                new_df.to_csv(f, index=False)
                tmp_csv = f.name

            # ── HPO phase (if enabled) ─────────────────────────────────────
            if auto_hpo and n_trials > 0:
                with st.spinner(f"🔍 Optuna HPO — {n_trials} trials en cours…"):
                    import sys as _sys
                    _sys.path.insert(0, str(ROOT / "src"))
                    from training.incremental_hpo import tune_incremental_hparams
                    from data.preprocessing import DynamicLabelEncoder
                    from utils.checkpoint import CheckpointManager
                    from models import build_model as _build_model
                    import copy as _copy

                    # Encode new_df for HPO
                    # Always reload checkpoint fresh (cache may be stale after previous incremental runs)
                    from utils.checkpoint import CheckpointManager as _CM
                    _fresh_ckpt = _CM.load(ckpt_path, torch.device("cpu"))
                    _ue = _fresh_ckpt.get("user_encoder")
                    _ie = _fresh_ckpt.get("item_encoder")
                    _n_u = len(_ue.classes_)

                    hpo_rows = []
                    _proxy_uid = 0  # fallback: use user 0 as proxy for unknown users
                    for _, r in new_df.iterrows():
                        # User: use real index if known, else proxy (HPO cares mainly about items)
                        if _ue.is_known(str(r["user_id"])):
                            uid = int(_ue.transform([str(r["user_id"])])[0])
                        else:
                            uid = _proxy_uid

                        # Item: must be known to compute embedding
                        biz_id = str(r["business_id"])
                        if _ie.is_known(biz_id):
                            iid = int(_ie.transform([biz_id])[0]) + _n_u
                            hpo_rows.append({"user_id": uid, "item_id": iid, "rating": float(r["stars"])})

                    if hpo_rows:
                        hpo_df = pd.DataFrame(hpo_rows)

                        trial_log = st.empty()
                        trial_rows = []

                        def _hpo_cb(trial_num, params, score):
                            trial_rows.append({
                                "Trial": trial_num,
                                "Epochs": params["finetune_epochs"],
                                "LR scale": params["finetune_lr_scale"],
                                "Replay": params["replay_ratio"],
                                "Score": round(score, 5),
                            })
                            trial_log.dataframe(pd.DataFrame(trial_rows), hide_index=True, use_container_width=True)

                        from config import Config
                        _cfg = Config()
                        _cfg.model_type = mc["model_type"]

                        best_params = tune_incremental_hparams(
                            ckpt=_fresh_ckpt,
                            new_df=hpo_df,
                            cfg_base=_cfg,
                            device=torch.device("cpu"),
                            n_trials=n_trials,
                            callback=_hpo_cb,
                        )

                        incr_epochs   = best_params["finetune_epochs"]
                        incr_lr_scale = best_params["finetune_lr_scale"]
                        incr_replay   = best_params["replay_ratio"]

                        st.success(
                            f"✅ Meilleurs params : **epochs={incr_epochs}** · "
                            f"**lr_scale={incr_lr_scale}** · **replay={incr_replay}** "
                            f"(score={best_params['best_score']:.5f})"
                        )
                    else:
                        st.warning("Impossible d'encoder les nouvelles interactions pour HPO.")

            new_ckpt_dir = str(ROOT / "checkpoints" / mc["model_type"])

            cmd = [
                sys.executable,
                str(ROOT / "src" / "main.py"),
                "--model",             mc["model_type"],
                "--mode",              "incremental",
                "--ckpt",              ckpt_path,
                "--new-data",          tmp_csv,
                "--ckpt-dir",          new_ckpt_dir,
                "--no-amp",
                "--finetune-epochs",   str(incr_epochs),
                "--finetune-lr-scale", str(incr_lr_scale),
                "--replay-ratio",      str(incr_replay),
            ]

            log_area = st.empty()
            progress_bar = st.progress(0)
            log_lines: list[str] = []

            with st.spinner("Entraînement incrémental en cours…"):
                t0 = time.perf_counter()
                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        cwd=str(ROOT),
                        env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
                    )

                    epoch_count = 0
                    for line in proc.stdout:
                        line = line.rstrip()
                        log_lines.append(line)
                        # Keep last 20 lines in UI
                        log_area.code("\n".join(log_lines[-20:]), language="bash")

                        # Parse epoch progress
                        if "epoch=" in line.lower() or "[train]" in line.lower():
                            epoch_count += 1
                            progress_bar.progress(min(epoch_count / incr_epochs, 1.0))

                    proc.wait()
                    t_elapsed = round(time.perf_counter() - t0, 1)

                    if proc.returncode == 0:
                        progress_bar.progress(1.0)
                        st.success(f"✅ Entraînement terminé en {t_elapsed}s")

                        # Invalidate caches and reload
                        _load_ckpt.clear()
                        _load_model_and_embs.clear()
                        _load_biz.clear()

                        # Load new checkpoint
                        new_best = Path(new_ckpt_dir) / f"{mc['model_type']}_best.pt"
                        if new_best.exists():
                            st.info(f"Nouveau checkpoint : `{new_best.relative_to(ROOT)}`")

                            new_ckpt = load_checkpoint(str(new_best))
                            new_model = build_model_from_ckpt(new_ckpt, torch.device("cpu"))
                            new_edge_index, new_n_users, new_n_items = build_edge_index_from_ckpt(new_ckpt)

                            if new_edge_index is not None:
                                new_embs = compute_embeddings(new_model, new_edge_index)
                                new_item_enc = new_ckpt.get("item_encoder")
                                new_user_enc = new_ckpt.get("user_encoder")

                                # Merge new business metadata so recs show names/categories
                                display_biz_df = biz_df.copy()
                                if "upload_biz_new" in st.session_state:
                                    _nb = st.session_state["upload_biz_new"]
                                    # Only add rows whose business_id is not already present
                                    existing_ids = set(display_biz_df["business_id"].tolist()) if "business_id" in display_biz_df.columns else set()
                                    _nb_new = _nb[~_nb["business_id"].isin(existing_ids)]
                                    if len(_nb_new) > 0:
                                        display_biz_df = pd.concat([display_biz_df, _nb_new], ignore_index=True)
                                else:
                                    display_biz_df = biz_df

                                # Get recs AFTER
                                if new_user_enc and new_user_enc.is_known(incr_user_id):
                                    new_user_idx = int(new_user_enc.transform([incr_user_id])[0])
                                    new_history  = get_user_history(new_user_idx, new_ckpt)
                                    new_seen     = set(new_history["item_idx"].tolist())
                                    recs_after   = recommend(
                                        new_user_idx, new_embs, new_n_users, new_n_items, new_seen, k=10
                                    )
                                else:
                                    liked_idx = []
                                    for inter in new_interactions:
                                        if new_item_enc and new_item_enc.is_known(inter["business_id"]):
                                            liked_idx.append(int(new_item_enc.transform([inter["business_id"]])[0]))
                                    recs_after = recommend_cold_start(liked_idx, new_embs, new_n_users, new_n_items, k=10)

                                recs_after_df = enrich_recommendations(recs_after, new_item_enc, display_biz_df)

                                st.markdown("---")
                                st.markdown("**Recommandations APRÈS :**")
                                display_cols = [c for c in ["Rang", "Nom", "Catégories", "Score"] if c in recs_after_df.columns]
                                st.dataframe(recs_after_df[display_cols].head(10), use_container_width=True, hide_index=True)

                                # Comparison chart
                                if "recs_before" in st.session_state and not recs_after_df.empty:
                                    before_names = set(st.session_state["recs_before"]["Nom"].tolist() if "Nom" in st.session_state["recs_before"].columns else [])
                                    after_names  = set(recs_after_df["Nom"].tolist() if "Nom" in recs_after_df.columns else [])
                                    new_recs  = after_names - before_names
                                    kept_recs = after_names & before_names

                                    st.markdown("---")
                                    c1, c2 = st.columns(2)
                                    c1.metric("Nouvelles recommandations", len(new_recs))
                                    c2.metric("Recommandations conservées", len(kept_recs))

                                    if new_recs:
                                        st.markdown("**🆕 Nouvelles dans le top-10 :**")
                                        for name in list(new_recs)[:5]:
                                            st.markdown(f"- {name}")

                            else:
                                st.warning("Graphe non reconstructible depuis le nouveau checkpoint.")
                        else:
                            st.warning("Checkpoint non trouvé après entraînement.")
                    else:
                        st.error(f"Erreur (code {proc.returncode}). Voir logs ci-dessus.")

                except FileNotFoundError:
                    st.error("Python introuvable. Vérifiez votre environnement.")
                except Exception as e:
                    st.error(f"Erreur inattendue : {e}")
                finally:
                    Path(tmp_csv).unlink(missing_ok=True)

        elif "recs_before" not in st.session_state:
            st.info(
                "👈 Configurez les nouvelles interactions à gauche, "
                "puis cliquez sur **Lancer l'apprentissage incrémental**."
            )
