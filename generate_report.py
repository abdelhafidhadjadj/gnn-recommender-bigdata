"""
Génère results_final/report.html — rapport complet avec figures embarquées.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import json, os, glob, sys, base64, io
sys.stdout.reconfigure(encoding='utf-8')

_HERE  = os.path.dirname(os.path.abspath(__file__))
BASE   = os.path.join(_HERE, "outputs")
OUT_H  = os.path.join(_HERE, "results_final", "report.html")
OUT_C  = os.path.join(_HERE, "charts")
os.makedirs(OUT_C, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_run(folder):
    path = os.path.join(BASE, folder, "metrics")
    if not os.path.exists(path):
        return {}
    files = [f for f in os.listdir(path) if f.endswith(".json")]
    if not files:
        return {}
    with open(os.path.join(path, files[0]), encoding="utf-8") as f:
        return json.load(f)

def fmt(v, decimals=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)

def rk(m, k, metric):
    try: return round(m["ranking"][str(k)][metric], 6)
    except: return None

def tv(m, key):
    try: return round(m[key], 4)
    except: return None

def tm(m, key):
    try: return round(m["timings"][key], 2)
    except: return None

def png_b64(path):
    """Encode un PNG en base64 pour l'embarquer dans le HTML."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def fig_b64(fig):
    """Encode une figure matplotlib en base64 sans la sauvegarder."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ─────────────────────────────────────────────────────────────────────────────
# Charger toutes les runs
# ─────────────────────────────────────────────────────────────────────────────

all_runs = {}
for f in glob.glob(os.path.join(BASE, "*", "metrics", "*.json")):
    folder = f.split(os.sep)[-3]
    try:
        with open(f, encoding="utf-8") as fp:
            all_runs[folder] = json.load(fp)
    except Exception:
        pass

MAIN = {"GAT": "gat_w4_full", "GraphSAGE": "sage_w4_full", "LightGCN": "lightgcn_w4_full"}
W1   = {"GAT": "gat_w1_full", "GraphSAGE": "sage_w1_full", "LightGCN": "lightgcn_w1_full"}
MODEL_CLS = {"GAT": "gat", "GraphSAGE": "sage", "LightGCN": "lgcn"}

# ─────────────────────────────────────────────────────────────────────────────
# Générer la figure pipeline Big Data
# ─────────────────────────────────────────────────────────────────────────────

def make_pipeline_fig():
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")
    fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")

    def box(x, y, w, h, color, text, sub=None, icon="", fontsize=10):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                              facecolor=color, edgecolor="white",
                              linewidth=1.5, alpha=0.92, zorder=3)
        ax.add_patch(rect)
        cy = y + h/2 + (0.15 if sub else 0)
        ax.text(x+w/2, cy, f"{icon} {text}" if icon else text,
                ha="center", va="center", fontsize=fontsize,
                color="white", fontweight="bold", zorder=4)
        if sub:
            ax.text(x+w/2, y+h/2-0.22, sub, ha="center", va="center",
                    fontsize=7.5, color="#cccccc", zorder=4)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                    arrowprops=dict(arrowstyle="-|>", color="#58a6ff",
                                   lw=2, mutation_scale=18), zorder=5)

    def label(x, y, text):
        ax.text(x, y, text, ha="center", va="center", fontsize=8,
                color="#8b949e", zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#161b22",
                          edgecolor="none", alpha=0.8))

    C1="#1f6feb"; C2="#388bfd"; C3="#f78166"; C4="#3fb950"; C5="#d29922"; C6="#8b949e"

    ax.text(8, 8.55, "Architecture Big Data — Pipeline GNN Recommender",
            ha="center", va="center", fontsize=15, color="white", fontweight="bold", zorder=6)

    # Ligne 1 : Données
    box(0.4, 6.8, 2.2, 0.95, C1, "Dataset CSV",   "Yelp Health & Medical\n350 MB / 188k reviews", "📁", 9)
    box(3.2, 6.8, 2.2, 0.95, C2, "HDFS NameNode", "hdfs://namenode:9000\nport 9870 (UI)",         "🗄️", 9)
    box(6.0, 6.8, 2.2, 0.95, C2, "HDFS DataNode", "Stockage distribué\nrep. factor = 1",          "💾", 9)
    arrow(2.6, 7.27, 3.2, 7.27); label(2.9, 7.5, "upload")
    arrow(5.4, 7.27, 6.0, 7.27); label(5.7, 7.5, "blocs 128MB")

    # Ligne 2 : Spark
    box(0.4, 5.3, 2.2, 0.95, C3, "Spark Master", "spark://master:7077\nport 8081 (UI)", "⚡", 9)
    for i, w in enumerate(["Worker 1","Worker 2","Worker 3","Worker 4"]):
        box(3.2 + i*2.2 + 0.1, 5.3, 1.85, 0.95, "#e16a2f", w, "2 cores / 1GB RAM", "", 8)
        arrow(2.6, 5.77, 3.35 + i*2.2, 5.77)
    arrow(7.1, 6.8, 7.1, 6.25); label(7.5, 6.52, "read\nParquet")

    # Ligne 3 : Preprocessing + Graph
    box(0.4, 3.7, 3.2, 0.95, C3, "Spark Preprocessing",
        "Filtrage rating≥3 | encodage | split 70/15/15", "🔧", 9)
    box(4.2, 3.7, 3.5, 0.95, C1, "Graphe Bipartite",
        "101k users + 11.7k items | 255k arêtes", "🕸️", 9)
    box(8.4, 3.7, 3.2, 0.95, C1, "Embeddings GNN",
        "PyTorch Geometric | dim=64", "🧠", 9)
    arrow(2.6, 5.3, 1.9, 4.65); label(1.5, 4.97, "données\nencodées")
    arrow(3.6, 4.17, 4.2, 4.17); label(3.9, 4.4, "edge_index")
    arrow(7.7, 4.17, 8.4, 4.17); label(8.05, 4.4, "forward()")

    # Ligne 4 : Modèles
    for mx, mc, mn, mndcg in [
        (0.4,  "#4C72B0", "GraphSAGE", "NDCG@10=0.0031"),
        (4.2,  "#DD8452", "GAT",       "NDCG@10=0.0086"),
        (8.0,  "#55A868", "LightGCN",  "NDCG@10=0.0050"),
    ]:
        box(mx, 2.1, 3.3, 0.95, mc, mn, mndcg, "🔮", 9)
        arrow(mx+3.3/2, 3.7, mx+3.3/2, 3.05)

    # Ligne 5 : Sortie
    box(0.4,  0.5, 3.2, 0.95, C4, "Checkpoint",         "best_model.pt\nembeddings sauvegardés", "💾", 9)
    box(4.2,  0.5, 3.5, 0.95, C4, "Interface Streamlit", "localhost:8501\nTop-K / Cold-start / Incrémental", "🖥️", 8)
    box(8.4,  0.5, 3.2, 0.95, C5, "ELK Monitoring",     "Kibana :5601 | Logstash\nFilebeat + Metricbeat", "📊", 9)
    box(12.2, 0.5, 3.4, 0.95, C6, "Utilisateur Final",  "Recommandations\nCold-start / Incrémental", "👤", 9)

    for mx, _, _, _ in [(0.4,"","",""), (4.2,"","",""), (8.0,"","","")]:
        arrow(mx+3.3/2, 2.1, 1.95, 1.45)
    arrow(3.6, 0.97, 4.2, 0.97); arrow(7.7, 0.97, 8.4, 0.97); arrow(11.6, 0.97, 12.2, 0.97)

    plt.tight_layout(pad=0.2)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Générer les charts à embarquer
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {"GAT": "#DD8452", "GraphSAGE": "#4C72B0", "LightGCN": "#55A868"}
KS = [5, 10, 20]

def chart_ndcg_bar():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.patch.set_facecolor("#f8f9ff")
    for ax, k in zip(axes, KS):
        vals  = [rk(all_runs.get(MAIN[m], {}), k, "NDCG") or 0 for m in MAIN]
        bars  = ax.bar(list(MAIN.keys()), vals,
                       color=[COLORS[m] for m in MAIN], width=0.5, edgecolor="white")
        ax.set_title(f"NDCG@{k}", fontweight="bold", fontsize=11)
        ax.set_ylim(0, max(vals)*1.3 if max(vals) > 0 else 0.01)
        ax.set_facecolor("#f0f2f5"); ax.grid(axis="y", alpha=0.4)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.00005,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return fig

def chart_precision_recall():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#f8f9ff")
    x = np.arange(len(KS)); w = 0.25
    for ax, metric, title in zip(axes, ["P", "R"], ["Precision@K", "Recall@K"]):
        for i, (mname, folder) in enumerate(MAIN.items()):
            vals = [rk(all_runs.get(folder, {}), k, metric) or 0 for k in KS]
            ax.bar(x + i*w, vals, w, label=mname, color=COLORS[mname], edgecolor="white")
        ax.set_xticks(x + w); ax.set_xticklabels([f"@{k}" for k in KS])
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_facecolor("#f0f2f5"); ax.grid(axis="y", alpha=0.4)
        ax.legend(fontsize=9)
    fig.tight_layout()
    return fig

def chart_speedup():
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#f8f9ff"); ax.set_facecolor("#f0f2f5")
    models = list(MAIN.keys()); x = np.arange(len(models)); w = 0.35
    t1s = [tm(all_runs.get(W1[m], {}), "t_train") or 0 for m in models]
    t4s = [tm(all_runs.get(MAIN[m], {}), "t_train") or 0 for m in models]
    ax.bar(x - w/2, t1s, w, label="1 worker (Standard)", color="#95aec7", edgecolor="white")
    ax.bar(x + w/2, t4s, w, label="4 workers (Big Data)", color="#4C72B0", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("Temps d'entraînement (s)"); ax.set_title("Speedup Big Data vs Standard", fontweight="bold")
    ax.grid(axis="y", alpha=0.4); ax.legend()
    for i, (t1, t4) in enumerate(zip(t1s, t4s)):
        if t1 > 0 and t4 > 0:
            sp = t1 / t4
            ax.text(i + w/2, t4 + max(t1s)*0.02, f"{sp:.1f}×",
                    ha="center", fontsize=10, color="#27ae60", fontweight="bold")
    fig.tight_layout()
    return fig

def chart_ndcg_by_size():
    sizes_lbl = ["1k", "5k", "10k", "50k", "100k", "full"]
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#f8f9ff"); ax.set_facecolor("#f0f2f5")
    for mname, mshort in [("GAT","gat"), ("GraphSAGE","sage"), ("LightGCN","lightgcn")]:
        vals = []
        for sz in sizes_lbl:
            best = 0
            for w in ["w1","w2","w3","w4"]:
                m = all_runs.get(f"{mshort}_{w}_{sz}", {})
                v = rk(m, 10, "NDCG") or 0
                best = max(best, v)
            vals.append(best if best > 0 else None)
        xs = [i for i, v in enumerate(vals) if v is not None]
        ys = [v for v in vals if v is not None]
        if xs:
            ax.plot(xs, ys, marker="o", label=mname,
                    color=COLORS[mname], linewidth=2, markersize=7)
    ax.set_xticks(range(len(sizes_lbl))); ax.set_xticklabels(sizes_lbl)
    ax.set_ylabel("NDCG@10"); ax.set_title("NDCG@10 par taille de dataset", fontweight="bold")
    ax.grid(alpha=0.4); ax.legend()
    fig.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Construire le HTML
# ─────────────────────────────────────────────────────────────────────────────

def img_tag(b64, alt="", width="75%"):
    if not b64:
        return f'<p style="color:#aaa;text-align:center">Figure non disponible</p>'
    return f'<div style="text-align:center"><img src="data:image/png;base64,{b64}" alt="{alt}" style="width:{width};border-radius:8px;margin-top:10px"></div>'

HTML = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>GNN Recommender — Rapport Métriques</title>
<style>
  * { box-sizing:border-box; margin:0; padding:0; }
  body { font-family:'Segoe UI',Arial,sans-serif; background:#f0f2f5; color:#222; padding:30px; }
  h1 { color:#1a1a2e; font-size:2rem; margin-bottom:6px; }
  h2 { color:#16213e; font-size:1.2rem; margin:28px 0 10px;
       border-left:4px solid #4C72B0; padding-left:10px; }
  .subtitle { color:#555; margin-bottom:28px; font-size:.95rem; }
  .card { background:white; border-radius:10px; padding:20px;
          box-shadow:0 2px 8px rgba(0,0,0,.08); margin-bottom:22px; }
  table { width:100%; border-collapse:collapse; font-size:.88rem; }
  th { background:#1a1a2e; color:white; padding:9px 12px; text-align:center; font-weight:600; }
  td { padding:8px 12px; text-align:center; border-bottom:1px solid #eee; }
  tr:nth-child(even) td { background:#f8f9ff; }
  tr:hover td { background:#e8eeff; }
  .best { background:#d4edda !important; font-weight:bold; color:#155724; }
  .badge { display:inline-block; padding:2px 8px; border-radius:12px; font-size:.8rem; font-weight:bold; }
  .badge-gat  { background:#FFF3E0; color:#DD8452; }
  .badge-sage { background:#E3F2FD; color:#4C72B0; }
  .badge-lgcn { background:#E8F5E9; color:#55A868; }
  .grid3 { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-bottom:22px; }
  .metric-box { background:white; border-radius:10px; padding:16px; text-align:center;
                box-shadow:0 2px 8px rgba(0,0,0,.08); }
  .metric-box .val { font-size:1.8rem; font-weight:bold; color:#1a1a2e; }
  .metric-box .lbl { color:#666; font-size:.82rem; margin-top:4px; }
  .speedup-high { color:#27ae60; font-weight:bold; }
  .speedup-low  { color:#e67e22; font-weight:bold; }
  footer { text-align:center; color:#999; font-size:.8rem; margin-top:40px; }
</style>
</head>
<body>
<h1>GNN Recommender — Rapport des Métriques</h1>
<p class="subtitle">Dataset : Yelp Health &amp; Medical | Full dataset | 4 workers Spark | rating_thresh=3</p>
"""

# KPIs
kpis = [
    ("127 569", "Interactions dans le graphe", "rating ≥ 3★"),
    ("101 005", "Utilisateurs",                "après filtrage"),
    ("11 719",  "Items (businesses)",          "dataset full"),
    ("5 000",   "Utilisateurs évalués",        "ensemble de test"),
    ("112 724", "Nœuds du graphe",             "users + items"),
    ("255 138", "Arêtes bidirectionnelles",    "graphe bipartite"),
]
HTML += '<div class="grid3">\n'
for val, lbl, sub in kpis:
    HTML += f'<div class="metric-box"><div class="val">{val}</div><div class="lbl">{lbl}<br><small style="color:#aaa">{sub}</small></div></div>\n'
HTML += "</div>\n"

# Table 1 — Performances complètes (P, R, NDCG uniquement)
HTML += '<div class="card"><h2>Performances sur le test set — w4 full dataset</h2><table>\n'
HTML += '<tr><th>Modèle</th><th>RMSE</th><th>MAE</th>'
HTML += '<th>P@5</th><th>R@5</th><th>NDCG@5</th>'
HTML += '<th>P@10</th><th>R@10</th><th>NDCG@10</th>'
HTML += '<th>P@20</th><th>R@20</th><th>NDCG@20</th></tr>\n'

best_ndcg10 = max((rk(all_runs.get(MAIN[m], {}), 10, "NDCG") or 0) for m in MAIN)
for mname, folder in MAIN.items():
    m   = all_runs.get(folder, {})
    cls = MODEL_CLS[mname]
    n10 = rk(m, 10, "NDCG")
    bc  = ' class="best"' if n10 and abs(n10 - best_ndcg10) < 1e-9 else ""
    HTML += (f'<tr><td><span class="badge badge-{cls}">{mname}</span></td>'
             f'<td>{fmt(tv(m,"rmse"))}</td><td>{fmt(tv(m,"mae"))}</td>'
             f'<td>{fmt(rk(m,5,"P"))}</td><td>{fmt(rk(m,5,"R"))}</td><td>{fmt(rk(m,5,"NDCG"))}</td>'
             f'<td{bc}>{fmt(rk(m,10,"P"))}</td><td{bc}>{fmt(rk(m,10,"R"))}</td><td{bc}>{fmt(rk(m,10,"NDCG"))}</td>'
             f'<td>{fmt(rk(m,20,"P"))}</td><td>{fmt(rk(m,20,"R"))}</td><td>{fmt(rk(m,20,"NDCG"))}</td></tr>\n')
HTML += '</table></div>\n'

# Table 2 — Accuracy & Global Precision (sans HR, MRR, F1)
HTML += '<div class="card"><h2>Métriques de classification</h2><table>\n'
HTML += '<tr><th>Modèle</th><th>Accuracy</th><th>Global Precision</th></tr>\n'
for mname, folder in MAIN.items():
    m   = all_runs.get(folder, {})
    cls = MODEL_CLS[mname]
    HTML += (f'<tr><td><span class="badge badge-{cls}">{mname}</span></td>'
             f'<td>{fmt(tv(m,"accuracy"))}</td><td>{fmt(tv(m,"global_precision"))}</td></tr>\n')
HTML += '</table></div>\n'

# Table 3 — Baselines
HTML += '<div class="card"><h2>Baselines comparatives</h2><table>\n'
HTML += '<tr><th>Baseline</th><th>NDCG@5</th><th>NDCG@10</th><th>NDCG@20</th><th>Remarque</th></tr>\n'
HTML += '<tr><td>Popularité</td><td>0.0082</td><td>0.0108</td><td>0.0141</td><td>Items les plus fréquents</td></tr>\n'
HTML += '<tr><td>Aléatoire</td><td>0.0005</td><td>0.0006</td><td>0.0006</td><td>Recommandation uniforme</td></tr>\n'
HTML += '</table></div>\n'

# Table 4 — Speedup
HTML += '<div class="card"><h2>Standard vs Big Data — Timings &amp; Speedup</h2><table>\n'
HTML += '<tr><th>Modèle</th><th>Mode</th><th>Workers</th><th>Chargement (s)</th><th>Entraînement (s)</th><th>Speedup</th><th>NDCG@10</th></tr>\n'
for mname in ["GraphSAGE", "GAT", "LightGCN"]:
    m1 = all_runs.get(W1[mname], {})
    m4 = all_runs.get(MAIN[mname], {})
    cls = MODEL_CLS[mname]
    tt1 = tm(m1, "t_train"); tt4 = tm(m4, "t_train")
    speedup = round(tt1/tt4, 2) if tt1 and tt4 else None
    sp_str  = f'<span class="speedup-high">{speedup}×</span>' if isinstance(speedup, float) and speedup > 1.5 else (f'{speedup}×' if speedup else '—')
    HTML += (f'<tr><td rowspan="2"><span class="badge badge-{cls}">{mname}</span></td>'
             f'<td>Standard</td><td>1</td><td>{fmt(tm(m1,"t_load"),2)}</td><td>{fmt(tt1,2)}</td><td>1.00×</td><td>{fmt(rk(m1,10,"NDCG"))}</td></tr>\n'
             f'<tr><td>Big Data</td><td>4</td><td>{fmt(tm(m4,"t_load"),2)}</td><td>{fmt(tt4,2)}</td><td>{sp_str}</td><td>{fmt(rk(m4,10,"NDCG"))}</td></tr>\n')
HTML += '</table></div>\n'

# Table 5 — NDCG@10 par taille
HTML += '<div class="card"><h2>NDCG@10 par taille de dataset</h2><table>\n'
HTML += '<tr><th>Modèle</th><th>1k</th><th>5k</th><th>10k</th><th>50k</th><th>100k</th><th>Full</th></tr>\n'
for model_short, mname in [("gat","GAT"), ("sage","GraphSAGE"), ("lightgcn","LightGCN")]:
    cls = MODEL_CLS[mname]
    HTML += f'<tr><td><span class="badge badge-{cls}">{mname}</span></td>'
    for sz in ["1k","5k","10k","50k","100k","full"]:
        best = max(
            (rk(all_runs.get(f"{model_short}_{w}_{sz}", {}), 10, "NDCG") or 0)
            for w in ["w1","w2","w3","w4"]
        )
        HTML += f'<td>{"—" if best == 0 else f"{best:.4f}"}</td>'
    HTML += '</tr>\n'
HTML += '</table></div>\n'

# ── Figures ──────────────────────────────────────────────────────────────────

# Chart 1 — NDCG bar
fig1 = chart_ndcg_bar()
b1   = fig_b64(fig1); plt.close(fig1)
HTML += f'<div class="card"><h2>NDCG@K par modèle</h2>{img_tag(b1,"NDCG@K")}</div>\n'

# Chart 2 — Precision / Recall
fig2 = chart_precision_recall()
b2   = fig_b64(fig2); plt.close(fig2)
HTML += f'<div class="card"><h2>Precision@K et Recall@K</h2>{img_tag(b2,"P@K R@K")}</div>\n'

# Chart 3 — Speedup
fig3 = chart_speedup()
b3   = fig_b64(fig3); plt.close(fig3)
HTML += f'<div class="card"><h2>Speedup Big Data vs Standard</h2>{img_tag(b3,"Speedup")}</div>\n'

# Chart 4 — NDCG par taille
fig4 = chart_ndcg_by_size()
b4   = fig_b64(fig4); plt.close(fig4)
HTML += f'<div class="card"><h2>NDCG@10 par taille de dataset</h2>{img_tag(b4,"NDCG size")}</div>\n'

# Chart 5 — Courbes BPR (w4_full)
bpr_paths = {
    "GAT":       os.path.join(BASE, "gat_w4_full",       "plots", "gat_training_curve.png"),
    "GraphSAGE": os.path.join(BASE, "sage_w4_full",      "plots", "sage_training_curve.png"),
    "LightGCN":  os.path.join(BASE, "lightgcn_w4_full",  "plots", "lightgcn_training_curve.png"),
}
bpr_b64s = {m: png_b64(p) for m, p in bpr_paths.items()}
if any(bpr_b64s.values()):
    HTML += '<div class="card"><h2>Courbes de perte BPR — Full dataset (4 workers)</h2>\n'
    HTML += '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:10px">\n'
    for mname, b in bpr_b64s.items():
        cls = MODEL_CLS[mname]
        HTML += f'<div style="text-align:center">'
        HTML += f'<p style="font-weight:bold;margin-bottom:6px"><span class="badge badge-{cls}">{mname}</span></p>'
        if b:
            HTML += f'<img src="data:image/png;base64,{b}" style="width:100%;border-radius:8px">'
        else:
            HTML += '<p style="color:#aaa">Non disponible</p>'
        HTML += '</div>\n'
    HTML += '</div></div>\n'

# Charts externes (generate_charts.py)
for i, (label, fname) in enumerate([
    ("Courbe d'entraînement", "chart1_training_curve.png"),
    ("Distribution des scores", "chart2_score_dist.png"),
    ("Comparaison des modèles", "chart3_model_comparison.png"),
    ("Cold-start performance", "chart4_coldstart.png"),
], start=6):
    path = os.path.join(OUT_C, fname)
    b = png_b64(path)
    if b:
        HTML += f'<div class="card"><h2>{label}</h2>{img_tag(b, label)}</div>\n'

# ── Tableau complet de toutes les expériences ────────────────────────────────
HTML += '<div class="card"><h2>Toutes les expériences — résultats complets</h2>\n'
HTML += ('<table><tr>'
         '<th>#</th><th>Run</th><th>Modèle</th><th>Workers</th><th>Taille</th>'
         '<th>RMSE</th><th>MAE</th><th>Accuracy</th><th>Global Precision</th>'
         '<th>P@5</th><th>R@5</th><th>NDCG@5</th>'
         '<th>P@10</th><th>R@10</th><th>NDCG@10</th>'
         '<th>P@20</th><th>R@20</th><th>NDCG@20</th>'
         '<th>t_train (s)</th><th>Loader</th>'
         '</tr>\n')

SIZE_ORDER  = {"1k":0,"5k":1,"10k":2,"50k":3,"100k":4,"full":5}
MODEL_ORDER = {"gat":0,"sage":1,"lightgcn":2}

def parse_run(folder):
    """Extraire (model, workers, size) depuis le nom du dossier."""
    import re
    m = re.match(r'^(gat|sage|lightgcn)_w(\d+)_(.+)$', folder)
    if not m:
        return None, None, None
    return m.group(1), int(m.group(2)), m.group(3)

# Collecter tous les runs avec leurs métriques
rows = []
for folder, data in all_runs.items():
    model, workers, size = parse_run(folder)
    if model is None:
        continue
    rows.append((model, workers, size, folder, data))

# Trier : modèle → taille → workers
rows.sort(key=lambda x: (MODEL_ORDER.get(x[0], 9), SIZE_ORDER.get(x[2], 9), x[1]))

MODEL_LABEL = {"gat": "GAT", "sage": "GraphSAGE", "lightgcn": "LightGCN"}
prev_model = None

for idx, (model, workers, size, folder, m) in enumerate(rows, 1):
    # Ligne de séparation entre modèles
    if model != prev_model:
        colspan = 19
        mlabel  = MODEL_LABEL.get(model, model)
        cls     = {"gat":"gat","sage":"sage","lightgcn":"lgcn"}[model]
        HTML   += (f'<tr><td colspan="{colspan}" style="background:#1a1a2e;color:white;'
                   f'font-weight:bold;text-align:left;padding:8px 12px">'
                   f'<span class="badge badge-{cls}">{mlabel}</span></td></tr>\n')
        prev_model = model

    loader = (m.get("timings") or {}).get("loader", "—")
    mcls   = {"gat":"gat","sage":"sage","lightgcn":"lgcn"}[model]
    HTML += (f'<tr>'
             f'<td>{idx}</td>'
             f'<td style="font-size:.8rem;color:#555">{folder}</td>'
             f'<td><span class="badge badge-{mcls}">{MODEL_LABEL[model]}</span></td>'
             f'<td>{workers}</td>'
             f'<td><b>{size}</b></td>'
             f'<td>{fmt(tv(m,"rmse"))}</td>'
             f'<td>{fmt(tv(m,"mae"))}</td>'
             f'<td>{fmt(tv(m,"accuracy"))}</td>'
             f'<td>{fmt(tv(m,"global_precision"))}</td>'
             f'<td>{fmt(rk(m,5,"P"))}</td><td>{fmt(rk(m,5,"R"))}</td><td>{fmt(rk(m,5,"NDCG"))}</td>'
             f'<td>{fmt(rk(m,10,"P"))}</td><td>{fmt(rk(m,10,"R"))}</td><td>{fmt(rk(m,10,"NDCG"))}</td>'
             f'<td>{fmt(rk(m,20,"P"))}</td><td>{fmt(rk(m,20,"R"))}</td><td>{fmt(rk(m,20,"NDCG"))}</td>'
             f'<td>{fmt(tm(m,"t_train"),1)}</td>'
             f'<td style="font-size:.8rem">{loader}</td>'
             f'</tr>\n')

HTML += '</table></div>\n'

HTML += '<footer>Généré automatiquement — GNN Recommender System | Yelp Health &amp; Medical Dataset</footer>\n'
HTML += '</body></html>'

with open(OUT_H, "w", encoding="utf-8") as f:
    f.write(HTML)
print(f"report.html OK -> {OUT_H}")
