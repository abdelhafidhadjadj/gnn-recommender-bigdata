"""
Génération des charts pour le mémoire.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = r"C:\Users\hafid\Desktop\gnn_recommender\charts"
os.makedirs(OUT, exist_ok=True)

C_SAGE   = "#4C72B0"
C_GAT    = "#DD8452"
C_LGCN   = "#55A868"
C_DARK   = "#2d2d2d"
FONTSIZE = 13

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        FONTSIZE,
    "axes.titlesize":   15,
    "axes.titleweight": "bold",
    "axes.labelsize":   FONTSIZE,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "figure.dpi":       150,
    "savefig.dpi":      200,
    "savefig.bbox":     "tight",
})

models = ["GraphSAGE", "GAT", "LightGCN"]
colors = [C_SAGE, C_GAT, C_LGCN]

# ─────────────────────────────────────────────────────────────────────────────
# CHART 1 — Speedup
# ─────────────────────────────────────────────────────────────────────────────
speedups = [1.55, 7.95, 1.61]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(models, speedups, color=colors, width=0.5, zorder=3,
              edgecolor="white", linewidth=1.2)
ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5,
           label="Référence (mode standard = 1×)", zorder=2)
for bar, val in zip(bars, speedups):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.12,
            f"{val:.2f}×", ha="center", va="bottom",
            fontweight="bold", fontsize=13, color=C_DARK)
ax.set_title("")
ax.set_ylabel("Speedup (×)")
ax.set_ylim(0, 10)
ax.legend(fontsize=11)
ax.set_yticks([0, 1, 2, 4, 6, 8, 10])
plt.tight_layout()
plt.savefig(os.path.join(OUT, "chart1_speedup.png"))
plt.close()
print("chart1_speedup.png  OK")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 2 — Temps d'entraînement
# ─────────────────────────────────────────────────────────────────────────────
t_standard = [257.78, 264.43, 250.01]
t_bigdata  = [166.54,  33.25, 155.72]
x     = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5.5))
b1 = ax.bar(x - width/2, t_standard, width, label="Standard (1 worker)",
            color=["#4C72B0aa","#DD8452aa","#55A868aa"],
            edgecolor="white", linewidth=1.2, zorder=3)
b2 = ax.bar(x + width/2, t_bigdata, width, label="Big Data (4 workers Spark)",
            color=colors, edgecolor="white", linewidth=1.2, zorder=3)
for bar in b1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 3,
            f"{h:.0f}s", ha="center", va="bottom", fontsize=11, color=C_DARK)
for bar in b2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 3,
            f"{h:.0f}s", ha="center", va="bottom", fontsize=11,
            color=C_DARK, fontweight="bold")
ax.set_title("")
ax.set_ylabel("Temps d'entraînement (secondes)")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 320)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "chart2_training_time.png"))
plt.close()
print("chart2_training_time.png  OK")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 3 — Accuracy & Global Precision
# ─────────────────────────────────────────────────────────────────────────────
accuracy         = [0.5658, 0.5914, 0.5239]
global_precision = [0.6843, 0.7545, 0.6851]
x     = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5.5))
b1 = ax.bar(x - width/2, accuracy, width, label="Accuracy",
            color=["#4C72B0aa","#DD8452aa","#55A868aa"],
            edgecolor="white", linewidth=1.2, zorder=3)
b2 = ax.bar(x + width/2, global_precision, width, label="Global Precision",
            color=colors, edgecolor="white", linewidth=1.2, zorder=3)
for bar in b1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
            f"{h:.4f}", ha="center", va="bottom", fontsize=10, color=C_DARK)
for bar in b2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
            f"{h:.4f}", ha="center", va="bottom", fontsize=10,
            color=C_DARK, fontweight="bold")
ax.set_title("")
ax.set_ylabel("Score")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 0.90)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "chart3_accuracy_precision.png"))
plt.close()
print("chart3_accuracy_precision.png  OK")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 4 — Métriques ranking @K=10
# ─────────────────────────────────────────────────────────────────────────────
p10    = [0.0008, 0.0020, 0.0012]
r10    = [0.0068, 0.0154, 0.0094]
ndcg10 = [0.0031, 0.0086, 0.0050]
x     = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
b1 = ax.bar(x - width, p10,    width, label="Precision@10",
            color="#4878CF", edgecolor="white", linewidth=1.2, zorder=3)
b2 = ax.bar(x,         r10,    width, label="Recall@10",
            color="#6ACC65", edgecolor="white", linewidth=1.2, zorder=3)
b3 = ax.bar(x + width, ndcg10, width, label="NDCG@10",
            color="#D65F5F", edgecolor="white", linewidth=1.2, zorder=3)
for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.00015,
                f"{h:.4f}", ha="center", va="bottom", fontsize=9,
                color=C_DARK, rotation=45)
ax.set_title("")
ax.set_ylabel("Score")
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=13)
ax.set_ylim(0, 0.022)
ax.legend(fontsize=11)
for i, (model, color) in enumerate(zip(models, colors)):
    ax.get_xticklabels()[i].set_color(color)
    ax.get_xticklabels()[i].set_fontweight("bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "chart4_ranking_metrics.png"))
plt.close()
print("chart4_ranking_metrics.png  OK")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 5 — CSV vs Parquet+Snappy (taille fichiers)
# ─────────────────────────────────────────────────────────────────────────────
fichiers  = ["business", "review", "user", "Total"]
csv_mb    = [5.16,  136.68, 208.74, 350.58]
parq_mb   = [1.52,   82.87, 202.93, 287.33]
gains_pct = [-70.5,  -39.4,   -2.8,  -18.0]

x     = np.arange(len(fichiers))
width = 0.35

fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

C_CSV  = "#E07B54"
C_PARQ = "#5B9BD5"
C_GAIN = "#2D7D46"

b1 = ax1.bar(x - width/2, csv_mb,  width, label="CSV",
             color=C_CSV,  edgecolor="white", linewidth=1.2, zorder=3, alpha=0.9)
b2 = ax1.bar(x + width/2, parq_mb, width, label="Parquet+Snappy",
             color=C_PARQ, edgecolor="white", linewidth=1.2, zorder=3, alpha=0.9)

# Annotations tailles
for bar in b1:
    h = bar.get_height()
    if h > 5:
        ax1.text(bar.get_x() + bar.get_width()/2, h + 2,
                 f"{h:.1f}", ha="center", va="bottom", fontsize=9, color=C_DARK)
for bar in b2:
    h = bar.get_height()
    if h > 5:
        ax1.text(bar.get_x() + bar.get_width()/2, h + 2,
                 f"{h:.1f}", ha="center", va="bottom", fontsize=9,
                 color=C_DARK, fontweight="bold")

# Ligne gain % sur axe secondaire
ax2.plot(x, gains_pct, color=C_GAIN, marker="o", markersize=9,
         linewidth=2.5, label="Gain (%)", zorder=4)
for i, (xi, g) in enumerate(zip(x, gains_pct)):
    ax2.annotate(f"{g:.1f}%", (xi, g),
                 textcoords="offset points", xytext=(0, 10),
                 ha="center", fontsize=10, color=C_GAIN, fontweight="bold")

ax1.set_title("")
ax1.set_ylabel("Taille (MB)", color=C_DARK)
ax1.set_xlabel("Fichier")
ax1.set_xticks(x)
ax1.set_xticklabels(fichiers, fontsize=13)
ax1.set_ylim(0, 420)
ax1.tick_params(axis="y", labelcolor=C_DARK)

ax2.set_ylabel("Gain de compression (%)", color=C_GAIN)
ax2.tick_params(axis="y", labelcolor=C_GAIN)
ax2.set_ylim(-90, 10)
ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

# Légendes combinées
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc="upper left")

# Highlight colonne Total
ax1.axvspan(2.5, 3.5, alpha=0.06, color="gray", zorder=0)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "chart5_csv_vs_parquet.png"))
plt.close()
print("chart5_csv_vs_parquet.png  OK")

print(f"\nTous les charts generes dans : {OUT}")
