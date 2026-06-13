import json, os, sys
sys.stdout.reconfigure(encoding='utf-8')

base = r"C:\Users\hafid\Desktop\gnn_recommender\outputs"

def load(folder):
    path = os.path.join(base, folder, "metrics")
    if not os.path.exists(path):
        return None
    files = [f for f in os.listdir(path) if f.endswith(".json")]
    if not files:
        return None
    with open(os.path.join(path, files[0])) as f:
        return json.load(f)

# ── Modèles principaux : w4_full ─────────────────────────────
gat   = load("gat_w4_full")
sage  = load("sage_w4_full")
lgcn  = load("lightgcn_w4_full")

# ── Standard w1 pour speedup ─────────────────────────────────
gat_w1   = load("gat_w1_full")
sage_w1  = load("sage_w1_full")
lgcn_w1  = load("lightgcn_w1_full")

def r(m, k, metric):
    try: return round(m["ranking"][str(k)][metric], 4)
    except: return "N/A"

def v(m, key):
    try: return round(m[key], 4)
    except: return "N/A"

def t(m, key):
    try: return round(m["timings"][key], 2)
    except: return "N/A"

print("TABLE 1")
for name, m in [("GraphSAGE", sage), ("GAT", gat), ("LightGCN", lgcn)]:
    print(f"{name}: RMSE={v(m,'rmse')} MAE={v(m,'mae')} "
          f"P5={r(m,5,'P')} R5={r(m,5,'R')} N5={r(m,5,'NDCG')} "
          f"P10={r(m,10,'P')} R10={r(m,10,'R')} N10={r(m,10,'NDCG')} "
          f"P20={r(m,20,'P')} R20={r(m,20,'R')} N20={r(m,20,'NDCG')}")

print("\nTABLE 2")
for name, m in [("GraphSAGE", sage), ("GAT", gat), ("LightGCN", lgcn)]:
    print(f"{name}: Accuracy={v(m,'accuracy')} GlobalPrecision={v(m,'global_precision')}")

print("\nTABLE 4 — timings")
for name, mw1, mw4 in [("GraphSAGE",sage_w1,sage),("GAT",gat_w1,gat),("LightGCN",lgcn_w1,lgcn)]:
    tl1=t(mw1,"t_load"); tt1=t(mw1,"t_train"); n1=r(mw1,10,"NDCG")
    tl4=t(mw4,"t_load"); tt4=t(mw4,"t_train"); n4=r(mw4,10,"NDCG")
    speedup = round(float(tt1)/float(tt4), 2) if tt1 != "N/A" and tt4 != "N/A" else "N/A"
    print(f"{name} w1: t_load={tl1}s t_train={tt1}s NDCG@10={n1}")
    print(f"{name} w4: t_load={tl4}s t_train={tt4}s NDCG@10={n4} speedup={speedup}x")
