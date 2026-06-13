import json, os, glob, sys
sys.stdout.reconfigure(encoding='utf-8')

base = r"C:\Users\hafid\Desktop\gnn_recommender\results_final"
results = []

for f in glob.glob(base + r"\*\metrics\*.json"):
    with open(f) as fp:
        m = json.load(fp)
    folder = f.split(os.sep)[-3]
    parts = folder.split("_")
    if len(parts) < 3:
        continue
    model   = parts[0]
    workers = parts[1]
    size    = parts[2]

    ranking = m.get("ranking", {})
    if ranking:
        k = list(ranking.keys())[0]
        r = ranking[k]
        results.append({
            "model":   model,
            "workers": workers,
            "size":    size,
            "NDCG":    round(r.get("NDCG", 0), 4),
            "P":       round(r.get("P", 0), 4),
            "R":       round(r.get("R", 0), 4),
            "HR":      round(r.get("HR", 0), 4),
        })

best = {}
for r in results:
    key = (r["model"], r["size"])
    if key not in best or r["NDCG"] > best[key]["NDCG"]:
        best[key] = r

def level(ndcg):
    if ndcg == 0:        return "NUL"
    elif ndcg < 0.003:   return "TRES FAIBLE"
    elif ndcg < 0.006:   return "FAIBLE"
    elif ndcg < 0.008:   return "LIMITE"
    elif ndcg < 0.015:   return "ACCEPTABLE"
    else:                return "BON"

for model in ["gat", "sage", "lightgcn"]:
    print(f"\n{'='*70}")
    print(f"  {model.upper()}")
    print(f"{'='*70}")
    print(f"  {'Dataset':<10} {'NDCG@5':<10} {'P@5':<10} {'R@5':<10} {'HR@5':<10} Niveau")
    print(f"  {'-'*65}")
    for size in ["1k","5k","10k","50k","100k","full"]:
        key = (model, size)
        if key in best:
            r = best[key]
            print(f"  {size:<10} {r['NDCG']:<10} {r['P']:<10} {r['R']:<10} {r['HR']:<10} {level(r['NDCG'])}")
        else:
            print(f"  {size:<10} {'---':<10} {'---':<10} {'---':<10} {'---':<10} (manquant)")
