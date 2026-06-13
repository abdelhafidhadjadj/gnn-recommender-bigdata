import json, os, glob

base = r"C:\Users\hafid\Desktop\gnn_recommender\results_final"
results = []

for f in glob.glob(base + r"\*\metrics\*.json"):
    with open(f) as fp:
        m = json.load(fp)
    folder = f.split(os.sep)[-3]
    ranking = m.get("ranking", {})
    if ranking:
        k = list(ranking.keys())[0]
        r = ranking[k]
        results.append({
            "run":  folder,
            "NDCG": round(r.get("NDCG", 0), 4),
            "P":    round(r.get("P", 0), 4),
            "R":    round(r.get("R", 0), 4),
            "HR":   round(r.get("HR", 0), 4),
            "K":    k,
        })

results.sort(key=lambda x: x["NDCG"], reverse=True)
for r in results:
    print(r["run"].ljust(35), "NDCG="+str(r["NDCG"]).ljust(7),
          "P="+str(r["P"]).ljust(7), "R="+str(r["R"]).ljust(7),
          "HR="+str(r["HR"]).ljust(7), "@K="+str(r["K"]))
