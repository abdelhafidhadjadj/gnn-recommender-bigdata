import json, sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')

rev = pd.read_csv(r"C:\Users\hafid\Desktop\gnn_recommender\data\raw\full\yelp_academic_dataset_review_healthandmedical.csv", low_memory=False)
rev = rev[rev["stars"] >= 3]

n_items = rev["business_id"].nunique()

# Split 70/15/15 (same random_state=1 as config)
rev = rev.sample(frac=1, random_state=1).reset_index(drop=True)
n = len(rev)
n_test = int(n * 0.15)
n_val  = int(n * 0.15)
test      = rev.iloc[:n_test]
train_val = rev.iloc[n_test:]

# Popularity top items from train
item_counts = train_val["business_id"].value_counts()
top_items   = item_counts.index.tolist()

# Test users
test_user_items = test.groupby("user_id")["business_id"].apply(set).to_dict()
eval_users = list(test_user_items.keys())[:5000]
all_items_list = list(rev["business_id"].unique())

rng = np.random.default_rng(42)

pop_p5,pop_p10,pop_p20 = [],[],[]
pop_r5,pop_r10,pop_r20 = [],[],[]
pop_ndcg10 = []
rnd_p10, rnd_r10, rnd_ndcg10 = [], [], []

for u in eval_users:
    relevant = test_user_items[u]
    n_rel = len(relevant)
    if n_rel == 0:
        continue

    for K, top_set, p_list, r_list in [
        (5,  set(top_items[:5]),  pop_p5,  pop_r5),
        (10, set(top_items[:10]), pop_p10, pop_r10),
        (20, set(top_items[:20]), pop_p20, pop_r20),
    ]:
        hits = len(relevant & top_set)
        p_list.append(hits / K)
        r_list.append(hits / n_rel)

    idcg = sum(1/np.log2(i+2) for i in range(min(n_rel,10)))
    dcg  = sum(1/np.log2(i+2) for i,item in enumerate(top_items[:10]) if item in relevant)
    pop_ndcg10.append(dcg/idcg if idcg > 0 else 0)

    rand_items = rng.choice(all_items_list, size=10, replace=False)
    hits_r = sum(1 for it in rand_items if it in relevant)
    rnd_p10.append(hits_r / 10)
    rnd_r10.append(hits_r / n_rel)
    dcg_r = sum(1/np.log2(i+2) for i,it in enumerate(rand_items) if it in relevant)
    rnd_ndcg10.append(dcg_r/idcg if idcg > 0 else 0)

def avg(lst): return round(float(np.mean(lst)), 6) if lst else 0.0

print("POPULARITE:")
print(f"  P@5={avg(pop_p5)}  P@10={avg(pop_p10)}  P@20={avg(pop_p20)}")
print(f"  R@5={avg(pop_r5)}  R@10={avg(pop_r10)}  R@20={avg(pop_r20)}")
print(f"  NDCG@10={avg(pop_ndcg10)}")
print()
print("ALEATOIRE:")
print(f"  P@10={avg(rnd_p10)}  R@10={avg(rnd_r10)}  NDCG@10={avg(rnd_ndcg10)}")
