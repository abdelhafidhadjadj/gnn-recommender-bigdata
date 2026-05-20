"""
Generate a new-batch CSV that simulates fresh interactions arriving after
the model was trained on data/medium.

Output: data/medium_new_batch/yelp_academic_dataset_review_healthandmedical.csv

Contains:
  - 200 brand-new users  (not in the medium dataset)
  - 50  brand-new businesses (not in the medium dataset)
  - ~5,000 interactions mixing old+new users and old+new businesses
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import uuid
import numpy as np
import pandas as pd

RNG = np.random.default_rng(99)   # different seed -> different data

MEDIUM_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "medium")
OUT_DIR     = os.path.join(os.path.dirname(__file__), "..", "data", "medium_new_batch")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load existing ids ─────────────────────────────────────────────────────────
existing_users = pd.read_csv(
    os.path.join(MEDIUM_DIR, "yelp_academic_dataset_user_healthandmedical.csv")
)["user_id"].tolist()

existing_biz = pd.read_csv(
    os.path.join(MEDIUM_DIR, "yelp_academic_dataset_business_healthandmedical.csv")
)["business_id"].tolist()

print(f"Existing users: {len(existing_users):,}  |  Existing businesses: {len(existing_biz):,}")

# ── New entities ──────────────────────────────────────────────────────────────
N_NEW_USERS = 200
N_NEW_BIZ   = 50

new_user_ids = [uuid.uuid4().hex[:22] for _ in range(N_NEW_USERS)]
new_biz_ids  = [uuid.uuid4().hex[:22] for _ in range(N_NEW_BIZ)]

all_users = existing_users + new_user_ids   # 2,200 total
all_biz   = existing_biz   + new_biz_ids    # 550   total

# ── Generate interactions ─────────────────────────────────────────────────────
# Sample ~5,000 (user, business) pairs — prefer new entities to stress the
# incremental encoder extension path.
TARGET = 5_000

rows = []
seen = set()

# New users × any business (weighted toward new businesses)
for u in new_user_ids:
    n_rev = int(RNG.integers(5, 30))
    weights = np.ones(len(all_biz))
    weights[-N_NEW_BIZ:] *= 3           # 3× more likely to review new businesses
    weights /= weights.sum()
    chosen = RNG.choice(len(all_biz), size=min(n_rev, len(all_biz)), replace=False, p=weights)
    for b_idx in chosen:
        key = (u, all_biz[b_idx])
        if key not in seen:
            seen.add(key)
            rows.append((u, all_biz[b_idx], int(RNG.integers(1, 6))))

# Some existing users also leave new reviews (simulates ongoing activity)
existing_sample = RNG.choice(existing_users, size=300, replace=False)
for u in existing_sample:
    n_rev = int(RNG.integers(1, 10))
    chosen = RNG.choice(len(all_biz), size=min(n_rev, len(all_biz)), replace=False)
    for b_idx in chosen:
        key = (u, all_biz[b_idx])
        if key not in seen:
            seen.add(key)
            rows.append((u, all_biz[b_idx], int(RNG.integers(1, 6))))

review_df = pd.DataFrame(rows, columns=["user_id", "business_id", "stars"])
review_df["review_id"] = [uuid.uuid4().hex[:22] for _ in range(len(review_df))]
review_df["date"]      = "2024-01-15"
review_df["text"]      = "New batch review."
review_df["useful"]    = 0
review_df["funny"]     = 0
review_df["cool"]      = 0
review_df = review_df[["review_id","user_id","business_id","stars","date","text","useful","funny","cool"]]
review_df = review_df.sample(frac=1, random_state=42).reset_index(drop=True)

out_path = os.path.join(OUT_DIR, "yelp_academic_dataset_review_healthandmedical.csv")
review_df.to_csv(out_path, index=False)

n_new_u_used = len(set(review_df["user_id"]) & set(new_user_ids))
n_new_b_used = len(set(review_df["business_id"]) & set(new_biz_ids))

print(f"\nNew-batch CSV written: {out_path}")
print(f"  Total interactions : {len(review_df):,}")
print(f"  Unique users       : {review_df['user_id'].nunique():,}  ({n_new_u_used} brand-new)")
print(f"  Unique businesses  : {review_df['business_id'].nunique():,}  ({n_new_b_used} brand-new)")
print(f"  Rating distribution: {dict(review_df['stars'].value_counts().sort_index())}")
