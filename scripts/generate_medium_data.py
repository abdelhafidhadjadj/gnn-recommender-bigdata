"""
Generate a medium-sized synthetic Yelp Health & Medical dataset.

Output: data/medium/
  - yelp_academic_dataset_business_healthandmedical.csv   (500 businesses)
  - yelp_academic_dataset_user_healthandmedical.csv       (2,000 users)
  - yelp_academic_dataset_review_healthandmedical.csv     (~40,000 reviews)

Density: ~40 reviews/user on average → much less sparse than the tiny test set.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)

N_USERS      = 2_000
N_BUSINESSES = 500
TARGET_REVIEWS = 40_000
OUT_DIR      = os.path.join(os.path.dirname(__file__), "..", "data", "medium")

os.makedirs(OUT_DIR, exist_ok=True)

# ── helpers ───────────────────────────────────────────────────────────────────

def uid() -> str:
    return uuid.uuid4().hex[:22]

FIRST_NAMES = [
    "Alice","Bob","Carol","David","Eve","Frank","Grace","Henry",
    "Irene","Jack","Karen","Leo","Mia","Noah","Olivia","Paul",
    "Quinn","Rachel","Sam","Tina","Uma","Victor","Wendy","Xander",
    "Yvonne","Zach","Amy","Brian","Claire","Derek",
]
LAST_NAMES = [
    "Smith","Jones","Williams","Brown","Davis","Miller","Wilson",
    "Moore","Taylor","Anderson","Thomas","Jackson","White","Harris",
    "Martin","Thompson","Garcia","Martinez","Robinson","Clark",
]

def rand_name():
    return f"{RNG.choice(FIRST_NAMES)} {RNG.choice(LAST_NAMES)}"

CITIES = ["Las Vegas","Phoenix","Toronto","Charlotte","Pittsburgh",
          "Montréal","Calgary","Cleveland","Madison","Tucson"]
STATES = ["NV","AZ","ON","NC","PA","QC","AB","OH","WI","AZ"]

SPECIALTIES = [
    "Doctors, Internal Medicine, Health & Medical",
    "Dentists, General Dentistry, Health & Medical",
    "Chiropractors, Health & Medical",
    "Optometrists, Health & Medical",
    "Pediatricians, Health & Medical",
    "Dermatologists, Health & Medical",
    "Obstetricians & Gynecologists, Health & Medical",
    "Orthopedists, Health & Medical",
    "Physical Therapy, Health & Medical",
    "Mental Health, Health & Medical",
    "Cardiologists, Health & Medical",
    "Urgent Care, Health & Medical",
    "Family Practice, Health & Medical",
    "Nutritionists, Health & Medical",
    "Acupuncture, Health & Medical",
]

REVIEW_TEXTS = [
    "Great experience, highly recommend!",
    "Staff was very professional and kind.",
    "Long wait times but excellent care.",
    "Could not be happier with the service.",
    "Average experience, nothing special.",
    "Very clean and modern facilities.",
    "Doctor explained everything clearly.",
    "Quick appointment and thorough checkup.",
    "Will definitely come back again.",
    "Friendly staff and convenient location.",
    "Disappointing experience overall.",
    "The best specialist I have ever visited.",
    "Pricing was fair and service was good.",
    "Hard to get an appointment but worth it.",
    "Excellent follow-up care provided.",
]

# ── 1. Businesses ─────────────────────────────────────────────────────────────
print(f"Generating {N_BUSINESSES} businesses ...")
biz_ids = [uid() for _ in range(N_BUSINESSES)]
biz_city_idx = RNG.integers(0, len(CITIES), N_BUSINESSES)
biz_df = pd.DataFrame({
    "business_id":  biz_ids,
    "name":         [f"Health Clinic {i+1}" for i in range(N_BUSINESSES)],
    "address":      [f"{RNG.integers(100,9999)} Main St" for _ in range(N_BUSINESSES)],
    "city":         [CITIES[i] for i in biz_city_idx],
    "state":        [STATES[i] for i in biz_city_idx],
    "postal_code":  RNG.integers(10000, 99999, N_BUSINESSES),
    "stars":        np.clip(RNG.normal(3.8, 0.8, N_BUSINESSES), 1, 5).round(1),
    "review_count": RNG.integers(10, 400, N_BUSINESSES),
    "is_open":      RNG.choice([0, 1], N_BUSINESSES, p=[0.1, 0.9]),
    "categories":   [RNG.choice(SPECIALTIES) for _ in range(N_BUSINESSES)],
})
biz_df.to_csv(os.path.join(OUT_DIR, "yelp_academic_dataset_business_healthandmedical.csv"), index=False)
print(f"  -> {len(biz_df)} businesses written.")

# ── 2. Users ──────────────────────────────────────────────────────────────────
print(f"Generating {N_USERS} users ...")
user_ids = [uid() for _ in range(N_USERS)]

# Users with more reviews tend to be more active -> realistic power-law
review_counts = np.clip(RNG.exponential(30, N_USERS).astype(int), 3, 500)
avg_stars_u   = np.clip(RNG.normal(3.9, 0.7, N_USERS), 1.0, 5.0).round(2)

base_date = datetime(2010, 1, 1)
yelping_since = [
    (base_date + timedelta(days=int(RNG.integers(0, 365 * 12)))).strftime("%Y-%m-%d")
    for _ in range(N_USERS)
]

user_df = pd.DataFrame({
    "user_id":       user_ids,
    "name":          [rand_name() for _ in range(N_USERS)],
    "review_count":  review_counts,
    "yelping_since": yelping_since,
    "average_stars": avg_stars_u,
    "fans":          RNG.integers(0, 200, N_USERS),
    "useful":        RNG.integers(0, 500, N_USERS),
    "funny":         RNG.integers(0, 200, N_USERS),
    "cool":          RNG.integers(0, 300, N_USERS),
})
user_df.to_csv(os.path.join(OUT_DIR, "yelp_academic_dataset_user_healthandmedical.csv"), index=False)
print(f"  -> {len(user_df)} users written.")

# ── 3. Reviews ────────────────────────────────────────────────────────────────
# Each user reviews a Poisson-sampled subset of businesses.
# Users have a latent "taste" vector (specialty preference) to create
# meaningful signal (not purely random ratings).
print(f"Generating ~{TARGET_REVIEWS:,} reviews ...")

N_SPEC = len(SPECIALTIES)
# Each user has a random preference distribution over specialties
user_pref = RNG.dirichlet(np.ones(N_SPEC) * 0.5, size=N_USERS)   # (N_USERS, N_SPEC)

# Each business belongs to one specialty (index)
biz_spec_idx = np.array([SPECIALTIES.index(c) for c in biz_df["categories"]])
biz_ids_arr  = np.array(biz_ids)

rows = []
base_date = datetime(2015, 1, 1)

reviews_per_user = np.clip(
    RNG.poisson(TARGET_REVIEWS / N_USERS, N_USERS), 1, 80
).astype(int)

for u_idx, u_id in enumerate(user_ids):
    n_rev = reviews_per_user[u_idx]
    # Sample businesses weighted by user's specialty preference
    biz_probs = user_pref[u_idx][biz_spec_idx]
    biz_probs = biz_probs / biz_probs.sum()
    chosen = RNG.choice(N_BUSINESSES, size=min(n_rev, N_BUSINESSES), replace=False, p=biz_probs)

    for b_idx in chosen:
        b_id = biz_ids[b_idx]
        # Rating influenced by user avg_stars + business stars, with noise
        base = 0.5 * avg_stars_u[u_idx] + 0.5 * float(biz_df.iloc[b_idx]["stars"])
        rating = int(np.clip(round(base + RNG.normal(0, 0.8)), 1, 5))

        rev_date = base_date + timedelta(days=int(RNG.integers(0, 365 * 8)))
        rows.append({
            "review_id":   uid(),
            "user_id":     u_id,
            "business_id": b_id,
            "stars":       rating,
            "date":        rev_date.strftime("%Y-%m-%d"),
            "text":        RNG.choice(REVIEW_TEXTS),
            "useful":      int(RNG.integers(0, 20)),
            "funny":       int(RNG.integers(0, 10)),
            "cool":        int(RNG.integers(0, 15)),
        })

review_df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
review_df.to_csv(
    os.path.join(OUT_DIR, "yelp_academic_dataset_review_healthandmedical.csv"), index=False
)
print(f"  -> {len(review_df):,} reviews written.")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Dataset summary ===")
print(f"  Users:      {len(user_df):,}")
print(f"  Businesses: {len(biz_df):,}")
print(f"  Reviews:    {len(review_df):,}")
print(f"  Avg reviews/user:     {len(review_df)/len(user_df):.1f}")
print(f"  Avg reviews/business: {len(review_df)/len(biz_df):.1f}")
print(f"  Rating distribution:")
for s in range(1, 6):
    cnt = (review_df["stars"] == s).sum()
    print(f"    {s} stars: {cnt:,}  ({100*cnt/len(review_df):.1f}%)")
print(f"\nOutput -> {os.path.abspath(OUT_DIR)}")
