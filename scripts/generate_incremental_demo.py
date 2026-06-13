"""
Génère les 3 fichiers CSV pour le dataset incrémental de démo :
  - yelp_academic_dataset_user_healthandmedical.csv     (existants + 50 nouveaux)
  - yelp_academic_dataset_business_healthandmedical.csv (existants + 10 nouveaux)
  - yelp_academic_dataset_review_healthandmedical.csv   (~1000 nouvelles interactions)

Output : data/raw/incremental_demo/
Usage  : python scripts/generate_incremental_demo.py
"""
import pandas as pd
import numpy as np
import uuid
import os
import json
from datetime import datetime, timedelta

RNG      = np.random.default_rng(2024)
FULL_DIR = "data/raw/full"
OUT_DIR  = "data/raw/incremental_demo"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Charger les données existantes ────────────────────────────────────────────
biz_df  = pd.read_csv(f"{FULL_DIR}/yelp_academic_dataset_business_healthandmedical.csv")
user_df = pd.read_csv(f"{FULL_DIR}/yelp_academic_dataset_user_healthandmedical.csv")

existing_biz_ids  = biz_df["business_id"].tolist()
existing_user_ids = user_df["user_id"].tolist()

print(f"Businesses existants : {len(existing_biz_ids):,}")
print(f"Users existants      : {len(existing_user_ids):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. 10 nouveaux businesses médicaux
# ─────────────────────────────────────────────────────────────────────────────
MEDICAL_NAMES = [
    ("HealthPlus Medical Center",     "Family Practice, Internal Medicine, Health & Medical"),
    ("CareFirst Dental Clinic",       "General Dentistry, Cosmetic Dentistry, Health & Medical"),
    ("MedExpress Urgent Care",        "Urgent Care, Emergency Medicine, Health & Medical"),
    ("WellBeing Pharmacy",            "Pharmacy, Vitamins & Supplements, Health & Medical"),
    ("PeakHealth Physical Therapy",   "Physical Therapy, Sports Medicine, Health & Medical"),
    ("MindCare Psychology",           "Counseling & Mental Health, Psychologists, Health & Medical"),
    ("OptiVision Eye Center",         "Optometrists, Ophthalmologists, Health & Medical"),
    ("HeartCare Cardiology",          "Cardiologists, Internists, Health & Medical"),
    ("GreenLeaf Naturopathy",         "Naturopathic/Holistic, Traditional Chinese Medicine, Health & Medical"),
    ("SkinHealth Dermatology",        "Dermatologists, Medical Spas, Health & Medical"),
]
CITIES = [
    ("Las Vegas", "NV", "89101"),
    ("Phoenix",   "AZ", "85001"),
    ("Charlotte", "NC", "28201"),
    ("Pittsburgh","PA", "15201"),
    ("Toronto",   "ON", "M5H 2N2"),
]

new_biz_rows = []
for i, (name, cats) in enumerate(MEDICAL_NAMES):
    city, state, postal = CITIES[i % len(CITIES)]
    new_biz_rows.append({
        "business_id":  uuid.uuid4().hex[:22],
        "name":         name,
        "address":      f"{int(RNG.integers(100, 9999))} Medical Blvd, Ste {int(RNG.integers(1, 20))}",
        "city":         city,
        "state":        state,
        "postal_code":  postal,
        "latitude":     round(float(RNG.uniform(33.0, 40.0)), 6),
        "longitude":    round(float(RNG.uniform(-115.0, -80.0)), 6),
        "stars":        float(RNG.choice([3.5, 4.0, 4.5, 5.0])),
        "review_count": int(RNG.integers(5, 80)),
        "is_open":      1,
        "attributes":   json.dumps({"ByAppointmentOnly": "True", "AcceptsInsurance": "True"}),
        "categories":   cats,
        "hours":        json.dumps({
            "Monday":    "8:0-18:0", "Tuesday":   "8:0-18:0",
            "Wednesday": "8:0-18:0", "Thursday":  "8:0-18:0",
            "Friday":    "8:0-17:0",
        }),
    })

new_biz_df = pd.DataFrame(new_biz_rows)
all_biz_df = pd.concat([biz_df, new_biz_df], ignore_index=True)
out_biz = f"{OUT_DIR}/yelp_academic_dataset_business_healthandmedical.csv"
all_biz_df.to_csv(out_biz, index=False)
print(f"\nBusiness : {len(all_biz_df):,} total ({len(new_biz_rows)} nouveaux) -> {out_biz}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. 50 nouveaux utilisateurs
# ─────────────────────────────────────────────────────────────────────────────
FIRST_NAMES = [
    "Alice","Bob","Carlos","Diana","Eric","Fatima","George","Helena",
    "Ivan","Julia","Kevin","Laura","Marc","Nina","Omar","Priya",
    "Quinn","Rosa","Sam","Tina","Umar","Vera","Will","Xia","Yann",
    "Zara","Ahmed","Bella","Cesar","Dina","Ethan","Farida","Greg",
    "Hana","Issa","Jana","Karim","Lena","Mehdi","Nadia","Oscar",
    "Paula","Rafik","Sara","Theo","Uma","Victor","Wendy","Yusuf","Zoe",
]

new_user_rows = []
base_date = datetime(2023, 6, 1)
for i in range(50):
    days_off = int(RNG.integers(0, 400))
    new_user_rows.append({
        "user_id":             uuid.uuid4().hex[:22],
        "name":                FIRST_NAMES[i],
        "review_count":        int(RNG.integers(2, 25)),
        "yelping_since":       (base_date + timedelta(days=days_off)).strftime("%Y-%m-%d %H:%M:%S"),
        "useful":              int(RNG.integers(0, 15)),
        "funny":               int(RNG.integers(0, 8)),
        "cool":                int(RNG.integers(0, 8)),
        "elite":               "",
        "friends":             "",
        "fans":                int(RNG.integers(0, 5)),
        "average_stars":       round(float(RNG.uniform(3.0, 5.0)), 2),
        "compliment_hot":      int(RNG.integers(0, 3)),
        "compliment_more":     0,
        "compliment_profile":  0,
        "compliment_cute":     0,
        "compliment_list":     0,
        "compliment_note":     int(RNG.integers(0, 3)),
        "compliment_plain":    int(RNG.integers(0, 5)),
        "compliment_cool":     int(RNG.integers(0, 3)),
        "compliment_funny":    0,
        "compliment_writer":   int(RNG.integers(0, 2)),
        "compliment_photos":   0,
    })

new_user_df = pd.DataFrame(new_user_rows)
all_user_df = pd.concat([user_df, new_user_df], ignore_index=True)
out_user = f"{OUT_DIR}/yelp_academic_dataset_user_healthandmedical.csv"
all_user_df.to_csv(out_user, index=False)
print(f"Users    : {len(all_user_df):,} total ({len(new_user_rows)} nouveaux) -> {out_user}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. ~1000 nouvelles reviews
# ─────────────────────────────────────────────────────────────────────────────
new_user_ids = [r["user_id"] for r in new_user_rows]
new_biz_ids  = [r["business_id"] for r in new_biz_rows]
all_user_ids = existing_user_ids + new_user_ids
all_biz_ids  = existing_biz_ids  + new_biz_ids

TEXTS = [
    "Great service and very professional staff. Highly recommended!",
    "Good experience overall. The doctor was attentive and thorough.",
    "Average visit, nothing special but got the job done.",
    "Excellent care! The staff was friendly and efficient.",
    "Very clean facility and short wait time. Will return.",
    "The specialist was knowledgeable and explained everything clearly.",
    "Decent place but the wait was a bit long.",
    "Outstanding medical care. Best clinic in the area!",
    "Professional and caring team. Felt very comfortable.",
    "Good clinic, would recommend to family and friends.",
    "Friendly staff and modern equipment. Very satisfied.",
    "Quick appointment scheduling and minimal wait time.",
]

rows = []
seen = set()

# Nouveaux users : 10-18 reviews chacun, biaisés vers nouveaux businesses
for u in new_user_ids:
    n = int(RNG.integers(10, 19))
    weights = np.ones(len(all_biz_ids))
    weights[-len(new_biz_ids):] *= 5
    weights /= weights.sum()
    chosen = RNG.choice(len(all_biz_ids), size=min(n, len(all_biz_ids)), replace=False, p=weights)
    for b_idx in chosen:
        key = (u, all_biz_ids[b_idx])
        if key not in seen:
            seen.add(key)
            rows.append((u, all_biz_ids[b_idx], int(RNG.choice([3, 4, 4, 5, 5]))))

# Users existants actifs : 2-5 nouvelles reviews
existing_sample = RNG.choice(existing_user_ids, size=200, replace=False)
for u in existing_sample:
    n = int(RNG.integers(2, 6))
    chosen = RNG.choice(len(all_biz_ids), size=min(n, len(all_biz_ids)), replace=False)
    for b_idx in chosen:
        key = (u, all_biz_ids[b_idx])
        if key not in seen:
            seen.add(key)
            rows.append((u, all_biz_ids[b_idx], int(RNG.choice([3, 4, 4, 5, 5]))))

rev_df = pd.DataFrame(rows, columns=["user_id", "business_id", "stars"])
rev_df = rev_df.head(1000).copy()
rev_df["review_id"] = [uuid.uuid4().hex[:22] for _ in range(len(rev_df))]

base = datetime(2024, 3, 1)
rev_df["date"] = [
    (base + timedelta(hours=int(i * 0.7))).strftime("%Y-%m-%d")
    for i in range(len(rev_df))
]
rev_df["text"]   = [TEXTS[int(RNG.integers(0, len(TEXTS)))] for _ in range(len(rev_df))]
rev_df["useful"] = RNG.integers(0, 6, size=len(rev_df))
rev_df["funny"]  = RNG.integers(0, 3, size=len(rev_df))
rev_df["cool"]   = RNG.integers(0, 3, size=len(rev_df))
rev_df = rev_df[["review_id","user_id","business_id","stars","date","text","useful","funny","cool"]]
rev_df = rev_df.sample(frac=1, random_state=42).reset_index(drop=True)

out_rev = f"{OUT_DIR}/yelp_academic_dataset_review_healthandmedical.csv"
rev_df.to_csv(out_rev, index=False)

n_new_u = len(set(rev_df["user_id"]) & set(new_user_ids))
n_new_b = len(set(rev_df["business_id"]) & set(new_biz_ids))
print(f"Reviews  : {len(rev_df):,} total -> {out_rev}")
print(f"  Users uniques  : {rev_df['user_id'].nunique():,}  ({n_new_u} nouveaux)")
print(f"  Business uniq  : {rev_df['business_id'].nunique():,}  ({n_new_b} nouveaux)")
print(f"  Ratings        : {dict(rev_df['stars'].value_counts().sort_index())}")
print(f"  Periode        : {rev_df['date'].min()} -> {rev_df['date'].max()}")
print(f"\nDone. Dossier : {OUT_DIR}/")
