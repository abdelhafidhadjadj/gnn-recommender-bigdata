"""
Generateur de dataset incremental Yelp-compatible (3 fichiers).

Structure identique aux fichiers data/1k/, data/5k/, etc. :
  - incremental_users.csv    : user_id, name, review_count, yelping_since,
                               average_stars, fans, useful, funny, cool
  - incremental_business.csv : business_id, name, address, city, state,
                               postal_code, stars, review_count, is_open, categories
  - incremental_reviews.csv  : user_id, review_id, business_id, stars,
                               date, text, useful, funny, cool

Design:
  - 1 200 reviews  (apres deduplication : ~1 150 uniques)
  - 80 utilisateurs  : 60 existants (depuis existing_users.json)
                     + 20 nouveaux  (IDs generes)
  - 50 businesses   : 38 existants  (depuis existing_items.json)
                     + 12 nouveaux  (IDs generes)
  - Sparsite ~ 70 %  (1 - 1200/4000)
  - 5 groupes de preferences coherents
  - Notes : mu = 2.5 + 2.5*group_match,  sigma = 0.9  -> moy ~3.88
  - Textes de review synthetiques (positif / neutre / negatif)

Usage:
  python3.13 generate_incremental_dataset.py
  -> data/incremental/incremental_users.csv
  -> data/incremental/incremental_business.csv
  -> data/incremental/incremental_reviews.csv
  -> data/incremental/stats.txt
"""

import json
import random
import string
import uuid
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
SEED           = 42
N_EXISTING_USR = 60
N_NEW_USR      = 20
N_EXISTING_ITM = 38
N_NEW_ITM      = 12
TARGET_ROWS    = 1_200

N_USERS = N_EXISTING_USR + N_NEW_USR   # 80
N_ITEMS = N_EXISTING_ITM + N_NEW_ITM   # 50

OUTPUT_DIR = Path("data/incremental")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)
random.seed(SEED)

# ── Helpers ID ────────────────────────────────────────────────────────────────
def _rand_b64(n=22):
    """Genere un ID alphanum de 22 chars (style Yelp)."""
    chars = string.ascii_letters + string.digits
    return ''.join(random.choices(chars, k=n))

def _rand_review_id():
    return _rand_b64(22)

# ── Charger les IDs existants ─────────────────────────────────────────────────
print("Chargement des IDs existants...")
with open("existing_users.json") as f:
    all_existing_users = json.load(f)
with open("existing_items.json") as f:
    all_existing_items = json.load(f)

existing_users = random.sample(all_existing_users, N_EXISTING_USR)
existing_items = random.sample(all_existing_items, N_EXISTING_ITM)

new_users = [_rand_b64(22) for _ in range(N_NEW_USR)]
new_items = [_rand_b64(22) for _ in range(N_NEW_ITM)]

all_users = existing_users + new_users
all_items = existing_items + new_items

print(f"  Users  : {N_EXISTING_USR} existants + {N_NEW_USR} nouveaux = {N_USERS}")
print(f"  Items  : {N_EXISTING_ITM} existants + {N_NEW_ITM} nouveaux = {N_ITEMS}")

# ── Groupes de preferences ────────────────────────────────────────────────────
N_GROUPS = 5

user_group = {u: int(rng.integers(0, N_GROUPS)) for u in all_users}
item_groups = {}
for item in all_items:
    g1 = int(rng.integers(0, N_GROUPS))
    if rng.random() < 0.3:
        g2 = (g1 + int(rng.integers(1, N_GROUPS))) % N_GROUPS
        item_groups[item] = {g1, g2}
    else:
        item_groups[item] = {g1}

# ── Note realiste ─────────────────────────────────────────────────────────────
def generate_rating(user, item):
    ug = user_group[user]
    ig = item_groups[item]
    group_match = 1.0 if ug in ig else 0.0
    adjacent = (ug + 1) % N_GROUPS
    if adjacent in ig:
        group_match = max(group_match, 0.4)
    mu = 2.5 + 2.5 * group_match
    raw = rng.normal(mu, 0.9)
    return int(np.clip(round(raw), 1, 5))

# ── Textes de review synthetiques ────────────────────────────────────────────
_POSITIVE = [
    "Excellent service, staff very attentive and professional.",
    "Great experience overall. Clean facility and knowledgeable doctors.",
    "Highly recommend. Short wait time and thorough examination.",
    "Very satisfied. The doctor explained everything clearly.",
    "Outstanding care. Will definitely return for future visits.",
    "Friendly staff and top-notch medical expertise.",
    "Best clinic in the area. Fast and efficient service.",
    "Impressive facilities and compassionate team.",
]
_NEUTRAL = [
    "Decent experience. Nothing exceptional but got the job done.",
    "Average service. Wait time was reasonable.",
    "OK visit. Doctor was fine, nothing special.",
    "Fairly standard medical visit. Met expectations.",
    "Acceptable care. Could improve on communication.",
    "Not bad, not great. Would consider returning.",
    "Mediocre experience. Staff was polite but rushed.",
    "So-so. The facility was clean but service was slow.",
]
_NEGATIVE = [
    "Disappointing. Long wait and unfriendly staff.",
    "Poor experience. Doctor seemed distracted.",
    "Would not recommend. Billing issues and rude receptionist.",
    "Very bad service. Waited 2 hours with no explanation.",
    "Terrible. Felt like just a number, not a patient.",
    "Unprofessional staff and outdated equipment.",
    "Worst clinic I have visited. No follow-up care.",
    "Extremely disappointing. Wrong prescription given.",
]

def generate_text(stars: int) -> str:
    if stars >= 4:
        return random.choice(_POSITIVE)
    elif stars == 3:
        return random.choice(_NEUTRAL)
    else:
        return random.choice(_NEGATIVE)

# ── Generation des reviews ────────────────────────────────────────────────────
print("Generation des interactions...")
start_date = datetime(2024, 1, 1)
activity = {u: int(rng.integers(10, 26)) for u in all_users}

rows = []
seen_pairs = set()

for user in all_users:
    quota = activity[user]
    ug = user_group[user]

    same_group  = [i for i in all_items if ug in item_groups[i]]
    other_group = [i for i in all_items if ug not in item_groups[i]]

    weights = (
        [0.75 / max(len(same_group), 1)]  * len(same_group) +
        [0.25 / max(len(other_group), 1)] * len(other_group)
    )
    pool = same_group + other_group
    w = np.array(weights, dtype=float)
    w /= w.sum()

    n_sample = min(quota, len(pool))
    chosen = rng.choice(len(pool), size=n_sample, replace=False, p=w)

    for idx in chosen:
        item = pool[idx]
        if (user, item) in seen_pairs:
            continue
        seen_pairs.add((user, item))

        stars = generate_rating(user, item)
        delta = timedelta(
            days=int(rng.integers(0, 180)),
            hours=int(rng.integers(0, 24)),
        )
        date = (start_date + delta).strftime("%Y-%m-%d")

        rows.append({
            "user_id":     user,
            "review_id":   _rand_review_id(),
            "business_id": item,
            "stars":       stars,
            "date":        date,
            "text":        generate_text(stars),
            "useful":      int(rng.integers(0, 25)),
            "funny":       int(rng.integers(0, 15)),
            "cool":        int(rng.integers(0, 15)),
        })

print(f"  Reviews generes : {len(rows)}")

# Completer jusqu'a TARGET_ROWS
if len(rows) < TARGET_ROWS:
    extra_needed = TARGET_ROWS - len(rows)
    print(f"  Completion : +{extra_needed} interactions...")
    attempts = 0
    while len(rows) < TARGET_ROWS and attempts < 50_000:
        u = random.choice(all_users)
        i = random.choice(all_items)
        if (u, i) not in seen_pairs:
            seen_pairs.add((u, i))
            stars = generate_rating(u, i)
            delta = timedelta(days=int(rng.integers(0, 180)))
            date  = (start_date + delta).strftime("%Y-%m-%d")
            rows.append({
                "user_id":     u,
                "review_id":   _rand_review_id(),
                "business_id": i,
                "stars":       stars,
                "date":        date,
                "text":        generate_text(stars),
                "useful":      int(rng.integers(0, 25)),
                "funny":       int(rng.integers(0, 15)),
                "cool":        int(rng.integers(0, 15)),
            })
        attempts += 1

df_reviews = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

# ── Fichier 1 : incremental_reviews.csv ──────────────────────────────────────
rev_path = OUTPUT_DIR / "incremental_reviews.csv"
df_reviews.to_csv(rev_path, index=False)
print(f"\n[1/3] Reviews -> {rev_path}  ({len(df_reviews):,} lignes)")

# ── Fichier 2 : incremental_users.csv ────────────────────────────────────────
FIRST_NAMES = [
    "James","Mary","John","Patricia","Robert","Jennifer","Michael","Linda",
    "William","Barbara","David","Susan","Richard","Jessica","Joseph","Sarah",
    "Thomas","Karen","Charles","Lisa","Emily","Daniel","Sofia","Lucas",
    "Emma","Noah","Olivia","Liam","Ava","Ethan","Isabella","Mason",
    "Mia","Logan","Amelia","Oliver","Harper","Elijah","Evelyn","Benjamin",
    "Abigail","Alexander","Ella","Henry","Scarlett","Sebastian","Grace",
    "Jack","Chloe","Owen","Victoria","Samuel","Riley","Grayson","Aria",
    "Jayden","Lily","Lincoln","Aurora","Christopher","Zoey","Joshua"
]
LAST_NAMES = [
    "Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis",
    "Rodriguez","Martinez","Hernandez","Lopez","Gonzalez","Wilson","Anderson",
    "Thomas","Taylor","Moore","Jackson","Martin","Lee","Perez","Thompson",
    "White","Harris","Sanchez","Clark","Ramirez","Lewis","Robinson","Walker",
    "Young","Allen","King","Wright","Scott","Torres","Nguyen","Hill","Flores",
    "Green","Adams","Nelson","Baker","Hall","Rivera","Campbell","Mitchell",
    "Carter","Roberts","Reed","Cook","Morgan","Bell","Murphy","Bailey"
]

rng2 = np.random.default_rng(SEED + 1)

def _rand_name():
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"

def _rand_date_since():
    base = datetime(2010, 1, 1)
    days = int(rng2.integers(0, 365 * 12))
    return (base + timedelta(days=days)).strftime("%Y-%m-%d")

# Calculer review_count et average_stars depuis les reviews generees
user_stats = df_reviews.groupby("user_id").agg(
    review_count=("stars", "count"),
    average_stars=("stars", "mean"),
).reset_index()
user_stats["average_stars"] = user_stats["average_stars"].round(2)

user_rows = []
for uid in all_users:
    stats = user_stats[user_stats["user_id"] == uid]
    rc  = int(stats["review_count"].iloc[0]) if len(stats) > 0 else 0
    avg = float(stats["average_stars"].iloc[0]) if len(stats) > 0 else round(float(rng2.uniform(1.5, 4.5)), 2)
    user_rows.append({
        "user_id":       uid,
        "name":          _rand_name(),
        "review_count":  rc,
        "yelping_since": _rand_date_since(),
        "average_stars": avg,
        "fans":          int(rng2.integers(0, 200)),
        "useful":        int(rng2.integers(0, 500)),
        "funny":         int(rng2.integers(0, 300)),
        "cool":          int(rng2.integers(0, 300)),
    })

df_users = pd.DataFrame(user_rows)
usr_path = OUTPUT_DIR / "incremental_users.csv"
df_users.to_csv(usr_path, index=False)
print(f"[2/3] Users    -> {usr_path}  ({len(df_users):,} lignes)")

# ── Fichier 3 : incremental_business.csv ─────────────────────────────────────
CITIES   = ["Phoenix","Las Vegas","Toronto","Charlotte","Pittsburgh",
            "Cleveland","Tampa","Austin","Denver","Seattle"]
STATES   = ["AZ","NV","ON","NC","PA","OH","FL","TX","CO","WA"]
STREETS  = ["Oak St","Elm St","Main Ave","Park Blvd","Cedar Rd",
            "Maple Dr","Lake Ave","River Rd","Hill St","Pine Blvd"]
CATEGORIES = [
    "Doctors, Internal Medicine, Health & Medical",
    "Dentists, General Dentistry, Health & Medical",
    "Hospitals, Health & Medical",
    "Optometrists, Eyewear & Opticians, Health & Medical",
    "Chiropractors, Health & Medical",
    "Physical Therapy, Health & Medical",
    "Dermatologists, Health & Medical",
    "Cardiologists, Health & Medical",
    "Pediatricians, Health & Medical",
    "Psychiatrists, Mental Health, Health & Medical",
]
BIZ_NAMES = [
    "City Medical Center","Valley Health Clinic","Metro Internal Medicine",
    "Premier Family Practice","Sunrise Dental Group","Pacific Eye Care",
    "Advanced Spine & Rehab","Elite Dermatology","HeartCare Associates",
    "Kids First Pediatrics","MindWell Psychiatry","Urban Urgent Care",
    "Northside Cardiology","Southgate Orthopedics","Eastside ENT Specialists",
    "Westbrook Radiology","Lakeside Wellness Center","Ridgeline Rheumatology",
    "Greenfield Sports Medicine","Clearview Ophthalmology","BlueSky Allergy Clinic",
    "Harmony Integrative Medicine","Precision Oncology Group","Community Hospice Care",
    "Apex Physical Therapy","Nexus Neurology","Crestview OB-GYN","Sagebrush Urology",
    "Golden Gate Gastroenterology","Harbor Hematology","Summit Sleep Center",
    "Riverside Pain Management","Meadow Nephrology","Cascade Pulmonology",
    "Timberline Toxicology","Bridgeview Bariatric Surgery","Cornerstone Colon & Rectal",
    "Oakwood Vascular Surgery","Pinecrest Plastic Surgery","Thornwood Transplant Center",
    "Fairview Foot & Ankle","Clearwater Hand Surgery","Milestone Microsurgery",
    "Coppice Cranio-Facial","Dune Dental Implants","Tidewater Endodontics",
    "Glenwood Periodontics","Sandstone Orthodontics","Ironwood Oral Surgery",
    "Copperfield Cosmetic Dentistry",
]

rng3 = np.random.default_rng(SEED + 2)

# Calculer review_count et stars moyennes depuis les reviews
biz_stats = df_reviews.groupby("business_id").agg(
    review_count=("stars", "count"),
    stars=("stars", "mean"),
).reset_index()
biz_stats["stars"] = biz_stats["stars"].round(1)

biz_rows = []
for i, bid in enumerate(all_items):
    stats = biz_stats[biz_stats["business_id"] == bid]
    rc    = int(stats["review_count"].iloc[0]) if len(stats) > 0 else 0
    avg   = float(stats["stars"].iloc[0]) if len(stats) > 0 else round(float(rng3.uniform(2.0, 5.0)), 1)
    city_idx = int(rng3.integers(0, len(CITIES)))
    biz_rows.append({
        "business_id":  bid,
        "name":         BIZ_NAMES[i % len(BIZ_NAMES)],
        "address":      f"{int(rng3.integers(100, 9999))} {random.choice(STREETS)}",
        "city":         CITIES[city_idx],
        "state":        STATES[city_idx],
        "postal_code":  str(int(rng3.integers(10000, 99999))),
        "stars":        avg,
        "review_count": rc,
        "is_open":      int(rng3.choice([0, 1], p=[0.1, 0.9])),
        "categories":   CATEGORIES[i % len(CATEGORIES)],
    })

df_business = pd.DataFrame(biz_rows)
biz_path = OUTPUT_DIR / "incremental_business.csv"
df_business.to_csv(biz_path, index=False)
print(f"[3/3] Business -> {biz_path}  ({len(df_business):,} lignes)")

# ── Stats qualite ─────────────────────────────────────────────────────────────
n_pairs   = N_USERS * N_ITEMS
sparsity  = 1 - len(df_reviews) / n_pairs
avg_u     = df_reviews.groupby("user_id").size().mean()
min_u     = df_reviews.groupby("user_id").size().min()
avg_i     = df_reviews.groupby("business_id").size().mean()
min_i     = df_reviews.groupby("business_id").size().min()
rat_mu    = df_reviews["stars"].mean()
rat_std   = df_reviews["stars"].std()
new_u_cnt = df_reviews["user_id"].isin(new_users).sum()
new_i_cnt = df_reviews["business_id"].isin(new_items).sum()
pct_pos   = (df_reviews["stars"] >= 4).mean() * 100

sep  = "=" * 60
sep2 = "-" * 60
stats = "\n".join([
    sep,
    "  RAPPORT QUALITE - DATASET INCREMENTAL (3 fichiers)",
    sep,
    f"  Fichiers generes :",
    f"    incremental_reviews.csv   ({len(df_reviews):>5,} lignes)",
    f"    incremental_users.csv     ({len(df_users):>5,} lignes)",
    f"    incremental_business.csv  ({len(df_business):>5,} lignes)",
    sep2,
    f"  Utilisateurs uniques    : {df_reviews['user_id'].nunique():>4}",
    f"    dont existants        : {N_EXISTING_USR}",
    f"    dont nouveaux         : {N_NEW_USR}",
    f"  Businesses uniques      : {df_reviews['business_id'].nunique():>4}",
    f"    dont existants        : {N_EXISTING_ITM}",
    f"    dont nouveaux         : {N_NEW_ITM}",
    sep2,
    f"  SPARSITE                : {sparsity*100:>5.1f} %",
    f"  Reviews/user  (moyenne) : {avg_u:>5.1f}   (min={min_u})",
    f"  Reviews/biz   (moyenne) : {avg_i:>5.1f}   (min={min_i})",
    sep2,
    f"  Note moyenne            : {rat_mu:>5.2f}   (std={rat_std:.2f})",
    f"  % notes >= 4 (positif)  : {pct_pos:>5.1f} %",
    f"  Reviews new users       : {new_u_cnt:>5,}",
    f"  Reviews new businesses  : {new_i_cnt:>5,}",
    sep2,
    "  COLONNES PAR FICHIER",
    "  reviews  : user_id, review_id, business_id, stars,",
    "             date, text, useful, funny, cool",
    "  users    : user_id, name, review_count, yelping_since,",
    "             average_stars, fans, useful, funny, cool",
    "  business : business_id, name, address, city, state,",
    "             postal_code, stars, review_count, is_open, categories",
    sep,
]) + "\n"

print("\n" + stats)

stat_path = OUTPUT_DIR / "stats.txt"
with open(stat_path, "w", encoding="utf-8") as f:
    f.write(stats)

print(f"Stats -> {stat_path}")
print("\nApercu reviews :")
print(df_reviews.head(5).to_string(index=False))
print("\nApercu users :")
print(df_users.head(3).to_string(index=False))
print("\nApercu business :")
print(df_business.head(3).to_string(index=False))
