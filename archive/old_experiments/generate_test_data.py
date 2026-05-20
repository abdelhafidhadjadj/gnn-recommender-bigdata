"""
Synthetic Yelp Health-&-Medical dataset generator.

Default output (data/test/):
    yelp_academic_dataset_business_healthandmedical.csv  — 150 businesses
    yelp_academic_dataset_user_healthandmedical.csv      — 300 users
    yelp_academic_dataset_review_healthandmedical.csv    — ~4 000 reviews (~8.9% density)

Run:
    python generate_test_data.py
    python generate_test_data.py --out-dir data/raw
"""
import argparse, os, random, string, datetime
import pandas as pd
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

_CHARS = string.ascii_letters + string.digits
def _uid(n=22):
    return "".join(random.choices(_CHARS, k=n))

# ── 30 speciality templates × 5 instances = 150 businesses ────────────────────
SPECS = [
    ("Internal Medicine",     "Doctors, Internal Medicine, Health & Medical"),
    ("Dental Care",           "Dentists, Cosmetic Dentists, Health & Medical"),
    ("General Hospital",      "Hospitals, Emergency Rooms, Health & Medical"),
    ("Physical Therapy",      "Physical Therapy, Sports Medicine, Health & Medical"),
    ("Optometry",             "Optometrists, Eyewear & Opticians, Health & Medical"),
    ("Chiropractic",          "Chiropractors, Sports Medicine, Health & Medical"),
    ("Urgent Care",           "Urgent Care, Walk-in Clinics, Health & Medical"),
    ("Pharmacy",              "Pharmacy, Health & Medical"),
    ("Mental Health",         "Mental Health, Counseling & Mental Health, Health & Medical"),
    ("Nutrition",             "Nutritionists, Health Coaches, Dietitians, Health & Medical"),
    ("Dermatology",           "Dermatologists, Skin Care, Health & Medical"),
    ("Pediatrics",            "Pediatricians, Child & Adolescent Psychiatry, Health & Medical"),
    ("OB-GYN",                "Obstetricians & Gynecologists, Midwives, Health & Medical"),
    ("Acupuncture",           "Acupuncture, Traditional Chinese Medicine, Health & Medical"),
    ("Massage Therapy",       "Massage Therapy, Physical Therapy, Health & Medical"),
    ("Orthopedics",           "Orthopedists, Sports Medicine, Health & Medical"),
    ("Cardiology",            "Cardiologists, Vascular Medicine, Health & Medical"),
    ("Neurology",             "Neurologists, Sleep Specialists, Health & Medical"),
    ("Allergy Center",        "Allergists, Immunologists, Health & Medical"),
    ("Endocrinology",         "Endocrinologists, Diabetes Care, Health & Medical"),
    ("Oncology",              "Oncologists, Hematologists, Health & Medical"),
    ("Radiology",             "Radiologists, Diagnostic Imaging, Health & Medical"),
    ("Diagnostics Lab",       "Labs, Diagnostic Services, Health & Medical"),
    ("Medical Spa",           "Medical Spas, Laser Hair Removal, Health & Medical"),
    ("Audiology",             "Hearing Aid Providers, Audiologists, Health & Medical"),
    ("Podiatry",              "Podiatrists, Orthotics, Health & Medical"),
    ("Rheumatology",          "Rheumatologists, Immunologists, Health & Medical"),
    ("Gastroenterology",      "Gastroenterologists, Colonoscopy, Health & Medical"),
    ("Urology",               "Urologists, Health & Medical"),
    ("Pulmonology",           "Pulmonologists, Sleep Medicine, Health & Medical"),
]
INSTANCES = 5        # 30 types × 5 = 150 businesses

PREFIXES  = ["Valley", "Metro", "City", "Summit", "Lake", "Bright", "Premier",
             "Elite", "Advanced", "Family", "Pacific", "Central", "North", "South"]
SUFFIXES  = ["Clinic", "Center", "Associates", "Group", "Partners", "Institute", "Health"]
CITIES    = ["Phoenix", "Las Vegas", "Chicago", "Houston", "Miami",
             "Seattle", "Denver", "Atlanta", "Boston", "Dallas",
             "Portland", "San Diego", "Austin", "Nashville", "Charlotte"]
STATES    = ["AZ","NV","IL","TX","FL","WA","CO","GA","MA","TX",
             "OR","CA","TX","TN","NC"]

def make_businesses():
    rows = []
    for spec_i, (base_name, cats) in enumerate(SPECS):
        for inst in range(INSTANCES):
            city_i = (spec_i * INSTANCES + inst) % len(CITIES)
            name = f"{random.choice(PREFIXES)} {base_name} {random.choice(SUFFIXES)}"
            rows.append({
                "business_id":  _uid(),
                "name":         name,
                "address":      f"{random.randint(100,9999)} {random.choice(['Main','Oak','Elm','Park','Lake'])} St",
                "city":         CITIES[city_i],
                "state":        STATES[city_i],
                "postal_code":  f"{random.randint(10000,99999)}",
                "stars":        round(random.uniform(2.5, 5.0), 1),
                "review_count": random.randint(10, 600),
                "is_open":      1,
                "categories":   cats,
            })
    return pd.DataFrame(rows)


# ── 300 users ─────────────────────────────────────────────────────────────────
N_USERS = 300
FIRST = ["Alice","Bob","Carlos","Diana","Eve","Frank","Grace","Hank","Iris","Jake",
         "Karen","Leo","Mia","Noah","Olivia","Paul","Quinn","Rosa","Sam","Tina",
         "Uma","Victor","Wendy","Xander","Yara","Zoe","Aaron","Beth","Cole","Dana",
         "Elena","Felix","Gina","Harry","Irene","Joel","Kim","Luke","Maya","Nick",
         "Ola","Pete","Rita","Steve","Tara","Ugo","Vera","Will","Xena","Yasmin"]
LAST  = ["Smith","Johnson","Williams","Brown","Jones","Garcia","Martinez","Davis",
         "Wilson","Anderson","Thomas","Taylor","Moore","Jackson","Martin","Lee",
         "Perez","White","Harris","Clark","Lewis","Robinson","Walker","Hall","Allen"]

def make_users(n=N_USERS):
    rows = []
    for _ in range(n):
        rows.append({
            "user_id":       _uid(),
            "name":          f"{random.choice(FIRST)} {random.choice(LAST)}",
            "review_count":  random.randint(1, 250),
            "yelping_since": str(datetime.date(random.randint(2010,2020),
                                               random.randint(1,12),
                                               random.randint(1,28))),
            "average_stars": round(random.uniform(1.5, 5.0), 2),
            "fans":          random.randint(0, 80),
            "useful":        random.randint(0, 150),
            "funny":         random.randint(0, 60),
            "cool":          random.randint(0, 60),
        })
    return pd.DataFrame(rows)


# ── ~4 000 reviews with realistic distribution ─────────────────────────────────
# Star weights (Yelp-realistic for health): 1★4%  2★5%  3★13%  4★28%  5★50%
STAR_W = [4, 5, 13, 28, 50]

TEXTS = {
    5: ["Absolutely fantastic! Staff was professional and caring.",
        "Best visit ever — thorough, efficient, highly recommend.",
        "Top-notch facility. The doctor explained everything clearly.",
        "Outstanding service. Will definitely return.",
        "Five stars without hesitation. Exceptional care.",
        "So impressed with the professionalism and attention to detail.",
        "Excellent experience from check-in to check-out.",
        "The staff made me feel completely at ease. Wonderful.",],
    4: ["Great experience overall. Minor wait but quality care.",
        "Very professional staff. Would return for future visits.",
        "Good service and knowledgeable doctors. Parking could be better.",
        "Pleasant visit. The nurse practitioner was very attentive.",
        "Solid clinic with friendly staff. Scheduling was a bit tricky.",
        "Good but not perfect — the wait was a bit long.",
        "Mostly positive. Clean facility and caring doctors.",],
    3: ["Average experience. Nothing bad, nothing exceptional.",
        "Decent care but the wait was longer than expected.",
        "The doctor seemed rushed. My concerns were not fully addressed.",
        "Mediocre front desk but the actual care was fine.",
        "Three stars — met my basic needs with room for improvement.",
        "OK visit. Would only return if closer options are unavailable.",],
    2: ["Disappointing. Long wait and unfriendly staff.",
        "Below expectations. Had to follow up multiple times.",
        "Not great. Facility needs updates and service was slow.",
        "Poor communication between staff made it frustrating.",
        "Would not rush back. Felt disorganized.",],
    1: ["Terrible experience. Very rude staff and extreme wait times.",
        "Avoid this place. Messed up my prescription with no concern.",
        "One star is too generous. Complete lack of professionalism.",
        "Worst clinic I have been to. No follow-up, dismissive doctors.",
        "Unacceptable. Required three visits for a simple diagnosis.",],
}

def make_reviews(business_df: pd.DataFrame, user_df: pd.DataFrame):
    biz_ids  = business_df["business_id"].tolist()
    user_ids = user_df["user_id"].tolist()

    # Assign each user a random activity level: light / moderate / heavy
    activity = np.random.choice(["light","moderate","heavy"],
                                size=len(user_ids),
                                p=[0.35, 0.45, 0.20])
    reviews_per_user = {"light": (4, 9), "moderate": (10, 20), "heavy": (21, 35)}

    pairs: set = set()

    # Primary seeding: every user reviews according to their activity level
    for uid, act in zip(user_ids, activity):
        lo, hi = reviews_per_user[act]
        k = min(random.randint(lo, hi), len(biz_ids))
        for bid in random.sample(biz_ids, k=k):
            pairs.add((uid, bid))

    # Secondary: ensure every business has at least 20 reviews
    for bid in biz_ids:
        reviewers = [u for u, b in pairs if b == bid]
        deficit = max(0, 20 - len(reviewers))
        candidates = [u for u in user_ids if (u, bid) not in pairs]
        for uid in random.sample(candidates, k=min(deficit, len(candidates))):
            pairs.add((uid, bid))

    pairs = list(pairs)
    random.shuffle(pairs)

    start_date = datetime.date(2020, 1, 1)
    end_date   = datetime.date(2025, 4, 30)
    date_range = (end_date - start_date).days

    rows = []
    for uid, bid in pairs:
        stars = random.choices([1,2,3,4,5], weights=STAR_W, k=1)[0]
        date  = start_date + datetime.timedelta(days=random.randint(0, date_range))
        rows.append({
            "review_id":   _uid(),
            "user_id":     uid,
            "business_id": bid,
            "stars":       stars,
            "date":        str(date),
            "text":        random.choice(TEXTS[stars]),
            "useful":      random.randint(0, 25),
            "funny":       random.randint(0, 12),
            "cool":        random.randint(0, 12),
        })

    return pd.DataFrame(rows)


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/test")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Generating 150 businesses ...")
    bdf = make_businesses()

    print("Generating 300 users ...")
    udf = make_users()

    print("Generating reviews ...")
    rdf = make_reviews(bdf, udf)

    b_path = os.path.join(args.out_dir, "yelp_academic_dataset_business_healthandmedical.csv")
    u_path = os.path.join(args.out_dir, "yelp_academic_dataset_user_healthandmedical.csv")
    r_path = os.path.join(args.out_dir, "yelp_academic_dataset_review_healthandmedical.csv")

    bdf.to_csv(b_path, index=False)
    udf.to_csv(u_path, index=False)
    rdf.to_csv(r_path, index=False)

    density = len(rdf) / (len(bdf) * len(udf)) * 100
    print(f"\nDataset written to: {os.path.abspath(args.out_dir)}")
    print(f"  Businesses : {len(bdf):>5}")
    print(f"  Users      : {len(udf):>5}")
    print(f"  Reviews    : {len(rdf):>5}  (density {density:.1f}%)")
    print(f"  Relevant (stars>=4): {(rdf.stars>=4).sum()} ({100*(rdf.stars>=4).mean():.1f}%)")
    print(f"\nRating distribution:")
    print(rdf.stars.value_counts().sort_index().to_string())
    print(f"\nReviews per user  — min:{rdf.groupby('user_id').size().min()}  "
          f"median:{rdf.groupby('user_id').size().median():.0f}  "
          f"max:{rdf.groupby('user_id').size().max()}")
    print(f"Reviews per biz   — min:{rdf.groupby('business_id').size().min()}  "
          f"median:{rdf.groupby('business_id').size().median():.0f}  "
          f"max:{rdf.groupby('business_id').size().max()}")

if __name__ == "__main__":
    main()
