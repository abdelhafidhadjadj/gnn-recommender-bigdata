import pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('data/raw/full/yelp_academic_dataset_review_healthandmedical.csv', low_memory=False)
top = df.groupby('user_id').size().sort_values(ascending=False).head(10)

print('Top 10 utilisateurs par nombre interactions:')
for uid, cnt in top.items():
    stars = df[df['user_id']==uid]['stars'].mean()
    print(f'  {uid}  ->  {cnt} interactions  (moy: {stars:.1f} etoiles)')
