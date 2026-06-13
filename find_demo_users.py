import pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('data/raw/100k/yelp_academic_dataset_review_healthandmedical.csv', low_memory=False)

# Tous les users triés alphabétiquement (comme user_enc.classes_)
all_users = sorted(df['user_id'].unique())
first_200 = all_users[:200]

# Compter les interactions pour chacun
counts = df[df['user_id'].isin(first_200)].groupby('user_id').size()
top = counts.sort_values(ascending=False).head(10)

print(f'Top 10 users parmi les 200 premiers (alphabetique) :')
print(f'(ce sont les users visibles dans Streamlit)\n')
for uid, cnt in top.items():
    avg = df[df['user_id']==uid]['stars'].mean()
    print(f'  {uid}  ->  {cnt} interactions  (moy: {avg:.1f})')
