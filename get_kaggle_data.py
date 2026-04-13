from datasets import load_dataset
import pandas as pd

print("Kaggle/HF'den dataset indiriliyor...")
# Load real job descriptions
ds = load_dataset("jacob-hugging-face/job-descriptions", split="train[:500]")
df = ds.to_pandas()

print("Sütunlar:", df.columns.tolist())
print(df.head(2))

# Map to our structure
df_new = pd.DataFrame()
df_new['title'] = df['position_title']
df_new['company'] = df['company_name']
df_new['skills'] = "Various IT Skills" # Placeholder if not present
df_new['description'] = df['job_description']

# Fill 'skills' with some mocked Kaggle mapping or keep generic if column not found
# Save
df_new.to_csv("job_dataset.csv", index=False)
print("Başarıyla job_dataset.csv güncellendi! Yeni Veri boyutu:", len(df_new))
