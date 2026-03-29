from pathlib import Path
import os

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from huggingface_hub import HfApi
except Exception:
    HfApi = None

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_DIR / "data" / "raw" / "tourism.csv"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW_FILE)
clean_df = df.drop(columns=["Unnamed: 0", "CustomerID"]).copy()
for column in clean_df.select_dtypes(include="object").columns:
    clean_df[column] = clean_df[column].str.strip()

X = clean_df.drop(columns=["ProdTaken"])
y = clean_df["ProdTaken"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_df = X_train.copy()
train_df["ProdTaken"] = y_train.values
test_df = X_test.copy()
test_df["ProdTaken"] = y_test.values

clean_path = PROCESSED_DIR / "cleaned_tourism.csv"
train_path = PROCESSED_DIR / "train.csv"
test_path = PROCESSED_DIR / "test.csv"

clean_df.to_csv(clean_path, index=False)
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print("Processed files created:")
print(f"- {clean_path}")
print(f"- {train_path}")
print(f"- {test_path}")

dataset_repo_id = os.getenv("HF_DATASET_REPO_ID", "mahipalyenreddy/tourism-dataset")
hf_token = os.getenv("HF_TOKEN")

if HfApi and hf_token:
    api = HfApi(token=hf_token)
    for local_file, remote_name in [
        (clean_path, "data/processed/cleaned_tourism.csv"),
        (train_path, "data/processed/train.csv"),
        (test_path, "data/processed/test.csv"),
    ]:
        api.upload_file(
            path_or_fileobj=str(local_file),
            path_in_repo=remote_name,
            repo_id=dataset_repo_id,
            repo_type="dataset",
        )
    print("Processed datasets uploaded to Hugging Face.")
else:
    print("Hugging Face upload skipped. Add HF_TOKEN to enable it.")
