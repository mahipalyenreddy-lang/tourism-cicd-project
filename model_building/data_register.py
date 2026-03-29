from pathlib import Path
import os

from huggingface_hub import HfApi

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_DIR / "data" / "raw" / "tourism.csv"

DATASET_REPO_ID = os.getenv("HF_DATASET_REPO_ID", "mahipalyenreddy/tourism-dataset")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is required to register the dataset on Hugging Face.")

api = HfApi(token=HF_TOKEN)
api.create_repo(repo_id=DATASET_REPO_ID, repo_type="dataset", exist_ok=True)
api.upload_file(
    path_or_fileobj=str(RAW_FILE),
    path_in_repo="data/raw/tourism.csv",
    repo_id=DATASET_REPO_ID,
    repo_type="dataset",
)

print(f"Raw dataset uploaded to https://huggingface.co/datasets/{DATASET_REPO_ID}")
