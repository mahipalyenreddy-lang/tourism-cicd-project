from pathlib import Path
import os

from huggingface_hub import HfApi

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEPLOYMENT_DIR = PROJECT_DIR / "deployment"

SPACE_REPO_ID = os.getenv("HF_SPACE_ID", "mahipalyenreddy/tourism-prediction-space")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is required for hosting upload.")

api = HfApi(token=HF_TOKEN)
api.create_repo(repo_id=SPACE_REPO_ID, repo_type="space", space_sdk="static", exist_ok=True)
api.upload_folder(
    folder_path=str(DEPLOYMENT_DIR),
    repo_id=SPACE_REPO_ID,
    repo_type="space",
)

print(f"Deployment folder pushed to https://huggingface.co/spaces/{SPACE_REPO_ID}")
