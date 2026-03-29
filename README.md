# Tourism MLOps Project

An end-to-end MLOps pipeline for predicting customer tourism package purchases, built with GitHub Actions for CI/CD.

## Business Context

"Visit with Us," a leading travel company, uses data-driven strategies to optimize operations and customer engagement.

## Hugging Face Repositories

- **Dataset**: https://huggingface.co/datasets/mahipalyenreddy/tourism-dataset
- **Model**: https://huggingface.co/models/mahipalyenreddy/tourism-purchase-model
- **Space**: https://huggingface.co/spaces/mahipalyenreddy/tourism-prediction-space

## Project Structure

```
tourism_cicd_project/
├── data/
│   ├── raw/
│   └── processed/
├── model_building/
│   ├── data_register.py
│   ├── prep.py
│   ├── train.py
│   ├── best_model.joblib
│   ├── best_model_metadata.json
│   └── experiment_tracking.csv
├── hosting/
│   └── hosting.py
├── pipeline.yml
├── requirements.txt
├── deployment/
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── .github/
│   └── workflows/
│       └── pipeline.yml
├── .gitignore
└── README.md
```

## Steps to Run

### 1. Clone the repository:
```bash
git clone https://github.com/mahipalyenreddy/tourism-cicd-project.git
cd tourism-cicd-project
```

### 2. Set up virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux
pip install -r requirements.txt
```

### 3. Prepare data:
```bash
python model_building/data_register.py
python model_building/prep.py
```

### 4. Train the model:
```bash
python model_building/train.py
```

### 5. Run the Streamlit app:
```bash
streamlit run deployment/app.py
```

## CI/CD Pipeline

The GitHub Actions workflow automatically triggers on push to the `main` branch:
- Installs dependencies
- Runs data preparation
- Runs model training
- Uploads model artifacts
- Deploys to Hugging Face Space (if HF_TOKEN secret is set)

## Secrets Configuration

Add the following secrets in GitHub Repository Settings > Secrets and variables > Actions:
- `HF_TOKEN`: Your Hugging Face access token
- `HF_DATASET_REPO_ID`: `mahipalyenreddy/tourism-dataset`
- `HF_MODEL_REPO_ID`: `mahipalyenreddy/tourism-purchase-model`
- `HF_SPACE_ID`: `mahipalyenreddy/tourism-prediction-space`

## Author

**mahipalyenreddy**

## License

MIT License
