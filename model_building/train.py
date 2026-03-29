from pathlib import Path
import json
import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except Exception:
    mlflow = None
    MLFLOW_AVAILABLE = False

try:
    from huggingface_hub import HfApi
except Exception:
    HfApi = None

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "model_building"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
test_df = pd.read_csv(PROCESSED_DIR / "test.csv")

target_column = "ProdTaken"
X_train = train_df.drop(columns=[target_column])
y_train = train_df[target_column]
X_test = test_df.drop(columns=[target_column])
y_test = test_df[target_column]

numeric_features = X_train.select_dtypes(include="number").columns.tolist()
categorical_features = X_train.select_dtypes(exclude="number").columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_features,
        ),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            categorical_features,
        ),
    ]
)

model_grid = {
    "DecisionTree": (
        DecisionTreeClassifier(random_state=42),
        {"model__max_depth": [6, None], "model__min_samples_split": [2, 10]},
    ),
    "RandomForest": (
        RandomForestClassifier(random_state=42),
        {
            "model__n_estimators": [100],
            "model__max_depth": [None, 10],
            "model__min_samples_split": [2, 5],
        },
    ),
    "GradientBoosting": (
        GradientBoostingClassifier(random_state=42),
        {
            "model__n_estimators": [100],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
        },
    ),
}

results = []
best_score = -1
best_model = None
best_name = None
best_params = None
best_metrics = None

if MLFLOW_AVAILABLE:
    mlflow.set_experiment("tourism_mlops_experiment")

for model_name, (model, param_grid) in model_grid.items():
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    search = GridSearchCV(pipeline, param_grid=param_grid, cv=3, scoring="f1", n_jobs=1)
    search.fit(X_train, y_train)

    candidate = search.best_estimator_
    preds = candidate.predict(X_test)
    probs = candidate.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "best_params": json.dumps(search.best_params_),
        "cv_best_f1": round(search.best_score_, 4),
        "test_accuracy": round(accuracy_score(y_test, preds), 4),
        "test_precision": round(precision_score(y_test, preds), 4),
        "test_recall": round(recall_score(y_test, preds), 4),
        "test_f1": round(f1_score(y_test, preds), 4),
        "test_roc_auc": round(roc_auc_score(y_test, probs), 4),
    }
    results.append(metrics)

    if MLFLOW_AVAILABLE:
        with mlflow.start_run(run_name=model_name):
            mlflow.log_params(search.best_params_)
            for key, value in metrics.items():
                if key not in {"model", "best_params"}:
                    mlflow.log_metric(key, value)

    if search.best_score_ > best_score:
        best_score = search.best_score_
        best_model = candidate
        best_name = model_name
        best_params = search.best_params_
        best_metrics = metrics

results_df = pd.DataFrame(results).sort_values(by=["test_f1", "test_roc_auc"], ascending=False)
results_path = MODEL_DIR / "experiment_tracking.csv"
results_df.to_csv(results_path, index=False)

model_path = MODEL_DIR / "best_model.joblib"
joblib.dump(best_model, model_path)

metadata = {
    "best_model_name": best_name,
    "best_params": best_params,
    "test_metrics": best_metrics,
    "target_column": target_column,
}
metadata_path = MODEL_DIR / "best_model_metadata.json"
metadata_path.write_text(json.dumps(metadata, indent=2))

print("Experiment tracking saved to:", results_path)
print("Best model saved to:", model_path)
print("Metadata saved to:", metadata_path)

model_repo_id = os.getenv("HF_MODEL_REPO_ID", "mahipalyenreddy/tourism-purchase-model")
hf_token = os.getenv("HF_TOKEN")
if HfApi and hf_token:
    api = HfApi(token=hf_token)
    api.create_repo(repo_id=model_repo_id, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(model_path),
        path_in_repo="best_model.joblib",
        repo_id=model_repo_id,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj=str(metadata_path),
        path_in_repo="best_model_metadata.json",
        repo_id=model_repo_id,
        repo_type="model",
    )
    print("Best model uploaded to Hugging Face model hub.")
else:
    print("Model upload skipped. Add HF_TOKEN to enable it.")
