from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_DIR / "model_building"
DATA_DIR = PROJECT_DIR / "data" / "processed"

model = joblib.load(MODEL_DIR / "best_model.joblib")
metadata = json.loads((MODEL_DIR / "best_model_metadata.json").read_text())
reference_df = pd.read_csv(DATA_DIR / "cleaned_tourism.csv")

st.set_page_config(page_title="Tourism Purchase Prediction", layout="wide")
st.title("Tourism Purchase Prediction")
st.write("Enter customer details to predict whether the customer is likely to purchase the tourism package.")

features = [column for column in reference_df.columns if column != metadata["target_column"]]
inputs = {}
for column in features:
    series = reference_df[column]
    if pd.api.types.is_numeric_dtype(series):
        inputs[column] = st.number_input(column, value=float(series.median()))
    else:
        options = sorted(series.astype(str).unique().tolist())
        inputs[column] = st.selectbox(column, options=options, index=0)

input_df = pd.DataFrame([inputs])
prediction = int(model.predict(input_df)[0])
probability = float(model.predict_proba(input_df)[0, 1])

st.subheader("Prediction Output")
st.write({"predicted_class": prediction, "purchase_probability": round(probability, 4)})
