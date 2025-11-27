import os
from pathlib import Path

import joblib
import pandas as pd

FEATURES = [
    "age_years",
    "gender",
    "cholesterol",
    "height",
    "weight",
    "bmi",
    "ap_hi",
    "ap_lo",
    "gluc",
    "smoke",
    "alco",
    "active"
]

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "heart_risk_logreg.joblib"

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

def predict_from_features(model, patient_data: dict):
    """
    patient_data: dict with keys matching FEATURES
    returns (pred_class, prob_heart_disease)
    """
    df_new = pd.DataFrame([patient_data])
    df_new = df_new[FEATURES]

    pred_class = int(model.predict(df_new)[0])
    prob_heart_disease = float(model.predict_proba(df_new)[0,1])

    return pred_class, prob_heart_disease