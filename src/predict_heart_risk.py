import os
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

MODEL_PATH = os.path.join("models", "heart_risk_logreg.joblib")

def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    return joblib.load(model_path)

def make_prediction(model, patient_data: dict):
    """
    patient_data: dict with keys matching FEATURES
    returns (pred_class, prob_heart_disease)
    """
    #turning dict into one-row dataframe
    df_new = pd.DataFrame([patient_data])

    # ensure correct column order
    df_new = df_new[FEATURES]

    #predict class and probability
    pred_class = model.predict(df_new)[0]
    prob_heart_disease = model.predict_proba(df_new)[0,1]

    return pred_class, prob_heart_disease

if __name__ == "__main__":
    # Load the trained model
    model = load_model(MODEL_PATH)

    #example patient (made up values)

    example_patient = {
        "age_years": 52.0,
        "gender": 1,
        "cholesterol": 2,
        "height": 165,
        "weight": 72.0,
        "bmi": 72.0 / (1.65 ** 2),
        "ap_hi": 130,
        "ap_lo": 80,
        "gluc": 1,
        "smoke": 0,
        "alco": 0,
        "active": 1,
    }

    pred_class, prob = make_prediction(model, example_patient)

    print("Predicted class (cardio):", pred_class)
    print(f"Probability of heart disease: {prob:.3f}")
    