from model_utils import load_model, predict_from_features, FEATURES

if __name__ == "__main__":
    # Load the trained model
    model = load_model()

    # Example patient (made up values)
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

    pred_class, prob = predict_from_features(model, example_patient)

    print("Predicted class (cardio):", pred_class)
    print(f"Probability of heart disease: {prob:.3f}")
