import os
from typing import Dict, Any

from openai import OpenAI
from model_utils import load_model, predict_from_features, FEATURES

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

QUESTION_ORDER = [
    ("age_years", "Ask the user how old they are in years."),
    ("gender", "Ask the user for their biological sex assigned at birth as male or female."),
    ("height", "Ask the user for their height in centimeters."),
    ("weight", "Ask the user for their weight in kilograms."),
    ("cholesterol", "Ask the user if their cholesterol level is normal, above normal, or well above normal"),
    ("ap_hi", "Ask the user for their usual systolic blood pressure (the top number, e.g., 120)."),
    ("ap_lo", "Ask the user for their usual diastolic blood pressure (the bottom number, e.g., 80)."),
    ("gluc", "Ask if their blood sugar/glucose is normal, above normal, or well above normal."),
    ("smoke", "Ask if they currently smoke (yes or no)."),
    ("alco", "Ask if they regularly drink alcohol (yes or no)."),
    ("active", "Ask if they are physically active at least a few days per week (yes or no)."),
]
def approx_bp_from_category(category: str):
    """
    Map a rough category (low/normal/high) to approximate systolic/diastolic values.
    These are just rough values for the model, not medical measurements.
    """
    category = category.strip().lower()
    if "high" in category:
        return 140.0, 90.0
    elif "low" in category:
        return 100.0, 60.0
    else:
        #treat anything else as "normal"
        return 120.0, 80.0
    
def ask_openai(instruction: str) -> str:
    """
    Use OpenAI to turn a short instruction into a natural-language question.
    Example instruction: "Ask the user how old they are in years."
    """
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a concise, friendly health assistant. "
                    "Ask the user a single clear question, no follow-up, "
                    "and do not add explanations."
                ),
            },
            {"role": "user", "content": instruction},
        ],
    )
    return resp.choices[0].message.content.strip()

def explain_result(prob: float) -> str:
    """
    Ask OpenAI to explain the model's probability in simple language.
    """
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a health communication assistant. "
                    "You are NOT a doctor and you must remind the user that this is not medical advice. "
                    "Explain the risk gently and clearly in 3-5 sentences with no fearmongering."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"The predicted probability of heart disease from a model is {prob:.3f}. "
                    "Explain what this means for a layperson, and suggest next steps "
                    "like talking to a doctor or improving lifestyle."
                ),
            },
        ],

    )
    return resp.choices[0].message.content.strip()

def parse_yes_no(ans: str) -> int:
    """
    Turn yes/no style answers into 1/0.
    """
    ans = ans.strip().lower()
    if ans in ["yes", "y", "yeah", "yep"]:
        return 1
    return 0

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set in environment.")
        print("Run: export OPENAI_API_KEY='your-key-here'")
        return
    
    # Load trained ML model
    model = load_model()
    answers: Dict[str, Any] = {}

    print("Welcome! I'll ask you a few questions to estimate your heart-risk using a machine learning model.")
    print("This is NOT medical advice. Please talk to a doctor for real medical guidance.\n")

    for feature, instruction in QUESTION_ORDER:
        #use open ai to phrase question nicely
        question = ask_openai(instruction)
        user_ans = input(question + " ")

        #if we've already set blood pressure from a category skip asking again for ap_lo
        if feature == "ap_lo" and "ap_lo" in answers and "ap_hi" in answers:
            continue

        if feature in ["age_years", "height", "weight", "ap_hi", "ap_lo"]:
            txt = user_ans.strip().lower()

            if feature in ["ap_hi", "ap_lo"] and any (
                k in txt for k in ["idk", "don't know", "dont know", "not sure", "no idea"]
            ):
                follow = input(
                    "If you had to guess, would you say your blood pressure is usually low, normal, or high? "
                )
                hi_val, lo_val = approx_bp_from_category(follow)
                answers["ap_hi"] = hi_val
                answers["ap_lo"] = lo_val
                print(
                    f"Okay, I'll approximate your blood pressure as about {hi_val:.0f}/{lo_val:.0f}. "
                    "This is just for the model and not medical advice.\n"
                )
                continue
            try:
                answers[feature] = float(user_ans)
            except ValueError:
                print("Couldn't parse that as a number, defaulting to 0. In a real app, you'd re-ask.")
                answers[feature] = 0.0

        elif feature == "gender":
            val = user_ans.strip().lower()
            #adjust mapping if dataset encodes gender differently
            if val.startswith("m"):
                answers[feature] = 1
            else:
                answers[feature] = 2

        elif feature in ["cholesterol", "gluc"]:
            val = user_ans.strip().lower()
            if "well above" in val:
                answers[feature] = 3
            elif "above" in val:
                answers[feature] = 2
            else:
                answers[feature] = 1
        elif feature in ["smoke", "alco", "active"]:
            answers[feature] = parse_yes_no(user_ans)

    #compute BMI from height and weight 
    height_m = answers["height"] / 100.0 if answers["height"] else 0.0
    if height_m > 0:
        bmi = answers["weight"] / (height_m ** 2)
    else:
        bmi = 0.0

    answers["bmi"] = bmi

    # build exact patient data dict in correct order
    patient_data = {f: answers[f] for f in FEATURES}

    # run model prediction
    pred_class, prob = predict_from_features(model, patient_data)

    print(f"\nRaw model output -> class: {pred_class}, probability: {prob:.3f}\n")


    #get a friendly explanation from OpenAI
    explanation = explain_result(prob)
    print(explanation)
    print("\nThanks for trying the demo! Remember, this is not medical advice.")

if __name__ == "__main__":
    main()
        