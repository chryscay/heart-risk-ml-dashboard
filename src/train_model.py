import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/cardio_train.csv", sep=";")

#print(df.shape)
#df.info()
#print(df.describe())
#print(df.isnull().sum())

df["age_years"] = df["age"] / 365.25 
df["bmi"] = df["weight"] / (df["height"] / 100) ** 2
df = df.drop(columns=["id"])

print(df.head())

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
TARGET_COL = "cardio"

X = df[FEATURES]
y = df[TARGET_COL]

print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state=42
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000))
])
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:,1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print("Accuracy:", acc)
print("ROC AUC:", auc)