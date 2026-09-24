"""
predict.py

Loads the trained fraud model and predicts on new transactions.
Uses the same scaler that was fit during preprocessing.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "fraud_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"


def load_artifacts():
    """Load the trained model and the scaler."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def predict(transaction: dict):
    """
    Predict whether a transaction is fraud.

    transaction: dict with keys V1..V28, Time, Amount
    Returns: (label, probability)
    """
    model, scaler = load_artifacts()

    feature_order = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    df = pd.DataFrame([transaction])[feature_order]

    df[["Time", "Amount"]] = scaler.transform(df[["Time", "Amount"]])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return int(prediction), float(probability)


def main():
    raw = pd.read_csv(BASE_DIR / "data" / "creditcard.csv")

    print("=" * 50)
    print("FRAUD DETECTION DEMO")
    print("=" * 50)

    for label_name, label_value in [("NORMAL", 0), ("FRAUD", 1)]:
        sample = raw[raw["Class"] == label_value].iloc[0]
        transaction = sample.drop("Class").to_dict()

        pred, prob = predict(transaction)

        print(f"\nActual    : {label_name}")
        print(f"Predicted : {'FRAUD' if pred == 1 else 'NORMAL'}")
        print(f"Fraud probability: {prob:.4f}")


if __name__ == "__main__":
    main()