"""
main.py

FastAPI service that loads the trained fraud detection model
and exposes a /predict endpoint.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from pathlib import Path


# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "fraud_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"


# ---------- Input Schema ----------
class TransactionInput(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


# ---------- Load Artifacts at Startup ----------
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model or scaler: {e}")


# ---------- App ----------
app = FastAPI(
    title="Fraud Detection API",
    description="Predicts whether a credit card transaction is fraudulent.",
    version="1.0.0",
)


@app.get("/")
def read_root():
    return {
        "message": "Fraud Detection API is running.",
        "endpoint": "POST /predict with transaction data",
    }


@app.post("/predict")
def predict_fraud(transaction: TransactionInput):
    try:
        data = transaction.dict()

        feature_order = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        df = pd.DataFrame([data])[feature_order]

        # Scale Time and Amount (same scaler used at training)
        df[["Time", "Amount"]] = scaler.transform(df[["Time", "Amount"]])

        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        return {
            "is_fraud": bool(prediction),
            "fraud_probability": round(float(probability), 4),
            "status": "success",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))