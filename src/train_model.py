"""
train_model.py

Trains two models on the processed fraud data:
1. Logistic Regression (baseline)
2. Random Forest (stronger)

Evaluates both with precision, recall, F1, and ROC-AUC.
Saves the best model to models/fraud_model.pkl
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "fraud_model.pkl"


def load_processed():
    """Load preprocessed arrays."""
    X_train = np.load(PROCESSED_DIR / "X_train.npy")
    X_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")
    return X_train, X_test, y_train, y_test


def evaluate(name, model, X_test, y_test):
    """Print evaluation metrics for a model."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    print("\n" + "=" * 50)
    print(f"{name.upper()} RESULTS")
    print("=" * 50)
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print(f"\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=["Normal", "Fraud"]))

    return roc_auc


def main():
    X_train, X_test, y_train, y_test = load_processed()

    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples:     {len(X_test):,}")

    # Model 1: Logistic Regression
    print("\n" + "=" * 50)
    print("TRAINING LOGISTIC REGRESSION...")
    print("=" * 50)
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_train)
    log_auc = evaluate("Logistic Regression", log_reg, X_test, y_test)

    # Model 2: Random Forest
    print("\n" + "=" * 50)
    print("TRAINING RANDOM FOREST...")
    print("=" * 50)
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    rf_auc = evaluate("Random Forest", rf, X_test, y_test)

    # Pick the best
    if rf_auc >= log_auc:
        best_model = rf
        best_name = "Random Forest"
        best_auc = rf_auc
    else:
        best_model = log_reg
        best_name = "Logistic Regression"
        best_auc = log_auc

    # Save it
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    print("\n" + "=" * 50)
    print("FINAL DECISION")
    print("=" * 50)
    print(f"Best model : {best_name}")
    print(f"ROC-AUC    : {best_auc:.4f}")
    print(f"Saved to   : {MODEL_PATH}")


if __name__ == "__main__":
    main()