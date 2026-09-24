"""
preprocess.py

Prepares the fraud dataset for training:
- Scales Time and Amount
- Splits into stratified train/test sets
- Applies SMOTE to the training set only

Saves the processed arrays and the scaler.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
SCALER_PATH = MODELS_DIR / "scaler.pkl"


def load_data():
    """Load raw CSV."""
    return pd.read_csv(DATA_PATH)


def preprocess():
    """Full preprocessing pipeline. Returns X_train, X_test, y_train, y_test."""
    df = load_data()

    # Separate features and target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Scale Time and Amount (V1-V28 are already PCA-scaled)
    scaler = StandardScaler()
    X = X.copy()
    X[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])

    # Save the scaler in models/
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)

    # Stratified split — preserves fraud ratio in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("=" * 50)
    print("BEFORE SMOTE")
    print("=" * 50)
    print(f"Train size: {len(X_train):,}")
    print(f"  Normal: {(y_train == 0).sum():,}")
    print(f"  Fraud : {(y_train == 1).sum():,}")
    print(f"Test size:  {len(X_test):,}")
    print(f"  Normal: {(y_test == 0).sum():,}")
    print(f"  Fraud : {(y_test == 1).sum():,}")

    # Apply SMOTE ONLY to training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print("\n" + "=" * 50)
    print("AFTER SMOTE (training set only)")
    print("=" * 50)
    print(f"Train size: {len(X_train_resampled):,}")
    print(f"  Normal: {(y_train_resampled == 0).sum():,}")
    print(f"  Fraud : {(y_train_resampled == 1).sum():,}")

    # Save processed arrays
    np.save(PROCESSED_DIR / "X_train.npy", X_train_resampled)
    np.save(PROCESSED_DIR / "X_test.npy", X_test)
    np.save(PROCESSED_DIR / "y_train.npy", y_train_resampled)
    np.save(PROCESSED_DIR / "y_test.npy", y_test)

    print(f"\nProcessed data saved to: {PROCESSED_DIR}")
    print(f"Scaler saved to: {SCALER_PATH}")

    return X_train_resampled, X_test, y_train_resampled, y_test


if __name__ == "__main__":
    preprocess()