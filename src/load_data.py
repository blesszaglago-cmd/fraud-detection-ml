"""
load_data.py

Loads the credit card fraud dataset and prints a summary.
"""

import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"


def load_data():
    """Load the dataset and return a DataFrame."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    return df


def main():
    df = load_data()

    print("=" * 50)
    print("CREDIT CARD FRAUD DATASET")
    print("=" * 50)
    print(f"\nRows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")
    print(f"\nColumns: {list(df.columns)}")

    print(f"\nClass distribution:")
    counts = df["Class"].value_counts()
    for label, count in counts.items():
        name = "Fraud" if label == 1 else "Normal"
        pct = (count / len(df)) * 100
        print(f"  {name:8s}: {count:,} ({pct:.3f}%)")

    print(f"\nFirst 5 rows:")
    print(df.head())

    print(f"\nBasic stats:")
    print(df[["Time", "Amount"]].describe())


if __name__ == "__main__":
    main()