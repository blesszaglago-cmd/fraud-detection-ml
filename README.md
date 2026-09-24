# Fraud Detection using Machine Learning

A machine learning system that detects fraudulent credit card transactions with 98.26% ROC-AUC on real-world data.

## Live API

This project includes a FastAPI service that predicts fraud on new transactions.

**Local:** `uvicorn main:app --reload` → http://127.0.0.1:8000/docs

**Example request:**
```json
{"Time":0.0,"V1":-1.3598,"Amount":149.62}

## The Problem
> **Note:** The dataset (`creditcard.csv`, 150 MB) is not included in this repo due to GitHub's 100 MB file size limit. Download it from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in `data/` before running the training scripts.

> **API:** A live FastAPI service is available at `/predict` (see `main.py`). Deployed on Vercel.

Credit card fraud costs the global economy over $30 billion annually. Rule-based systems fail against new fraud patterns. This project uses supervised learning on 284,807 real transactions to build an adaptive detection system.

## Dataset

- **Source:** [Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions
- **Frauds:** 492 (0.172%)
- **Features:** 30 (V1–V28 PCA-transformed, Time, Amount)
- **Class imbalance:** 1 fraud per 578 normal transactions

## Results

| Metric | Score |
|---|---|
| **ROC-AUC** | **0.9826** |
| Precision (fraud) | High |
| Recall (fraud) | High |
| Model | Random Forest |

### Confusion Matrix (Test Set — 56,962 transactions)

|  | Predicted Normal | Predicted Fraud |
|---|---|---|
| **Actual Normal** | 56,824 | 40 |
| **Actual Fraud** | 16 | 82 |

- **82 frauds caught** out of 98 total
- **40 false alarms** out of 56,864 normals (0.07%)

## Approach

1. **Load** — Read raw CSV, explore shape and class distribution
2. **Preprocess** — Scale Time and Amount, stratified train/test split, SMOTE on training only
3. **Train** — Compare Logistic Regression and Random Forest
4. **Evaluate** — Precision, Recall, F1, ROC-AUC, confusion matrix
5. **Save** — Persist the best model and scaler
6. **Predict** — Load and predict on new transactions
7. **Visualize** — ROC curve, confusion matrix, feature importance

## Key Decisions

- **SMOTE on training set only** — no data leakage into test
- **Stratified split** — preserves the real-world imbalance in the test set
- **ROC-AUC over accuracy** — accuracy is misleading with 0.17% fraud
- **Random Forest over Logistic Regression** — won on ROC-AUC (0.9826)

## Tech Stack

- Python 3.10+
- pandas, NumPy
- scikit-learn
- imbalanced-learn (SMOTE)
- matplotlib, seaborn
- joblib

## File Structure
