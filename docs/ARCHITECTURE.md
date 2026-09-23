# Architecture

## Tech Stack
- Python 3.10+
- pandas, NumPy
- scikit-learn
- imbalanced-learn (SMOTE)
- matplotlib, seaborn
- joblib

## File Structure
fraud-detection-ml/
├── docs/
├── data/
│   └── creditcard.csv
├── models/
│   └── fraud_model.pkl
├── reports/
├── src/
│   ├── load_data.py
│   ├── preprocess.py
│   ├── train_model.py
│   ├── predict.py
│   └── visualize.py
└── README.md

## Flow
1. `load_data.py` — read CSV, print summary
2. `preprocess.py` — scale Amount and Time, split data, apply SMOTE
3. `train_model.py` — train models, evaluate, save best
4. `predict.py` — load model, predict on new input
5. `visualize.py` — generate charts for reports

## Data
- 284,807 transactions
- 492 frauds (0.172%)
- 30 features (V1-V28 PCA, Time, Amount)
- Target: Class (0 = normal, 1 = fraud)