"""
visualize.py

Creates three charts:
1. Confusion matrix
2. ROC curve
3. Feature importance (Random Forest only)

Saves them to reports/
"""

import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_PATH = BASE_DIR / "models" / "fraud_model.pkl"
REPORTS_DIR = BASE_DIR / "reports"


def main():
    X_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")
    model = joblib.load(MODEL_PATH)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    # 1. Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Normal", "Fraud"],
        yticklabels=["Normal", "Fraud"],
    )
    plt.title("Confusion Matrix — Fraud Detection", fontweight="bold")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    # 2. ROC curve
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="#2E86AB", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "--", color="#999999")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Fraud Detection", fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "roc_curve.png", dpi=150)
    plt.close()

    # 3. Feature importance (Random Forest only)
    if hasattr(model, "feature_importances_"):
        feature_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        importances = model.feature_importances_

        # Top 10
        indices = np.argsort(importances)[-10:]
        top_features = [feature_names[i] for i in indices]
        top_values = importances[indices]

        plt.figure(figsize=(9, 6))
        plt.barh(top_features, top_values, color="#2E86AB")
        plt.xlabel("Importance")
        plt.title("Top 10 Most Important Features", fontweight="bold")
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "feature_importance.png", dpi=150)
        plt.close()

    print("Charts saved to:", REPORTS_DIR)
    print("  - confusion_matrix.png")
    print("  - roc_curve.png")
    if hasattr(model, "feature_importances_"):
        print("  - feature_importance.png")


if __name__ == "__main__":
    main()