"""
Logistic Regression module - Heart Disease Risk Prediction
Topic: cholesterol level (X) -> heart disease risk, Low (0) / High (1) (Y)

TODO (Manuel): replace DATASET_PATH with the real Kaggle CSV once downloaded,
and update the column names below (COL_X / COL_Y) to match that file.
Until then, this module generates a synthetic placeholder dataset so the
rest of the team (and the deployed app) is never broken while you work.
"""

import io
import os
import base64

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score,
)

DATASET_PATH = "logreg_dataset.csv"   # TODO: point to the real Kaggle CSV
COL_X = "cholesterol"                 # TODO: match the real column name
COL_Y = "risk"                        # TODO: match the real column name (0/1)


def _load_dataset() -> pd.DataFrame:
    if os.path.exists(DATASET_PATH):
        return pd.read_csv(DATASET_PATH)

    # ---- Synthetic placeholder dataset (remove once the real CSV is in place) ----
    rng = np.random.default_rng(42)
    n = 500
    cholesterol = rng.normal(210, 35, n).clip(120, 340)
    prob_high = 1 / (1 + np.exp(-(cholesterol - 220) / 15))
    risk = (rng.random(n) < prob_high).astype(int)
    return pd.DataFrame({COL_X: cholesterol, COL_Y: risk})


df = _load_dataset()

X = df[[COL_X]]
y = df[COL_Y]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

MODEL = LogisticRegression()
MODEL.fit(X_train, y_train)

_y_pred_test = MODEL.predict(X_test)

DATASET_INFO = {
    "n_records": len(df),
    "n_train": len(X_train),
    "n_test": len(X_test),
    "independent_variable": "Cholesterol Level",
    "independent_unit": "mg/dL",
    "dependent_variable": "Heart Disease Risk",
    "classes": "0 = Low Risk, 1 = High Risk",
    "source": "TODO (Manuel): cite the real Kaggle dataset here once selected.",
}

EVAL_METRICS = {
    "confusion_matrix": confusion_matrix(y_test, _y_pred_test).tolist(),
    "accuracy": round(accuracy_score(y_test, _y_pred_test), 4),
    "precision": round(precision_score(y_test, _y_pred_test), 4),
    "recall": round(recall_score(y_test, _y_pred_test), 4),
    "f1_score": round(f1_score(y_test, _y_pred_test), 4),
}


def get_plot_base64() -> str:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=110)

    for cls, color, label in [(0, "#4C72B0", "Low Risk"), (1, "#C44E52", "High Risk")]:
        subset = df[df[COL_Y] == cls]
        ax.scatter(
            subset[COL_X], subset[COL_Y],
            alpha=0.5, s=16, color=color, label=label
        )

    ax.set_title("Heart Disease Risk vs. Cholesterol Level (Logistic Regression)", fontsize=12)
    ax.set_xlabel("Cholesterol Level (mg/dL)", fontsize=11)
    ax.set_ylabel("Risk Class (0 = Low, 1 = High)", fontsize=11)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode("utf-8")


def predict_risk(cholesterol: float) -> dict:
    proba = MODEL.predict_proba(pd.DataFrame({COL_X: [cholesterol]}))[0]
    predicted_class = int(MODEL.predict(pd.DataFrame({COL_X: [cholesterol]}))[0])
    return {
        "class": predicted_class,
        "label": "High Risk" if predicted_class == 1 else "Low Risk",
        "probability": round(float(proba[predicted_class]), 4),
    }


if __name__ == "__main__":
    print("Dataset info:", DATASET_INFO)
    print("Eval metrics:", EVAL_METRICS)
    print("Prediction for 250 mg/dL:", predict_risk(250))
    print("Prediction for 180 mg/dL:", predict_risk(180))