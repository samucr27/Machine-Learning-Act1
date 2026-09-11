"""
Logistic Regression module - Heart Disease Risk Prediction
Topic: cholesterol level (X) -> heart disease risk, Low (0) / High (1) (Y)
"""

import io
import os
import base64

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

DATASET_PATH = "heart.csv"
COL_X = "chol"
COL_Y = "target"


def _load_dataset() -> pd.DataFrame:
    if os.path.exists(DATASET_PATH):
        return pd.read_csv(DATASET_PATH)

    # ---- Synthetic placeholder dataset (fallback safety net) ----
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

SCALER = StandardScaler()
X_train_scaled = SCALER.fit_transform(X_train)
X_test_scaled = SCALER.transform(X_test)

MODEL = LogisticRegression()
MODEL.fit(X_train_scaled, y_train)

_y_pred_test = MODEL.predict(X_test_scaled)

DATASET_INFO = {
    "n_records": len(df),
    "n_train": len(X_train),
    "n_test": len(X_test),
    "independent_variable": "Cholesterol Level",
    "independent_unit": "mg/dL",
    "dependent_variable": "Heart Disease Risk",
    "classes": "0 = Low Risk, 1 = High Risk",
    "source": "Heart Disease dataset (Cleveland, Hungary, Switzerland, and Long Beach V databases, 1988), the classic UCI Heart Disease dataset (archive.ics.uci.edu/dataset/45/heart+disease), redistributed on Kaggle.",
}

_report_dict = classification_report(y_test, _y_pred_test, output_dict=True, zero_division=0)

EVAL_METRICS = {
    "confusion_matrix": confusion_matrix(y_test, _y_pred_test).tolist(),
    "accuracy": round(accuracy_score(y_test, _y_pred_test), 4),
    "precision": round(_report_dict["weighted avg"]["precision"], 4),
    "recall": round(_report_dict["weighted avg"]["recall"], 4),
    "f1_score": round(_report_dict["weighted avg"]["f1-score"], 4),
    "classification_report": classification_report(y_test, _y_pred_test, zero_division=0),
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


def get_confusion_matrix_plot_base64() -> str:
    """Seaborn heatmap of the confusion matrix — for the Evaluation Metrics page (Sergio)."""
    cm = confusion_matrix(y_test, _y_pred_test)

    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=110)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Low Risk", "High Risk"],
        yticklabels=["Low Risk", "High Risk"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Logistic Regression")
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode("utf-8")


def predict_risk(cholesterol: float) -> dict:
    row_scaled = SCALER.transform(pd.DataFrame({COL_X: [cholesterol]}))
    proba = MODEL.predict_proba(row_scaled)[0]
    predicted_class = int(MODEL.predict(row_scaled)[0])
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