"""
Extra Trees Classifier module - Activity 2, Part 2
Topic and variables: TODO (Jonathan) - define with the team, needs 3+ independent variables.

TODO (Jonathan): replace DATASET_PATH with the real CSV once downloaded, and
update COL_X1 / COL_X2 / COL_X3 / COL_Y to match your chosen dataset's columns
(add more COL_X* if you use more than 3 variables). Until then, this module
generates a synthetic placeholder dataset so the rest of the team (and the
deployed app) is never broken while you work.
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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

DATASET_PATH = "et_dataset.csv"   # TODO: point to the real CSV
COL_X1 = "feature_1"              # TODO: match the real column name
COL_X2 = "feature_2"              # TODO: match the real column name
COL_X3 = "feature_3"              # TODO: match the real column name
COL_Y = "target"                  # TODO: match the real column name (0/1)

FEATURE_COLS = [COL_X1, COL_X2, COL_X3]


def _load_dataset() -> pd.DataFrame:
    if os.path.exists(DATASET_PATH):
        return pd.read_csv(DATASET_PATH)

    # ---- Synthetic placeholder dataset (remove once the real CSV is in place) ----
    rng = np.random.default_rng(7)
    n = 500
    f1 = rng.normal(50, 12, n)
    f2 = rng.normal(30, 8, n)
    f3 = rng.normal(100, 20, n)
    score = 0.05 * f1 + 0.08 * f2 - 0.03 * f3 + rng.normal(0, 2, n)
    target = (score > np.median(score)).astype(int)
    return pd.DataFrame({COL_X1: f1, COL_X2: f2, COL_X3: f3, COL_Y: target})


df = _load_dataset()

X = df[FEATURE_COLS]
y = df[COL_Y]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling (as shown in class). Tree-based models like Extra Trees don't
# strictly need it, but it's kept for consistency with the rest of the project.
SCALER = StandardScaler()
X_train_scaled = SCALER.fit_transform(X_train)
X_test_scaled = SCALER.transform(X_test)

MODEL = ExtraTreesClassifier(random_state=42)
MODEL.fit(X_train_scaled, y_train)

_y_pred_test = MODEL.predict(X_test_scaled)

DATASET_INFO = {
    "n_records": len(df),
    "n_train": len(X_train),
    "n_test": len(X_test),
    "independent_variables": FEATURE_COLS,
    "dependent_variable": "Target Class",
    "classes": "0 = Class A, 1 = Class B",  # TODO (Jonathan): rename to match your topic
    "source": "TODO (Jonathan): cite the real dataset here once selected.",
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

    for cls, color, label in [(0, "#4C72B0", "Class A"), (1, "#C44E52", "Class B")]:
        subset = df[df[COL_Y] == cls]
        ax.scatter(
            subset[COL_X1], subset[COL_X2],
            alpha=0.5, s=16, color=color, label=label
        )

    ax.set_title("Extra Trees Classifier — Feature 1 vs. Feature 2 by Class", fontsize=12)
    ax.set_xlabel(COL_X1, fontsize=11)
    ax.set_ylabel(COL_X2, fontsize=11)
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
        xticklabels=["Class A", "Class B"],
        yticklabels=["Class A", "Class B"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Extra Trees Classifier")
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode("utf-8")


def predict_class(x1: float, x2: float, x3: float) -> dict:
    row_scaled = SCALER.transform(pd.DataFrame({COL_X1: [x1], COL_X2: [x2], COL_X3: [x3]}))
    predicted_class = int(MODEL.predict(row_scaled)[0])
    proba = MODEL.predict_proba(row_scaled)[0]
    return {
        "class": predicted_class,
        "label": "Class B" if predicted_class == 1 else "Class A",
        "probability": round(float(proba[predicted_class]), 4),
    }


if __name__ == "__main__":
    print("Dataset info:", DATASET_INFO)
    print("Eval metrics:", EVAL_METRICS)
    print("Prediction:", predict_class(55, 32, 95))