
import io
import base64

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

DATASET_PATH = "heart.csv"

COL_CHOL = "chol"
COL_AGE = "age"
COL_TRESTBPS = "trestbps"
COL_THALACH = "thalach"
COL_Y = "target"

FEATURE_COLS = [COL_CHOL, COL_AGE, COL_TRESTBPS, COL_THALACH]

CLASS_LABELS = {0: "Low Risk", 1: "High Risk"}


df = pd.read_csv(DATASET_PATH)

X = df[FEATURE_COLS]
y = df[COL_Y]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)



SCALER = StandardScaler()
X_train_scaled = SCALER.fit_transform(X_train)
X_test_scaled = SCALER.transform(X_test)
 
MODEL = ExtraTreesClassifier(n_estimators=300, random_state=42)
MODEL.fit(X_train_scaled, y_train)
 
_y_pred_test = MODEL.predict(X_test_scaled)
 
DATASET_INFO = {
    "n_records": len(df),
    "n_train": len(X_train),
    "n_test": len(X_test),
    "independent_variables": [
        "Cholesterol (chol) : serum cholesterol in mg/dL",
        "Age (age) : in years",
        "Resting Blood Pressure (trestbps) : in mm Hg",
        "Maximum Heart Rate Achieved (thalach) : in bpm",
    ],
    "dependent_variable": "Heart Disease Risk (target)",
    "classes": "0 = Low Risk, 1 = High Risk",
    "source": "Heart Disease dataset (Cleveland, Hungary, Switzerland, and Long Beach V "
              "databases, 1988) the classic UCI Heart Disease dataset "
              "(archive.ics.uci.edu/dataset/45/heart+disease), widely redistributed on Kaggle. "
              "Only the 4 variables used by this model are kept in heart.csv.",
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
 
    for cls, color in [(0, "#4C72B0"), (1, "#C44E52")]:
        subset = df[df[COL_Y] == cls]
        ax.scatter(
            subset[COL_CHOL], subset[COL_AGE],
            alpha=0.5, s=16, color=color, label=CLASS_LABELS[cls]
        )
 
    ax.set_title("Heart Disease Risk by Cholesterol and Age", fontsize=12)
    ax.set_xlabel("Cholesterol (mg/dL)", fontsize=11)
    ax.set_ylabel("Age (years)", fontsize=11)
    ax.legend(title="Class")
    ax.grid(alpha=0.25)
    fig.tight_layout()
 
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
 
    return base64.b64encode(buffer.read()).decode("utf-8")
 

def get_confusion_matrix_plot_base64() -> str:
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
    ax.set_title("Confusion Matrix — Extra Trees Classifier")
    fig.tight_layout()
 
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
 
    return base64.b64encode(buffer.read()).decode("utf-8")
 
 

def predict_class(chol: float, age: float, trestbps: float, thalach: float) -> dict:
    row = pd.DataFrame({
        COL_CHOL: [chol],
        COL_AGE: [age],
        COL_TRESTBPS: [trestbps],
        COL_THALACH: [thalach],
    })
    row_scaled = SCALER.transform(row)
    predicted_class = int(MODEL.predict(row_scaled)[0])
    proba = MODEL.predict_proba(row_scaled)[0]
    return {
        "class": predicted_class,
        "label": CLASS_LABELS[predicted_class],
        "probability": round(float(proba[predicted_class]), 4),
    }
 
 
if __name__ == "__main__":
    print("Dataset info:", DATASET_INFO)
    print("Eval metrics:", {k: v for k, v in EVAL_METRICS.items() if k != "classification_report"})
    print("Prediction (chol=280, age=60, trestbps=150, thalach=120):",
          predict_class(280, 60, 150, 120))
    print("Prediction (chol=180, age=35, trestbps=110, thalach=170):",
          predict_class(180, 35, 110, 170))