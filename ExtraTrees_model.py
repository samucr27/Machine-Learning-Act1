
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
        "Cholesterol (chol) — serum cholesterol in mg/dL",
        "Age (age) — in years",
        "Resting Blood Pressure (trestbps) — in mm Hg",
        "Maximum Heart Rate Achieved (thalach) — in bpm",
    ],
    "dependent_variable": "Heart Disease Risk (target)",
    "classes": "0 = Low Risk, 1 = High Risk",
    "source": "Heart Disease dataset (Cleveland, Hungary, Switzerland, and Long Beach V "
              "databases, 1988) — the classic UCI Heart Disease dataset "
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