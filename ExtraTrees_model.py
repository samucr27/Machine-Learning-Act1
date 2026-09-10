
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

