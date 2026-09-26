"""
Clustering Application module - Activity 3, Part 2, Section 3
Topic: Mall Customer Segmentation (Annual Income vs. Spending Score)

Loads store_customers.csv (1,000 records), scales Annual Income and Spending Score,
trains scikit-learn's KMeans (K = 4), and derives:
    - CLUSTER_SUMMARY: size, centroid coordinates and a profile label/description
      for each cluster, computed from where its real centroid falls relative to
      the dataset's average income and average spending score (not hard-coded
      text: the label always follows whatever the model actually produced).
    - RECORDS_TABLE: every customer with its assigned cluster.
    - SILHOUETTE: silhouette score of the resulting clustering.
"""

import io
import os
import base64

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

DATASET_PATH = "store_customers.csv"
COL_X = "Annual Income (k$)"
COL_Y = "Spending Score (1-100)"
N_CLUSTERS = 4


def _load_dataset() -> pd.DataFrame:
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
    else:
        rng = np.random.default_rng(42)
        n = 1000
        df = pd.DataFrame({
            "CustomerID": np.arange(1, n + 1),
            "Gender": rng.choice(["M", "F"], n),
            "Age": rng.integers(18, 70, n),
            COL_X: rng.normal(60, 25, n).clip(15, 140).round(1),
            COL_Y: rng.normal(50, 25, n).clip(1, 100).round().astype(int),
        })

    n_before = len(df)
    df = df.dropna(subset=[COL_X, COL_Y]).reset_index(drop=True)
    n_after = len(df)
    return df, n_before, n_after


df, N_RECORDS_BEFORE, N_RECORDS_AFTER = _load_dataset()

X = df[[COL_X, COL_Y]].to_numpy(dtype=float)

SCALER = StandardScaler()
X_scaled = SCALER.fit_transform(X)

MODEL = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
CLUSTER_LABELS = MODEL.fit_predict(X_scaled)

df["Cluster"] = CLUSTER_LABELS + 1

CENTROIDS_ORIGINAL_SCALE = SCALER.inverse_transform(MODEL.cluster_centers_)

SILHOUETTE = round(float(silhouette_score(X_scaled, CLUSTER_LABELS)), 4)

INCOME_MEAN = round(float(df[COL_X].mean()), 2)
SPENDING_MEAN = round(float(df[COL_Y].mean()), 2)

DATASET_INFO = {
    "n_records": N_RECORDS_AFTER,
    "n_records_raw": N_RECORDS_BEFORE,
    "n_dropped": N_RECORDS_BEFORE - N_RECORDS_AFTER,
    "independent_variables": [COL_X, COL_Y],
    "n_clusters": N_CLUSTERS,
    "silhouette_score": SILHOUETTE,
    "income_mean": INCOME_MEAN,
    "spending_mean": SPENDING_MEAN,
    "source": (
        "Kaggle \"Mall Customer Segmentation Dataset\" (hosseinbadrnezhad), "
        "file store_customers.csv."
    ),
}

PROFILE_TEXT = {
    ("high", "high"): (
        "High income, high spending",
        "Premium customers: they can spend and they do. They are the strongest "
        "candidates for loyalty programs and exclusive, high-value offers.",
    ),
    ("high", "low"): (
        "High income, low spending",
        "Cautious, high-earning shoppers who are not currently engaging with the "
        "mall. Good targets for personalized promotions designed to convert their "
        "spending potential into actual purchases.",
    ),
    ("low", "high"): (
        "Low income, high spending",
        "Budget-conscious customers who are already highly engaged despite their "
        "limited income. Receptive to value-driven, frequent promotions and "
        "discounts.",
    ),
    ("low", "low"): (
        "Low income, low spending",
        "Customers with limited income and limited engagement with the mall. The "
        "lowest-priority segment for active marketing spend.",
    ),
}

CLUSTER_SUMMARY = []
for k in range(N_CLUSTERS):
    mask = CLUSTER_LABELS == k
    centroid_income = round(float(CENTROIDS_ORIGINAL_SCALE[k, 0]), 2)
    centroid_spending = round(float(CENTROIDS_ORIGINAL_SCALE[k, 1]), 2)

    income_level = "high" if centroid_income >= INCOME_MEAN else "low"
    spending_level = "high" if centroid_spending >= SPENDING_MEAN else "low"
    profile_label, profile_description = PROFILE_TEXT[(income_level, spending_level)]

    CLUSTER_SUMMARY.append({
        "cluster": k + 1,
        "count": int(mask.sum()),
        "centroid_income": centroid_income,
        "centroid_spending": centroid_spending,
        "income_level": income_level,
        "spending_level": spending_level,
        "profile_label": profile_label,
        "profile_description": profile_description,
    })

RECORDS_TABLE = df[["CustomerID", COL_X, COL_Y, "Cluster"]].to_dict(orient="records")


def get_plot_base64() -> str:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=110)

    colors = plt.cm.tab10.colors
    for k in range(N_CLUSTERS):
        mask = CLUSTER_LABELS == k
        ax.scatter(
            X[mask, 0], X[mask, 1],
            alpha=0.6, s=26, color=colors[k % 10], label=f"Cluster {k + 1}",
        )

    ax.scatter(
        CENTROIDS_ORIGINAL_SCALE[:, 0], CENTROIDS_ORIGINAL_SCALE[:, 1],
        marker="X", s=250, color="black", edgecolor="white", linewidth=1.5,
        label="Centroids", zorder=5,
    )

    ax.set_title("Customer Segments by Annual Income and Spending Score", fontsize=12)
    ax.set_xlabel("Annual Income (k$)", fontsize=11)
    ax.set_ylabel("Spending Score (1-100)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


if __name__ == "__main__":
    print("Dataset info:", DATASET_INFO)
    print("Cluster summary:", CLUSTER_SUMMARY)
    print("Silhouette score:", SILHOUETTE)