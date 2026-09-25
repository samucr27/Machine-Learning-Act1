"""
Clustering Application module - Activity 3, Part 2, Section 3
Topic: Mall Customer Segmentation (Annual Income vs. Spending Score)

TODO (Manuel): you own this file. The dataset is already connected (store_customers.csv,
1,000 records). What's left is mostly analysis/interpretation work, not wiring:
- Justify the choice of dataset/variables in clustering_application.html (this module
  only computes things, it doesn't write the narrative).
- Decide and justify the number of clusters (K) -- N_CLUSTERS below defaults to 4,
  change it if your analysis (e.g. elbow method) suggests a different value.
- Write the per-cluster interpretation (what each cluster represents) in the template,
  using CLUSTER_SUMMARY below as your data source.
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
N_CLUSTERS = 4  # TODO (Manuel): justify this number in the Application page (e.g. elbow method)


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

DATASET_INFO = {
    "n_records": N_RECORDS_AFTER,
    "n_records_raw": N_RECORDS_BEFORE,
    "n_dropped": N_RECORDS_BEFORE - N_RECORDS_AFTER,
    "independent_variables": [COL_X, COL_Y],
    "n_clusters": N_CLUSTERS,
    "silhouette_score": SILHOUETTE,
    "source": (
        "Kaggle \"Mall Customer Segmentation Dataset\" (hosseinbadrnezhad), "
        "file store_customers.csv."
    ),
}

CLUSTER_SUMMARY = []
for k in range(N_CLUSTERS):
    mask = CLUSTER_LABELS == k
    CLUSTER_SUMMARY.append({
        "cluster": k + 1,
        "count": int(mask.sum()),
        "centroid_income": round(float(CENTROIDS_ORIGINAL_SCALE[k, 0]), 2),
        "centroid_spending": round(float(CENTROIDS_ORIGINAL_SCALE[k, 1]), 2),
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