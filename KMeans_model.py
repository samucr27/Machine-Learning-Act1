"""
Clustering Application module - Activity 3, Part 2, Section 3
Topic: Mall Customer Segmentation (Annual Income vs. Spending Score)

Pipeline:
    CSV Dataset -> Pandas -> Drop Nulls -> Feature Scaling -> K-Means Training
    -> Cluster Assignment -> Evaluation -> Visualization

Configuration: KMeans(n_clusters=4, random_state=42, n_init=10)
    - n_clusters=4: selected with the elbow method (see K_SELECTION below).
    - random_state=42: fixed initialization, so the same clusters are obtained
      every time the application starts.
    - n_init=10: the algorithm runs 10 times with different initial centroids
      (k-means++) and keeps the solution with the lowest inertia.
"""

import io
import os
import base64

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "store_customers.csv")

COL_X = "Annual Income (k$)"
COL_Y = "Spending Score (1-100)"

N_CLUSTERS = 4
RANDOM_STATE = 42
N_INIT = 10
K_RANGE = range(2, 7)


# ---------------------------------------------------------------------------
# 1. Load and clean the dataset
# ---------------------------------------------------------------------------
def _load_dataset():
    """Load store_customers.csv and remove records with null values in the clustering variables."""
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    n_before = len(df)
    df = df.dropna(subset=[COL_X, COL_Y]).reset_index(drop=True)
    n_after = len(df)
    return df, n_before, n_after


df, N_RECORDS_BEFORE, N_RECORDS_AFTER = _load_dataset()

X = df[[COL_X, COL_Y]].to_numpy(dtype=float)

# ---------------------------------------------------------------------------
# 2. Feature scaling
# ---------------------------------------------------------------------------
SCALER = StandardScaler()
X_scaled = SCALER.fit_transform(X)

# ---------------------------------------------------------------------------
# 3. Selection of K: inertia (elbow method) and silhouette score for K = 2..6
# ---------------------------------------------------------------------------
K_SELECTION = []
_previous_inertia = None
for k in K_RANGE:
    candidate = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
    labels = candidate.fit_predict(X_scaled)
    inertia = float(candidate.inertia_)
    K_SELECTION.append({
        "k": k,
        "inertia": round(inertia, 2),
        "reduction": None if _previous_inertia is None else round(_previous_inertia - inertia, 2),
        "silhouette": round(float(silhouette_score(X_scaled, labels)), 4),
        "selected": k == N_CLUSTERS,
    })
    _previous_inertia = inertia

# ---------------------------------------------------------------------------
# 4. Final model, cluster assignment and evaluation
# ---------------------------------------------------------------------------
MODEL = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=N_INIT)
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
    "random_state": RANDOM_STATE,
    "n_init": N_INIT,
    "k_selection": K_SELECTION,
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


# ---------------------------------------------------------------------------
# 5. Visualization
# ---------------------------------------------------------------------------
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
    print("Dataset info:", {k: v for k, v in DATASET_INFO.items() if k != "k_selection"})
    print("K selection:")
    for row in K_SELECTION:
        print(" ", row)
    print("Cluster summary:", CLUSTER_SUMMARY)
    print("Silhouette score:", SILHOUETTE)