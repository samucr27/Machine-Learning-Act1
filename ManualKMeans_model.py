"""
Manual K-Means Simulation module - Activity 3, Part 1
Topic: Mall Customer Segmentation (Annual Income vs. Spending Score)

This module performs the full manual K-Means walkthrough required by the activity:
- Loads the 100-record sample.
- Picks 3 initial centroids (based on the 25th/50th/75th percentiles of the sample,
  so they start reasonably spread across the data).
- Runs exactly 3 iterations, computing for each one: Euclidean distances from every
  point to the 3 centroids, cluster assignment, updated centroids, a results table,
  a scatter plot, and the within-cluster variance.

Sergio (P3): you generally do NOT need to change the logic here — just import
ITERATIONS, INITIAL_CENTROIDS, INITIAL_PLOT, and FINAL_SUMMARY into your Flask route
and display them in manual_exercise.html. Only touch DATASET_PATH / COL_X / COL_Y
below if you end up using a different sample file or column names.
"""

import io
import os
import base64

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASET_PATH = "store_customers_100_sample.csv"
COL_X = "Annual Income (k$)"
COL_Y = "Spending Score (1-100)"

CLUSTER_COLORS = ["#4C72B0", "#DD8452", "#55A868"]
CLUSTER_NAMES = ["Cluster 1", "Cluster 2", "Cluster 3"]


def _load_dataset() -> pd.DataFrame:
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        df = df.dropna(subset=[COL_X, COL_Y]).reset_index(drop=True)
        return df

    # ---- Synthetic placeholder sample (remove once the real 100-row CSV is in place) ----
    rng = np.random.default_rng(42)
    n = 100
    income = rng.normal(60, 25, n).clip(15, 140).round(1)
    spending = rng.normal(50, 25, n).clip(1, 100).round().astype(int)
    return pd.DataFrame({
        "CustomerID": np.arange(1, n + 1),
        COL_X: income,
        COL_Y: spending,
    })


df = _load_dataset()
points = df[[COL_X, COL_Y]].to_numpy(dtype=float)
customer_ids = df["CustomerID"].tolist() if "CustomerID" in df.columns else list(range(1, len(df) + 1))

# ---------- Initial centroids: 25th / 50th / 75th percentile of the sample ----------
q25 = np.percentile(points, 25, axis=0)
q50 = np.percentile(points, 50, axis=0)
q75 = np.percentile(points, 75, axis=0)
INITIAL_CENTROIDS = np.array([q25, q50, q75])


def _plot_points(centroids, assignments=None, title="", highlight_new_centroids=True):
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=110)

    if assignments is None:
        ax.scatter(points[:, 0], points[:, 1], alpha=0.6, s=28, color="#6b7280", label="Customers")
    else:
        for k in range(3):
            mask = assignments == k
            ax.scatter(
                points[mask, 0], points[mask, 1],
                alpha=0.65, s=28, color=CLUSTER_COLORS[k], label=CLUSTER_NAMES[k],
            )

    marker_color = "black" if highlight_new_centroids else "red"
    ax.scatter(
        centroids[:, 0], centroids[:, 1],
        marker="X", s=220, color=marker_color, edgecolor="white", linewidth=1.5,
        label="Centroids", zorder=5,
    )

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Annual Income (k$)", fontsize=11)
    ax.set_ylabel("Spending Score (1-100)", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


INITIAL_PLOT = _plot_points(INITIAL_CENTROIDS, assignments=None, title="Initial Data and Centroids")


def _run_iteration(centroids):
    distances = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
    assignments = np.argmin(distances, axis=1)

    table = []
    for i in range(len(points)):
        table.append({
            "customer_id": customer_ids[i],
            "x": round(float(points[i, 0]), 2),
            "y": round(float(points[i, 1]), 2),
            "dist_c1": round(float(distances[i, 0]), 2),
            "dist_c2": round(float(distances[i, 1]), 2),
            "dist_c3": round(float(distances[i, 2]), 2),
            "cluster": int(assignments[i]) + 1,
        })

    new_centroids = centroids.copy()
    variance = {}
    for k in range(3):
        mask = assignments == k
        if mask.sum() > 0:
            new_centroids[k] = points[mask].mean(axis=0)
            variance[f"cluster_{k + 1}"] = round(
                float(np.mean(np.sum((points[mask] - new_centroids[k]) ** 2, axis=1))), 2
            )
        else:
            variance[f"cluster_{k + 1}"] = 0.0

    total_variance = round(sum(variance.values()), 2)

    return {
        "table": table,
        "old_centroids": [tuple(round(v, 2) for v in c) for c in centroids],
        "new_centroids": [tuple(round(v, 2) for v in c) for c in new_centroids],
        "assignments": assignments,
        "variance": variance,
        "total_variance": total_variance,
    }


ITERATIONS = []
_current_centroids = INITIAL_CENTROIDS.copy()

for iteration_num in range(1, 4):
    result = _run_iteration(_current_centroids)
    plot = _plot_points(
        np.array(result["new_centroids"]),
        assignments=result["assignments"],
        title=f"Iteration {iteration_num}: Cluster Assignments and Updated Centroids",
    )
    ITERATIONS.append({
        "number": iteration_num,
        "table": result["table"],
        "old_centroids": result["old_centroids"],
        "new_centroids": result["new_centroids"],
        "variance": result["variance"],
        "total_variance": result["total_variance"],
        "plot": plot,
    })
    _current_centroids = np.array(result["new_centroids"])

# ---------- Final cluster characteristics, based on the last iteration's assignments ----------
_final_assignments = _run_iteration(_current_centroids)["assignments"]
FINAL_CLUSTER_STATS = []
for k in range(3):
    mask = _final_assignments == k
    cluster_points = points[mask]
    FINAL_CLUSTER_STATS.append({
        "cluster": k + 1,
        "count": int(mask.sum()),
        "avg_income": round(float(cluster_points[:, 0].mean()), 2) if mask.sum() > 0 else 0,
        "avg_spending": round(float(cluster_points[:, 1].mean()), 2) if mask.sum() > 0 else 0,
    })

FINAL_SUMMARY = {
    "initial_centroids": [tuple(round(v, 2) for v in c) for c in INITIAL_CENTROIDS],
    "final_centroids": ITERATIONS[-1]["new_centroids"],
    "variance_by_iteration": [it["total_variance"] for it in ITERATIONS],
    "cluster_stats": FINAL_CLUSTER_STATS,
}


if __name__ == "__main__":
    print("Initial centroids:", FINAL_SUMMARY["initial_centroids"])
    for it in ITERATIONS:
        print(f"\nIteration {it['number']}: total variance = {it['total_variance']}")
        print("New centroids:", it["new_centroids"])
    print("\nFinal cluster stats:", FINAL_CLUSTER_STATS)