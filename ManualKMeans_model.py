"""
Manual K-Means Simulation - Activity 3, Part 1
Author: Sergio Steeven Moreno Forero

Topic: Mall customer segmentation (Annual Income vs. Spending Score).

This module reproduces, step by step, what K-Means does internally, without using
scikit-learn. Every step of the algorithm is written as its own function so that
it can be followed and explained by hand:

    1. Load the 100-record sample (2 numerical variables).
    2. Select 3 initial centroids from real customers of the sample.
    3. For each of the 3 iterations:
         a. Euclidean distance from every customer to each centroid.
         b. Assign every customer to the cluster of its nearest centroid.
         c. New centroid = average of the customers assigned to each cluster.
         d. Within-cluster variance, centroid movement and reassignments.
         e. Scatter plot with the clusters and the updated centroids.
    4. Final summary: variance per iteration and characteristics of each cluster.

Values exported to app.py (same names used by the Flask route):
    INITIAL_CENTROIDS, INITIAL_PLOT, ITERATIONS, FINAL_SUMMARY
"""

import io
import os
import base64
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "store_customers_100_sample.csv")

COL_ID = "CustomerID"
COL_X = "Annual Income (k$)"
COL_Y = "Spending Score (1-100)"

K = 3
N_ITERATIONS = 3

CLUSTER_COLORS = ["#4C72B0", "#DD8452", "#55A868"]
CLUSTER_NAMES = ["Cluster 1", "Cluster 2", "Cluster 3"]


# ---------------------------------------------------------------------------
# Step 1 - Load the dataset
# ---------------------------------------------------------------------------
def load_dataset(path=DATASET_PATH):
    """Load the 100-record sample and keep only the columns used by K-Means."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Sample dataset not found: {path}")

    df = pd.read_csv(path)
    df = df[[COL_ID, COL_X, COL_Y]].dropna().reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Step 2 - Initial centroids (3 real customers of the sample)
# ---------------------------------------------------------------------------
def select_initial_centroids(df):
    """
    Pick 3 real customers as starting centroids, spread along the income axis:
      C1 -> customer with the lowest annual income
      C2 -> customer closest to the median point (median income, median score)
      C3 -> customer with the highest annual income
    Ties are broken by the lowest CustomerID so the result is always the same.
    """
    ordered = df.sort_values([COL_X, COL_ID]).reset_index(drop=True)
    lowest = ordered.iloc[0]
    highest = df.sort_values([COL_X, COL_ID], ascending=[False, True]).iloc[0]

    median_x = df[COL_X].median()
    median_y = df[COL_Y].median()
    dist_to_median = np.sqrt((df[COL_X] - median_x) ** 2 + (df[COL_Y] - median_y) ** 2)
    closest_to_median = df.loc[dist_to_median.idxmin()]

    chosen = [
        (lowest, "Customer with the lowest annual income"),
        (closest_to_median, "Customer closest to the median point"),
        (highest, "Customer with the highest annual income"),
    ]

    centroids = np.array([[float(row[COL_X]), float(row[COL_Y])] for row, _ in chosen])
    info = [
        {
            "centroid": f"C{i + 1}",
            "rule": rule,
            "customer_id": int(row[COL_ID]),
            "x": round(float(row[COL_X]), 2),
            "y": round(float(row[COL_Y]), 2),
        }
        for i, (row, rule) in enumerate(chosen)
    ]
    return centroids, info


# ---------------------------------------------------------------------------
# Step 3a - Euclidean distance
# ---------------------------------------------------------------------------
def euclidean_distance(x1, y1, x2, y2):
    """d = sqrt((x1 - x2)^2 + (y1 - y2)^2)"""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def distance_matrix(points, centroids):
    """Distance from every point to every centroid (rows = points, columns = centroids)."""
    distances = np.zeros((len(points), len(centroids)))
    for i, (px, py) in enumerate(points):
        for k, (cx, cy) in enumerate(centroids):
            distances[i, k] = euclidean_distance(px, py, cx, cy)
    return distances


# ---------------------------------------------------------------------------
# Step 3b - Assignment to the nearest centroid
# ---------------------------------------------------------------------------
def assign_clusters(distances):
    """Index (0, 1 or 2) of the nearest centroid for every point."""
    return np.argmin(distances, axis=1)


# ---------------------------------------------------------------------------
# Step 3c - New centroids (average of the assigned points)
# ---------------------------------------------------------------------------
def update_centroids(points, assignments, old_centroids):
    """New centroid = (mean X, mean Y) of its cluster. An empty cluster keeps its centroid."""
    new_centroids = old_centroids.copy()
    for k in range(K):
        members = points[assignments == k]
        if len(members) > 0:
            new_centroids[k] = [members[:, 0].mean(), members[:, 1].mean()]
    return new_centroids


# ---------------------------------------------------------------------------
# Step 3d - Within-cluster variance
# ---------------------------------------------------------------------------
def within_cluster_variance(points, assignments, centroids):
    """
    Variance of cluster k = average squared distance between its points and its centroid.
    The total is the sum of the 3 clusters. Lower total = more compact clusters.
    """
    variance = {}
    for k in range(K):
        members = points[assignments == k]
        if len(members) == 0:
            variance[f"cluster_{k + 1}"] = 0.0
            continue
        squared = (members[:, 0] - centroids[k][0]) ** 2 + (members[:, 1] - centroids[k][1]) ** 2
        variance[f"cluster_{k + 1}"] = round(float(squared.mean()), 2)
    total = round(sum(variance.values()), 2)
    return variance, total


# ---------------------------------------------------------------------------
# Step 3e - Scatter plot
# ---------------------------------------------------------------------------
def scatter_plot(points, centroids, title, assignments=None, previous_centroids=None):
    """Scatter plot of the customers. Returns the image as a base64 string for the template."""
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=110)

    if assignments is None:
        ax.scatter(points[:, 0], points[:, 1], s=28, alpha=0.6, color="#6b7280", label="Customers")
    else:
        for k in range(K):
            members = points[assignments == k]
            ax.scatter(members[:, 0], members[:, 1], s=28, alpha=0.7,
                       color=CLUSTER_COLORS[k], label=CLUSTER_NAMES[k])

    # Previous centroids and arrows showing how far each centroid moved
    if previous_centroids is not None:
        ax.scatter(previous_centroids[:, 0], previous_centroids[:, 1], marker="X", s=140,
                   color="white", edgecolor="black", linewidth=1.2, label="Previous centroids", zorder=4)
        for old, new in zip(previous_centroids, centroids):
            ax.annotate("", xy=(new[0], new[1]), xytext=(old[0], old[1]),
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.2), zorder=4)

    ax.scatter(centroids[:, 0], centroids[:, 1], marker="X", s=230, color="black",
               edgecolor="white", linewidth=1.5, label="Centroids", zorder=5)
    for k, (cx, cy) in enumerate(centroids):
        ax.annotate(f"C{k + 1}", (cx, cy), textcoords="offset points", xytext=(9, 7),
                    fontsize=10, fontweight="bold", zorder=6)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel(COL_X, fontsize=11)
    ax.set_ylabel(COL_Y, fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# One full iteration (steps 3a to 3e)
# ---------------------------------------------------------------------------
def run_iteration(number, points, ids, centroids, previous_assignments=None):
    distances = distance_matrix(points, centroids)
    assignments = assign_clusters(distances)
    new_centroids = update_centroids(points, assignments, centroids)
    variance, total_variance = within_cluster_variance(points, assignments, new_centroids)

    table = []
    for i in range(len(points)):
        table.append({
            "customer_id": int(ids[i]),
            "x": round(float(points[i, 0]), 2),
            "y": round(float(points[i, 1]), 2),
            "dist_c1": round(float(distances[i, 0]), 2),
            "dist_c2": round(float(distances[i, 1]), 2),
            "dist_c3": round(float(distances[i, 2]), 2),
            "cluster": int(assignments[i]) + 1,
        })

    centroid_shift = [
        round(euclidean_distance(o[0], o[1], n[0], n[1]), 2)
        for o, n in zip(centroids, new_centroids)
    ]
    reassigned = None if previous_assignments is None else int((assignments != previous_assignments).sum())

    plot = scatter_plot(
        points, new_centroids,
        title=f"Iteration {number}: Cluster Assignments and Updated Centroids",
        assignments=assignments, previous_centroids=centroids,
    )

    return {
        "number": number,
        "table": table,
        "old_centroids": [(round(float(c[0]), 2), round(float(c[1]), 2)) for c in centroids],
        "new_centroids": [(round(float(c[0]), 2), round(float(c[1]), 2)) for c in new_centroids],
        "cluster_sizes": [int((assignments == k).sum()) for k in range(K)],
        "centroid_shift": centroid_shift,
        "reassigned": reassigned,
        "variance": variance,
        "total_variance": total_variance,
        "plot": plot,
        "_assignments": assignments,
        "_new_centroids": new_centroids,
    }


# ---------------------------------------------------------------------------
# Run the simulation (executed once when app.py imports this module)
# ---------------------------------------------------------------------------
df = load_dataset()
points = df[[COL_X, COL_Y]].to_numpy(dtype=float)
customer_ids = df[COL_ID].to_numpy()

INITIAL_CENTROIDS, INITIAL_CENTROID_INFO = select_initial_centroids(df)
INITIAL_PLOT = scatter_plot(points, INITIAL_CENTROIDS, title="Initial Data and Centroids")

ITERATIONS = []
_centroids = INITIAL_CENTROIDS.copy()
_previous_assignments = None

for n in range(1, N_ITERATIONS + 1):
    result = run_iteration(n, points, customer_ids, _centroids, _previous_assignments)
    _centroids = result.pop("_new_centroids")
    _previous_assignments = result.pop("_assignments")
    ITERATIONS.append(result)

# ---------------------------------------------------------------------------
# Step 4 - Final characteristics of each cluster (points assigned in iteration 3)
# ---------------------------------------------------------------------------
FINAL_CLUSTER_STATS = []
for k in range(K):
    members = points[_previous_assignments == k]
    FINAL_CLUSTER_STATS.append({
        "cluster": k + 1,
        "count": int(len(members)),
        "avg_income": round(float(members[:, 0].mean()), 2) if len(members) else 0.0,
        "avg_spending": round(float(members[:, 1].mean()), 2) if len(members) else 0.0,
        "min_income": round(float(members[:, 0].min()), 2) if len(members) else 0.0,
        "max_income": round(float(members[:, 0].max()), 2) if len(members) else 0.0,
        "min_spending": round(float(members[:, 1].min()), 2) if len(members) else 0.0,
        "max_spending": round(float(members[:, 1].max()), 2) if len(members) else 0.0,
    })

FINAL_SUMMARY = {
    "n_records": int(len(points)),
    "initial_centroid_info": INITIAL_CENTROID_INFO,
    "initial_centroids": [(round(float(c[0]), 2), round(float(c[1]), 2)) for c in INITIAL_CENTROIDS],
    "final_centroids": ITERATIONS[-1]["new_centroids"],
    "variance_by_iteration": [it["total_variance"] for it in ITERATIONS],
    "cluster_stats": FINAL_CLUSTER_STATS,
}


if __name__ == "__main__":
    print("Initial centroids:")
    for c in INITIAL_CENTROID_INFO:
        print(f"  {c['centroid']}: customer {c['customer_id']} -> ({c['x']}, {c['y']})  [{c['rule']}]")
    for it in ITERATIONS:
        print(f"\nIteration {it['number']}")
        print("  Cluster sizes:     ", it["cluster_sizes"])
        print("  New centroids:     ", it["new_centroids"])
        print("  Centroid movement: ", it["centroid_shift"])
        print("  Reassigned points: ", it["reassigned"])
        print("  Variance:          ", it["variance"], "total =", it["total_variance"])
    print("\nFinal clusters:")
    for s in FINAL_CLUSTER_STATS:
        print(" ", s)