"""
OptiCrop - Crop Statistics & Clustering Analysis
Generates:
  1. Per-crop min/max/mean/std for each feature (used for Scenario 2: suitability check)
  2. K-Means clustering of crops by their average environmental needs (Scenario 3: research view)
"""

import pandas as pd
import numpy as np
import pickle
import os
import json

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Crop_recommendation.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

df = pd.read_csv(DATA_PATH)

# ---------------------------------------------------------
# 1. Per-crop statistics (min, max, mean, std) for suitability checks
# ---------------------------------------------------------
stats = df.groupby("label")[FEATURES].agg(["min", "max", "mean", "std"])
stats.columns = ["_".join(col) for col in stats.columns]
stats = stats.reset_index()
stats.to_csv(os.path.join(MODEL_DIR, "crop_stats.csv"), index=False)

# Also save as a nested dict for easy lookup in Flask
stats_dict = {}
for _, row in stats.iterrows():
    crop = row["label"]
    stats_dict[crop] = {}
    for feat in FEATURES:
        stats_dict[crop][feat] = {
            "min": round(row[f"{feat}_min"], 2),
            "max": round(row[f"{feat}_max"], 2),
            "mean": round(row[f"{feat}_mean"], 2),
            "std": round(row[f"{feat}_std"], 2),
        }

with open(os.path.join(MODEL_DIR, "crop_stats.json"), "w") as f:
    json.dump(stats_dict, f, indent=2)

print("Saved crop_stats.csv and crop_stats.json")

# ---------------------------------------------------------
# 2. K-Means clustering of crops by average environmental needs
# ---------------------------------------------------------
crop_means = df.groupby("label")[FEATURES].mean()

scaler = StandardScaler()
crop_means_scaled = scaler.fit_transform(crop_means)

# Use 4 clusters - a reasonable grouping for 22 crops (elbow method generally
# supports 3-5 clusters for this dataset)
N_CLUSTERS = 4
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(crop_means_scaled)

crop_clusters = pd.DataFrame({
    "crop": crop_means.index,
    "cluster": cluster_labels
}).sort_values(["cluster", "crop"])

crop_clusters.to_csv(os.path.join(MODEL_DIR, "crop_clusters.csv"), index=False)
print("\nCrop clusters:")
print(crop_clusters.to_string(index=False))

# Save cluster model + scaler for potential reuse
with open(os.path.join(MODEL_DIR, "kmeans_model.pkl"), "wb") as f:
    pickle.dump(kmeans, f)
with open(os.path.join(MODEL_DIR, "cluster_scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# ---------------------------------------------------------
# 3. Visualize clusters with PCA (2D projection)
# ---------------------------------------------------------
from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(crop_means_scaled)

plt.figure(figsize=(10, 8))
palette = sns.color_palette("Set2", N_CLUSTERS)
for cluster_id in range(N_CLUSTERS):
    mask = cluster_labels == cluster_id
    plt.scatter(coords[mask, 0], coords[mask, 1], s=150,
                color=palette[cluster_id], label=f"Cluster {cluster_id}")

for i, crop in enumerate(crop_means.index):
    plt.annotate(crop, (coords[i, 0], coords[i, 1]), fontsize=9,
                 xytext=(5, 5), textcoords="offset points")

plt.title("Crop Clusters by Environmental Needs (PCA Projection)")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "crop_clusters_pca.png"), dpi=120)
plt.close()

# ---------------------------------------------------------
# 4. Cluster profile summary (avg conditions per cluster)
# ---------------------------------------------------------
crop_means_with_cluster = crop_means.copy()
crop_means_with_cluster["cluster"] = cluster_labels
cluster_profile = crop_means_with_cluster.groupby("cluster")[FEATURES].mean().round(2)
cluster_profile.to_csv(os.path.join(MODEL_DIR, "cluster_profile.csv"))
print("\nCluster environmental profile (avg conditions per cluster):")
print(cluster_profile)

print("\nSaved crop_clusters.csv, crop_clusters_pca.png, cluster_profile.csv, kmeans_model.pkl")
