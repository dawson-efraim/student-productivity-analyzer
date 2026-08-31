"""
Script 04: Clustering — Identify Student Profiles
- K-Means clustering on lifestyle + performance features
- Elbow method to find optimal k
- PCA for 2D visualization
- Cluster profiling (summary stats per cluster)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in {DATA_DIR}. Run 01_download_data.py first.")
    return pd.read_csv(os.path.join(DATA_DIR, csv_files[0]))


def select_features(df):
    """Select numeric features relevant for clustering."""
    # Known columns from this dataset
    feature_candidates = [
        "Hours_Studied", "Sleep_Hours", "Screen_Time", "Attendance",
        "Stress_Level", "Exam_Anxiety_Score", "Tutoring_Sessions_Per_Week",
        "Previous_GPA", "Final_Score",
    ]
    # Fallback: use all numeric columns except IDs
    available = [c for c in feature_candidates if c in df.columns]
    if len(available) < 4:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        available = [c for c in numeric_cols if "id" not in c.lower()]

    print(f"Clustering features: {available}")
    return available


def elbow_method(X_scaled, max_k=10):
    """Plot inertia vs k to find optimal cluster count."""
    inertias = []
    K_range = range(2, max_k + 1)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(K_range, inertias, "bo-", linewidth=2, markersize=8)
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method — Optimal k")
    ax.set_xticks(list(K_range))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "04_elbow_method.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Auto-pick k: biggest drop in inertia
    drops = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]
    best_k = list(K_range)[np.argmax(drops)]
    print(f"Suggested k: {best_k}")
    return best_k


def run_clustering(df, features, k):
    """Run K-Means and return cluster labels."""
    X = df[features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    # Map back to original df
    df_clustered = df.loc[X.index].copy()
    df_clustered["Cluster"] = labels
    return df_clustered, X_scaled, labels, km


def plot_pca(X_scaled, labels, k):
    """PCA scatter plot of clusters."""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="Set2",
                         alpha=0.6, s=15, edgecolors="white", linewidth=0.3)

    # Plot centroids in PCA space
    centroids_pca = pca.transform(km_clusterer.cluster_centers_)  # noqa: will pass from caller
    ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c="black", marker="X",
               s=200, edgecolors="white", linewidth=2, zorder=10, label="Centroids")
    ax.legend()

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title("Student Profiles — PCA Visualization")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "04_pca_clusters.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def profile_clusters(df_clustered, features):
    """Print summary stats per cluster."""
    print("\n" + "=" * 60)
    print("CLUSTER PROFILES")
    print("=" * 60)

    profile = df_clustered.groupby("Cluster")[features].agg(["mean", "std", "median"]).round(2)

    # Mean per cluster
    mean_profile = df_clustered.groupby("Cluster")[features].mean().round(2)
    print("\nCluster Means:")
    print(mean_profile.to_string())

    # Size
    sizes = df_clustered["Cluster"].value_counts().sort_index()
    print(f"\nCluster Sizes:\n{sizes.to_string()}")

    # Save profile table
    path = os.path.join(OUTPUT_DIR, "04_cluster_profiles.csv")
    mean_profile.to_csv(path)
    print(f"\nSaved: {path}")

    # Bar chart comparing clusters
    n_features = len(features)
    fig, axes = plt.subplots(1, n_features, figsize=(4 * n_features, 5))
    if n_features == 1:
        axes = [axes]

    colors = sns.color_palette("Set2", n_colors=len(mean_profile))

    for i, feat in enumerate(features):
        mean_profile[feat].plot(kind="bar", ax=axes[i], color=colors, edgecolor="white")
        axes[i].set_title(feat, fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Cluster")
        axes[i].set_ylabel("Mean Value")
        axes[i].tick_params(axis="x", rotation=0)

    fig.suptitle("Cluster Comparison — Key Features", fontsize=16, fontweight="bold", y=1.03)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "04_cluster_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# Module-level var to share centroid ref
km_clusterer = None


def main():
    global km_clusterer

    df = load_data()
    features = select_features(df)

    X = df[features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow
    best_k = elbow_method(X_scaled)

    # Run clustering with best k
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    km_clusterer = km

    df_clustered = df.loc[X.index].copy()
    df_clustered["Cluster"] = labels

    # PCA plot
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="Set2",
                         alpha=0.6, s=15, edgecolors="white", linewidth=0.3)
    centroids_pca = pca.transform(km.cluster_centers_)
    ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c="black", marker="X",
               s=200, edgecolors="white", linewidth=2, zorder=10, label="Centroids")
    ax.legend()
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title("Student Profiles — PCA Visualization")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "04_pca_clusters.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Profiles
    profile_clusters(df_clustered, features)
    print("\nClustering complete.")


if __name__ == "__main__":
    main()
