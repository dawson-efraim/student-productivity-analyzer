"""
Script 04: Clustering — Identify Student Profiles (K-Means + PCA)
- K-Means with elbow method
- Exclude Final_Score from features to prevent target leakage
- Silhouette score to validate cluster separation
- PCA 2D projection
- Profile each cluster by lifestyle features only
- Derive behavioral labels from lifestyle patterns

Dataset: synthetic student lifestyle data (1,000 rows).
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
from sklearn.metrics import silhouette_score

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Numeric lifestyle features (exclude Final_Score to avoid target leakage)
FEATURE_NAMES = ["Hours_Studied", "Sleep_Hours", "Screen_Time", "Attendance", "Stress_Level"]

# Additional categorical features to include (one-hot encoded internally)
EXTRA_FEATURES = ["Extracurricular"]


def load_data():
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in {DATA_DIR}. Run generate_data.py first.")
    return pd.read_csv(os.path.join(DATA_DIR, csv_files[0]))


def build_features(df):
    """Build the feature matrix: numeric features + one-hot encoded categoricals."""
    numeric = [c for c in FEATURE_NAMES if c in df.columns]
    if len(numeric) < 4:
        raise ValueError(f"Need at least 4 numeric features. Found: {numeric}")

    X = df[numeric].copy()
    extras = [c for c in EXTRA_FEATURES if c in df.columns]
    if extras:
        X = pd.concat([X, pd.get_dummies(df[extras].astype("object"), drop_first=True)], axis=1)

    return X, numeric, extras


def elbow_method(X_scaled, max_k=10):
    """Elbow plot + silhouette-sweep for optimal k; return best k by silhouette."""
    K_range = list(range(2, min(max_k + 1, len(X_scaled) // 10 + 1)))
    inertias = []
    silhouettes = []
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        sil_k = silhouette_score(X_scaled, km.labels_) if 1 < k < len(X_scaled) else np.nan
        silhouettes.append(sil_k)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(K_range, inertias, "bo-", linewidth=2, markersize=8)
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia (within-cluster sum of squares)")
    ax1.set_title("Elbow Method", fontsize=11, fontweight="bold")
    ax1.set_xticks(K_range)
    ax1.grid(True, alpha=0.3)

    ax2.plot(K_range, silhouettes, "gs-", linewidth=2, markersize=8)
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Silhouette Score (higher = better)")
    ax2.set_title("Silhouette Score by k", fontsize=11, fontweight="bold")
    ax2.set_xticks(K_range)
    ax2.grid(True, alpha=0.3)
    fig.suptitle("Choosing k — Elbow vs Silhouette", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "04_elbow_method.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: outputs/04_elbow_method.png")

    # Pick k with the highest silhouette score
    sil_arr = np.array(silhouettes)
    best_idx = int(np.nanargmax(sil_arr))
    best_k = K_range[best_idx]
    print(f"k candidates: {K_range}")
    print(f"silhouettes:  {[round(s, 3) for s in silhouettes]}")
    print(f"Best k by silhouette: {best_k} (silhouette = {sil_arr[best_idx]:.3f})")
    return best_k


def run_clustering(df, k, X_scaled):
    """Run K-Means on the scaled feature matrix."""
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_scaled)

    df_clustered = df.copy()
    df_clustered["Cluster"] = labels

    sil = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else np.nan
    print(f"Silhouette score: {sil:.3f}")

    return df_clustered, km, labels, sil


def plot_pca(X_scaled, labels, k):
    """PCA visualization with cluster centroids."""
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="Set2", alpha=0.6, s=15,
                    edgecolors="white", linewidth=0.3)

    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    km.fit(X_scaled)
    centroids_pca = pca.transform(km.cluster_centers_)
    ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c="black", marker="X",
               s=200, edgecolors="white", linewidth=2, zorder=10, label="Centroids")
    ax.legend(scatterpoints=1, frameon=False, fontsize=9)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} variance)")
    ax.set_title("Student Profiles — PCA Visualization (Lifestyle Features)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "04_pca_clusters.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: outputs/04_pca_clusters.png")


def profile_clusters(df_clustered, df):
    """Mean per cluster for lifestyle features; behavioral labels from lifestyle patterns."""
    lifestyle = [c for c in FEATURE_NAMES if c in df_clustered.columns]
    if not lifestyle:
        raise ValueError("No lifestyle features found for profiling.")

    mean_vals = df_clustered.groupby("Cluster")[lifestyle].mean().round(2)
    if "Extracurricular" in df_clustered.columns:
        ec_props = df_clustered.groupby("Cluster")["Extracurricular"].apply(
            lambda s: (s == "Yes").mean()).round(2)
        mean_vals["Extracurricular_%_Yes"] = ec_props

    # Overall averages (across clusters) for reference in labeling
    overall = mean_vals.mean(numeric_only=True)

    cluster_labels = {}
    for cid in mean_vals.index:
        row = mean_vals.loc[cid]
        parts = []
        # Study hours
        if row["Hours_Studied"] > overall["Hours_Studied"]:
            parts.append("High study")
        else:
            parts.append("Moderate/Low study")
        # Sleep
        if row["Sleep_Hours"] >= 7:
            parts.append("Healthy sleep")
        else:
            parts.append("Short sleep")
        # Screen time
        if row["Screen_Time"] <= 3:
            parts.append("Low screen")
        else:
            parts.append("High screen")
        # Attendance
        if row["Attendance"] >= 85:
            parts.append("High attendance")
        else:
            parts.append("Moderate attendance")
        # Stress
        if row["Stress_Level"] <= 4:
            parts.append("Low stress")
        else:
            parts.append("High stress")
        cluster_labels[int(cid)] = " / ".join(parts)

    print("\n--- Behavioral profile per cluster ---")
    for cid, label in cluster_labels.items():
        print(f"  Cluster {cid}: {label}")

    # Bar chart per feature
    n_feat = len(lifestyle)
    fig, axes = plt.subplots(1, n_feat, figsize=(4 * n_feat, 5))
    if n_feat == 1:
        axes = [axes]
    colors = sns.color_palette("Set2", n_colors=len(mean_vals))
    for i, feat in enumerate(lifestyle):
        mean_vals[feat].plot(kind="bar", ax=axes[i], color=colors, edgecolor="white")
        axes[i].set_title(feat, fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Cluster")
        axes[i].set_ylabel("Mean Value")
        axes[i].tick_params(axis="x", rotation=0)

    fig.suptitle("Cluster Comparison — Lifestyle Features", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "04_cluster_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: outputs/04_cluster_comparison.png")

    # Save profile CSV (lifestyle means only — no Final_Score)
    prof_path = os.path.join(OUTPUT_DIR, "04_cluster_profiles.csv")
    mean_vals.to_csv(prof_path)
    print(f"Saved cluster profiles CSV: {prof_path}")

    return mean_vals, cluster_labels


def main():
    df = load_data()
    print(f"Loaded {df.shape[0]:,} records with columns: {list(df.columns)}")

    X, numeric_feats, extras = build_features(df)
    print(f"Clustering features (numeric): {numeric_feats}")
    if extras:
        print(f"Encoded categorical features: {extras}")

    X_scaled = StandardScaler().fit_transform(X)

    best_k = elbow_method(X_scaled)
    df_clustered, km, labels, sil = run_clustering(df, best_k, X_scaled)
    print(f"Final k = {best_k}; Silhouette = {sil:.3f}")

    plot_pca(X_scaled, labels, best_k)
    mean_vals, cluster_labels = profile_clusters(df_clustered, df)

    print("\nClustering complete.")


if __name__ == "__main__":
    main()
