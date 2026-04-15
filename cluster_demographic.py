import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings

warnings.filterwarnings("ignore")

DIR = Path("__file__").resolve().parent
CLUSTER_RESULTS = DIR / "clustering" / "demographics"
DATA_SOURCE = DIR / "strd_data" / "MERGED_POPULATION_DEMOGRAPHIC_DATA.csv"


def split_demographics_by_gender(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, index_col="ZIP")
    df_male = df.filter(regex="Male", axis=1).dropna()
    df_female = df.filter(regex="Female", axis=1).dropna()

    common_idx = df_male.index.intersection(df_female.index)
    df_male = df_male.loc[common_idx]
    df_female = df_female.loc[common_idx]

    df_male.columns = df_male.columns.str.replace("Male", "", regex=False).str.strip()
    df_female.columns = df_female.columns.str.replace("Female", "", regex=False).str.strip()

    common_cols = df_male.columns.intersection(df_female.columns)
    df_male = df_male[common_cols]
    df_female = df_female[common_cols]

    log_ratio_df = np.log1p(df_male) - np.log1p(df_female)
    log_ratio_df.columns = [f"{col}_log_ratio" for col in common_cols]

    CLUSTER_RESULTS.mkdir(parents=True, exist_ok=True)
    log_ratio_df.to_csv(CLUSTER_RESULTS / "gender_log_ratio.csv")

    return log_ratio_df


def find_optimal_k(scaled_data: np.ndarray, max_k: int = 10) -> int:
    """Elbow (inertia), Silhouette, Davies-Bouldin, and Calinski-Harabasz to find optimal k."""
    inertias = []
    silhouette_scores = []
    db_scores = []
    ch_scores = []
    k_range = range(2, max_k + 1)

    print("Evaluating K-Means cluster counts...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(scaled_data)

        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(scaled_data, labels))
        db_scores.append(davies_bouldin_score(scaled_data, labels))
        ch_scores.append(calinski_harabasz_score(scaled_data, labels))

        print(
            f"  k={k} | Inertia: {kmeans.inertia_:.1f} | "
            f"Silhouette: {silhouette_scores[-1]:.4f} | "
            f"Davies-Bouldin: {db_scores[-1]:.4f} | "
            f"Calinski-Harabasz: {ch_scores[-1]:.1f}"
        )

    # --- Plot all four metrics ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("K-Means Cluster Selection Metrics", fontsize=14, fontweight="bold")

    axes[0, 0].plot(k_range, inertias, marker="o", color="steelblue")
    axes[0, 0].set_title("Elbow — Inertia (lower = better)")
    axes[0, 0].set_xlabel("k")
    axes[0, 0].set_ylabel("Inertia")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(k_range, silhouette_scores, marker="o", color="seagreen")
    axes[0, 1].set_title("Silhouette Score (higher = better)")
    axes[0, 1].set_xlabel("k")
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(k_range, db_scores, marker="o", color="tomato")
    axes[1, 0].set_title("Davies-Bouldin Score (lower = better)")
    axes[1, 0].set_xlabel("k")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(k_range, ch_scores, marker="o", color="darkorange")
    axes[1, 1].set_title("Calinski-Harabasz Score (higher = better)")
    axes[1, 1].set_xlabel("k")
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(CLUSTER_RESULTS / "kmeans_k_selection.png", dpi=150)
    plt.close()
    print("K selection plots saved.")

    # Optimal k via majority vote across metrics
    best_silhouette_k = list(k_range)[np.argmax(silhouette_scores)]
    best_db_k = list(k_range)[np.argmin(db_scores)]
    best_ch_k = list(k_range)[np.argmax(ch_scores)]

    votes = [best_silhouette_k, best_db_k, best_ch_k]
    optimal_k = max(set(votes), key=votes.count)

    print(f"\n  Best k by Silhouette:         {best_silhouette_k}")
    print(f"  Best k by Davies-Bouldin:     {best_db_k}")
    print(f"  Best k by Calinski-Harabasz:  {best_ch_k}")
    print(f"  >>> Optimal k (majority vote): {optimal_k}")

    return optimal_k


def plot_clusters(pca_coords, labels, optimal_k, explained_variance, output_path):
    """Scatter plot of clusters in 2D PCA space."""
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = cm.tab10(np.linspace(0, 1, optimal_k))

    for cluster_id, color in zip(range(optimal_k), colors):
        mask = labels == cluster_id
        ax.scatter(
            pca_coords[mask, 0],
            pca_coords[mask, 1],
            label=f"Cluster {cluster_id} (n={mask.sum()})",
            color=color,
            alpha=0.6,
            s=30,
            edgecolors="none",
        )

    ax.set_title(
        f"K-Means Clusters in PCA Space "
        f"(k={optimal_k}, {explained_variance:.1f}% variance explained)"
    )
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.legend(loc="best", framealpha=0.7)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_cluster_profiles(summary_df: pd.DataFrame, output_path: Path):
    """Heatmap of mean log-ratios per cluster for interpretability."""
    fig, ax = plt.subplots(figsize=(14, max(4, len(summary_df) * 0.8)))

    # Drop probability columns if any leaked in
    plot_data = summary_df.filter(regex="log_ratio")

    im = ax.imshow(plot_data.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Mean Log Ratio (male / female)")

    ax.set_xticks(range(len(plot_data.columns)))
    ax.set_xticklabels(
        [c.replace("_log_ratio", "") for c in plot_data.columns],
        rotation=45, ha="right", fontsize=8
    )
    ax.set_yticks(range(len(plot_data.index)))
    ax.set_yticklabels([f"Cluster {i}" for i in plot_data.index])
    ax.set_title("Cluster Profile Heatmap — Mean Log Ratio per Demographic")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_kmeans_clustering(log_ratio_df: pd.DataFrame, max_k: int = 10) -> pd.DataFrame:

    # --- 1. Scale ---
    scaler = StandardScaler()
    scaled = scaler.fit_transform(log_ratio_df)

    # --- 2. PCA for visualization only ---
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(scaled)
    explained = pca.explained_variance_ratio_.sum() * 100
    print(f"PCA 2-component explained variance: {explained:.1f}%\n")

    # --- 3. Find optimal k ---
    optimal_k = find_optimal_k(scaled, max_k=max_k)
    optimal_k = 3

    # --- 4. Fit final K-Means ---
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20, max_iter=500)
    labels = kmeans.fit_predict(scaled)

    # --- 5. Assemble results ---
    result_df = log_ratio_df.copy()
    result_df["cluster"] = labels
    result_df["pca_1"] = pca_coords[:, 0]
    result_df["pca_2"] = pca_coords[:, 1]

    # Distance to each centroid (analog to GMM soft probabilities)
    centroids = kmeans.cluster_centers_
    for i in range(optimal_k):
        dist = np.linalg.norm(scaled - centroids[i], axis=1)
        result_df[f"dist_to_centroid_{i}"] = dist

    # --- 6. Plots ---
    plot_clusters(pca_coords, labels, optimal_k, explained, CLUSTER_RESULTS / "kmeans_clusters_pca.png")
    print("Cluster PCA plot saved.")

    # --- 7. Cluster summary ---
    summary = (
        result_df.drop(columns=["pca_1", "pca_2"])
        .groupby("cluster")
        .mean()
        .round(4)
    )

    plot_cluster_profiles(summary, CLUSTER_RESULTS / "kmeans_cluster_profiles.png")
    print("Cluster profile heatmap saved.")

    print("\nCluster mean log-ratios:")
    print(summary.filter(regex="log_ratio").to_string())

    # --- 8. Save outputs ---
    result_df.to_csv(CLUSTER_RESULTS / "kmeans_cluster_results.csv")
    summary.to_csv(CLUSTER_RESULTS / "kmeans_cluster_summary.csv")
    print(f"\nAll results saved to: {CLUSTER_RESULTS}")

    return result_df


if __name__ == "__main__":
    log_ratio_df = split_demographics_by_gender(DATA_SOURCE)
    cluster_df = run_kmeans_clustering(log_ratio_df, max_k=10)
    print(f"\nFinal dataframe shape: {cluster_df.shape}")
    print(cluster_df[["cluster"]].value_counts().sort_index())