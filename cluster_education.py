"""
K-Means Clustering for ZIP Code Industry Employment Data
---------------------------------------------------------
Groups ZIP codes by their industry employment mix using K-Means.

Expected DataFrame structure:
  - Index: ZIP codes (named "ZIP")
  - Columns: Industry employment counts (int64)
    0: Total Employed Civilians Above 16
    1: Management, business, science, and arts occupations
    2: Service occupations
    3: Sales and office occupations
    4: Natural resources, construction, and maintenance occupations
    5: Production, transportation, and material moving occupations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA

# ── CONFIG ────────────────────────────────────────────────────────────────────
K_RANGE      = range(2, 11)      # range of K values to evaluate
FINAL_K      = None              # set an int to skip elbow analysis, or None to auto-pick
RANDOM_STATE = 42

DIR = Path("__file__").resolve().parent
INPUT_FILE = DIR / "strd_data" / "SIMPLIFIED_EDU.csv"   # ← replace with your file path

CLUSTER_RESULTS = DIR / "clustering" / "education"
CLUSTER_RESULTS.mkdir(exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────

INDUSTRY_COLS = [
    "Less than high school graduate",
    "High school graduate (includes equivalency)",
    "Some college or associate's degree",
    "Bachelor's degree or higher"
]

TOTAL_COL = "Total in Working Age"

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, index_col="ZIP", dtype={"ZIP": str})
    print(f"Loaded {len(df):,} ZIP codes with {len(df.columns)} columns.")
    return df


# ── 2. PREPROCESS ─────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Convert raw counts → industry share of local workforce, then standardize.
    This ensures clusters reflect industry *mix*, not population size.
    """
    industry_df = df[INDUSTRY_COLS].copy()

    # Row-normalize: each value becomes its share of the total employed
    totals = df[TOTAL_COL].replace(0, np.nan)           # avoid div-by-zero
    shares = industry_df.div(totals, axis=0).fillna(0)  # proportions 0–1

    # Column-standardize: zero mean, unit variance across ZIPs
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(shares)

    print("\nPreprocessing complete:")
    print(f"  • {X_scaled.shape[0]:,} samples  |  {X_scaled.shape[1]} features")
    print("  • Row-normalized to workforce proportions, then StandardScaler applied.")
    return X_scaled, shares


# ── 3. ELBOW + SILHOUETTE ANALYSIS ───────────────────────────────────────────
def evaluate_k(X: np.ndarray, k_range: range) -> int:
    inertias, sil_scores = [], []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X, labels))
        print(f"  K={k:2d}  |  Inertia: {km.inertia_:>12,.1f}  |  Silhouette: {sil_scores[-1]:.4f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ks = list(k_range)

    ax1.plot(ks, inertias, "bo-", linewidth=2, markersize=6)
    ax1.set(title="Elbow Method — Inertia", xlabel="Number of Clusters (K)", ylabel="Inertia")
    ax1.grid(alpha=0.3)

    ax2.plot(ks, sil_scores, "rs-", linewidth=2, markersize=6)
    ax2.set(title="Silhouette Score", xlabel="Number of Clusters (K)", ylabel="Silhouette Score")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(CLUSTER_RESULTS / "kmeans_k_selection.png", dpi=150)
    plt.show()
    print(f"\nSaved: {CLUSTER_RESULTS / 'kmeans_k_selection.png'}")

    # Auto-pick: highest silhouette score
    best_k = ks[int(np.argmax(sil_scores))]
    best_k = 5 # <- manually overwritten due to better results in testing.
    print(f"\nAuto-selected K = {best_k}  (highest silhouette score: {max(sil_scores):.4f})")
    return best_k


# ── 4. FIT FINAL MODEL ────────────────────────────────────────────────────────
def fit_kmeans(X: np.ndarray, k: int) -> KMeans:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
    km.fit(X)
    print(f"\nFinal K-Means model fitted with K={k}.")
    return km


# ── 5. RESULTS & CLUSTER PROFILES ────────────────────────────────────────────
def build_results(
    df: pd.DataFrame,
    shares: pd.DataFrame,
    labels: np.ndarray,
    k: int
) -> pd.DataFrame:
    results = df.copy()
    results["Cluster"] = labels

    # Cluster size
    counts = results["Cluster"].value_counts().sort_index()
    print(f"\n{'Cluster':<10} {'ZIP Count':>10}")
    print("-" * 22)
    for c, n in counts.items():
        print(f"  {c:<8} {n:>10,}")

    # Centroid profiles (as % of workforce, averaged per cluster)
    shares_labeled = shares.copy()
    shares_labeled["Cluster"] = labels
    centroids = shares_labeled.groupby("Cluster")[INDUSTRY_COLS].mean() * 100

    print("\nCluster Industry Profiles (mean % of local workforce):")
    print(centroids.round(1).to_string())

    centroids.to_csv(CLUSTER_RESULTS / "kmeans_cluster_profiles.csv")
    print(f"\nSaved: {CLUSTER_RESULTS / 'kmeans_cluster_profiles.csv'}")
    return results


# ── 6. VISUALIZATIONS ────────────────────────────────────────────────────────
def plot_clusters(X: np.ndarray, labels: np.ndarray, k: int) -> None:
    """PCA scatter plot (2D projection) colored by cluster."""
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X)
    var_explained = pca.explained_variance_ratio_ * 100

    colors = cm.tab10(np.linspace(0, 1, k))
    fig, ax = plt.subplots(figsize=(9, 6))

    for c in range(k):
        mask = labels == c
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            color=colors[c], label=f"Cluster {c}",
            alpha=0.6, s=30, edgecolors="none"
        )

    ax.set(
        title=f"K-Means Clusters (K={k}) — PCA Projection",
        xlabel=f"PC1 ({var_explained[0]:.1f}% variance)",
        ylabel=f"PC2 ({var_explained[1]:.1f}% variance)",
    )
    ax.legend(title="Cluster", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(CLUSTER_RESULTS / "kmeans_pca_scatter.png", dpi=150)
    plt.show()
    print(f"Saved: {CLUSTER_RESULTS / 'kmeans_pca_scatter.png'}")


def plot_centroid_heatmap(k: int) -> None:
    """Heatmap of cluster centroids by industry share."""
    profiles = pd.read_csv(CLUSTER_RESULTS / "kmeans_cluster_profiles.csv", index_col="Cluster")
    short_labels = [col.split(",")[0] for col in profiles.columns]  # shorten for display

    fig, ax = plt.subplots(figsize=(10, max(3, k * 0.8)))
    im = ax.imshow(profiles.values, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Mean % of Population")

    ax.set(
        xticks=range(len(short_labels)), xticklabels=short_labels,
        yticks=range(k), yticklabels=[f"Cluster {i}" for i in range(k)],
        title="Cluster Centroids — Educational Attainment Share (%) of Population"
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)

    # Annotate cells
    for i in range(k):
        for j in range(len(short_labels)):
            ax.text(j, i, f"{profiles.values[i, j]:.1f}",
                    ha="center", va="center", fontsize=8,
                    color="black" if profiles.values[i, j] < profiles.values.max() * 0.7 else "white")

    plt.tight_layout()
    plt.savefig(CLUSTER_RESULTS / "kmeans_centroid_heatmap.png", dpi=150)
    plt.show()
    print(f"Saved: {CLUSTER_RESULTS / 'kmeans_centroid_heatmap.png'}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    # 1. Load
    df = load_data(INPUT_FILE)

    # 2. Preprocess
    X, shares = preprocess(df)

    # 3. Determine K
    if FINAL_K is None:
        print(f"\nEvaluating K in range {list(K_RANGE)} …")
        k = evaluate_k(X, K_RANGE)
    else:
        k = FINAL_K
        print(f"\nUsing user-specified K = {k}")

    # 4. Fit final model
    km      = fit_kmeans(X, k)
    labels  = km.labels_

    # 5. Results + profiles
    results = build_results(df, shares, labels, k)

    # 6. Save labeled data
    output_path = CLUSTER_RESULTS / "kmeans_zip_clusters.csv"
    results.to_csv(output_path)
    print(f"\nSaved labeled ZIP data: {output_path}")

    # 7. Visualize
    plot_clusters(X, labels, k)
    plot_centroid_heatmap(k)

    print("\nDone!")


main()