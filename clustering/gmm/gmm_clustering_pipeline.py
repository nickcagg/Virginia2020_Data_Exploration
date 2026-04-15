"""
GMM Clustering Pipeline for Virginia ZIP Code Demographics
==========================================================
Topics covered:
  - Educational attainment (population-scaled)
  - Gender disparity log ratios (age-group level)
  - Occupation segments (population-scaled)

Requirements: pandas, numpy, scikit-learn, matplotlib, seaborn
"""

import numpy as np
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 0. COLUMN DEFINITIONS (from data_legend.csv)
# ──────────────────────────────────────────────

EDUCATION_COLS = [
    "Less than high school graduate",
    "High school graduate (includes equivalency)",
    "Some college or associate's degree",
    "Bachelor's degree or higher",
]

LOG_RATIO_COLS = [
    "Population_log_ratio",
    "Under 5 years_log_ratio",
    "5 to 9 years_log_ratio",
    "10 to 14 years_log_ratio",
    "15 to 19 years_log_ratio",
    "20 to 24 years_log_ratio",
    "25 to 29 years_log_ratio",
    "30 to 34 years_log_ratio",
    "35 to 39 years_log_ratio",
    "40 to 44 years_log_ratio",
    "45 to 49 years_log_ratio",
    "50 to 54 years_log_ratio",
    "55 to 59 years_log_ratio",
    "60 to 64 years_log_ratio",
    "65 to 69 years_log_ratio",
    "70 to 74 years_log_ratio",
    "75 to 79 years_log_ratio",
    "80 to 84 years_log_ratio",
    "85 years and over_log_ratio",
]

OCCUPATION_COLS = [
    "Management, business, science, and arts occupations",
    "Service occupations",
    "Sales and office occupations",
    "Natural resources, construction, and maintenance occupations",
    "Production, transportation, and material moving occupations",
]

SCALE_COL_EDUCATION   = "Total in Working Age"
SCALE_COL_OCCUPATION  = "Total Employed Civillians Above 16"
POPULATION_COL        = "Total Population"
ZIP_COL               = "ZIP"

ALL_FEATURE_COLS = EDUCATION_COLS + LOG_RATIO_COLS + OCCUPATION_COLS


# ──────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load the Virginia ZIP-code dataset."""
    df = pd.read_excel(filepath, dtype={"ZIP": str})
    print(f"[load]  Shape: {df.shape}")
    print(f"[load]  Missing values:\n{df[ALL_FEATURE_COLS].isnull().sum()[lambda s: s > 0]}\n")
    return df


# ──────────────────────────────────────────────
# 2. PREPROCESS
# ──────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns
    -------
    features_raw : DataFrame of unscaled, rate-converted features (for inspection)
    features_scaled : DataFrame of StandardScaler-transformed features (model input)
    """
    feat = pd.DataFrame(index=df.index)

    # --- Education: convert counts → rates relative to working-age pop ---
    for col in EDUCATION_COLS:
        denom = df[SCALE_COL_EDUCATION]
        feat[col] = np.where(denom == 0, 0.0, df[col] / denom)

    # --- Gender log ratios: already in log-ratio form, use directly ---
    for col in LOG_RATIO_COLS:
        feat[col] = df[col]

    # --- Occupation: convert counts → rates relative to employed civilians ---
    for col in OCCUPATION_COLS:
        denom = df[SCALE_COL_OCCUPATION]
        feat[col] = np.where(denom == 0, 0.0, df[col] / denom)
    # Drop rows with any remaining NaN (very small ZIPs with 0 denominators)
    n_before = len(feat)
    feat = feat.dropna(subset=LOG_RATIO_COLS)
    print(f"[prep]  Dropped {n_before - len(feat)} rows with NaN after rate conversion.")
    print(f"[prep]  Feature matrix: {feat.shape}\n")

    features_raw = feat.copy()

    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(feat)
    features_scaled = pd.DataFrame(scaled_array, columns=feat.columns, index=feat.index)

    return features_raw, features_scaled


# ──────────────────────────────────────────────
# 3. PCA — DIMENSIONALITY REDUCTION
# ──────────────────────────────────────────────

def run_pca(
    features_scaled: pd.DataFrame,
    variance_target: float = 0.88,
    plot: bool = True,
) -> tuple[np.ndarray, PCA]:
    """
    Fit PCA and retain enough components to explain `variance_target` of variance.
    Returns the reduced array and the fitted PCA object.
    """
    pca_full = PCA(random_state=42)
    pca_full.fit(features_scaled)

    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumvar, variance_target) + 1)
    print(f"[pca]   Components to explain {variance_target:.0%} variance: {n_components}")

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(features_scaled)
    print(f"[pca]   Reduced shape: {X_pca.shape}\n")

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].bar(range(1, len(pca_full.explained_variance_ratio_) + 1),
                    pca_full.explained_variance_ratio_, color="steelblue", alpha=0.7)
        axes[0].set_title("Explained Variance per Component")
        axes[0].set_xlabel("Principal Component")
        axes[0].set_ylabel("Explained Variance Ratio")
        axes[0].axvline(n_components, color="red", linestyle="--", label=f"n={n_components}")
        axes[0].legend()

        axes[1].plot(range(1, len(cumvar) + 1), cumvar, marker="o", color="steelblue")
        axes[1].axhline(variance_target, color="red", linestyle="--",
                        label=f"{variance_target:.0%} target")
        axes[1].axvline(n_components, color="orange", linestyle="--", label=f"n={n_components}")
        axes[1].set_title("Cumulative Explained Variance")
        axes[1].set_xlabel("Number of Components")
        axes[1].set_ylabel("Cumulative Variance")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig("pca_variance.png", dpi=150)
        plt.show()
        print("[pca]   Plot saved → pca_variance.png\n")

    return X_pca, pca


# ──────────────────────────────────────────────
# 4. BIC / SILHOUETTE SWEEP — SELECT k
# ──────────────────────────────────────────────

def select_k(
    X_pca: np.ndarray,
    k_range: range = range(2, 16),
    covariance_type: str = "full",
    plot: bool = True,
) -> int:
    """
    Sweep over candidate k values; select k that minimises BIC.
    Also reports Silhouette Score for reference.
    """
    bic_scores  = []
    sil_scores  = []

    print("[sweep] Fitting GMM for k =", list(k_range))
    for k in k_range:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            n_init=5,
            random_state=42,
        )
        gmm.fit(X_pca)
        labels = gmm.predict(X_pca)

        bic_scores.append(gmm.bic(X_pca))
        sil = silhouette_score(X_pca, labels) if len(set(labels)) > 1 else 0
        sil_scores.append(sil)
        print(f"  k={k:2d}  BIC={gmm.bic(X_pca):,.1f}  Silhouette={sil:.4f}")

    best_k = k_range[int(np.argmin(bic_scores))]
    print(f"\n[sweep] Best k by BIC: {best_k}\n")

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(k_range, bic_scores, marker="o", color="steelblue")
        axes[0].axvline(best_k, color="red", linestyle="--", label=f"Best k={best_k}")
        axes[0].set_title("BIC Score vs. Number of Clusters")
        axes[0].set_xlabel("k")
        axes[0].set_ylabel("BIC")
        axes[0].legend()

        axes[1].plot(k_range, sil_scores, marker="o", color="seagreen")
        axes[1].axvline(best_k, color="red", linestyle="--", label=f"Best k={best_k}")
        axes[1].set_title("Silhouette Score vs. Number of Clusters")
        axes[1].set_xlabel("k")
        axes[1].set_ylabel("Silhouette Score")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig("bic_silhouette_sweep.png", dpi=150)
        plt.show()
        print("[sweep] Plot saved → bic_silhouette_sweep.png\n")

    return best_k


# ──────────────────────────────────────────────
# 5. FIT FINAL GMM
# ──────────────────────────────────────────────

def fit_gmm(
    X_pca: np.ndarray,
    k: int,
    covariance_type: str = "full",
) -> GaussianMixture:
    """Fit the final GMM with the selected k."""
    gmm = GaussianMixture(
        n_components=k,
        covariance_type=covariance_type,
        n_init=10,           # more restarts for final model
        max_iter=300,
        random_state=42,
    )
    gmm.fit(X_pca)
    print(f"[gmm]   Converged: {gmm.converged_}")
    print(f"[gmm]   Final BIC: {gmm.bic(X_pca):,.1f}\n")
    return gmm


# ──────────────────────────────────────────────
# 6. ANALYSE & VISUALISE CLUSTERS
# ──────────────────────────────────────────────

def analyse_clusters(
    df: pd.DataFrame,
    features_raw: pd.DataFrame,
    gmm: GaussianMixture,
    X_pca: np.ndarray,
    zip_col: str = ZIP_COL,
) -> pd.DataFrame:
    """
    Attach cluster labels + soft probabilities to the original DataFrame,
    compute per-cluster feature means, and produce visualisations.
    """
    labels      = gmm.predict(X_pca)
    proba       = gmm.predict_proba(X_pca)          # shape (n, k)
    max_proba   = proba.max(axis=1)                  # confidence of assignment

    result = df.loc[features_raw.index].copy()
    result["Cluster"]     = labels
    result["Confidence"]  = max_proba
    for i in range(gmm.n_components):
        result[f"P_cluster_{i}"] = proba[:, i]

    # ── Cluster summary (raw feature means) ──────────────────────────────
    summary = (
        features_raw
        .assign(Cluster=labels)
        .groupby("Cluster")
        .mean()
        .round(4)
    )
    print("[analysis] Cluster feature means (rates / log-ratios):")
    print(summary.to_string())
    print()

    cluster_sizes = result["Cluster"].value_counts().sort_index()
    print("[analysis] Cluster sizes:")
    print(cluster_sizes.to_string(), "\n")

    # ── Plot 1: PCA scatter coloured by cluster ───────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=labels, cmap="tab10", alpha=0.7, s=30,
    )
    plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_title("GMM Clusters — PCA Space (PC1 vs PC2)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    plt.tight_layout()
    plt.savefig("gmm_pca_scatter.png", dpi=150)
    plt.show()
    print("[analysis] Plot saved → gmm_pca_scatter.png\n")

    # ── Plot 2: Heatmap of cluster means (Z-scored for readability) ───────
    summary_z = (summary - summary.mean()) / summary.std()

    fig, ax = plt.subplots(figsize=(18, max(4, gmm.n_components)))
    sns.heatmap(
        summary_z,
        annot=False,
        cmap="RdBu_r",
        center=0,
        linewidths=0.4,
        ax=ax,
        cbar_kws={"label": "Z-score relative to cluster means"},
    )
    ax.set_title("Cluster Profiles — Feature Z-Scores")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Cluster")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.tight_layout()
    plt.savefig("gmm_cluster_heatmap.png", dpi=150)
    plt.show()
    print("[analysis] Plot saved → gmm_cluster_heatmap.png\n")

    # ── Plot 3: Cluster size bar chart ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    cluster_sizes.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Number of ZIP Codes per Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig("gmm_cluster_sizes.png", dpi=150)
    plt.show()
    print("[analysis] Plot saved → gmm_cluster_sizes.png\n")

    return result


# ──────────────────────────────────────────────
# 7. SAVE RESULTS
# ──────────────────────────────────────────────

def save_results(result: pd.DataFrame, output_path: str = "virginia_zip_clusters.csv"):
    result.to_csv(output_path, index=False)
    print(f"[save]  Results saved → {output_path}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main(filepath: str):
    # 1. Load
    df = load_data(filepath)

    # 2. Preprocess
    features_raw, features_scaled = preprocess(df)

    # 3. PCA  (retain 88% variance by default)
    X_pca, pca = run_pca(features_scaled, variance_target=0.88, plot=True)

    # 4. Select k via BIC sweep
    best_k = select_k(X_pca, k_range=range(2, 16), covariance_type="full", plot=True)

    # 5. Fit final GMM
    gmm = fit_gmm(X_pca, k=best_k, covariance_type="full")

    # 6. Analyse clusters
    result = analyse_clusters(df, features_raw, gmm, X_pca)

    # 7. Save
    save_results(result, "virginia_zip_clusters.csv")

    return result, gmm, pca


if __name__ == "__main__":
    import sys
    filepath = sys.argv[1] if len(sys.argv) > 1 else "your_dataset.csv"
    result, gmm, pca = main(filepath)
