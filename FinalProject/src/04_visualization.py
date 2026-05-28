from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

FEATURE_COLUMNS = [
    "avg_rating",
    "log_rating_count",
    "rating_std",
    "median_rating",
    "rating_range",
    "high_rating_ratio",
    "low_rating_ratio",
    "rating_1_ratio",
    "rating_2_ratio",
    "rating_3_ratio",
    "rating_4_ratio",
    "rating_5_ratio",
    "rating_entropy",
    "extreme_rating_ratio",
    "active_days",
    "rating_frequency",
]


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    clustered_path = PROCESSED_DATA_DIR / "clustered_users.csv"
    k_result_path = PROCESSED_DATA_DIR / "k_selection_results.csv"

    df = pd.read_csv(clustered_path)
    k_results = pd.read_csv(k_result_path)

    # -------------------------
    # Elbow plot
    # -------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(k_results["k"], k_results["inertia"], marker="o")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for K Selection")
    plt.grid(True, alpha=0.3)
    plt.savefig(FIGURE_DIR / "elbow_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------
    # Silhouette plot
    # -------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(k_results["k"], k_results["silhouette_score"], marker="o")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score by K")
    plt.grid(True, alpha=0.3)
    plt.savefig(FIGURE_DIR / "silhouette_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------
    # PCA visualization
    # -------------------------
    X = df[FEATURE_COLUMNS]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    df["PC1"] = X_pca[:, 0]
    df["PC2"] = X_pca[:, 1]

    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(
        df["PC1"],
        df["PC2"],
        c=df["cluster"],
        alpha=0.7
    )
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("User Clusters Visualized by PCA")
    plt.colorbar(scatter, label="Cluster")
    plt.grid(True, alpha=0.3)
    plt.savefig(FIGURE_DIR / "pca_clusters.png", dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------
    # Cluster summary
    # -------------------------
    summary = df.groupby("cluster")[FEATURE_COLUMNS].mean()
    summary["user_count"] = df.groupby("cluster").size()
    summary = summary.reset_index()

    summary_path = PROCESSED_DATA_DIR / "cluster_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("Saved figures:")
    print(FIGURE_DIR / "elbow_plot.png")
    print(FIGURE_DIR / "silhouette_plot.png")
    print(FIGURE_DIR / "pca_clusters.png")

    print(f"\nSaved cluster summary to: {summary_path}")

    print("\nCluster summary:")
    print(summary)


if __name__ == "__main__":
    main()