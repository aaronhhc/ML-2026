from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

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
    input_path = PROCESSED_DATA_DIR / "user_features.csv"
    output_cluster_path = PROCESSED_DATA_DIR / "clustered_users.csv"
    output_k_result_path = PROCESSED_DATA_DIR / "k_selection_results.csv"

    print(f"Loading from: {input_path}")
    df = pd.read_csv(input_path)

    X = df[FEATURE_COLUMNS]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = []

    print("\nTesting different K values...")

    for k in range(2, 9):
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = kmeans.fit_predict(X_scaled)

        inertia = kmeans.inertia_
        silhouette = silhouette_score(X_scaled, labels)

        results.append({
            "k": k,
            "inertia": inertia,
            "silhouette_score": silhouette
        })

        print(
            f"K={k}, "
            f"inertia={inertia:.2f}, "
            f"silhouette={silhouette:.4f}"
        )

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_k_result_path, index=False)

    
    best_k = 3

    print(f"\nTraining final K-means model with K={best_k}...")

    final_model = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=10
    )

    df["cluster"] = final_model.fit_predict(X_scaled)

    df.to_csv(output_cluster_path, index=False)

    print("\nCluster counts:")
    print(df["cluster"].value_counts().sort_index())

    print(f"\nSaved clustered users to: {output_cluster_path}")
    print(f"Saved K selection results to: {output_k_result_path}")


if __name__ == "__main__":
    main()