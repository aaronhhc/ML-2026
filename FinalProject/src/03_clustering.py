from pathlib import Path

import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


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

K_RANGE = range(2, 9)
BEST_K = 3

# MiniBatchKMeans is used for the V4 large-scale experiment.
BATCH_SIZE = 4096
N_INIT = 10
RANDOM_STATE = 42

# Full silhouette computation is expensive for 154k+ users.
# We estimate it using a fixed random sample for efficiency and reproducibility.
SILHOUETTE_SAMPLE_SIZE = 10_000


def main():
    input_path = PROCESSED_DATA_DIR / "user_features.csv"
    output_cluster_path = PROCESSED_DATA_DIR / "clustered_users.csv"
    output_k_result_path = PROCESSED_DATA_DIR / "k_selection_results.csv"

    print(f"Loading from: {input_path}")
    df = pd.read_csv(input_path)

    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")

    X = df[FEATURE_COLUMNS]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = []

    print("\nTesting different K values...")

    for k in K_RANGE:
        print(f"Running MiniBatchKMeans for K={k}...")

        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            batch_size=BATCH_SIZE,
            n_init=N_INIT,
        )

        labels = model.fit_predict(X_scaled)

        inertia = model.inertia_
        silhouette = silhouette_score(
            X_scaled,
            labels,
            sample_size=SILHOUETTE_SAMPLE_SIZE,
            random_state=RANDOM_STATE,
        )

        results.append(
            {
                "k": k,
                "inertia": inertia,
                "silhouette_score": silhouette,
            }
        )

        print(
            f"K={k}, "
            f"inertia={inertia:.2f}, "
            f"silhouette={silhouette:.4f}"
        )

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_k_result_path, index=False)

    print(f"\nTraining final MiniBatchKMeans model with K={BEST_K}...")

    final_model = MiniBatchKMeans(
        n_clusters=BEST_K,
        random_state=RANDOM_STATE,
        batch_size=BATCH_SIZE,
        n_init=N_INIT,
    )

    df["cluster"] = final_model.fit_predict(X_scaled)

    df.to_csv(output_cluster_path, index=False)

    print("\nCluster counts:")
    print(df["cluster"].value_counts().sort_index())

    print(f"\nSaved clustered users to: {output_cluster_path}")
    print(f"Saved K selection results to: {output_k_result_path}")


if __name__ == "__main__":
    main()