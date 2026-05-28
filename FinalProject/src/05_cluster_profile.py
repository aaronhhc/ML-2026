from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"


SUMMARY_PROFILE_FEATURES = [
    "avg_rating",
    "log_rating_count",
    "rating_std",
    "high_rating_ratio",
    "low_rating_ratio",
    "rating_5_ratio",
    "rating_entropy",
    "extreme_rating_ratio",
    "active_days",
    "rating_frequency",
]


ALL_PROFILE_FEATURES = [
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

def create_cluster_profile_plot(df, features, output_path, title, figsize=(12, 6)):
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df[features])

    scaled_df = pd.DataFrame(
        scaled_values,
        columns=features
    )

    scaled_df["cluster"] = df["cluster"].values

    profile = scaled_df.groupby("cluster")[features].mean()

    print(f"\n{title}")
    print(profile)

    ax = profile.T.plot(
        kind="bar",
        figsize=figsize,
        rot=45
    )

    ax.set_title(title)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Standardized Mean Value")
    ax.legend(title="Cluster")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved to: {output_path}")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    input_path = PROCESSED_DATA_DIR / "clustered_users.csv"

    print(f"Loading from: {input_path}")
    df = pd.read_csv(input_path)

    create_cluster_profile_plot(
        df=df,
        features=SUMMARY_PROFILE_FEATURES,
        output_path=FIGURE_DIR / "cluster_profile_standardized.png",
        title="Standardized Cluster Profile Comparison",
        figsize=(12, 6),
    )

    create_cluster_profile_plot(
        df=df,
        features=ALL_PROFILE_FEATURES,
        output_path=FIGURE_DIR / "cluster_profile_all_features.png",
        title="Full Standardized Cluster Profile Comparison",
        figsize=(16, 7),
    )


if __name__ == "__main__":
    main()