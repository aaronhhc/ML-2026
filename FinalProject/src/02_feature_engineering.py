from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def build_user_features(df):
    df["date"] = pd.to_datetime(df["date"])

    grouped = df.groupby("user_id")

    # Basic user-level statistics
    features = grouped.agg(
        avg_rating=("rating", "mean"),
        rating_count=("rating", "count"),
        rating_std=("rating", "std"),
        median_rating=("rating", "median"),
        min_rating=("rating", "min"),
        max_rating=("rating", "max"),
        first_rating_date=("date", "min"),
        last_rating_date=("date", "max"),
    ).reset_index()

    # Rating range
    features["rating_range"] = features["max_rating"] - features["min_rating"]

    # Rating distribution count table
    rating_counts = (
        df.pivot_table(
            index="user_id",
            columns="rating",
            values="movie_id",
            aggfunc="count",
            fill_value=0,
        )
    )

    # Make sure all rating columns 1 to 5 exist
    for rating_value in range(1, 6):
        if rating_value not in rating_counts.columns:
            rating_counts[rating_value] = 0

    rating_counts = rating_counts[[1, 2, 3, 4, 5]]

    # Convert counts to ratios
    rating_ratios = rating_counts.div(rating_counts.sum(axis=1), axis=0)
    rating_ratios.columns = [
        f"rating_{rating_value}_ratio" for rating_value in rating_ratios.columns
    ]
    rating_ratios = rating_ratios.reset_index()

    features = features.merge(rating_ratios, on="user_id")

    # High / low rating ratios
    features["high_rating_ratio"] = (
        features["rating_4_ratio"] + features["rating_5_ratio"]
    )
    features["low_rating_ratio"] = (
        features["rating_1_ratio"] + features["rating_2_ratio"]
    )

    # Log-scaled rating count to reduce heavy-user effect
    features["log_rating_count"] = np.log1p(features["rating_count"])

    # Rating entropy: measures how diverse the user's rating distribution is
    rating_ratio_cols = [
        "rating_1_ratio",
        "rating_2_ratio",
        "rating_3_ratio",
        "rating_4_ratio",
        "rating_5_ratio",
    ]

    eps = 1e-12
    features["rating_entropy"] = -(
        features[rating_ratio_cols] *
        np.log(features[rating_ratio_cols] + eps)
    ).sum(axis=1)

    # Extreme rating behavior: ratio of 1-star and 5-star ratings
    features["extreme_rating_ratio"] = (
        features["rating_1_ratio"] + features["rating_5_ratio"]
    )

    # Users with only one rating have NaN std
    features["rating_std"] = features["rating_std"].fillna(0)

    # Active period
    features["active_days"] = (
        features["last_rating_date"] - features["first_rating_date"]
    ).dt.days

    # Avoid extremely large frequency from very short active periods
    features["active_days"] = features["active_days"].clip(lower=7)

    # Rating frequency
    features["rating_frequency"] = (
        features["rating_count"] / features["active_days"]
    )

    # Clip high-frequency outliers
    features["rating_frequency"] = features["rating_frequency"].clip(
        upper=features["rating_frequency"].quantile(0.99)
    )

    features = features.drop(
        columns=[
            "first_rating_date",
            "last_rating_date",
            "min_rating",
            "max_rating",
        ]
    )

    return features


def main():
    input_path = PROCESSED_DATA_DIR / "ratings_sample.csv"
    output_path = PROCESSED_DATA_DIR / "user_features.csv"

    print(f"Loading from: {input_path}")
    df = pd.read_csv(input_path)

    print("Building user-level features...")
    features = build_user_features(df)

    print(f"\nBefore filtering: {len(features)} users")

    # Keep users with enough rating history
    features = features[features["rating_count"] >= 20]

    print(f"After filtering rating_count >= 20: {len(features)} users")

    print("\nPreview:")
    print(features.head())

    print("\nFeature summary:")
    print(features.describe())

    features.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()