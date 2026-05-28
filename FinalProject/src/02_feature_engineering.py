from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def build_user_features(df):
    df["date"] = pd.to_datetime(df["date"])

    grouped = df.groupby("user_id")

    features = grouped.agg(
        avg_rating=("rating", "mean"),
        rating_count=("rating", "count"),
        rating_std=("rating", "std"),
        median_rating=("rating", "median"),
        first_rating_date=("date", "min"),
        last_rating_date=("date", "max"),
    ).reset_index()

    high_ratio = (
        grouped.apply(lambda x: (x["rating"] >= 4).mean())
        .reset_index(name="high_rating_ratio")
    )

    low_ratio = (
        grouped.apply(lambda x: (x["rating"] <= 2).mean())
        .reset_index(name="low_rating_ratio")
    )

    features = features.merge(high_ratio, on="user_id")
    features = features.merge(low_ratio, on="user_id")

    features["rating_std"] = features["rating_std"].fillna(0)

    features["active_days"] = (
        features["last_rating_date"] - features["first_rating_date"]
    ).dt.days

    features["active_days"] = features["active_days"].clip(lower=7)

    features["rating_frequency"] = (
        features["rating_count"] / features["active_days"]
    )

    features["rating_frequency"] = features["rating_frequency"].clip(
    upper=features["rating_frequency"].quantile(0.99)
    )

    features = features.drop(
        columns=["first_rating_date", "last_rating_date"]
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