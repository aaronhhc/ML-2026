from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


CLUSTER_NAME_MAP = {
    0: "Conservative Moderate Raters",
    1: "Diverse Critical Raters",
    2: "Generous Extreme Raters",
}


CLUSTER_DESCRIPTION_MAP = {
    0: "Users who avoid extreme scores and show stable, moderate rating behavior.",
    1: "Users who rate more critically and show greater variation in rating behavior.",
    2: "Users who tend to give high ratings and frequently use strong positive ratings.",
}


def add_cluster_names_to_users():
    input_path = PROCESSED_DATA_DIR / "clustered_users.csv"
    output_path = PROCESSED_DATA_DIR / "clustered_users_named.csv"

    df = pd.read_csv(input_path)

    df["cluster_name"] = df["cluster"].map(CLUSTER_NAME_MAP)
    df["cluster_description"] = df["cluster"].map(CLUSTER_DESCRIPTION_MAP)

    df.to_csv(output_path, index=False)

    print(f"Saved named clustered users to: {output_path}")
    print(df[["user_id", "cluster", "cluster_name"]].head())


def add_cluster_names_to_summary():
    input_path = PROCESSED_DATA_DIR / "cluster_summary.csv"
    output_path = PROCESSED_DATA_DIR / "cluster_summary_named.csv"

    summary = pd.read_csv(input_path)

    summary["cluster_name"] = summary["cluster"].map(CLUSTER_NAME_MAP)
    summary["cluster_description"] = summary["cluster"].map(CLUSTER_DESCRIPTION_MAP)

    # Move names near the front
    cols = ["cluster", "cluster_name", "cluster_description"] + [
        col for col in summary.columns
        if col not in ["cluster", "cluster_name", "cluster_description"]
    ]

    summary = summary[cols]
    summary.to_csv(output_path, index=False)

    print(f"\nSaved named cluster summary to: {output_path}")
    print(summary[["cluster", "cluster_name", "user_count"]])


def main():
    add_cluster_names_to_users()
    add_cluster_names_to_summary()


if __name__ == "__main__":
    main()