from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

DATA_FILE_NAME = "combined_data_1.txt"

# Use the first 10M lines for the V4 large-scale experiment.
# This produces about 9.99M rating records after excluding movie header lines.
MAX_LINES = 10_000_000


def load_netflix_file(file_path, max_lines=MAX_LINES):
    """
    Load Netflix Prize combined_data file.

    The raw file format uses movie ID headers such as '1:',
    followed by user rating records in the format:
    user_id,rating,date
    """
    rows = []
    current_movie_id = None

    with open(file_path, "r", encoding="latin1") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break

            line = line.strip()

            if not line:
                continue

            if line.endswith(":"):
                current_movie_id = int(line[:-1])
            else:
                user_id, rating, date = line.split(",")

                rows.append({
                    "movie_id": current_movie_id,
                    "user_id": int(user_id),
                    "rating": int(rating),
                    "date": date
                })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    return df


def main():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    input_path = RAW_DATA_DIR / DATA_FILE_NAME
    output_path = PROCESSED_DATA_DIR / "ratings_sample.csv"

    print(f"Loading from: {input_path}")

    df = load_netflix_file(input_path, max_lines=MAX_LINES)

    print("\nPreview:")
    print(df.head())

    print("\nInfo:")
    print(df.info())

    print("\nRating distribution:")
    print(df["rating"].value_counts().sort_index())

    df.to_csv(output_path, index=False)

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()