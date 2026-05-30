# Netflix User Rating Behavior Clustering

## Project Overview

This project analyzes Netflix Prize rating data and clusters users based on their rating behavior.

Instead of predicting movie ratings, this project focuses on unsupervised user segmentation. The goal is to discover meaningful user behavior groups from historical rating records.

The final version uses nearly 10 million Netflix rating records and clusters 154,655 valid users into three interpretable groups.

---

## Research Question

Can Netflix users be grouped into meaningful behavioral clusters using rating records alone?

---

## Dataset

The project uses the Netflix Prize dataset.

For the final version, we use:

```text
combined_data_1.txt
```

Data scale:

| Item | Count |
|---|---:|
| Rating records | 9,998,038 |
| Users before filtering | 447,835 |
| Users after filtering | 154,655 |
| Features used for clustering | 16 |

Raw Netflix data is not included in the repository.

---

## Method

The project pipeline contains six main steps:

1. Load and parse Netflix Prize rating data
2. Convert raw text records into a structured table
3. Build user-level rating behavior features
4. Standardize features
5. Apply MiniBatchKMeans clustering
6. Generate cluster summaries, visualizations, and reports

---

## Feature Engineering

Each user is represented by 16 behavior-level features:

- `avg_rating`
- `log_rating_count`
- `rating_std`
- `median_rating`
- `rating_range`
- `high_rating_ratio`
- `low_rating_ratio`
- `rating_1_ratio`
- `rating_2_ratio`
- `rating_3_ratio`
- `rating_4_ratio`
- `rating_5_ratio`
- `rating_entropy`
- `extreme_rating_ratio`
- `active_days`
- `rating_frequency`

---

## Final Result

The final model uses:

```text
K = 3
```

Final clusters:

| Cluster | User Count | Name |
|---:|---:|---|
| 0 | 66,310 | Conservative Moderate Raters |
| 1 | 54,032 | Diverse Critical Raters |
| 2 | 34,313 | Generous Extreme Raters |

---

## Cluster Interpretation

### Conservative Moderate Raters

Users who avoid extreme scores and show stable, moderate rating behavior.

### Diverse Critical Raters

Users who rate more critically and show greater variation in rating behavior.

### Generous Extreme Raters

Users who tend to give high ratings and frequently use strong positive ratings.

---

## Project Structure

```text
FinalProject/
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       ├── cluster_summary.csv
│       ├── cluster_summary_named.csv
│       └── k_selection_results.csv
├── reports/
│   ├── figures/
│   │   ├── elbow_plot.png
│   │   ├── silhouette_plot.png
│   │   ├── pca_clusters.png
│   │   ├── cluster_profile_standardized.png
│   │   └── cluster_profile_all_features.png
│   ├── v1_report.md
│   ├── v2_report.md
│   ├── v3_report.md
│   └── v4_report.md
├── src/
│   ├── 01_load_data.py
│   ├── 02_feature_engineering.py
│   ├── 03_clustering.py
│   ├── 04_visualization.py
│   ├── 05_cluster_profile.py
│   └── 06_add_cluster_names.py
└── requirements.txt
```

---

## How to Run

Run the scripts from:

```bash
cd FinalProject/src
```

Then execute:

```bash
python 01_load_data.py
python 02_feature_engineering.py
python 03_clustering.py
python 04_visualization.py
python 05_cluster_profile.py
python 06_add_cluster_names.py
```

---

## Script Description

| Script | Description |
|---|---|
| `01_load_data.py` | Loads and parses Netflix Prize raw rating data |
| `02_feature_engineering.py` | Builds user-level behavior features |
| `03_clustering.py` | Runs MiniBatchKMeans and K selection |
| `04_visualization.py` | Generates elbow, silhouette, PCA plots, and cluster summary |
| `05_cluster_profile.py` | Generates standardized cluster profile plots |
| `06_add_cluster_names.py` | Adds human-readable cluster names |

---

## Generated Outputs

Important output files:

| File | Description |
|---|---|
| `cluster_summary.csv` | Cluster-level feature summary |
| `cluster_summary_named.csv` | Cluster summary with cluster names |
| `k_selection_results.csv` | Inertia and silhouette results for different K values |
| `elbow_plot.png` | Elbow method visualization |
| `silhouette_plot.png` | Silhouette score visualization |
| `pca_clusters.png` | PCA-based cluster visualization |
| `cluster_profile_standardized.png` | Simplified cluster profile plot |
| `cluster_profile_all_features.png` | Full cluster profile plot |

---

## Notes

Raw Netflix Prize data and large generated intermediate files are excluded using `.gitignore`.

Excluded files include:

- `data/raw/`
- `ratings_sample.csv`
- `user_features.csv`
- `clustered_users.csv`
- `clustered_users_named.csv`

---

## Conclusion

The project shows that Netflix users can be meaningfully segmented using rating behavior alone.

The final V4 pipeline identifies three interpretable user groups and scales the experiment to nearly 10 million rating records, making the result more robust and suitable for final project presentation.

---

## Alert

Due to file size and license constraints, the raw Netflix Prize dataset is not included in this repository. Users should place `combined_data_1.txt` under `FinalProject/data/raw/` before running the pipeline.