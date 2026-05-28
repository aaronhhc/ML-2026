# V1 Report: Netflix User Rating Behavior Clustering

## 1. Project Objective

The goal of this project is to analyze Netflix Prize rating data and cluster users based on their rating behavior. Instead of predicting future movie ratings, this version focuses on discovering meaningful user groups using unsupervised learning.

The main research question is:

> Can Netflix users be grouped into meaningful behavior-based clusters using only rating records?

---

## 2. Dataset

This project uses a sample from the Netflix Prize dataset. The original data contains user ratings for movies in the following format:

- `movie_id`
- `user_id`
- `rating`
- `date`

For V1, we loaded the first 500,000 lines from `combined_data_1.txt`.

After preprocessing, the dataset contains:

- 499,852 rating records
- 214,992 unique users before filtering
- 237 users after filtering users with fewer than 20 ratings

The rating distribution is:

| Rating | Count |
|---:|---:|
| 1 | 25,316 |
| 2 | 52,325 |
| 3 | 147,151 |
| 4 | 174,248 |
| 5 | 100,812 |

---

## 3. Methodology

The V1 pipeline contains four main steps:

1. Load and parse Netflix Prize rating data
2. Build user-level behavior features
3. Apply K-means clustering
4. Visualize clusters using PCA

Since the task is unsupervised learning, there are no ground-truth labels. Therefore, the model is evaluated using:

- Elbow method
- Silhouette score
- Cluster size distribution
- Interpretability of cluster profiles

---

## 4. Feature Engineering

Each user is represented by rating behavior features.

The V1 features are:

| Feature | Description |
|---|---|
| `avg_rating` | Average rating given by the user |
| `rating_count` | Number of ratings given by the user |
| `rating_std` | Standard deviation of the user's ratings |
| `median_rating` | Median rating given by the user |
| `high_rating_ratio` | Ratio of ratings greater than or equal to 4 |
| `low_rating_ratio` | Ratio of ratings less than or equal to 2 |
| `active_days` | Number of days between the user's first and last rating |
| `rating_frequency` | Average number of ratings per active day |

To reduce the impact of extreme values, `rating_frequency` was clipped at the 99th percentile. In addition, `active_days` was clipped with a minimum value of 7 days to avoid extremely large frequency values caused by very short activity periods.

---

## 5. K-means Clustering Results

K-means clustering was tested with K values from 2 to 8.

| K | Inertia | Silhouette Score |
|---:|---:|---:|
| 2 | 1302.48 | 0.3195 |
| 3 | 1069.32 | 0.2363 |
| 4 | 893.71 | 0.2581 |
| 5 | 784.91 | 0.2698 |
| 6 | 699.77 | 0.2526 |
| 7 | 635.62 | 0.2433 |
| 8 | 586.47 | 0.2300 |

Although K=2 achieved the highest silhouette score, V1 uses K=3 because it provides more interpretable user segments while keeping cluster sizes balanced.

The final cluster sizes are:

| Cluster | User Count |
|---:|---:|
| 0 | 101 |
| 1 | 67 |
| 2 | 69 |

---

## 6. Cluster Interpretation

The cluster summary is:

| Cluster | Avg Rating | Rating Count | Low Rating Ratio | Active Days | Rating Frequency | User Count |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3.05 | 26.48 | 0.26 | 1020.47 | 0.036 | 101 |
| 1 | 1.96 | 38.33 | 0.71 | 732.37 | 0.094 | 67 |
| 2 | 3.78 | 28.03 | 0.09 | 390.28 | 0.133 | 69 |

### Cluster 0: Moderate Long-term Users

Users in this cluster have an average rating close to 3.0 and the longest active period. They rate movies over a long time span but with low frequency. This group can be interpreted as stable, moderate users.

### Cluster 1: Critical Active Users

Users in this cluster have the lowest average rating and the highest low-rating ratio. They also have the highest rating count. This suggests that they are more active but stricter in their evaluations.

### Cluster 2: Generous Short-term Users

Users in this cluster have the highest average rating and the lowest low-rating ratio. They also have the highest rating frequency but the shortest active period. This group may represent users who rate more positively within a shorter time window.

---

## 7. Generated Outputs

The V1 pipeline produces the following files:

### Processed Data

- `data/processed/ratings_sample.csv`
- `data/processed/user_features.csv`
- `data/processed/k_selection_results.csv`
- `data/processed/clustered_users.csv`
- `data/processed/cluster_summary.csv`

### Figures

- `reports/figures/elbow_plot.png`
- `reports/figures/silhouette_plot.png`
- `reports/figures/pca_clusters.png`

---

## 8. Limitations

V1 has several limitations:

1. The dataset sample is relatively small.
2. Only 237 users remain after filtering users with fewer than 20 ratings.
3. The current features are based only on rating behavior.
4. Movie genre and metadata are not included.
5. K-means assumes spherical clusters and may not fully capture complex user behavior.
6. PCA visualization is only a two-dimensional projection of higher-dimensional features.

---

## 9. Next Steps for V2

The next version should improve the project in the following ways:

1. Increase the sample size to 2,000,000 or more lines.
2. Add rating distribution features such as rating 1–5 ratios.
3. Compare different K values more carefully.
4. Improve cluster visualization and interpretation.
5. Consider comparing K-means with Gaussian Mixture Model.
6. Prepare final presentation slides and a polished final report.