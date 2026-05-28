# V3 Report: Netflix User Rating Behavior Clustering

## 1. Objective

V3 further improves the Netflix user clustering project by enhancing feature engineering and cluster interpretability.

The goal remains the same:

> Cluster Netflix users into meaningful behavioral groups using only rating records.

Compared with V2, V3 focuses on adding more behavior-aware features that capture user rating diversity and rating intensity.

---

## 2. V3 Improvements

V3 introduces three new user-level features:

| Feature | Description |
|---|---|
| `log_rating_count` | Log-transformed rating count to reduce the effect of heavy users |
| `rating_entropy` | Measures how diverse a user's rating distribution is |
| `extreme_rating_ratio` | Ratio of extreme ratings, defined as 1-star plus 5-star ratings |

These features improve the behavioral interpretation of clusters.

---

## 3. Dataset

V3 uses the same data scale as V2:

- 1,999,639 rating records
- 342,416 users before filtering
- 15,451 users after filtering users with fewer than 20 ratings

Users with fewer than 20 ratings are removed to reduce noise from users with insufficient rating history.

---

## 4. V3 Feature Set

The V3 clustering model uses 16 features:

| Feature | Description |
|---|---|
| `avg_rating` | Average rating given by the user |
| `log_rating_count` | Log-transformed number of ratings |
| `rating_std` | Standard deviation of the user's ratings |
| `median_rating` | Median rating given by the user |
| `rating_range` | Difference between maximum and minimum rating |
| `high_rating_ratio` | Ratio of ratings greater than or equal to 4 |
| `low_rating_ratio` | Ratio of ratings less than or equal to 2 |
| `rating_1_ratio` | Ratio of 1-star ratings |
| `rating_2_ratio` | Ratio of 2-star ratings |
| `rating_3_ratio` | Ratio of 3-star ratings |
| `rating_4_ratio` | Ratio of 4-star ratings |
| `rating_5_ratio` | Ratio of 5-star ratings |
| `rating_entropy` | Diversity of the user's rating distribution |
| `extreme_rating_ratio` | Ratio of 1-star and 5-star ratings |
| `active_days` | Days between first and last rating |
| `rating_frequency` | Average number of ratings per active day |

`rating_count` is retained in the processed dataset for reporting, but `log_rating_count` is used in clustering to reduce the dominance of very active users.

---

## 5. K-means Results

K-means clustering was tested with K values from 2 to 8.

| K | Inertia | Silhouette Score |
|---:|---:|---:|
| 2 | 188335.42 | 0.2235 |
| 3 | 160968.08 | 0.1901 |
| 4 | 143917.96 | 0.1832 |
| 5 | 130018.47 | 0.1821 |
| 6 | 120356.81 | 0.1823 |
| 7 | 112114.35 | 0.1902 |
| 8 | 107144.02 | 0.1771 |

Although K=2 has the highest silhouette score, K=3 is selected because it provides more interpretable behavioral segments and balanced cluster sizes.

Final cluster sizes:

| Cluster | User Count | Cluster Name |
|---:|---:|---|
| 0 | 5,809 | Generous Extreme Raters |
| 1 | 5,034 | Conservative Moderate Raters |
| 2 | 4,608 | Diverse Critical Raters |

---

## 6. Cluster Interpretation

### Cluster 0: Generous Extreme Raters

Cluster 0 has the highest average rating and a high extreme rating ratio. Because this group also has high positive rating behavior, the extreme rating ratio is mainly driven by frequent 5-star ratings.

This cluster represents users who tend to give high ratings and use strong positive ratings to express preference.

**Behavioral interpretation:**

> Positive users who frequently give strong high ratings.

---

### Cluster 1: Conservative Moderate Raters

Cluster 1 has lower rating standard deviation and the lowest extreme rating ratio. These users avoid extreme ratings and tend to stay closer to moderate scores.

This cluster represents users with stable and conservative rating behavior.

**Behavioral interpretation:**

> Users who avoid extreme scores and prefer moderate ratings.

---

### Cluster 2: Diverse Critical Raters

Cluster 2 has the lowest average rating, the highest rating standard deviation, and relatively higher rating diversity. It also has higher activity and longer active days.

This cluster represents users who rate more critically and show stronger variation across movies.

**Behavioral interpretation:**

> Critical users who differentiate more strongly between movies.

---

## 7. V2 vs V3 Comparison

| Item | V2 | V3 |
|---|---:|---:|
| Features used for clustering | 14 | 16 |
| Added entropy feature | No | Yes |
| Added extreme rating feature | No | Yes |
| Used log-transformed rating count | No | Yes |
| Cluster sizes | 7446 / 3658 / 4347 | 5809 / 5034 / 4608 |
| Cluster balance | Good | Better |
| Interpretation depth | Good | Better |

V3 slightly reduces the silhouette score for K=3 compared with V2, but it produces more balanced clusters and clearer behavioral interpretation.

---

## 8. Outputs

V3 generates the following major outputs:

### Data outputs

- `cluster_summary.csv`
- `cluster_summary_named.csv`

### Figure outputs

- `elbow_plot.png`
- `silhouette_plot.png`
- `pca_clusters.png`
- `cluster_profile_standardized.png`
- `cluster_profile_all_features.png`

### Reports

- `v1_report.md`
- `v2_report.md`
- `v3_report.md`

---

## 9. Limitations

1. The project still uses only a sample of the Netflix Prize dataset.
2. Movie metadata such as genre, title, or release year is not included in clustering.
3. K-means assumes spherical clusters and may not capture more complex user behavior.
4. Silhouette score suggests K=2, while K=3 is selected for interpretability.
5. Rating behavior alone may not fully represent viewing preferences.

---

## 10. Future Work

Future improvements may include:

1. Using more Netflix Prize data files.
2. Incorporating movie metadata.
3. Comparing K-means with Gaussian Mixture Model.
4. Trying dimensionality reduction methods such as UMAP or t-SNE.
5. Connecting clusters to recommendation performance.
6. Studying how different user clusters may experience filter bubbles.

---

## 11. Conclusion

V3 improves the project by adding more behavior-aware features, including rating entropy and extreme rating ratio.

The final model identifies three meaningful user behavior groups:

1. **Generous Extreme Raters**
2. **Conservative Moderate Raters**
3. **Diverse Critical Raters**

These clusters suggest that Netflix users can be meaningfully segmented based on rating behavior alone, even without movie metadata or demographic information.