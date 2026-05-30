# V4 Report: Large-Scale Netflix User Rating Behavior Clustering

## 1. Objective

The goal of this project is to analyze Netflix Prize rating data and cluster users based on their rating behavior.

Instead of predicting movie ratings, this project focuses on unsupervised user segmentation. Each user is represented by behavior-level features extracted from historical rating records, and clustering is used to identify different types of users.

The main research question is:

> Can Netflix users be grouped into meaningful behavioral clusters using rating behavior alone?

V4 is the large-scale version of the project. It scales the experiment from 2 million rating records to nearly 10 million rating records to verify whether the clustering results remain stable with a much larger sample.

---

## 2. V4 Improvements

V4 builds on the previous versions:

| Version | Main Purpose | Data Size | Main Improvement |
|---|---:|---:|---|
| V1 | Baseline pipeline | 500K ratings | Basic user clustering |
| V2 | Larger data + richer features | 2M ratings | Rating distribution features |
| V3 | Better behavior features | 2M ratings | Entropy and extreme rating features |
| V4 | Large-scale robustness | 10M ratings | Scaled experiment and more stable user sample |

The key improvement in V4 is scale. V4 uses nearly 10 million rating records and obtains over 154,000 valid users after filtering.

---

## 3. Dataset

This project uses the Netflix Prize dataset.

For V4, the project uses the first 10 million lines from:

```text
combined_data_1.txt
```

After parsing the raw text file, the dataset contains:

| Item | Count |
|---|---:|
| Rating records | 9,998,038 |
| Users before filtering | 447,835 |
| Users after filtering | 154,655 |
| Features used for clustering | 16 |

The rating distribution is:

| Rating | Count |
|---:|---:|
| 1 | 442,214 |
| 2 | 974,633 |
| 3 | 2,842,781 |
| 4 | 3,445,626 |
| 5 | 2,292,784 |

Users with fewer than 20 ratings are removed to reduce noise from users with limited rating history.

---

## 4. Preprocessing

The original Netflix Prize data is stored in a raw text format where movie IDs appear as headers, followed by user rating records.

The preprocessing pipeline converts the raw file into a structured table with the following columns:

| Column | Description |
|---|---|
| `movie_id` | Movie identifier |
| `user_id` | User identifier |
| `rating` | Rating score from 1 to 5 |
| `date` | Rating date |

After loading the rating records, the project aggregates rating-level data into user-level behavior features.

---

## 5. Feature Engineering

The V4 model uses 16 user-level features for clustering:

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

Two important behavior features are `rating_entropy` and `extreme_rating_ratio`.

### Rating Entropy

Rating entropy measures how diverse a user's rating distribution is.

A user who gives many different rating scores has higher entropy, while a user who mostly gives the same score has lower entropy.

### Extreme Rating Ratio

Extreme rating ratio is defined as:

```text
rating_1_ratio + rating_5_ratio
```

It measures how often a user gives extreme ratings.

---

## 6. Method

The project uses MiniBatchKMeans for V4.

MiniBatchKMeans is a scalable version of K-means. It updates cluster centers using small batches of data, making it more efficient for large datasets.

The clustering pipeline is:

1. Load Netflix Prize rating data
2. Convert raw text records into a structured rating table
3. Aggregate rating records into user-level features
4. Remove users with fewer than 20 ratings
5. Standardize features using `StandardScaler`
6. Test different K values from 2 to 8
7. Evaluate clustering using inertia and silhouette score
8. Train final model with K = 3
9. Generate cluster summaries and visualizations
10. Assign human-readable cluster names

For the V4 large-scale experiment, silhouette score is computed using a fixed random sample of 10,000 users for efficiency. This reduces computation time while keeping the K comparison reproducible.

---

## 7. K Selection

MiniBatchKMeans was tested with K values from 2 to 8.

| K | Inertia | Silhouette Score |
|---:|---:|---:|
| 2 | 1,850,506.46 | 0.2242 |
| 3 | 1,596,239.12 | 0.1887 |
| 4 | 1,414,671.22 | 0.1793 |
| 5 | 1,271,192.03 | 0.1834 |
| 6 | 1,203,806.96 | 0.1626 |
| 7 | 1,147,379.98 | 0.1458 |
| 8 | 1,088,308.20 | 0.1514 |

Although K = 2 gives the highest silhouette score, it mainly separates users into broader high-rating and lower/more diverse-rating groups. Since the goal of this project is user behavior segmentation, K = 3 is selected because it provides a more informative behavioral structure. In particular, K = 3 separates moderate users from critical users and generous high-rating users, which is more useful for interpretation and recommendation-related analysis.

---

## 8. Final Clustering Result

The final model uses:

```text
K = 3
```

Final cluster sizes:

| Cluster | User Count | Cluster Name |
|---:|---:|---|
| 0 | 66,310 | Conservative Moderate Raters |
| 1 | 54,032 | Diverse Critical Raters |
| 2 | 34,313 | Generous Extreme Raters |

The three clusters are all large enough for meaningful interpretation. There is no very small outlier cluster.

---

## 9. Cluster Interpretation

| Cluster | Key Above-Average Features | Key Below-Average Features | Interpretation |
|---:|---|---|---|
| 0 | Slightly higher average rating | Lower rating standard deviation, lower extreme rating ratio | Stable and moderate rating behavior |
| 1 | Higher rating count, higher rating standard deviation, longer active days | Lower average rating | Critical and diverse rating behavior |
| 2 | Higher average rating, higher extreme rating ratio, higher rating frequency | Shorter active days, lower rating count | Generous and strongly positive rating behavior |

### Cluster 0: Conservative Moderate Raters

Cluster 0 is the largest group. These users have relatively stable rating behavior, lower rating variance, lower rating entropy, and lower extreme rating ratio.

They do not frequently give extreme ratings. Their rating behavior is more moderate and consistent.

**Interpretation:**

> Users who avoid extreme scores and show stable, moderate rating behavior.

---

### Cluster 1: Diverse Critical Raters

Cluster 1 has the lowest average rating, higher rating standard deviation, higher rating entropy, higher rating count, and longer active days.

These users rate more critically and show more diverse rating behavior. They are more likely to distinguish strongly between movies.

**Interpretation:**

> Users who rate more critically and show greater variation in rating behavior.

---

### Cluster 2: Generous Extreme Raters

Cluster 2 has the highest average rating, highest high-rating ratio, highest 5-star rating ratio, and highest extreme rating ratio.

Since the average rating is very high, the extreme rating behavior is mainly driven by 5-star ratings.

**Interpretation:**

> Users who tend to give high ratings and frequently use strong positive ratings.

---

## 10. V3 vs V4 Comparison

| Item | V3 | V4 |
|---|---:|---:|
| Rating records | 1,999,639 | 9,998,038 |
| Valid users | 15,451 | 154,655 |
| Features used | 16 | 16 |
| Clustering method | KMeans | MiniBatchKMeans |
| Final K | 3 | 3 |
| K=3 silhouette score | 0.1901 | 0.1887 |

The K=3 silhouette score remains very similar between V3 and V4. This suggests that increasing the dataset from 2 million to nearly 10 million ratings produces stable clustering behavior.

Therefore, V4 supports the robustness of the discovered user segments.

---

## 11. Visualization Outputs

V4 generates the following figures:

| Figure | Purpose |
|---|---|
| `elbow_plot.png` | Shows inertia across K values |
| `silhouette_plot.png` | Shows silhouette score across K values |
| `pca_clusters.png` | Visualizes clusters in 2D PCA space |
| `cluster_profile_standardized.png` | Shows representative standardized cluster profiles |
| `cluster_profile_all_features.png` | Shows standardized profiles across all 16 features |

The cluster profile plots are especially useful because they show how each cluster differs from the overall average.

---

## 12. Recommendation Implications

The clustering result can help interpret different user behavior types in recommendation systems.

| Cluster | Possible Recommendation Strategy |
|---|---|
| Conservative Moderate Raters | Recommend reliable and broadly popular content |
| Diverse Critical Raters | Use stronger personalization and diverse recommendations |
| Generous Extreme Raters | Recommend highly rated or emotionally engaging content |

These segments can also be used to discuss filter bubble risk. If a recommendation system repeatedly reinforces the same user behavior pattern, users may receive less diverse content over time.

---

## 13. Limitations

This project has several limitations:

1. Only `combined_data_1.txt` is used, not the full Netflix Prize dataset.
2. The project focuses on user rating behavior and does not include movie metadata.
3. Genres, directors, actors, and content descriptions are not used.
4. User demographic information is not available.
5. K-means assumes relatively simple cluster shapes.
6. K=2 has the highest silhouette score, while K=3 is selected for interpretability.
7. The Netflix Prize dataset is old and may not fully represent modern streaming behavior.

The V4 experiment uses the first 10 million lines from `combined_data_1.txt`. Since Netflix Prize data is organized by movie blocks rather than randomly shuffled records, this may introduce movie sampling bias. Therefore, the result should be interpreted as a large-scale exploratory analysis rather than a fully random sample of the entire Netflix Prize dataset.

---

## 14. Future Work

Future work may include:

1. Using all four Netflix Prize combined data files.
2. Adding movie metadata such as title, year, or genre.
3. Comparing MiniBatchKMeans with Gaussian Mixture Model.
4. Testing hierarchical clustering or density-based clustering.
5. Connecting user clusters to recommendation performance.
6. Studying filter bubble effects across different user clusters.
7. Building a simple recommendation strategy for each user segment.

---

## 15. Conclusion

V4 successfully scales the Netflix user behavior clustering project to nearly 10 million rating records and 154,655 valid users.

The final model identifies three meaningful user behavior segments:

1. **Conservative Moderate Raters**
2. **Diverse Critical Raters**
3. **Generous Extreme Raters**

The V4 result shows that rating behavior alone can reveal interpretable user groups. The similarity between V3 and V4 results also suggests that the discovered clusters are reasonably stable under a larger sample size.
