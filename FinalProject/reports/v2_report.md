# V2 Report: Netflix User Rating Behavior Clustering

## 1. Project Objective

The goal of this project is to analyze Netflix Prize rating data and cluster users based on their rating behavior.

Unlike traditional recommendation tasks that predict future ratings, this project focuses on unsupervised user segmentation. Each user is represented by behavior-level features extracted from their historical movie ratings. K-means clustering is then used to identify different types of users.

The main research question is:

> Can Netflix users be grouped into meaningful behavioral clusters using only rating records?

---

## 2. V2 Improvements over V1

V2 improves the baseline V1 pipeline in three major ways:

| Item | V1 | V2 |
|---|---:|---:|
| Rating records used | 499,852 | 1,999,639 |
| Users before filtering | 214,992 | 342,416 |
| Users after filtering | 237 | 15,451 |
| Number of features | 8 | 14 |
| Cluster stability | Lower | Higher |
| Interpretation depth | Basic | More detailed |

Compared with V1, V2 uses a larger data sample and adds rating distribution features. This allows the clustering result to better capture differences in user rating behavior.

---

## 3. Dataset

This project uses the Netflix Prize dataset. The original data format contains movie rating records in the following structure:

- `movie_id`
- `user_id`
- `rating`
- `date`

For V2, the first 2,000,000 lines from `combined_data_1.txt` were loaded.

After parsing and preprocessing, the dataset contains:

- 1,999,639 rating records
- 342,416 unique users before filtering
- 15,451 users after filtering users with fewer than 20 ratings

The rating distribution is:

| Rating | Count |
|---:|---:|
| 1 | 88,991 |
| 2 | 192,912 |
| 3 | 559,529 |
| 4 | 697,639 |
| 5 | 460,568 |

---

## 4. Feature Engineering

Each user is transformed into a user-level feature vector. Instead of using a sparse user-movie matrix, this project summarizes each user's rating behavior into interpretable numerical features.

The V2 feature set contains 14 features:

| Feature | Description |
|---|---|
| `avg_rating` | Average rating given by the user |
| `rating_count` | Number of ratings given by the user |
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
| `active_days` | Number of days between the user's first and last rating |
| `rating_frequency` | Average number of ratings per active day |

Users with fewer than 20 ratings were removed to reduce noise from users with very limited rating history.

To reduce the influence of extreme outliers, `active_days` was clipped with a lower bound of 7 days, and `rating_frequency` was clipped at the 99th percentile.

---

## 5. Methodology

The V2 pipeline consists of the following steps:

1. Load and parse Netflix Prize rating data
2. Convert raw rating records into a structured table
3. Build user-level rating behavior features
4. Remove users with fewer than 20 ratings
5. Apply outlier handling
6. Standardize features using `StandardScaler`
7. Test K-means clustering with K values from 2 to 8
8. Select the final K value
9. Visualize clusters using PCA
10. Interpret cluster profiles using standardized feature means

Because this is an unsupervised learning task, there is no ground-truth label. Therefore, model evaluation focuses on:

- Elbow method
- Silhouette score
- Cluster size balance
- Cluster interpretability

---

## 6. K Selection

K-means clustering was tested with K values from 2 to 8.

| K | Inertia | Silhouette Score |
|---:|---:|---:|
| 2 | 160498.33 | 0.2457 |
| 3 | 139957.20 | 0.2045 |
| 4 | 126206.99 | 0.1846 |
| 5 | 115880.89 | 0.1797 |
| 6 | 107193.19 | 0.1896 |
| 7 | 98875.33 | 0.1891 |
| 8 | 94495.21 | 0.1775 |

K=2 achieved the highest silhouette score. However, V2 uses K=3 as the final clustering setting because K=3 provides more interpretable user behavior segments while maintaining reasonably balanced cluster sizes.

The final cluster sizes are:

| Cluster | User Count |
|---:|---:|
| 0 | 7,446 |
| 1 | 3,658 |
| 2 | 4,347 |

---

## 7. Cluster Summary

The average feature values of the three clusters are summarized below:

| Cluster | Avg Rating | Rating Count | Rating Std | Rating 5 Ratio | Active Days | Rating Frequency | User Count |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3.86 | 26.40 | 0.92 | 0.284 | 658.12 | 0.088 | 7,446 |
| 1 | 3.01 | 30.35 | 1.19 | 0.134 | 810.94 | 0.087 | 3,658 |
| 2 | 3.27 | 26.99 | 0.83 | 0.074 | 716.62 | 0.086 | 4,347 |

---

## 8. Cluster Interpretation

### Cluster 0: Generous Positive Raters

Cluster 0 has the highest average rating, highest high-rating ratio, and highest 5-star rating ratio. It also has a relatively low low-rating ratio.

This suggests that users in this cluster tend to give positive ratings and are more likely to assign 4-star or 5-star ratings.

**Interpretation:**

> These users are positive raters who are generally more satisfied with the movies they rate.

---

### Cluster 1: Diverse Critical Raters

Cluster 1 has the lowest average rating, highest rating standard deviation, and highest low-rating ratio. It also has the highest average rating count and longest active period.

This suggests that users in this cluster are more critical, more active, and more diverse in their evaluations. They are more likely to give low ratings, but their high standard deviation also indicates that they differentiate more strongly between movies.

**Interpretation:**

> These users are critical raters who evaluate movies more strictly and show greater variation in rating behavior.

---

### Cluster 2: Conservative Moderate Raters

Cluster 2 has lower rating variance and the lowest 5-star rating ratio. Its average rating is below Cluster 0 but higher than Cluster 1.

This suggests that users in this cluster are more conservative in their rating behavior. They rarely give extreme high ratings and tend to stay closer to moderate scores.

**Interpretation:**

> These users are moderate raters who avoid extreme positive ratings and show more conservative rating behavior.

---

## 9. Visualization Outputs

The V2 pipeline generates the following figures:

| Figure | Purpose |
|---|---|
| `elbow_plot.png` | Shows how inertia changes across K values |
| `silhouette_plot.png` | Compares silhouette scores across K values |
| `pca_clusters.png` | Visualizes user clusters in 2D PCA space |
| `cluster_profile_standardized.png` | Shows representative standardized cluster profiles |
| `cluster_profile_all_features.png` | Shows standardized profiles across all 14 features |

The standardized cluster profile plot is especially useful for interpretation because it shows whether each cluster is above or below the overall average for each feature.

---

## 10. Business and Recommendation Implications

The clustering results suggest that users can be segmented based on rating behavior alone.

Possible recommendation implications include:

| User Segment | Possible Recommendation Strategy |
|---|---|
| Generous Positive Raters | Recommend popular or highly rated movies to maintain engagement |
| Diverse Critical Raters | Recommend more carefully matched movies with stronger personalization |
| Conservative Moderate Raters | Recommend safe, mainstream, and consistently well-rated content |

These clusters can help a streaming platform understand different rating behaviors and design more targeted recommendation strategies.

However, if a recommendation system repeatedly reinforces the same user behavior pattern, users may be exposed to a narrower range of content over time. This connects to the concept of a filter bubble, where recommendation systems may limit content diversity by repeatedly recommending similar items.

---

## 11. Limitations

This project has several limitations:

1. Only a sample of the Netflix Prize dataset is used.
2. The project uses rating behavior only and does not include movie genres or metadata.
3. User demographics such as age, gender, and location are not available.
4. K-means assumes roughly spherical clusters and may not capture more complex user behavior patterns.
5. PCA visualization is only a two-dimensional projection of higher-dimensional data.
6. The selected K value balances interpretability and cluster quality, but K=2 has a higher silhouette score.
7. The dataset is old and may not fully represent modern streaming behavior.

---

## 12. Future Work

Future improvements may include:

1. Increasing the sample size further.
2. Using all four Netflix Prize combined data files.
3. Incorporating movie metadata such as title, release year, or genre.
4. Comparing K-means with Gaussian Mixture Model or hierarchical clustering.
5. Applying dimensionality reduction methods such as t-SNE or UMAP for visualization.
6. Connecting user clusters with recommendation performance.
7. Investigating whether certain clusters are more likely to experience filter bubble effects.

---

## 13. Conclusion

V2 successfully improves the baseline clustering pipeline by using a larger sample size and a richer set of rating behavior features.

The final K-means model groups users into three interpretable behavioral segments:

1. **Generous Positive Raters**
2. **Diverse Critical Raters**
3. **Conservative Moderate Raters**

These results show that even without movie genre or demographic data, user rating behavior alone can reveal meaningful user segments. This supports the idea that unsupervised learning can be useful for understanding user behavior in recommendation systems.s