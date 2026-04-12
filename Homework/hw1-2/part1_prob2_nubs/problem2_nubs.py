import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA

# =========================
# 1. Load data
# =========================
def load_data(filepath):
    return np.loadtxt(filepath)

# =========================
# 2. Compute squared distances
# X: (N, 3)
# M: (K, 3)
# return: (N, K)
# =========================
def squared_distances(X, M):
    return np.sum((X[:, None, :] - M[None, :, :]) ** 2, axis=2)

# =========================
# 3. E-step
# =========================
def e_step(X, centroids):
    dists = squared_distances(X, centroids)
    labels = np.argmin(dists, axis=1)
    distortion = np.sum(np.min(dists, axis=1))
    return labels, distortion

# =========================
# 4. M-step
# =========================
def m_step(X, labels, old_centroids, K):
    new_centroids = old_centroids.copy()
    for k in range(K):
        cluster_points = X[labels == k]
        if len(cluster_points) > 0:
            new_centroids[k] = cluster_points.mean(axis=0)
    return new_centroids

# =========================
# 5. K-means(2) with PCA initialisation
# initialise along axis of maximum variance
# =========================
def run_kmeans2(X, max_iter=100):
    if len(X) < 2:
        return np.zeros(len(X), dtype=int), X.mean(axis=0, keepdims=True).repeat(2, axis=0)

    # axis of maximum variance via PCA (first principal component)
    pca = PCA(n_components=1)
    projections = pca.fit_transform(X).flatten()
    pc1 = pca.components_[0]
    
    # project points onto pc1, split at median
    mean = X.mean(axis=0)
    median = np.median(projections)
    init_centroids = np.array([
        X[projections <= median].mean(axis=0) if np.any(projections <= median) else mean - pc1,
        X[projections >  median].mean(axis=0) if np.any(projections >  median) else mean + pc1,
    ])

    centroids = init_centroids.copy().astype(float)
    prev_labels = None

    for _ in range(max_iter):
        labels, _ = e_step(X, centroids)

        if prev_labels is not None and np.array_equal(labels, prev_labels):
            break

        new_centroids = m_step(X, labels, centroids, 2)

        if np.allclose(new_centroids, centroids):
            centroids = new_centroids
            break

        centroids = new_centroids
        prev_labels = labels.copy()

    final_labels, _ = e_step(X, centroids)
    return final_labels, centroids

# =========================
# 6. Cluster distortion
# =========================
def cluster_distortion(X, mean):
    return np.sum(np.sum((X - mean) ** 2, axis=1))

# =========================
# 7. NUBS main loop
# returns list of snapshots:
#   each snapshot = (labels_global, centroids_list, split_cluster_idx, total_distortion)
# =========================
def run_nubs(X, K=5):
    N = len(X)

    # state: list of (indices_array, centroid) per cluster
    clusters = [(np.arange(N), X.mean(axis=0))]

    snapshots = []  # recorded after every split (including initial state)

    def build_global_labels(clusters):
        labels = np.empty(N, dtype=int)
        centroids = []
        for k, (idx, c) in enumerate(clusters):
            labels[idx] = k
            centroids.append(c)
        return labels, np.array(centroids)

    def total_distortion(clusters):
        return sum(cluster_distortion(X[idx], c) for idx, c in clusters)

    # record initial state (1 cluster)
    labels0, centroids0 = build_global_labels(clusters)
    snapshots.append({
        "num_clusters": 1,
        "split_cluster": None,
        "total_distortion": total_distortion(clusters),
        "labels": labels0,
        "centroids": centroids0,
    })

    while len(clusters) < K:
        # find cluster with highest distortion
        distortions = [cluster_distortion(X[idx], c) for idx, c in clusters]
        split_k = int(np.argmax(distortions))
        split_idx, _ = clusters[split_k]

        # split using K-means(2)
        local_labels, local_centroids = run_kmeans2(X[split_idx])

        # replace split cluster with two new clusters
        idx_a = split_idx[local_labels == 0]
        idx_b = split_idx[local_labels == 1]
        centroid_a = X[idx_a].mean(axis=0) if len(idx_a) > 0 else local_centroids[0]
        centroid_b = X[idx_b].mean(axis=0) if len(idx_b) > 0 else local_centroids[1]

        clusters = (
            clusters[:split_k]
            + [(idx_a, centroid_a), (idx_b, centroid_b)]
            + clusters[split_k + 1:]
        )

        labels_g, centroids_g = build_global_labels(clusters)
        snapshots.append({
            "num_clusters": len(clusters),
            "split_cluster": split_k + 1,   # 1-indexed for display
            "total_distortion": total_distortion(clusters),
            "labels": labels_g,
            "centroids": centroids_g,
        })

    return snapshots

# =========================
# 8. Plot grid of 3D scatter plots (one per step)
# =========================
def plot_nubs_grid(X, snapshots, savepath):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    n_steps = len(snapshots)  # should be K (1..K clusters)
    colors = ["blue", "black", "red", "green", "magenta"]

    fig = plt.figure(figsize=(5 * n_steps, 5))

    for col, snap in enumerate(snapshots):
        ax = fig.add_subplot(1, n_steps, col + 1, projection="3d")
        labels = snap["labels"]
        centroids = snap["centroids"]
        n_clusters = snap["num_clusters"]

        for k in range(n_clusters):
            pts = X[labels == k]
            ax.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2],
                s=8, c=colors[k % len(colors)], alpha=0.7, label=f"C{k+1}"
            )

        ax.scatter(
            centroids[:, 0], centroids[:, 1], centroids[:, 2],
            c="yellow", s=180, marker="*", edgecolors="black", zorder=5
        )

        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

        if snap["split_cluster"] is None:
            title = "Initial\n(1 cluster)"
        else:
            title = (
                f"Split C{snap['split_cluster']} → {n_clusters} clusters\n"
                f"D = {snap['total_distortion']:.1f}"
            )
        ax.set_title(title, fontsize=9)

    plt.suptitle("NUBS Hierarchical Clustering (K=5)", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close()

# =========================
# 9. Print NUBS split log
# =========================
def print_nubs_log(snapshots):
    print(f"{'Step':>4}  {'Clusters':>8}  {'Split':>10}  {'Total Distortion':>18}")
    print("-" * 50)
    for snap in snapshots:
        split_str = f"C{snap['split_cluster']}" if snap["split_cluster"] is not None else "—"
        print(
            f"{snapshots.index(snap):>4}  "
            f"{snap['num_clusters']:>8}  "
            f"{split_str:>10}  "
            f"{snap['total_distortion']:>18.4f}"
        )

# =========================
# 10. Main
# =========================
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "../hw1-2_data/Data_Exercise_2.txt"
    figure_dir = base_dir / "figure"
    figure_dir.mkdir(exist_ok=True)

    X = load_data(data_path)

    snapshots = run_nubs(X, K=5)

    print("===== NUBS Split Log =====")
    print_nubs_log(snapshots)

    plot_nubs_grid(
        X, snapshots,
        savepath=figure_dir / "nubs_splits_grid.png"
    )

    nubs_final_distortion = snapshots[-1]["total_distortion"]
    print(f"\nNUBS final distortion (K=5): {nubs_final_distortion:.4f}")