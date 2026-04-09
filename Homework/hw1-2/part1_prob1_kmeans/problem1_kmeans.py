import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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
# assign each point to nearest centroid
# =========================
def e_step(X, centroids):
    dists = squared_distances(X, centroids)
    labels = np.argmin(dists, axis=1)
    distortion = np.sum(np.min(dists, axis=1))
    return labels, distortion

# =========================
# 4. M-step
# recompute centroids
# if a cluster is empty, keep old centroid
# =========================
def m_step(X, labels, old_centroids, K):
    new_centroids = old_centroids.copy()
    for k in range(K):
        cluster_points = X[labels == k]
        if len(cluster_points) > 0:
            new_centroids[k] = cluster_points.mean(axis=0)
    return new_centroids

# =========================
# 5. K-means main loop
# logs distortion after every E-step and M-step
# =========================
def run_kmeans(X, init_centroids, max_iter=100):
    K = init_centroids.shape[0]
    centroids = init_centroids.copy().astype(float)

    history = []
    prev_labels = None

    for it in range(max_iter):
        # E-step
        labels, D_e = e_step(X, centroids)
        history.append(("E", it + 1, D_e))

        # convergence by unchanged assignments
        if prev_labels is not None and np.array_equal(labels, prev_labels):
            break

        # M-step
        new_centroids = m_step(X, labels, centroids, K)
        _, D_m = e_step(X, new_centroids)
        history.append(("M", it + 1, D_m))

        # convergence by unchanged centroids
        if np.allclose(new_centroids, centroids):
            centroids = new_centroids
            break

        centroids = new_centroids
        prev_labels = labels.copy()

    # final assignment with final centroids
    final_labels, final_distortion = e_step(X, centroids)
    return centroids, final_labels, final_distortion, history

# =========================
# 6. Plot distortion curve
# =========================
def plot_distortion_curve(history, title, savepath):
    x = np.arange(1, len(history) + 1)
    y = [h[2] for h in history]
    step_type = [h[0] for h in history]

    plt.figure(figsize=(8, 5))
    for i, (xi, yi, s) in enumerate(zip(x, y, step_type)):
        if s == "E":
            plt.scatter(xi, yi, marker="o", label="E-step" if i == 0 else "")
        else:
            plt.scatter(xi, yi, marker="s", label="M-step" if i == 1 else "")
    plt.plot(x, y, alpha=0.7)
    plt.xlabel("Step Index")
    plt.ylabel("Distortion D")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()

# =========================
# 7. Plot final 3D clustering
# =========================
def plot_3d_clusters(X, labels, centroids, title, savepath):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    colors = ["blue", "black", "red", "green", "magenta"]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    for k in range(5):
        pts = X[labels == k]
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            s=8, c=colors[k], label=f"C{k+1}", alpha=0.7
        )

    ax.scatter(
        centroids[:, 0], centroids[:, 1], centroids[:, 2],
        c="yellow", s=180, marker="*", edgecolors="black", label="Centroids"
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()

# =========================
# 8. Print distortion history
# =========================
def print_history(history):
    print("StepType  Iteration  Distortion")
    for step_type, iteration, distortion in history:
        print(f"{step_type:>6} {iteration:>10} {distortion:>12.4f}")

# =========================
# 9. Main
# =========================
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "../hw1-2_data/Data_Exercise_2.txt"
    figure_dir = base_dir / "figure"
    figure_dir.mkdir(exist_ok=True)

    X = load_data(data_path)

    init_a = np.array([
        [4.0, -0.5,  2.0],
        [2.0,  2.5,  1.0],
        [10.0, 2.0, -1.0],
        [7.0,  0.5, -0.5],
        [12.0, 1.0, -1.0],
    ])

    init_b = np.array([
        [0.0, -0.3, -2.0],
        [-1.3, 1.5,  4.0],
        [11.3, 3.0,  0.2],
        [5.7,  3.0, -2.0],
        [10.0, -1.0, 1.2],
    ])

    for name, init in [("a", init_a), ("b", init_b)]:
        centroids, labels, final_D, history = run_kmeans(X, init)

        print(f"\n===== Initialization {name} =====")
        print_history(history)
        print("\nFinal centroids:")
        print(centroids)
        print(f"\nFinal distortion: {final_D:.4f}")
        print(f"Iterations to convergence: {len([h for h in history if h[0] == 'E'])}")

        plot_distortion_curve(
            history,
            title=f"K-means Distortion Curve (Init {name})",
            savepath=figure_dir / f"distortion_init_{name}.png"
        )

        plot_3d_clusters(
            X, labels, centroids,
            title=f"Final 3D Clustering (Init {name})",
            savepath=figure_dir / f"clusters_init_{name}.png"
        )