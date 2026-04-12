import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# =========================
# Basic image utils
# =========================
def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    return np.array(img)

def image_to_pixels(img_np):
    h, w, c = img_np.shape
    pixels = img_np.reshape(-1, 3).astype(np.float64)
    return pixels, h, w

def rebuild_image_from_labels(labels, centroids, h, w):
    quantized_pixels = centroids[labels]
    quantized_img = quantized_pixels.reshape(h, w, 3).clip(0, 255).astype(np.uint8)
    return quantized_img

# =========================
# K-means(2) for splitting
# =========================
def squared_distances(X, M):
    return np.sum((X[:, None, :] - M[None, :, :]) ** 2, axis=2)

def e_step(X, centroids):
    dists = squared_distances(X, centroids)
    labels = np.argmin(dists, axis=1)
    distortion = np.sum(np.min(dists, axis=1))
    return labels, distortion

def m_step(X, labels, old_centroids, K):
    new_centroids = old_centroids.copy()
    for k in range(K):
        pts = X[labels == k]
        if len(pts) > 0:
            new_centroids[k] = pts.mean(axis=0)
    return new_centroids

def run_kmeans_2(X, init_centroids, max_iter=100):
    centroids = init_centroids.copy().astype(float)
    prev_labels = None

    for _ in range(max_iter):
        labels, _ = e_step(X, centroids)

        if prev_labels is not None and np.array_equal(labels, prev_labels):
            break

        new_centroids = m_step(X, labels, centroids, K=2)

        if np.allclose(new_centroids, centroids):
            centroids = new_centroids
            break

        centroids = new_centroids
        prev_labels = labels.copy()

    final_labels, final_distortion = e_step(X, centroids)
    return centroids, final_labels, final_distortion

# =========================
# NUBS helpers
# =========================
def cluster_centroid_and_distortion(X):
    centroid = X.mean(axis=0)
    distortion = np.sum((X - centroid) ** 2)
    return centroid, distortion

def initialize_split_centroids(X):
    """
    Initialize along the axis of maximum variance.
    """
    centroid = X.mean(axis=0)
    variances = np.var(X, axis=0)
    max_axis = np.argmax(variances)

    std_along_axis = np.std(X[:, max_axis])
    if std_along_axis == 0:
        std_along_axis = 1.0

    c1 = centroid.copy()
    c2 = centroid.copy()
    c1[max_axis] -= 0.5 * std_along_axis
    c2[max_axis] += 0.5 * std_along_axis

    return np.vstack([c1, c2])

def split_cluster(X_cluster):
    init_centroids = initialize_split_centroids(X_cluster)
    centroids, local_labels, distortion = run_kmeans_2(X_cluster, init_centroids)

    idx0 = np.where(local_labels == 0)[0]
    idx1 = np.where(local_labels == 1)[0]

    return centroids, local_labels, distortion, idx0, idx1

# =========================
# Main NUBS
# =========================
def run_nubs(pixels, target_k=8):
    """
    Returns:
        global_labels: shape (N,)
        centroids: shape (target_k, 3)
        total_distortion: float
        split_history: list of dict
    """
    N = len(pixels)

    # each cluster stores global indices
    clusters = [{
        "indices": np.arange(N)
    }]

    split_history = []

    while len(clusters) < target_k:
        # compute distortion for each cluster
        distortions = []
        for c in clusters:
            Xc = pixels[c["indices"]]
            _, d = cluster_centroid_and_distortion(Xc)
            distortions.append(d)

        # choose cluster with largest distortion
        split_idx = int(np.argmax(distortions))
        cluster_to_split = clusters.pop(split_idx)

        X_split = pixels[cluster_to_split["indices"]]

        centroids_2, local_labels, split_distortion, idx0, idx1 = split_cluster(X_split)

        global_idx0 = cluster_to_split["indices"][idx0]
        global_idx1 = cluster_to_split["indices"][idx1]

        clusters.append({"indices": global_idx0})
        clusters.append({"indices": global_idx1})

        split_history.append({
            "step": len(clusters),
            "split_from_cluster_distortion": distortions[split_idx],
            "resulting_num_clusters": len(clusters)
        })

    # assign final labels
    global_labels = np.empty(N, dtype=int)
    centroids = []

    for k, c in enumerate(clusters):
        Xc = pixels[c["indices"]]
        centroid, _ = cluster_centroid_and_distortion(Xc)
        centroids.append(centroid)
        global_labels[c["indices"]] = k

    centroids = np.array(centroids)
    total_distortion = np.sum((pixels - centroids[global_labels]) ** 2)

    return global_labels, centroids, total_distortion, split_history

# =========================
# Plot utils
# =========================
def plot_original_vs_quantized(original, quantized, title_right, savepath):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(quantized)
    plt.title(title_right)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()

def plot_palette(centroids, title, savepath):
    k = len(centroids)
    centroids_uint8 = centroids.clip(0, 255).astype(np.uint8)

    palette = np.zeros((50, 50 * k, 3), dtype=np.uint8)
    for i in range(k):
        palette[:, i * 50:(i + 1) * 50, :] = centroids_uint8[i]

    plt.figure(figsize=(max(6, k), 2))
    plt.imshow(palette)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()

# =========================
# Main
# =========================
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    image_path = base_dir / "../hw1-2_data/images/ladybug.jpg"
    figure_dir = base_dir / "figure"
    figure_dir.mkdir(exist_ok=True)

    original_img = load_image(image_path)
    pixels, h, w = image_to_pixels(original_img)

    global_labels, centroids, total_distortion, split_history = run_nubs(pixels, target_k=8)
    quantized_img = rebuild_image_from_labels(global_labels, centroids, h, w)

    plot_original_vs_quantized(
        original_img,
        quantized_img,
        title_right=f"NUBS (K = 8)\nDistortion = {total_distortion:.2f}",
        savepath=figure_dir / "nubs_k_8.png"
    )

    plot_palette(
        centroids,
        title="NUBS Color Palette (K = 8)",
        savepath=figure_dir / "nubs_palette_k_8.png"
    )

    print(f"NUBS K=8 Distortion = {total_distortion:.2f}")
    print("\nSplit history:")
    for item in split_history:
        print(
            f"Num clusters = {item['resulting_num_clusters']}, "
            f"split source distortion = {item['split_from_cluster_distortion']:.2f}"
        )

    print("\nStep 3 NUBS results saved successfully.")