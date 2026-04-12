import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from sklearn.cluster import KMeans

def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    return np.array(img)

def image_to_pixels(img_np):
    h, w, c = img_np.shape
    pixels = img_np.reshape(-1, 3).astype(np.float64)
    return pixels

def compute_distortion(pixels, centroids, labels):
    return np.sum((pixels - centroids[labels]) ** 2)

def run_kmeans_distortion(pixels, k, random_state=42):
    kmeans = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=10
    )
    labels = kmeans.fit_predict(pixels)
    centroids = kmeans.cluster_centers_
    distortion = compute_distortion(pixels, centroids, labels)
    return distortion

def plot_elbow(k_values, distortions, savepath):
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, distortions, marker="o")
    plt.xlabel("Number of Clusters K")
    plt.ylabel("Distortion")
    plt.title("Elbow Plot: Distortion vs K")
    plt.xticks(k_values)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    image_path = base_dir / "../hw1-2_data/images/ladybug.jpg"
    figure_dir = base_dir / "figure"
    figure_dir.mkdir(exist_ok=True)

    img_np = load_image(image_path)
    pixels = image_to_pixels(img_np)

    k_values = list(range(1, 17))
    distortions = []

    for k in k_values:
        print(f"Running K-means for K = {k} ...")
        distortion = run_kmeans_distortion(pixels, k)
        distortions.append(distortion)
        print(f"K = {k}, Distortion = {distortion:.2f}")

    plot_elbow(
        k_values,
        distortions,
        figure_dir / "elbow_plot.png"
    )

    print("\nElbow plot saved successfully.")
    print("Distortion values:")
    for k, d in zip(k_values, distortions):
        print(f"K = {k}: {d:.2f}")