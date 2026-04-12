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
    return pixels, h, w

def quantize_kmeans(img_np, k, random_state=42):
    pixels, h, w = image_to_pixels(img_np)

    kmeans = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=10
    )

    labels = kmeans.fit_predict(pixels)
    centroids = kmeans.cluster_centers_

    quantized_pixels = centroids[labels]
    quantized_img = quantized_pixels.reshape(h, w, 3).clip(0, 255).astype(np.uint8)

    distortion = np.sum((pixels - centroids[labels]) ** 2)

    return quantized_img, centroids, labels, distortion

def plot_original_vs_quantized(original, quantized, k, distortion, savepath):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(quantized)
    plt.title(f"K = {k}\nDistortion = {distortion:.2f}")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()

def plot_palette(centroids, k, savepath):
    centroids_uint8 = centroids.clip(0, 255).astype(np.uint8)

    palette = np.zeros((50, 50 * k, 3), dtype=np.uint8)

    for i in range(k):
        palette[:, i * 50:(i + 1) * 50, :] = centroids_uint8[i]

    plt.figure(figsize=(max(6, k), 2))
    plt.imshow(palette)
    plt.title(f"Color Palette (K = {k})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    image_path = base_dir / "../hw1-2_data/images/ladybug.jpg"
    figure_dir = base_dir / "figure"
    figure_dir.mkdir(exist_ok=True)

    original_img = load_image(image_path)

    k_values = [2, 4, 8, 16, 32]

    for k in k_values:
        print(f"Running K-means with K = {k} ...")

        quantized_img, centroids, labels, distortion = quantize_kmeans(original_img, k)

        plot_original_vs_quantized(
            original_img,
            quantized_img,
            k,
            distortion,
            figure_dir / f"kmeans_k_{k}.png"
        )

        plot_palette(
            centroids,
            k,
            figure_dir / f"palette_k_{k}.png"
        )

        print(f"K = {k}, Distortion = {distortion:.2f}")

    print("Step 2 K-means results saved successfully.")