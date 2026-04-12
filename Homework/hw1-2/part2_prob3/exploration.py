import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    return img_np

def save_original_image(img_np, savepath):
    plt.figure(figsize=(6, 6))
    plt.imshow(img_np)
    plt.title("Original Image")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()

def image_dimensions(img_np):
    height, width, _ = img_np.shape
    return width, height

def sample_pixels(img_np, max_samples=5000, random_state=42):
    pixels = img_np.reshape(-1, 3)
    n_pixels = len(pixels)

    rng = np.random.default_rng(random_state)

    if n_pixels > max_samples:
        indices = rng.choice(n_pixels, size=max_samples, replace=False)
        sampled_pixels = pixels[indices]
    else:
        sampled_pixels = pixels

    return sampled_pixels

def plot_rgb_scatter(sampled_pixels, savepath):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    colors = sampled_pixels / 255.0

    ax.scatter(
        sampled_pixels[:, 0],  # R
        sampled_pixels[:, 1],  # G
        sampled_pixels[:, 2],  # B
        c=colors,
        s=8,
        alpha=0.6
    )

    ax.set_xlabel("Red")
    ax.set_ylabel("Green")
    ax.set_zlabel("Blue")
    ax.set_title("RGB Color Space Scatter Plot")

    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    image_path = base_dir / "../hw1-2_data/images/ladybug.jpg"
    figure_dir = base_dir / "figure"
    figure_dir.mkdir(exist_ok=True)

    img_np = load_image(image_path)

    width, height = image_dimensions(img_np)
    print(f"Image size: {width} x {height}")

    save_original_image(img_np, figure_dir / "original_image.png")

    sampled_pixels = sample_pixels(img_np, max_samples=5000, random_state=42)
    plot_rgb_scatter(sampled_pixels, figure_dir / "rgb_scatter.png")

    print("Step 1 figures saved successfully.")