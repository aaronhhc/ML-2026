import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from pathlib import Path
import os
import sys

# Ensure the current directory is in sys.path so we can import local modules
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from kmeans_quantization import quantize_kmeans
from nubs_quantization import run_nubs, image_to_pixels, load_image

st.set_page_config(layout="wide", page_title="Color Quantization Demo")

def render_palette_bar(centroids, labels, h, w):
    """Render a color palette bar showing K cluster colors and their pixel share."""
    k = len(centroids)
    pixel_counts = np.bincount(labels, minlength=k)
    proportions = pixel_counts / (h * w)
    
    # Sort colors by proportion
    sorted_indices = np.argsort(proportions)[::-1]
    sorted_centroids = centroids[sorted_indices]
    sorted_proportions = proportions[sorted_indices]
    
    # Create the horizontal stacked bar
    fig, ax = plt.subplots(figsize=(8, 1))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.axis("off")
    
    left = 0
    for centroid, proportion in zip(sorted_centroids, sorted_proportions):
        color = centroid / 255.0
        ax.barh(y=0, width=proportion, left=left, color=color, edgecolor="none")
        left += proportion
        
    return fig

if __name__ == "__main__":
    st.title("Image Color Quantization Interactive Demo")
    
    # --- Sidebar Controls ---
    st.sidebar.header("Settings")
    
    # 1. Image Selection
    upload_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    preloaded_images = {
        "Ladybug": str(current_dir / "../hw1-2_data/images/ladybug.jpg"),
        # Add more if available, fallback appropriately.
    }
    
    selected_preset = st.sidebar.selectbox(
        "Or choose a preloaded image",
        options=["Ladybug"]
    )
    
    # Load image logic
    if upload_file is not None:
        img_np = np.array(Image.open(upload_file).convert("RGB"))
    else:
        # Fallback to preloaded
        img_path = preloaded_images[selected_preset]
        if os.path.exists(img_path):
            img_np = load_image(img_path)
        else:
            st.error(f"Default image not found at {img_path}. Please upload an image.")
            st.stop()
            
    # 2. Algorithm and K Selection
    algo = st.sidebar.radio("Select Quantization Algorithm", ["K-means", "NUBS"])
    k_val = st.sidebar.slider("Select K (Number of Colors)", min_value=1, max_value=32, value=8, step=1)
    
    st.sidebar.markdown("### Additional Visualizations")
    show_3d_scatter = st.sidebar.checkbox("Show 3D RGB Scatter (10% Subsample)", value=False)
    
    # --- Processing ---
    # Downsample slightly for speed if the image is extremely large, though not required here
    pixels, h, w = image_to_pixels(img_np)
    
    with st.spinner(f"Running {algo} for K={k_val}..."):
        if algo == "K-means":
            quantized_img, centroids, labels, distortion = quantize_kmeans(img_np, k_val)
        else:  # NUBS
            labels, centroids, distortion, _ = run_nubs(pixels, target_k=k_val)
            quantized_pixels = centroids[labels]
            quantized_img = quantized_pixels.reshape(h, w, 3).clip(0, 255).astype(np.uint8)

    # --- Display Results ---
    st.subheader("Image Comparison")
    
    # Display Images Side-by-Side
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_np, caption="Original Image", use_container_width=True)
    with col2:
        st.image(quantized_img, caption=f"Quantized Image (K={k_val})", use_container_width=True)
    
    # Display Stats
    st.subheader("Quantization Statistics")
    st.metric(label="Total Distortion (D)", value=f"{distortion:,.2f}")
    
    # Display Color Palette
    st.subheader(f"Color Palette Share (K={k_val})")
    st.markdown("Displays the cluster colors sorted by the proportion of pixels they represent.")
    palette_fig = render_palette_bar(centroids.clip(0, 255), labels, h, w)
    st.pyplot(palette_fig)
    
    # Optional 3D Scatter Plot for deeper exploration
    if show_3d_scatter:
        st.subheader("3D RGB Scatter (10% Subsample)")
        from mpl_toolkits.mplot3d import Axes3D
        
        # Subsample original pixels and labels to plot faster
        sample_size = max(1, len(pixels) // 10)
        indices = np.random.choice(len(pixels), sample_size, replace=False)
        sub_pixels = pixels[indices] / 255.0
        sub_labels = labels[indices]
        
        fig_scatter = plt.figure(figsize=(10, 8))
        ax = fig_scatter.add_subplot(111, projection='3d')
        
        # Scatter points using their quantized cluster color
        cluster_colors = centroids.clip(0, 255) / 255.0
        c_mapped = cluster_colors[sub_labels]
        
        ax.scatter(sub_pixels[:, 0], sub_pixels[:, 1], sub_pixels[:, 2], c=c_mapped, s=2, alpha=0.5)
        ax.scatter(cluster_colors[:, 0], cluster_colors[:, 1], cluster_colors[:, 2], 
                   c='red', s=100, edgecolor='black', marker='X', label='Centroids')
        
        ax.set_xlabel('Red')
        ax.set_ylabel('Green')
        ax.set_zlabel('Blue')
        ax.legend()
        st.pyplot(fig_scatter)
