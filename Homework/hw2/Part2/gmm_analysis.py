from pathlib import Path
import os

import matplotlib

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "Data" / "mall_customers.csv"
NUMERIC_FEATURES = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
COVARIANCE_TYPES = ["full", "tied", "diag", "spherical"]
K_RANGE = range(1, 11)

DATASET_NAME = "Mall Customers"
KAGGLE_URL = "https://www.kaggle.com/datasets/kandij/mall-customers"


def load_mall_customers():
    df = pd.read_csv("../Data/Mall_Customers.csv")

    # Some versions of this dataset use "Genre" instead of "Gender"
    if "Genre" in df.columns and "Gender" not in df.columns:
        df = df.rename(columns={"Genre": "Gender"})

    selected_columns = [
        "Gender",
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]

    clean_df = df[selected_columns].dropna().copy()
    return clean_df


def prepare_features(df, features):
    X_original = df[list(features)].to_numpy(dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_original)
    return X_original, X_scaled, scaler


def fit_gmm(X_scaled, n_components, covariance_type, random_state=42):
    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        init_params="kmeans",
        n_init=10,
        random_state=random_state,
    )
    model.fit(X_scaled)
    labels = model.predict(X_scaled)
    responsibilities = model.predict_proba(X_scaled)
    return model, labels, responsibilities


def fit_kmeans(X_scaled, n_clusters, random_state=42):
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = model.fit_predict(X_scaled)
    return model, labels


def run_model_selection(
    X_scaled,
    covariance_types=COVARIANCE_TYPES,
    k_values=K_RANGE,
    random_state=42,
):
    rows = []
    for covariance_type in covariance_types:
        for n_components in k_values:
            model, labels, _ = fit_gmm(
                X_scaled,
                n_components=n_components,
                covariance_type=covariance_type,
                random_state=random_state,
            )
            silhouette = np.nan
            if n_components > 1:
                silhouette = silhouette_score(X_scaled, labels)
            rows.append(
                {
                    "n_components": n_components,
                    "covariance_type": covariance_type,
                    "bic": model.bic(X_scaled),
                    "aic": model.aic(X_scaled),
                    "log_likelihood": model.score(X_scaled) * len(X_scaled),
                    "silhouette": silhouette,
                    "converged": model.converged_,
                    "n_iter": model.n_iter_,
                }
            )
    return pd.DataFrame(rows).sort_values(["bic", "aic"]).reset_index(drop=True)


def run_kmeans_baseline(X_scaled, k_values=range(2, 11), random_state=42):
    rows = []
    for n_clusters in k_values:
        model, labels = fit_kmeans(X_scaled, n_clusters, random_state=random_state)
        rows.append(
            {
                "n_clusters": n_clusters,
                "inertia": model.inertia_,
                "silhouette": silhouette_score(X_scaled, labels),
            }
        )
    return pd.DataFrame(rows)


def summarize_clusters(df, features, labels, responsibilities=None):
    summary_df = df.copy()
    summary_df["cluster"] = labels
    if responsibilities is not None:
        summary_df["max_responsibility"] = responsibilities.max(axis=1)

    aggregations = {feature: ["count", "mean", "std", "min", "max"] for feature in features}
    summary = summary_df.groupby("cluster").agg(aggregations)
    summary.columns = ["_".join(col).strip() for col in summary.columns.to_flat_index()]
    summary = summary.reset_index()

    if responsibilities is not None:
        confidence = (
            summary_df.groupby("cluster")["max_responsibility"]
            .mean()
            .rename("mean_max_responsibility")
        )
        summary = summary.merge(confidence, on="cluster", how="left")
    return summary


def component_parameters(model, scaler, features):
    means_original = scaler.inverse_transform(model.means_)
    covariances_original = covariance_matrices_original(model, scaler)
    rows = []
    for idx, mean in enumerate(means_original):
        row = {
            "component": idx,
            "weight": model.weights_[idx],
        }
        for feature, value in zip(features, mean):
            row[f"mean_{feature}"] = value
        row["covariance_matrix_original_units"] = np.array2string(
            covariances_original[idx],
            precision=4,
            suppress_small=True,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def covariance_matrices_original(model, scaler):
    d = len(scaler.scale_)
    scale = np.diag(scaler.scale_)

    if model.covariance_type == "full":
        covariances = model.covariances_
    elif model.covariance_type == "tied":
        covariances = np.repeat(model.covariances_[np.newaxis, :, :], model.n_components, axis=0)
    elif model.covariance_type == "diag":
        covariances = np.array([np.diag(cov) for cov in model.covariances_])
    elif model.covariance_type == "spherical":
        covariances = np.array([np.eye(d) * cov for cov in model.covariances_])
    else:
        raise ValueError(f"Unsupported covariance_type: {model.covariance_type}")

    return np.array([scale @ cov @ scale for cov in covariances])


def plot_assignment_scatter(X_original, labels, features, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        X_original[:, 0],
        X_original[:, 1],
        c=labels,
        cmap="tab10",
        s=45,
        alpha=0.82,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_gmm_contours(X_original, X_scaled, scaler, model, labels, features, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        X_original[:, 0],
        X_original[:, 1],
        c=labels,
        cmap="tab10",
        s=42,
        alpha=0.72,
        edgecolor="white",
        linewidth=0.35,
    )

    x_min, x_max = X_original[:, 0].min(), X_original[:, 0].max()
    y_min, y_max = X_original[:, 1].min(), X_original[:, 1].max()
    x_pad = (x_max - x_min) * 0.08
    y_pad = (y_max - y_min) * 0.08
    xx, yy = np.meshgrid(
        np.linspace(x_min - x_pad, x_max + x_pad, 180),
        np.linspace(y_min - y_pad, y_max + y_pad, 180),
    )
    grid_original = np.c_[xx.ravel(), yy.ravel()]
    grid_scaled = scaler.transform(grid_original)
    density = np.exp(model.score_samples(grid_scaled)).reshape(xx.shape)
    contour = ax.contour(xx, yy, density, levels=8, cmap="magma", linewidths=1.1)
    ax.clabel(contour, inline=True, fontsize=8)

    means_original = scaler.inverse_transform(model.means_)
    covariances_original = covariance_matrices_original(model, scaler)
    colors = plt.cm.tab10(np.linspace(0, 1, model.n_components))
    for idx, (mean, covariance) in enumerate(zip(means_original, covariances_original)):
        add_covariance_ellipse(ax, mean, covariance, colors[idx % len(colors)])

    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.25)
    fig.tight_layout()
    return fig


def add_covariance_ellipse(ax, mean, covariance, color):
    values, vectors = np.linalg.eigh(covariance)
    order = values.argsort()[::-1]
    values = values[order]
    vectors = vectors[:, order]
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))

    for n_std, alpha in [(1.0, 0.34), (2.0, 0.14)]:
        width, height = 2 * n_std * np.sqrt(np.maximum(values, 1e-12))
        ellipse = Ellipse(
            xy=mean,
            width=width,
            height=height,
            angle=angle,
            facecolor=color,
            edgecolor=color,
            linewidth=1.6,
            alpha=alpha,
        )
        ax.add_patch(ellipse)


def plot_model_selection(selection_df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for covariance_type in COVARIANCE_TYPES:
        subset = selection_df[selection_df["covariance_type"] == covariance_type].sort_values(
            "n_components"
        )
        axes[0].plot(subset["n_components"], subset["bic"], marker="o", label=covariance_type)
        axes[1].plot(subset["n_components"], subset["aic"], marker="o", label=covariance_type)

    axes[0].set_title("BIC by GMM setting")
    axes[1].set_title("AIC by GMM setting")
    for ax in axes:
        ax.set_xlabel("n_components")
        ax.set_xticks(list(K_RANGE))
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(title="covariance_type")
    axes[0].set_ylabel("BIC")
    axes[1].set_ylabel("AIC")
    fig.tight_layout()
    return fig


def plot_gmm_vs_kmeans(X_original, gmm_labels, kmeans_labels, features, gmm_title, kmeans_title):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    for ax, labels, title in zip(axes, [gmm_labels, kmeans_labels], [gmm_title, kmeans_title]):
        ax.scatter(
            X_original[:, 0],
            X_original[:, 1],
            c=labels,
            cmap="tab10",
            s=42,
            alpha=0.78,
            edgecolor="white",
            linewidth=0.35,
        )
        ax.set_title(title)
        ax.set_xlabel(features[0])
        ax.grid(True, linestyle="--", alpha=0.25)
    axes[0].set_ylabel(features[1])
    fig.tight_layout()
    return fig


def save_figure(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
