from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data1.txt"
FIG_DIR = BASE_DIR / "figures"
RESULT_DIR = BASE_DIR / "results"

FIG_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)


def load_data():
    """
    Load 2-D data from data1.txt.
    Try comma-separated format first, then whitespace-separated format.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Cannot find data file: {DATA_PATH}")

    try:
        X = np.loadtxt(DATA_PATH, delimiter=",")
    except ValueError:
        X = np.loadtxt(DATA_PATH)

    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(f"Expected shape (n_samples, 2), got {X.shape}")

    return X


def get_plot_limits(X, padding_ratio=0.15):
    """
    Compute plot limits based on actual data range.
    This avoids using too large a mesh grid for small-scale data.
    """
    x_range = X[:, 0].max() - X[:, 0].min()
    y_range = X[:, 1].max() - X[:, 1].min()

    x_pad = x_range * padding_ratio
    y_pad = y_range * padding_ratio

    x_min = X[:, 0].min() - x_pad
    x_max = X[:, 0].max() + x_pad
    y_min = X[:, 1].min() - y_pad
    y_max = X[:, 1].max() + y_pad

    return x_min, x_max, y_min, y_max


def fit_gmm(X, n_components=4, covariance_type="full", random_state=42):
    """
    Fit GMM using K-means initialization.
    """
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        init_params="kmeans",
        random_state=random_state,
        max_iter=300,
        n_init=5,
    )

    gmm.fit(X)
    labels = gmm.predict(X)

    return gmm, labels


def save_parameters(gmm, filename="gmm_k4_parameters.txt"):
    """
    Save estimated GMM parameters.
    """
    output_path = RESULT_DIR / filename

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("GMM Part 1 Estimated Parameters\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"n_components: {gmm.n_components}\n")
        f.write(f"covariance_type: {gmm.covariance_type}\n")
        f.write(f"Converged: {gmm.converged_}\n")
        f.write(f"Number of EM iterations: {gmm.n_iter_}\n")
        f.write(f"Lower bound / average log-likelihood: {gmm.lower_bound_:.6f}\n\n")

        f.write("Mixture Weights:\n")
        f.write(np.array2string(gmm.weights_, precision=6) + "\n\n")

        f.write("Means:\n")
        f.write(np.array2string(gmm.means_, precision=6) + "\n\n")

        f.write("Covariances:\n")
        f.write(np.array2string(gmm.covariances_, precision=6) + "\n")

    print(f"[Saved] {output_path}")


def plot_scatter(X, labels):
    """
    Scatter plot colored by GMM component assignment.
    """
    plt.figure(figsize=(6, 5))
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=25)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("GMM Component Assignment, K=4")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "scatter_gmm_k4.png", dpi=300)
    plt.close()

    print("[Saved] scatter_gmm_k4.png")


def plot_contour(X, gmm):
    """
    Contour plot of learned GMM density.
    """
    x_min, x_max, y_min, y_max = get_plot_limits(X)

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )

    grid = np.column_stack([xx.ravel(), yy.ravel()])
    log_density = gmm.score_samples(grid)
    density = np.exp(log_density).reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.scatter(X[:, 0], X[:, 1], s=12, alpha=0.45)

    levels = np.linspace(density.min(), density.max(), 15)
    plt.contour(xx, yy, density, levels=levels)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("GMM Density Contour, K=4")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "contour_gmm_k4.png", dpi=300)
    plt.close()

    print("[Saved] contour_gmm_k4.png")


def plot_3d_log_density(X, gmm):
    """
    3D plot of GMM log probability density.

    We use log density instead of raw density because the data scale is very small,
    which can make raw density extremely sharp and hard to visualize.
    """
    x_min, x_max, y_min, y_max = get_plot_limits(X)

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 160),
        np.linspace(y_min, y_max, 160),
    )

    grid = np.column_stack([xx.ravel(), yy.ravel()])
    log_density = gmm.score_samples(grid).reshape(xx.shape)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(xx, yy, log_density, linewidth=0, antialiased=True)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Log Density")
    ax.set_title("3D GMM Log Probability Density, K=4")

    ax.view_init(elev=30, azim=135)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="z", labelsize=8)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "density_3d_log_gmm_k4.png", dpi=300)
    plt.close()

    print("[Saved] density_3d_log_gmm_k4.png")


def plot_bic_aic(X):
    """
    Compare n_components and covariance_type using BIC and AIC.
    Lower BIC/AIC is better.
    """
    n_components_range = range(1, 11)
    covariance_types = ["full", "diag", "spherical"]

    rows = []

    plt.figure(figsize=(8, 5))

    for cov_type in covariance_types:
        bic_scores = []
        aic_scores = []

        for k in n_components_range:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=cov_type,
                init_params="kmeans",
                random_state=42,
                max_iter=300,
                n_init=5,
            )

            gmm.fit(X)

            bic = gmm.bic(X)
            aic = gmm.aic(X)

            bic_scores.append(bic)
            aic_scores.append(aic)

            rows.append((cov_type, k, bic, aic))

        plt.plot(
            list(n_components_range),
            bic_scores,
            marker="o",
            label=f"BIC-{cov_type}",
        )
        plt.plot(
            list(n_components_range),
            aic_scores,
            marker="x",
            linestyle="--",
            label=f"AIC-{cov_type}",
        )

    plt.xlabel("Number of Components")
    plt.ylabel("Score")
    plt.title("BIC/AIC Model Selection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "bic_aic_model_selection.png", dpi=300)
    plt.close()

    result_path = RESULT_DIR / "bic_aic_scores.csv"

    with open(result_path, "w", encoding="utf-8") as f:
        f.write("covariance_type,n_components,bic,aic\n")
        for cov_type, k, bic, aic in rows:
            f.write(f"{cov_type},{k},{bic:.6f},{aic:.6f}\n")

    print(f"[Saved] {result_path}")
    print("[Saved] bic_aic_model_selection.png")


def compare_covariance_types(X):
    """
    Compare at least two covariance_type settings for K=4.
    This satisfies the model comparison requirement.
    """
    covariance_types = ["full", "diag", "spherical", "tied"]
    rows = []

    for cov_type in covariance_types:
        gmm, labels = fit_gmm(
            X,
            n_components=4,
            covariance_type=cov_type,
            random_state=42,
        )

        bic = gmm.bic(X)
        aic = gmm.aic(X)

        rows.append(
            {
                "covariance_type": cov_type,
                "bic": bic,
                "aic": aic,
                "lower_bound": gmm.lower_bound_,
                "n_iter": gmm.n_iter_,
                "converged": gmm.converged_,
            }
        )

    output_path = RESULT_DIR / "covariance_type_comparison.csv"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("covariance_type,bic,aic,lower_bound,n_iter,converged\n")
        for row in rows:
            f.write(
                f"{row['covariance_type']},"
                f"{row['bic']:.6f},"
                f"{row['aic']:.6f},"
                f"{row['lower_bound']:.6f},"
                f"{row['n_iter']},"
                f"{row['converged']}\n"
            )

    print(f"[Saved] {output_path}")


def compare_with_kmeans(X, gmm_labels):
    """
    Optional baseline comparison with K-means.
    Useful for discussion.
    """
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)

    ari = adjusted_rand_score(kmeans_labels, gmm_labels)

    output_path = RESULT_DIR / "kmeans_vs_gmm.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("K-means vs GMM Comparison\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Adjusted Rand Index between K-means and GMM labels: {ari:.6f}\n\n")
        f.write("Discussion:\n")
        f.write("- K-means gives hard assignments.\n")
        f.write("- GMM gives probabilistic soft assignments.\n")
        f.write("- K-means assumes roughly spherical clusters.\n")
        f.write("- GMM can model elliptical clusters through covariance matrices.\n")

    print(f"[Saved] {output_path}")


def main():
    X = load_data()

    print("=" * 50)
    print("HW2 Part 1: GMM on data1.txt")
    print("=" * 50)
    print(f"Loaded data shape: {X.shape}")
    print(f"X range: {X[:, 0].min():.6f} to {X[:, 0].max():.6f}")
    print(f"Y range: {X[:, 1].min():.6f} to {X[:, 1].max():.6f}")
    print()

    # Final model required by assignment: K=4, K-means initialization.
    gmm, labels = fit_gmm(
        X,
        n_components=4,
        covariance_type="full",
        random_state=42,
    )

    save_parameters(gmm)

    plot_scatter(X, labels)
    plot_contour(X, gmm)
    plot_3d_log_density(X, gmm)
    plot_bic_aic(X)
    compare_covariance_types(X)
    compare_with_kmeans(X, labels)

    print()
    print("Done. Check the figures/ and results/ folders.")


if __name__ == "__main__":
    main()