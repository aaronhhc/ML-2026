from gmm_analysis import (
    BASE_DIR,
    DATASET_NAME,
    DATA_PATH,
    KAGGLE_URL,
    fit_gmm,
    fit_kmeans,
    load_mall_customers,
    plot_assignment_scatter,
    plot_gmm_contours,
    plot_gmm_vs_kmeans,
    plot_model_selection,
    prepare_features,
    run_kmeans_baseline,
    run_model_selection,
    save_figure,
)


PRIMARY_FEATURES = ["Annual Income (k$)", "Spending Score (1-100)"]
RANDOM_STATE = 42


def main():
    figure_dir = BASE_DIR / "figure"
    figure_dir.mkdir(exist_ok=True)

    raw_df, clean_df = load_mall_customers(DATA_PATH)
    X_original, X_scaled, scaler = prepare_features(clean_df, PRIMARY_FEATURES)

    selection_df = run_model_selection(X_scaled, random_state=RANDOM_STATE)
    best = selection_df.iloc[0]
    final_k = int(best["n_components"])
    final_covariance = str(best["covariance_type"])
    gmm, gmm_labels, responsibilities = fit_gmm(
        X_scaled,
        n_components=final_k,
        covariance_type=final_covariance,
        random_state=RANDOM_STATE,
    )

    kmeans_df = run_kmeans_baseline(X_scaled, random_state=RANDOM_STATE)
    best_kmeans = kmeans_df.sort_values("silhouette", ascending=False).iloc[0]
    kmeans, kmeans_labels = fit_kmeans(
        X_scaled,
        n_clusters=int(best_kmeans["n_clusters"]),
        random_state=RANDOM_STATE,
    )

    save_figure(
        plot_assignment_scatter(
            X_original,
            gmm_labels,
            PRIMARY_FEATURES,
            f"GMM assignments ({final_covariance}, K={final_k})",
        ),
        figure_dir / "gmm_assignment_scatter.png",
    )
    save_figure(
        plot_gmm_contours(
            X_original,
            X_scaled,
            scaler,
            gmm,
            gmm_labels,
            PRIMARY_FEATURES,
            f"GMM density contours ({final_covariance}, K={final_k})",
        ),
        figure_dir / "gmm_density_contours.png",
    )
    save_figure(
        plot_model_selection(selection_df),
        figure_dir / "bic_aic_model_selection.png",
    )
    save_figure(
        plot_gmm_vs_kmeans(
            X_original,
            gmm_labels,
            kmeans_labels,
            PRIMARY_FEATURES,
            f"GMM ({final_covariance}, K={final_k})",
            f"K-means (K={int(best_kmeans['n_clusters'])})",
        ),
        figure_dir / "gmm_vs_kmeans.png",
    )

    print_required_results(
        raw_rows=len(raw_df),
        clean_rows=len(clean_df),
        selection_df=selection_df,
        best=best,
        best_kmeans=best_kmeans,
        figure_dir=figure_dir,
    )


def print_required_results(
    raw_rows,
    clean_rows,
    selection_df,
    best,
    best_kmeans,
    figure_dir,
):
    final_k = int(best["n_components"])
    final_covariance = str(best["covariance_type"])

    print("HW2 Part 2 analysis complete.")
    print()
    print("Dataset")
    print(f"- Name: {DATASET_NAME}")
    print(f"- Kaggle URL: {KAGGLE_URL}")
    print(f"- Source file: {DATA_PATH}")
    print(f"- Rows before cleaning: {raw_rows}")
    print(f"- Rows after cleaning: {clean_rows}")
    print(f"- Features: {', '.join(PRIMARY_FEATURES)}")
    print("- Preprocessing: drop missing rows, then StandardScaler")
    print(f"- Random state: {RANDOM_STATE}")
    print()
    print("GMM model selection")
    print("- Compared n_components: 1 to 10")
    print("- Compared covariance_type: full, tied, diag, spherical")
    print(f"- covariance_type: {final_covariance}")
    print(f"- n_components: {final_k}")
    print(f"- BIC: {float(best['bic']):.4f}")
    print(f"- AIC: {float(best['aic']):.4f}")
    print()
    print("Top 5 GMM settings by BIC")
    print(
        selection_df[
            ["n_components", "covariance_type", "bic", "aic"]
        ]
        .head(5)
        .round(4)
        .to_string(index=False)
    )
    print()
    print("K-means baseline")
    print(f"- Best K by silhouette: {int(best_kmeans['n_clusters'])}")
    print(f"- Best silhouette: {float(best_kmeans['silhouette']):.4f}")
    print()
    print("Generated figures")
    print(f"- {figure_dir / 'gmm_assignment_scatter.png'}")
    print(f"- {figure_dir / 'gmm_density_contours.png'}")
    print(f"- {figure_dir / 'bic_aic_model_selection.png'}")
    print(f"- {figure_dir / 'gmm_vs_kmeans.png'}")


if __name__ == "__main__":
    main()
