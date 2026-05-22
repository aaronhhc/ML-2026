from pathlib import Path
import sys

import pandas as pd
import streamlit as st

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from gmm_analysis import (
    COVARIANCE_TYPES,
    DATASET_NAME,
    KAGGLE_URL,
    NUMERIC_FEATURES,
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
    summarize_clusters,
)


st.set_page_config(page_title="HW2 Part 2 GMM Demo", layout="wide")


@st.cache_data
def cached_mall_customers():
    return load_mall_customers()


@st.cache_data
def read_uploaded_csv(file_bytes):
    from io import BytesIO

    return pd.read_csv(BytesIO(file_bytes))


def numeric_feature_candidates(df):
    numeric_columns = list(df.select_dtypes(include="number").columns)
    without_ids = [
        column
        for column in numeric_columns
        if column.lower() not in {"id", "customerid", "customer_id"}
    ]
    return without_ids if len(without_ids) >= 2 else numeric_columns


def default_feature_index(options, preferred):
    return options.index(preferred) if preferred in options else 0


st.title("HW2 Part 2: Real-World GMM Demo")

st.sidebar.header("Dataset")
dataset_choice = st.sidebar.selectbox("Dataset source", ["Mall Customers", "Upload CSV"])

if dataset_choice == "Mall Customers":
    raw_df, _ = cached_mall_customers()
    dataset_label = DATASET_NAME
    st.sidebar.caption(KAGGLE_URL)
else:
    uploaded_file = st.sidebar.file_uploader("Upload a CSV dataset", type=["csv"])
    if uploaded_file is None:
        st.info("Upload a CSV file with at least two numeric columns to run GMM.")
        st.stop()
    raw_df = read_uploaded_csv(uploaded_file.getvalue())
    dataset_label = uploaded_file.name

feature_options = numeric_feature_candidates(raw_df)
if len(feature_options) < 2:
    st.error("This dataset needs at least two numeric columns for a 2-D GMM demo.")
    st.stop()

st.sidebar.caption(f"Loaded: {dataset_label}")
st.sidebar.caption(f"Rows before cleaning: {len(raw_df)}")

default_x = "Annual Income (k$)" if dataset_choice == "Mall Customers" else feature_options[0]
feature_x = st.sidebar.selectbox(
    "X feature",
    feature_options,
    index=default_feature_index(feature_options, default_x),
)

feature_y_options = [feature for feature in feature_options if feature != feature_x]
default_y = "Spending Score (1-100)" if dataset_choice == "Mall Customers" else feature_y_options[0]
feature_y = st.sidebar.selectbox(
    "Y feature",
    feature_y_options,
    index=default_feature_index(feature_y_options, default_y),
)
features = [feature_x, feature_y]

clean_df = raw_df.dropna(subset=features).copy()
if len(clean_df) < 2:
    st.error("After dropping missing values, the selected feature pair has fewer than two rows.")
    st.stop()

st.sidebar.caption(f"Rows after cleaning: {len(clean_df)}")

max_components = min(10, len(clean_df))

st.sidebar.header("GMM")
n_components = st.sidebar.slider(
    "n_components",
    min_value=1,
    max_value=max_components,
    value=min(7, max_components),
    step=1,
)
covariance_type = st.sidebar.selectbox(
    "covariance_type",
    COVARIANCE_TYPES,
    index=COVARIANCE_TYPES.index("tied"),
)
random_state = st.sidebar.number_input("random_state", min_value=0, value=42, step=1)
show_kmeans = st.sidebar.checkbox("Compare with K-means", value=True)
auto_pick_bic = st.sidebar.checkbox("Use current BIC winner", value=False)

X_original, X_scaled, scaler = prepare_features(clean_df, features)

with st.spinner("Running GMM model selection..."):
    selection_df = run_model_selection(
        X_scaled,
        k_values=range(1, max_components + 1),
        random_state=int(random_state),
    )

if auto_pick_bic:
    best_row = selection_df.iloc[0]
    n_components = int(best_row["n_components"])
    covariance_type = str(best_row["covariance_type"])

model, labels, responsibilities = fit_gmm(
    X_scaled,
    n_components=n_components,
    covariance_type=covariance_type,
    random_state=int(random_state),
)
summary = summarize_clusters(clean_df, features, labels, responsibilities)

st.caption(
    f"Dataset: {dataset_label}. {len(raw_df)} rows before cleaning, "
    f"{len(clean_df)} rows after cleaning. Features are standardized before fitting "
    "and plotted in original units."
)

metrics = st.columns(5)
metrics[0].metric("BIC", f"{model.bic(X_scaled):,.2f}")
metrics[1].metric("AIC", f"{model.aic(X_scaled):,.2f}")
metrics[2].metric("Log likelihood", f"{model.score(X_scaled) * len(X_scaled):,.2f}")
metrics[3].metric("Iterations", f"{model.n_iter_}")
metrics[4].metric("Converged", "Yes" if model.converged_ else "No")

left, right = st.columns((1.05, 1))
with left:
    contour_fig = plot_gmm_contours(
        X_original,
        X_scaled,
        scaler,
        model,
        labels,
        features,
        f"GMM density contours ({covariance_type}, K={n_components})",
    )
    st.pyplot(contour_fig, use_container_width=True)

with right:
    scatter_fig = plot_assignment_scatter(
        X_original,
        labels,
        features,
        f"GMM assignments ({covariance_type}, K={n_components})",
    )
    st.pyplot(scatter_fig, use_container_width=True)

st.subheader("Model Selection")
selection_fig = plot_model_selection(selection_df)
st.pyplot(selection_fig, use_container_width=True)

best_bic = selection_df.iloc[0]
best_aic = selection_df.sort_values("aic").iloc[0]
model_cols = st.columns(2)
model_cols[0].dataframe(
    selection_df.head(10).round(4),
    use_container_width=True,
    hide_index=True,
)
model_cols[1].dataframe(
    pd.DataFrame(
        [
            {
                "criterion": "BIC",
                "n_components": int(best_bic["n_components"]),
                "covariance_type": best_bic["covariance_type"],
                "score": float(best_bic["bic"]),
            },
            {
                "criterion": "AIC",
                "n_components": int(best_aic["n_components"]),
                "covariance_type": best_aic["covariance_type"],
                "score": float(best_aic["aic"]),
            },
        ]
    ).round(4),
    use_container_width=True,
    hide_index=True,
)

st.subheader("Cluster Details")
weights_df = pd.DataFrame(
    {
        "component": range(n_components),
        "mixture_weight": model.weights_,
    }
)
detail_cols = st.columns(2)
detail_cols[0].dataframe(weights_df.round(4), use_container_width=True, hide_index=True)
detail_cols[1].dataframe(summary.round(3), use_container_width=True, hide_index=True)

if show_kmeans:
    st.subheader("GMM vs K-means")
    if n_components >= 2:
        kmeans, kmeans_labels = fit_kmeans(
            X_scaled,
            n_clusters=n_components,
            random_state=int(random_state),
        )
        comparison_fig = plot_gmm_vs_kmeans(
            X_original,
            labels,
            kmeans_labels,
            features,
            f"GMM ({covariance_type}, K={n_components})",
            f"K-means (K={n_components})",
        )
        st.pyplot(comparison_fig, use_container_width=True)

    if len(clean_df) >= 2:
        baseline_df = run_kmeans_baseline(
            X_scaled,
            k_values=range(2, max_components + 1),
            random_state=int(random_state),
        )
        st.dataframe(baseline_df.round(4), use_container_width=True, hide_index=True)

