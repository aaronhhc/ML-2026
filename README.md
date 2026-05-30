# ML-2026

Machine Learning Concept 2026 course repository containing homework code,
datasets, generated figures, reports, small interactive demos, and the final project.

## Members

- 黃竑喆 - 410410080
- 金晴宇 - 411410051
- 吳旻融 - 412410099

## Repository Layout

```text
Homework/
├── hw1-1/
│   ├── Part1-1/   # polynomial regression on dataset1
│   ├── Part1-2/   # higher-order regression on dataset2
│   └── Part2/     # advertising regression, ridge, lasso, Streamlit demo
├── hw1-2/
│   ├── part1_prob1_kmeans/   # K-means clustering from scratch
│   ├── part1_prob2_nubs/     # NUBS clustering
│   ├── part2_prob3/          # image exploration and color quantization
│   └── hw1-2_data/           # shared data and images
└── hw2/
    ├── Data/                 # customer and clustering datasets
    ├── Part1/                # GMM analysis, model selection, reports
    └── Part2/                # GMM/K-means comparison and interactive demo

FinalProject/
├── data/
│   ├── raw/                  # local Netflix Prize data, not committed
│   └── processed/            # cluster summaries and K-selection results
├── reports/
│   ├── figures/              # generated plots for the final report
│   ├── v1_report.md
│   ├── v2_report.md
│   ├── v3_report.md
│   └── v4_report.md
├── src/                      # final project pipeline scripts
└── requirements.txt
```

## Topics Covered

- Polynomial regression with feature scaling
- Train/test evaluation with mean squared error
- Regularization with Ridge and Lasso
- Interactive visualization with Streamlit
- K-means clustering and distortion tracking
- NUBS clustering and split visualization
- Image color quantization with K-means and NUBS
- Gaussian Mixture Models, EM-style clustering, and covariance analysis
- Model selection with BIC and AIC
- K-means versus GMM comparison
- Large-scale user behavior feature engineering
- MiniBatchKMeans clustering for Netflix user segmentation
- Cluster interpretation, naming, and report visualization

## Final Project

The final project analyzes Netflix Prize rating data and clusters users based
on rating behavior rather than predicting individual movie ratings.

Research question:

```text
Can Netflix users be grouped into meaningful behavioral clusters using rating records alone?
```

The final version uses `combined_data_1.txt`, containing nearly 10 million
rating records. Raw Netflix data is not included in this repository.

Final data scale:

| Item | Count |
|---|---:|
| Rating records | 9,998,038 |
| Users before filtering | 447,835 |
| Users after filtering | 154,655 |
| Features used for clustering | 16 |

The final pipeline builds user-level behavior features, standardizes them,
runs MiniBatchKMeans, generates visualizations, and assigns interpretable
cluster names.

Final clustering result:

| Cluster | User Count | Name |
|---:|---:|---|
| 0 | 66,310 | Conservative Moderate Raters |
| 1 | 54,032 | Diverse Critical Raters |
| 2 | 34,313 | Generous Extreme Raters |

See [FinalProject/README.md](FinalProject/README.md) and
[FinalProject/reports/v4_report.md](FinalProject/reports/v4_report.md) for
the full project description and final report.

## Environment

Most scripts use Python 3 and the following libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `streamlit`
- `Pillow`
- `seaborn`

Example setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib scikit-learn streamlit Pillow seaborn
```

For the final project only:

```bash
cd FinalProject
pip install -r requirements.txt
```

## Running the Homework Code

Run scripts from the directory that contains the target file unless the script
already resolves paths with `pathlib`.

Examples:

```bash
cd Homework/hw1-1/Part1-1
python problem_a.py

cd Homework/hw1-1/Part2
python problem3_step1.py
streamlit run problem3_interactive_demo.py

cd Homework/hw1-2/part1_prob1_kmeans
python problem1_kmeans.py

cd Homework/hw1-2/part2_prob3
python exploration.py
python kmeans_quantization.py
python nubs_quantization.py

cd Homework/hw2/Part1
python part1_gmm.py

cd Homework/hw2/Part2
python run_part2_analysis.py
streamlit run interactive_demo.py
```

## Running the Final Project

Place the Netflix Prize `combined_data_1.txt` file in
`FinalProject/data/raw/`, then run:

```bash
cd FinalProject/src
python 01_load_data.py
python 02_feature_engineering.py
python 03_clustering.py
python 04_visualization.py
python 05_cluster_profile.py
python 06_add_cluster_names.py
```

Final project scripts:

| Script | Description |
|---|---|
| `01_load_data.py` | Loads and parses Netflix Prize raw rating data |
| `02_feature_engineering.py` | Builds user-level behavior features |
| `03_clustering.py` | Runs MiniBatchKMeans and K selection |
| `04_visualization.py` | Generates elbow, silhouette, PCA plots, and cluster summary |
| `05_cluster_profile.py` | Generates standardized cluster profile plots |
| `06_add_cluster_names.py` | Adds human-readable cluster names |

## Outputs

- Figures are saved in each assignment's `figure/` or `figures/` directory.
- Text summaries and experiment logs are saved next to the relevant scripts,
  usually as `.txt` or `.csv` files.
- HW2 Part 2 also includes a demo video and screenshots for the interactive
  GMM comparison workflow.
- Final project figures are saved in `FinalProject/reports/figures/`.
- Final project processed summaries are saved in `FinalProject/data/processed/`,
  including `cluster_summary.csv`, `cluster_summary_named.csv`, and
  `k_selection_results.csv`.

## Notes

- The repository currently contains local virtual environment folders under
  `Homework/hw1-1/.venv/`, `Homework/hw2/.venv/`, and `FinalProject/.venv/`.
  These are useful locally but are usually excluded from version control in a
  clean submission.
- Some generated caches, such as `__pycache__/`, may appear after running the
  scripts and can be safely regenerated.
- Raw Netflix Prize data and large generated intermediate files are excluded
  from the final project with `.gitignore`.
