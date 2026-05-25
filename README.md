# ML-2026

Machine Learning Concept 2026 course repository containing homework code,
datasets, generated figures, reports, and small interactive demos.

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
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib scikit-learn streamlit Pillow seaborn
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

## Outputs

- Figures are saved in each assignment's `figure/` or `figures/` directory.
- Text summaries and experiment logs are saved next to the relevant scripts,
  usually as `.txt` or `.csv` files.
- HW2 Part 2 also includes a demo video and screenshots for the interactive
  GMM comparison workflow.

## Notes

- The repository currently contains local virtual environment folders under
  `Homework/hw1-1/.venv/` and `Homework/hw2/.venv/`. These are useful locally
  but are usually excluded from version control in a clean submission.
- Some generated caches, such as `__pycache__/`, may appear after running the
  scripts and can be safely regenerated.
