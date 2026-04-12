# ML-2026

Machine Learning Concept 2026 course repository containing homework code, datasets, figures, and small interactive demos.

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
└── hw1-2/
    ├── part1_prob1_kmeans/   # K-means clustering from scratch
    ├── part1_prob2_nubs/     # NUBS-related homework code
    ├── part2_prob3/          # image exploration and color quantization
    └── hw1-2_data/           # shared data and images
```

## Topics Covered

- Polynomial regression with feature scaling
- Train/test evaluation with mean squared error
- Regularization with Ridge and Lasso
- Interactive visualization with Streamlit
- K-means clustering and distortion tracking
- Image color quantization with K-means

## Environment

Most scripts use Python 3 and the following libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `streamlit`
- `Pillow`

Example setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib scikit-learn streamlit Pillow
```

## Running the Homework Code

Run scripts from the directory that contains the target file unless the script already resolves paths with `pathlib`.

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
```

## Notes

- Generated figures are saved inside each homework folder's `figure/` directory.
- Some folders also include `.txt` writeups or experiment logs alongside the code.
- `hw1-1` currently contains a local virtual environment folder; if you want a cleaner repo, that directory is usually better kept out of version control.
