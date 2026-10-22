# Reproducing the Experiments

This repository contains three notebooks for the Olivetti Faces assignment:
`answer_1.ipynb`, `answer_2.ipynb`, and `answer_3.ipynb`.

## 1. Requirements

Use the Python version recorded in `requirements.txt` or the version used to run the submitted notebooks. The commands below use Python's `venv` module.

### Windows (PowerShell)

From the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install jupyter
jupyter notebook
```

If PowerShell blocks activation, you can run the environment's Python directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install jupyter
.\.venv\Scripts\jupyter.exe notebook
```

### macOS / Linux

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install jupyter
jupyter notebook
```

## 2. Dataset download

The notebooks use scikit-learn's Olivetti Faces dataset through
`sklearn.datasets.fetch_olivetti_faces`. On the first run, scikit-learn may
download the dataset automatically, so an internet connection is required.
Subsequent runs can use the local scikit-learn dataset cache.

The images are 64 × 64 pixels and are flattened into feature vectors by the
notebook code.

## 3. Run the notebooks in this order

Open Jupyter from the repository root and execute every cell in each notebook
from top to bottom, using the same Python environment.

1. **`answer_1.ipynb` (Q1)** — run this first. It establishes the deterministic
   train/validation/test split and saves `olivetti_split.npz` when the notebook's
   split-creation cell is run. Keep this file in the repository root.
2. **`answer_2.ipynb` (Q2)** — run after Q1. It loads and reuses the saved split.
3. **`answer_3.ipynb` (Q3)** — run after Q1 and Q2. It also reuses the saved split.

Do not regenerate or replace `olivetti_split.npz` between notebooks. The split
uses seed `26016` (derived from roll number `M26CSA016`) and contains 240
training, 80 validation, and 80 test images. If the split file is already
provided in the repository, keep it and use it rather than creating a new split.

If a notebook contains Google Colab-only setup cells (for example, mounting
Google Drive or referring to a `drive/MyDrive/...` path), those cells are
specific to the original Colab environment. Before running locally, update the
path to point to the repository-root `olivetti_split.npz`; do not change the
split itself or its indices. Check the notebook's data-loading cell to confirm
the path.

## 4. Regenerate figures with the plotting scripts

Run these commands from the repository root, after installing the dependencies
and ensuring `olivetti_split.npz` is present:

```bash
python scripts/plot_q1.py
python scripts/plot_q2.py
python scripts/plot_q3.py
```

The scripts should be stored in `scripts/` and should save their outputs into
repository-relative `figures/` and, where applicable, `results/` directories.
Run them one at a time in the order shown above.

**Before submitting:** check `scripts/plot_q2.py`. If it still contains a
machine-specific path such as `C:\Users\JAHANVI\ml_assignment_2\figure`, change
it to a repository-relative path (for example, `ROOT / "figures"`), so the
commands work on another computer. Also verify that all three scripts exist at
the listed paths. If you are not including the plotting scripts in the
repository, remove this section or replace it with the actual plotting steps
used in your notebooks.

## 5. Saved split and reproducibility notes

- `olivetti_split.npz` is the shared split file.
- Q2 and Q3 must load the same saved train/validation/test indices produced by
  Q1; they should not independently create a different split.
- `configs/final_config.json` documents the main experiment settings.
- The notebooks may take time to run because they train multiple decision
  trees and ensembles.
- Exact training times can vary by computer and Python/library versions.
