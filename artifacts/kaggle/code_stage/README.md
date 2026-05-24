# Empirical Asset Pricing via Machine Learning
### Replication & Extensions — Hedge Funds Group Project

Replication of **Gu, Kelly & Xiu (2020), "Empirical Asset Pricing via Machine
Learning"**, *Review of Financial Studies*, 33(5), 2223-2273.

The paper systematically compares ~7 machine learning models (OLS, LASSO,
Random Forest, Gradient Boosting, Neural Networks) for predicting cross-sectional
stock returns on CRSP + Compustat using 94 characteristics. The main finding is
that non-linear methods — neural networks in particular — dominate linear models,
with profitability, momentum, and trading frictions being the most important
feature groups. Data is sourced from WRDS (CRSP + Compustat); factor benchmarks
use Kenneth French's data library (free access).

---

## Project Structure

```
.
├── data/
│   ├── raw/          # Downloaded datasets (ignored by git — see .gitignore)
│   └── processed/    # Cleaned HDF5 / parquet files (also ignored)
├── docs/             # Reference papers and assignment brief
├── notebooks/        # Jupyter notebooks: EDA, visualisations, final plots
├── report/           # LaTeX write-up (*.tex, *.bib); compiled PDFs are ignored
├── src/              # Pure Python: data pipelines, feature engineering, model training
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Our Three Extensions

| # | Extension | Brief Description |
|---|-----------|-------------------|
| 1 | **Out-of-sample post-2020** | The original paper ends ~2016. We extend the evaluation window to 2020-2023 (Covid crash, reflation, 2022 bear market) to test whether the neural network advantage persists in unseen macro regimes. |
| 2 | **Model ranking after transaction costs** | The paper reports gross returns. Using turnover estimates and TC assumptions, we show whether the model ranking changes once frictions are accounted for — an economically important result. |
| 3 | **Reduced characteristic set (quality/momentum)** | Instead of all 94 characteristics, we use the 15-20 most economically interpretable ones and measure the performance loss. This answers: *does the full complexity actually pay off?* |

---

## Getting Started

### 1 — Clone the repository

```bash
git clone <REPO_URL>
cd "Group project"
```

### 2 — Create and activate a virtual environment

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4 — Download raw data

*(Instructions TBD — data will be documented in `data/raw/README.md` once the
download pipeline is finalised.)*

---

## Daily Workflow (IMPORTANT — read before editing)

Always pull the latest changes **before** you start working:

```bash
git pull origin main
```

Create a feature branch for non-trivial changes:

```bash
git checkout -b feature/<your-initials>-<short-description>
# e.g.  git checkout -b feature/tb-shap-analysis
```

Commit often with descriptive messages:

```bash
git add .
git commit -m "feat: add SHAP waterfall plots for OLS baseline"
git push origin feature/tb-shap-analysis
```

Then open a **Pull Request** on GitHub and request a review before merging into
`main`.

---

## References

- Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine
  learning. *Review of Financial Studies*, 33(5), 2223-2273.
