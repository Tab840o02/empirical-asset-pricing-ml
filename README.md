# Empirical Asset Pricing via Machine Learning
### Replication & Extensions — Hedge Funds Group Project

Replication of **Gu, Kelly & Xiu (2020), "Empirical Asset Pricing via Machine
Learning"**, *Review of Financial Studies*, 33(5), 2223-2273.

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
| 1 | **Alternative feature set** | Augment the original 94 characteristics with additional macro/sentiment signals to test whether out-of-sample R² improves. |
| 2 | **Long-short portfolio construction** | Build decile-sorted long-short portfolios from model predictions and evaluate Sharpe ratio, max drawdown, and factor exposures (FF5 + MOM). |
| 3 | **Explainability via SHAP** | Apply SHAP values to the best-performing neural network to identify which characteristics drive return predictions in different market regimes. |

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
- Two Sigma (2020). A machine learning approach to risk and return. *(see
  docs/)*
