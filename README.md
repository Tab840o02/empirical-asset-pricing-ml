# Empirical Asset Pricing via Machine Learning
### Replication & Extensions — Hedge Funds Group Project

Replication of **Gu, Kelly & Xiu (2020), "Empirical Asset Pricing via Machine
Learning"**, *Review of Financial Studies*, 33(5), 2223-2273.

The paper systematically compares machine learning models (OLS, LASSO,
Random Forest, Gradient Boosting, Neural Networks NN1–NN5) for predicting
cross-sectional stock returns on CRSP + Compustat using 94 firm characteristics.
The main finding is that non-linear methods — neural networks in particular —
dominate linear models in out-of-sample prediction. Data is sourced from WRDS
(CRSP + Compustat); factor benchmarks use Kenneth French's data library (free
access).

**Phase 4 (replication) is complete and Phase 5a run execution is complete.**
All 13 models are trained and evaluated on the 1987–2016 test window
(18,165,186 stock-month predictions), and the post-2020 extension manifest run
covers 2017–2024 for all 13 models. Phase 5c (feature parsimony) remains in
progress. See
[docs/results_comparison.md](docs/results_comparison.md) for the full comparison
against the paper and a register of methodological deviations.

---

## Project Structure

```
.
├── data/
│   ├── raw/          # Downloaded datasets (ignored by git — see .gitignore)
│   └── processed/    # Cleaned parquet files (also ignored)
├── docs/             # Reference papers, project plan, status, results comparison
├── notebooks/        # Jupyter notebooks: EDA, visualisations, final plots
├── report/           # LaTeX write-up (*.tex, *.bib); compiled PDFs are ignored
├── scripts/          # One-off utility scripts (evaluation runner, Kaggle bundles)
├── src/              # Pure Python: data pipelines, feature engineering, model training
├── tests/            # Pytest: no-lookahead checks, feature tests
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Phase 4 Results Summary

> Full details in [docs/results_comparison.md](docs/results_comparison.md) and [docs/STATUS.md](docs/STATUS.md).

| Model | OOS R² | Sharpe | GKX OOS R² | Status |
|---|---|---|---|---|
| OLS-3 | +0.025% | 0.70 | +0.06% | ✅ Close |
| OLS-all | +0.152% | 0.74 | +0.09% | ✅ Close |
| PCR | +0.169% | 0.81 | +0.19% | ✅ Very close |
| PLS | +0.163% | 0.85 | +0.25% | ✅ Close |
| ElasticNet | +0.180% | 0.72 | +0.22% | ✅ Close |
| GLM | +0.063% | 0.56 | +0.06% | ✅ Exact (OOS R²) |
| RF | −0.509% | 0.10 | +0.39% | ❌ Negative |
| GBRT | −0.989% | 0.04 | +0.34% | ❌ Negative |
| NN1 | +0.344% | 1.60 | +0.39% | ✅ Close (10-seed) |
| NN2 | +0.178% | 1.51 | +0.40% | ⚠️ Below GKX (10-seed) |
| NN3 | +0.023% | 0.41 | +0.41% | ⚠️ 3-seed non-compliant |
| NN4 | +0.003% | 0.09 | +0.45% | ❌ 3-seed non-compliant |
| NN5 | +0.007% | 0.10 | +0.55% | ❌ 3-seed non-compliant |

**Key limitations:** Tree models produce negative OOS R² (root cause unresolved). NN3–NN5 used 3 seeds instead of GKX's 10 — their results are not directly comparable. Portfolio L/S monthly returns are systematically higher than GKX for linear and deep NN models; cause not isolated.

---

## Our Three Extensions

| # | Extension | File | Brief Description |
|---|-----------|------|-------------------|
| 1 | **Out-of-sample post-2020** | `src/extensions/post2020_eval.py` | Extend evaluation through Covid crash, 2022 bear market, and beyond to test whether NN advantage persists |
| 2 | **Model ranking after transaction costs** | `src/extensions/transaction_costs.py` | Apply turnover × spread costs to gross L/S returns; test whether model rankings change |
| 3 | **Reduced characteristic set** | `src/extensions/feature_parsimony.py` | Retrain on 15–20 most interpretable features; measure R² cost vs full 94-feature set |

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
