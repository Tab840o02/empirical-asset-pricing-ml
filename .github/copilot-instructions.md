# AI Developer Instructions — GKX (2020) Replication Project

## What This Project Is
A Python replication of **Gu, Kelly & Xiu (2020), "Empirical Asset Pricing via Machine Learning"** (RFS 33(5)), extended with three original contributions. The codebase builds a machine learning pipeline to predict cross-sectional US stock returns using 94 firm characteristics from CRSP and Compustat, then evaluates multiple models (linear, tree-based, neural networks) on an out-of-sample test set.

The full project plan is in [docs/project_plan.md](../docs/project_plan.md). Read it before making non-trivial changes.

---

## Tech Stack
- **Language:** Python 3.10+
- **Data source:** WRDS (PostgreSQL via the `wrds` Python package)
- **Storage format:** Parquet (via `pyarrow`) for all intermediate and processed data
- **ML:** `scikit-learn`, `lightgbm`, `tensorflow`/`keras` (NN1–NN5)
- **Notebooks:** Jupyter (in `notebooks/`) — for EDA and final plots only, not for pipeline logic
- **Testing:** `pytest` (in `tests/`)

---

## Project Structure
```
src/         Pure Python pipeline — data, features, models, evaluation, extensions
notebooks/   EDA and report figures only; no data transformations here
data/raw/    WRDS downloads (git-ignored; never commit)
data/processed/  Cleaned and featurised data (git-ignored; never commit)
docs/        Reference papers, project plan
report/      LaTeX write-up
tests/       Pytest unit tests
```

---

## Critical Rules — Do Not Violate

### 1. No Look-Ahead Bias
The single most important invariant in the entire codebase. Compustat annual data for fiscal year ending in month M may **not** appear in the feature panel before month M+6. Quarterly data may not appear before M+3. Any code that joins Compustat to CRSP must enforce this lag. When in doubt, check `src/data/compustat_cleaner.py` and its `public_date` column.

### 2. Credentials Are Never Hard-Coded
The WRDS username is read from the `WRDS_USER` environment variable. No passwords, tokens, or usernames may appear in source code or notebooks.

### 3. Raw and Processed Data Are Not Committed
`data/raw/` and `data/processed/` are in `.gitignore`. Never add Parquet files, CSVs, or any CRSP/Compustat data to the repository.

### 4. Normalisation Lives in One Place
Cross-sectional rank normalisation to $[-1, +1]$ is performed **only** in `src/features/feature_assembler.py`. Individual feature files return raw values.

### 5. Notebooks Call `src/`, Not the Other Way Around
All logic belongs in `src/`. Notebooks may import from `src/` but `src/` never imports from notebooks.

---

## Key Conventions

- **Config:** All paths and date constants are in `src/config.py`. Never hard-code a path or date elsewhere.
- **Predictions file:** `data/processed/predictions.parquet` has columns `permno, date, model, pred_ret`. All evaluation code reads from this file — do not change its schema without updating `src/evaluation/metrics.py` and `src/evaluation/portfolio.py`.
- **Target variable:** Predict **excess return** (stock return minus 1-month T-bill from the FF factors file), not total return.
- **NN ensemble:** Neural networks are trained with 10 different random seeds; predictions are averaged before any portfolio construction. Never evaluate a single-seed NN.
- **LightGBM for GBRT:** Use `lightgbm.LGBMRegressor` as the gradient boosting implementation (much faster than sklearn's). The interface is compatible with sklearn pipelines.

---

## The Three Extensions (Phases 5a, 5b, 5c)

| # | Name | File | One-line description |
|---|---|---|---|
| 1 | Post-2020 OOS | `src/extensions/post2020_eval.py` | Extend the test window past 2016 through the COVID crash, 2022 bear market, and beyond |
| 2 | Net of Transaction Costs | `src/extensions/transaction_costs.py` | Adjust gross L/S returns by turnover × estimated spread; check if model rankings change |
| 3 | Feature Parsimony | `src/extensions/feature_parsimony.py` | Retrain all models on a 15–20 feature subset selected from RF importance + interpretability filter; measure the R² cost |

---

## Common Pitfalls Encountered in This Domain
- **CCM link table:** Replace `linkenddt = 'E'` with `'9999-12-31'` before date filtering or all currently-active links will be dropped.
- **Compustat duplicates:** Always filter `indfmt='INDL', datafmt='STD', popsrc='D', consol='C'` on `comp.funda` or you get 2–4 rows per firm-year.
- **CRSP daily file size:** Never pull the full daily file in one WRDS query. Always loop by year.
- **Delisting returns:** Impute −30% for performance-related delistings (`dlstcd 500–584`) with missing `dlret`. Ignoring this creates survivorship bias.
- **Amihud ILLIQ outliers:** Winsorise at the 99th percentile each month before using in transaction cost calculations. Penny stocks produce extreme values.
