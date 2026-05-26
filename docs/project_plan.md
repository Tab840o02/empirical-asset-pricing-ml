# Execution Workflow — Empirical Asset Pricing via Machine Learning
### Replication of Gu, Kelly & Xiu (2020) + Three Extensions

> **Completed:** May 2026  
> **Data source:** WRDS (PostgreSQL via the `wrds` Python package)  
> **ML framework:** Python 3.10 · TensorFlow/Keras · scikit-learn · LightGBM  
> **Neural network training:** Kaggle (GPU); all other steps run locally on CPU

---

## Repository Layout

```
src/                    Core library — data, features, models, evaluation, extensions
scripts/                Execution drivers (see pipeline below)
  run_final_eval.py       Phase 4 evaluation
  run_phase5a.py          Extension 1: Post-2020 OOS
  run_phase5b.py          Extension 2: Transaction costs
  run_phase5c.py          Extension 3: Feature parsimony training (also used on Kaggle)
  run_phase5c_eval.py     Extension 3: Parsimony evaluation
  generate_tables.py      Generate LaTeX table fragments for the report
  kaggle/                 Kaggle-specific bootstrap and runners
    kaggle_bootstrap.py
    kaggle_nn_runner.py
    kaggle_phase5c_runner.py
    make_kaggle_bundles.ps1
tests/                  Pytest unit tests
notebooks/              EDA and result visualisations (01_eda through 05_final_plots)
report/                 LaTeX source (main.tex), compiled PDF (main.pdf), tables/, figures/
data/raw/               WRDS downloads — git-ignored, never commit
data/processed/         Cleaned data and evaluation outputs — git-ignored
artifacts/kaggle/       Kaggle upload bundles (gkx_code_bundle.zip, gkx_processed_bundle.zip)
docs/                   Reference paper and this workflow document
```

---

## Step 1 — WRDS Data Download

**Script:** `src/data/wrds_downloader.py`

Connected to WRDS via `wrds.Connection()` (credentials from `WRDS_USER` env var; never hard-coded). Tables downloaded:

| Table | WRDS path | Notes |
|---|---|---|
| CRSP monthly returns | `crsp.msf` + `crsp.msenames` | shrcd, exchcd, cfacshr, vol |
| CRSP delisting returns | `crsp.mse` | Required to avoid survivorship bias |
| CRSP daily returns | `crsp.dsf` | Pulled year-by-year to avoid query timeouts |
| Compustat annual | `comp.funda` | `indfmt='INDL', datafmt='STD', popsrc='D', consol='C'` |
| Compustat quarterly | `comp.fundq` | For roaq, sue, chtx, rsup |
| CCM link table | `crsp.ccmxpf_linktable` | `linktype IN ('LC','LU')` only |
| FF 5-factor monthly | Kenneth French website | Via `pandas_datareader` |

**Output:** Parquet files in `data/raw/`.

---

## Step 2 — Data Cleaning

**Scripts:** `src/data/crsp_cleaner.py`, `src/data/compustat_cleaner.py`, `src/data/ccm_merger.py`, `src/data/crsp_compustat_linker.py`

**CRSP cleaning:**
- Kept ordinary common shares (`shrcd IN (10, 11)`) on NYSE/AMEX/NASDAQ (`exchcd IN (1, 2, 3)`).
- Imputed delisting return of −30% for performance-related delistings (`dlstcd 500–584`) with missing `dlret` (Shumway & Warther 1999). Ignoring this creates survivorship bias.
- Computed `me = |prc| × shrout / 1000`; stored `me_lag1` for portfolio weighting.

**Compustat cleaning:**
- Filtered with `indfmt='INDL', datafmt='STD', popsrc='D', consol='C'` — mandatory to avoid 2–4 duplicate rows per firm-year.
- Enforced **look-ahead lag**: annual data usable from `datadate + 6 months`; quarterly from `datadate + 3 months`. This `public_date` column is the primary look-ahead bias guard.

**CCM merge:**
- Replaced `linkenddt = 'E'` with `'9999-12-31'` before date filtering (otherwise all active links are dropped).
- Joined on `permno = lpermno` and `crsp_date BETWEEN linkdt AND linkenddt`.

**Output:** `data/processed/crsp_clean.parquet`, `data/processed/compustat_clean.parquet`, `data/processed/merged_panel.parquet`

---

## Step 3 — Feature Assembly

**Script:** `src/features/feature_assembler.py`

Built all 94 firm characteristics from GKX Appendix A, grouped across five feature modules:

| File | Category |
|---|---|
| `value_features.py` | Book-to-market, cash flow yield, earnings yield, sales-to-price |
| `momentum_features.py` | Short- and long-horizon price momentum, industry momentum, turnover |
| `profitability_features.py` | ROE, ROA, gross profitability, R&D intensity |
| `investment_features.py` | Asset growth, net operating assets, capex changes |
| `trading_friction_features.py` | Amihud illiquidity, beta, idiosyncratic volatility, return volatility |

Each module returns raw values. Cross-sectional rank normalisation to [−1, +1] is applied exclusively in `feature_assembler.py`, never inside individual feature files.

The prediction target is `ret_exc` — monthly excess return over the 1-month T-bill from the FF factors file.

**Output:** `data/processed/features_panel.parquet` — one row per `(permno, date)` with all 94 characteristics and `ret_exc`.

---

## Step 4 — Model Training (Phase 4)

**Scripts:** `src/models/train_eval.py`, `src/models/linear_models.py`, `src/models/tree_models.py`, `src/models/neural_nets.py`

**Train / validation / test split:**

| Split | Window | Purpose |
|---|---|---|
| Training | 1957–(t−2) expanding | Model fitting |
| Validation | 1975–1986 | Hyperparameter selection |
| Test | 1987–2016 | OOS evaluation matching GKX |

**13 models trained:**

| Model | Key hyperparameters |
|---|---|
| OLS-3 | 3 characteristics (size, B/M, momentum) |
| OLS-all | All 94, no regularisation |
| PCR | Components ∈ {3, 5, 10, 20, 50}; selected on validation R² |
| PLS | Latent factors ∈ {3, 5, 10, 20, 50} |
| Elastic Net | α ∈ {0.001, 0.01, 0.1}, l1_ratio ∈ {0.1, 0.5, 0.9} |
| GLM | Group LASSO; λ via validation |
| Random Forest | max_depth = 2, n_estimators ≥ 300 |
| GBRT | lr = 0.01, max_depth = 2, subsample = 0.5 (`lightgbm.LGBMRegressor`) |
| NN1–NN5 | 1–5 hidden layers × 32 units; dropout = 0.50; L1 penalty; Adam lr = 0.001 |

**NN training on Kaggle:**
1. Run `scripts/kaggle/make_kaggle_bundles.ps1` to build `artifacts/kaggle/gkx_code_bundle.zip` and `gkx_processed_bundle.zip`.
2. Upload both as Kaggle Datasets.
3. In a Kaggle Notebook: run `kaggle_bootstrap.py` to install dependencies, then `kaggle_nn_runner.py` to train each NN model (one at a time, `--append --resume` to survive session interruptions). Each NN uses 10 random seeds; predictions are averaged before evaluation.
4. Download predictions and merge into `data/processed/predictions.parquet` (`permno, date, model, pred_ret` schema).

Linear and tree models were trained locally. Predictions are checkpointed per test year via `train_eval.py`.

**Output:** `data/processed/predictions.parquet`, `data/processed/run_manifest.json`

---

## Step 5 — Phase 4 Evaluation

**Script:** `scripts/run_final_eval.py`

```
python scripts/run_final_eval.py
```

Evaluated all 13 models on the 1987–2016 test set using:
- Pooled OOS R² (stock-weighted; benchmark = expanding-window mean excess return)
- Monthly Spearman rank IC and ICIR
- Long-short decile portfolio: annual return, Sharpe, FF5 alpha (Newey-West 12 lags)

**Output:** `data/processed/eval_oos_r2_latest.csv`, `eval_ic_stats_latest.csv`, `eval_portfolio_perf_latest.csv`, `eval_pred_std_latest.csv`

---

## Step 6 — Extension 1: Post-2020 OOS Evaluation

**Script:** `scripts/run_phase5a.py`

```
python scripts/run_phase5a.py
```

Extended the test window from 2016 to 2024 using the same expanding-window scheme. Results decomposed by sub-period: Pre-COVID (2017–2019), COVID (2020), Reflation (2021), Rate hikes (2022), Post-normalisation (2023+). NN4 and NN5 were excluded from this extension due to constant-prediction collapse under distributional shift.

**Output:** `data/processed/predictions_ext.parquet`, `run_manifest_ext.json`, `eval_ext_oos_r2.csv`, `eval_ext_ic_stats.csv`, `eval_ext_portfolio_perf.csv`

---

## Step 7 — Extension 2: Transaction Costs

**Script:** `scripts/run_phase5b.py`

```
python scripts/run_phase5b.py
```

Applied Amihud (2002) illiquidity-calibrated bid-ask spread estimates to compute net-of-TC Sharpe ratios. Amihud ILLIQ was winsorised at the 99th percentile each month to control penny-stock outliers. Turnover measured as monthly portfolio rebalancing from decile migration.

**Output:** `data/processed/eval_tc_monthly.csv`, `eval_tc_summary.csv`

---

## Step 8 — Extension 3: Feature Parsimony Training (Kaggle)

**Scripts:** `scripts/kaggle/kaggle_phase5c_runner.py`, `scripts/run_phase5c.py`

Retrained all Phase 4 models on a 15-feature subset selected by RF permutation importance on the training/validation window only (no test-period information enters the selection). The 15 features span value, momentum, profitability, investment, and liquidity.

Kaggle execution follows the same bundle workflow as Step 4. `kaggle_phase5c_runner.py` calls `run_phase5c.py` inside the Kaggle environment.

```
python scripts/run_phase5c.py --models all
```

**Output:** `data/processed/predictions_parsimony.parquet`, `run_manifest_parsimony.json`

---

## Step 9 — Extension 3: Parsimony Evaluation

**Script:** `scripts/run_phase5c_eval.py`

```
python scripts/run_phase5c_eval.py
```

Compared Phase 4 (94 features) vs. Phase 5c (15 features) on OOS R² and portfolio performance. NN4 and NN5 produced degenerate zero-variance predictions in 13% and 60% of test months respectively; audited metrics restrict to non-degenerate months only.

**Output:** `data/processed/eval_parsimony_oos_r2.csv`, `eval_parsimony_portfolio_perf_audited.csv`

---

## Step 10 — Report Generation

**Scripts:** `scripts/generate_tables.py`, `report/Makefile`

```
python scripts/generate_tables.py   # reads eval CSVs → writes report/tables/*.tex
cd report && make                   # pdflatex → bibtex → pdflatex × 2 → main.pdf
```

The compiled report is `report/main.pdf` — 13 pages: cover + 5 body pages (§1 Motivation, §2 Replication, §3 Extensions, §4 What AI Couldn't Tell Me, §5 Future Extensions) + 7 appendix pages including references.

---

## Reproducing the Full Pipeline

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download raw data (requires WRDS credentials)
$env:WRDS_USER = "your_wrds_username"
python -c "from src.data.wrds_downloader import download_all; download_all()"

# 3. Clean and merge data
python -c "from src.data.crsp_cleaner import clean; clean()"
python -c "from src.data.compustat_cleaner import clean; clean()"
python -c "from src.data.ccm_merger import merge; merge()"

# 4. Build feature panel
python -c "from src.features.feature_assembler import build_panel; build_panel()"

# 5. Train non-NN models locally
python -c "from src.models.train_eval import run; run(models=['ols3','ols_all','glm','pcr','pls','enet','rf','gbrt'])"

# 6. Train NN models on Kaggle (see Step 4 above for bundle workflow), then merge predictions

# 7. Run all evaluations and extensions
python scripts/run_final_eval.py
python scripts/run_phase5a.py
python scripts/run_phase5b.py
python scripts/run_phase5c.py --models all
python scripts/run_phase5c_eval.py

# 8. Generate report
python scripts/generate_tables.py
cd report; make
```

---

## Key Invariants

- **No look-ahead bias:** Compustat annual data may not enter the panel before `datadate + 6 months`; quarterly before `+3 months`. Enforced via `public_date` in `ccm_merger.py`. Verified by `tests/test_no_lookahead.py`.
- **Credentials never in code:** WRDS username from `WRDS_USER` env var only.
- **Data not committed:** `data/raw/` and `data/processed/` are in `.gitignore`.
- **Normalisation in one place:** Cross-sectional rank normalisation only in `feature_assembler.py`.
- **NN ensemble:** Always average 10-seed predictions before evaluation; single-seed results are not reported.
- **Prediction schema:** `predictions.parquet` columns: `permno, date, model, pred_ret`. Do not alter without updating `metrics.py` and `portfolio.py`.
