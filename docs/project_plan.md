# Project Plan — Empirical Asset Pricing via Machine Learning
### Replication of Gu, Kelly & Xiu (2020) + Three Custom Extensions

> **Last updated:** 2026-05-23  
> **Data source:** WRDS (all team members have accounts)  
> **ML framework:** Python · TensorFlow/Keras · scikit-learn · LightGBM  
> **Test window extended to:** most recent available WRDS month (~2026)

---

## Table of Contents

1. [Phase 0 — Environment Setup](#phase-0--environment-setup)
2. [Phase 1 — WRDS Data Extraction](#phase-1--wrds-data-extraction)
3. [Phase 2 — Data Cleaning & CCM Merge](#phase-2--data-cleaning--crspcompustat-merge)
4. [Phase 3 — Feature Engineering (94 Characteristics)](#phase-3--feature-engineering-94-characteristics)
5. [Phase 4 — Model Training & Evaluation (Replication)](#phase-4--model-training--evaluation-replication)
6. [Phase 5 — Extensions](#phase-5--extensions)
7. [Phase 6 — Notebooks & Visualisation](#phase-6--notebooks--visualisation)
8. [Phase 7 — Reporting](#phase-7--reporting)
9. [Full File Map](#full-file-map)
10. [Scope Boundaries & Standing Decisions](#scope-boundaries--standing-decisions)
11. [Verification Checklist](#verification-checklist)

---

## Phase 0 — Environment Setup

**Goal:** Reproducible, identical environment for every team member before any data work begins.

### Tasks
1. Expand `requirements.txt` with: `wrds`, `pyarrow`, `lightgbm`, `shap`, `pytest`, `pandas_datareader`.
2. Create the full source skeleton:
   ```
   src/
   ├── config.py
   ├── data/
   ├── features/
   ├── models/
   ├── evaluation/
   └── extensions/
   ```
   Each subdirectory gets an `__init__.py`.
3. Populate `src/config.py` with:
   - Absolute paths to `data/raw/` and `data/processed/` (derived from `__file__`).
   - Date constants: `TRAIN_END = "1974-12"`, `VAL_END = "1986-12"`, `REPLICATE_TEST_END = "2016-12"`, `EXT_END = None` (resolved at runtime from latest CRSP month).
   - WRDS username (read from environment variable `WRDS_USER`, never hard-coded).
   - Model hyperparameter defaults (can be overridden per run).
4. Smoke-test WRDS connection: `import wrds; conn = wrds.Connection()`. On first run this caches credentials interactively; subsequent runs are non-interactive.

### Files created
`src/config.py` · `src/data/__init__.py` · `src/features/__init__.py` · `src/models/__init__.py` · `src/evaluation/__init__.py` · `src/extensions/__init__.py` · updated `requirements.txt`

### ⚠ Bottlenecks
- The WRDS `wrds` package caches credentials in `~/.pgpass` (Linux/Mac) or `%APPDATA%\postgresql\pgpass.conf` (Windows). Each team member must run the interactive login once before any automated scripts work.
- Pin library versions in `requirements.txt` to avoid silent API changes in `sklearn` or `tensorflow` between team members' machines.

---

## Phase 1 — WRDS Data Extraction

**Goal:** Pull all necessary raw tables from WRDS and persist them locally as Parquet files under `data/raw/`.

**Main file:** `src/data/wrds_downloader.py` — one function per table, each accepting `start_year`/`end_year` arguments and writing a dated Parquet file.

### Tables to Pull

| Table | WRDS Path | Columns | Notes |
|---|---|---|---|
| CRSP Monthly Stock | `crsp.msf` | `permno, date, ret, prc, shrout, vol, cfacshr` | Core monthly returns & prices |
| CRSP Names/Exchange | `crsp.msenames` | `permno, shrcd, exchcd, namedt, nameendt` | Share class and exchange filters |
| CRSP Delisting Returns | `crsp.mse` | `permno, dlret, dlstcd, dlpdt` | Required to avoid survivorship bias |
| CRSP Daily Stock | `crsp.dsf` | `permno, date, ret, vol, prc` | For beta, idiovol, retvol, maxret, ill. **Pull year-by-year** |
| Compustat Annual | `comp.funda` | All fundamental items | Filter: `indfmt='INDL', datafmt='STD', popsrc='D', consol='C'` |
| Compustat Quarterly | `comp.fundq` | Quarterly fundamentals | For roaq, sue, chtx, rsup |
| CCM Link Table | `crsp.ccmxpf_linktable` | `gvkey, lpermno, linktype, linkdt, linkenddt` | Use `linktype IN ('LC','LU')` only |
| FF Factors (monthly) | Kenneth French website (free) | FF3, FF5 | Via `pandas_datareader.data.DataReader("F-F_Research_Data_5_Factors_2x3", "famafrench")` |

### Files created
`data/raw/crsp_monthly.parquet` · `data/raw/crsp_daily_{year}.parquet` · `data/raw/compustat_annual.parquet` · `data/raw/compustat_quarterly.parquet` · `data/raw/ccm_links.parquet` · `data/raw/ff_factors.parquet`

### ⚠ Bottlenecks
- **CRSP daily file is large (~3–5 GB total).** Always pull year-by-year in a loop. Never attempt a full pull in one query — WRDS will time out or truncate results.
- **Compustat `funda` duplicates:** The `indfmt/datafmt/popsrc/consol` filter combination is mandatory. Without it, one `(gvkey, datadate)` pair may have 2–4 rows with different formats, breaking every downstream join.
- **WRDS connection pooling:** Open one `wrds.Connection()` object per session and pass it into all download functions; do not re-open per query.

---

## Phase 2 — Data Cleaning & CRSP/Compustat Merge

**Goal:** Produce a clean, survivorship-bias-free monthly panel linking CRSP returns to Compustat fundamentals.

### Files
`src/data/crsp_cleaner.py` · `src/data/compustat_cleaner.py` · `src/data/ccm_merger.py`

### CRSP Cleaning (`crsp_cleaner.py`)
1. Keep only ordinary common shares: `shrcd IN (10, 11)`.
2. Keep only major US exchanges: `exchcd IN (1, 2, 3)` (NYSE, AMEX, NASDAQ).
3. **Delisting return adjustment (Shumway 1997):** When `dlstcd ∈ {500–584}` (performance-related delist) and `dlret` is missing, impute `−0.30`. Merge the delisting return into the final month's return as:
   $$r_{\text{adj}} = (1 + r_{\text{monthly}}) \cdot (1 + r_{\text{delist}}) - 1$$
4. Compute `me = |prc| × shrout / 1000` (market cap in $thousands). Store both `me` and `me_lag1` (lagged one month — used for weighting in portfolio sorts).
5. Adjust prices for stock splits using `cfacshr`.

### Compustat Cleaning (`compustat_cleaner.py`)
1. Drop rows with missing `gvkey` or `datadate`.
2. Construct **public availability date**: `public_date = datadate + 6 months`. This is the earliest month a characteristic derived from this fiscal-year filing may be used as a predictor. **This lag is the primary look-ahead bias guard.**
3. For quarterly data, use `public_date_q = datadate_q + 3 months`.
4. Deduplicate on `(gvkey, datadate)` — keep one row per firm-year.

### CCM Merge (`ccm_merger.py`)
1. Join `crsp_clean` to `ccm_links` on `permno = lpermno` and `crsp_date BETWEEN linkdt AND linkenddt`. Replace `linkenddt = 'E'` with `9999-12-31` before filtering.
2. For each `(permno, crsp_month)`, attach the most recent Compustat annual filing where `public_date ≤ crsp_month ≤ public_date + 12 months`.
3. Similarly attach the most recent quarterly filing where `public_date_q ≤ crsp_month`.
4. Output one row per `(permno, year_month)` with all raw accounting fields attached.

### Files created
`data/processed/crsp_clean.parquet` · `data/processed/compustat_clean.parquet` · `data/processed/merged_panel.parquet`

### ⚠ Bottlenecks
- **Look-ahead bias is the most dangerous and subtle bug in the project.** Write a unit test in `tests/test_no_lookahead.py` that asserts: for any row where the Compustat fiscal year ends in December of year Y, the feature may not appear in the panel before July of year Y+1.
- **Multiple CCM links per permno:** A firm may have two overlapping `gvkey` mappings during a merger. Prioritise `linktype = 'LC'` over `'LU'`; if still ambiguous, keep the link with the longer active period.
- **Quarterly Compustat coverage:** Not all firms report quarterly. Missing quarterly features must fall back to the annual filing value, not be treated as NaN.

---

## Phase 3 — Feature Engineering (94 Characteristics)

**Goal:** Construct all 94 firm characteristics from GKX Table A.1 and assemble the final feature matrix.

### File Structure
| File | Characteristics |
|---|---|
| `src/features/momentum_features.py` | mom1m, mom6m, mom12m, mom36m, chmom, indmom, maxret, turn, std_turn, dolvol, std_dolvol, mve_ia |
| `src/features/value_features.py` | bm, bm_ia, cfp, ep, sp, agr, pchsale_pchinvt, pchsale_pchxsga, cfp_ia, ep_ia, pchcapx_ia, ... |
| `src/features/profitability_features.py` | roaq, roe, roeq, gp, gma, niy, rdmq, ... |
| `src/features/investment_features.py` | invest, noa, chcsho, grprofits, pchcapex, chtx, ... |
| `src/features/trading_friction_features.py` | ill (Amihud), beta, betasq, idiovol, retvol, me, ... |
| `src/features/feature_assembler.py` | Merges all modules · rank normalization · missing imputation · final panel output |

### Cross-Sectional Rank Normalization (GKX §2.2)
Each characteristic $c_{i,t}$ is transformed to $[-1, +1]$ at every month $t$:

$$\tilde{c}_{i,t} = 2 \cdot \frac{\text{rank}(c_{i,t}) - 1}{N_t - 1} - 1$$

Stocks with missing values for a given characteristic are assigned 0 (the cross-sectional median after normalization). **Apply normalization in `feature_assembler.py`, never inside individual feature files** — individual files return raw values.

### Files created
`data/processed/features_panel.parquet` — columns: `permno, date` + 94 characteristics + `ret` (next-month return, the prediction target)

### ⚠ Bottlenecks
- **Daily-based features** (`beta`, `idiovol`, `retvol`, `maxret`, `ill`) require a rolling 12-month window of daily CRSP data. Compute using vectorised NumPy or `pandas.GroupBy.rolling`. This is the slowest step in Phase 3 — expect 30–60 minutes on a laptop.
- **Industry adjustment** (`bm_ia`, `cfp_ia`, etc.) requires an industry mapping. Use the Compustat `sich` (historical SIC code) field; map to 49 Fama-French industries using the FF49 SIC mapping table (available on Kenneth French's website).
- **Coverage check:** After assembly, plot feature coverage (% non-missing per characteristic per year) as a heatmap. Characteristics with < 30% coverage before 1970 are expected; flag any with < 50% coverage after 1980.
- **Verify against the paper:** Cross-check the average monthly Information Coefficient (IC = rank correlation of feature with next-month return) for the top 5 characteristics. GKX Figure 1 shows mean IC ≈ 0.02–0.05 for most predictors. Large deviations indicate a construction error.

---

## Phase 4 — Model Training & Evaluation (Replication)

**Goal:** Reproduce GKX Table 3 (OOS-R²) and the main portfolio results (Sharpe ratios, FF5 alphas).

### Rolling-Window Scheme

| Split | Date Range | Purpose |
|---|---|---|
| Training | 1957-01 → end of year (t − 2) | Model fitting (expanding window) |
| Validation | 1975-01 → 1986-12 | Hyperparameter selection |
| Test (replication) | 1987-01 → 2016-12 | OOS evaluation matching GKX |

### Models

| Model | File | Key Hyperparameters |
|---|---|---|
| OLS-3 | `linear_models.py` | 3 most important characteristics only (size, B/M, momentum) |
| OLS-all | `linear_models.py` | All 94 predictors, no regularisation |
| PCR | `linear_models.py` | # components ∈ {3, 5, 10, 20, 50} — chosen on validation |
| PLS | `linear_models.py` | # latent factors ∈ {3, 5, 10, 20, 50} |
| Elastic Net | `linear_models.py` | α ∈ {0.001, 0.01, 0.1}, l1_ratio ∈ {0.1, 0.5, 0.9} |
| GLM (group LASSO) | `linear_models.py` | Group penalty λ via validation |
| Random Forest | `tree_models.py` | max_depth ∈ {2, 3, 4}, n_estimators ≥ 300, min_samples_leaf = 1000; selected: max_depth=2 |
| GBRT | `tree_models.py` | learning_rate = 0.01 (fixed), max_depth ∈ {1, 2}, subsample = 0.5; selected: lr=0.01, depth=2 |
| NN1–NN5 | `neural_nets.py` | hidden units per layer = 32, dropout = 0.50, L1 penalty, Adam lr = 0.001; NN1–NN2 were run with **10 seeds**, NN3–NN5 switch to a **3-seed deadline-mode ensemble** |

### Current execution status (2026-05-26) — Phase 4 **complete**
- All non-NN models complete and evaluated on 1987–2016.
- NN1 complete and evaluated with 10 seeds (GKX-compliant).
- NN2 complete and evaluated with 10 seeds (GKX-compliant).
- NN3 complete — 3-seed deadline-mode, Kaggle GPU.
- NN4 complete — 3-seed deadline-mode, local CPU.
- NN5 complete — 3-seed deadline-mode, local CPU.
- Total predictions: 18,165,186 rows (13 models × 1,397,322 stock-months).
- All evaluation CSVs regenerated: `eval_oos_r2_latest.csv`, `eval_ic_stats_latest.csv`, `eval_portfolio_perf_latest.csv`, `eval_pred_std_latest.csv`.

### Training Loop (`src/models/train_eval.py`)
- Iterate month-by-month from 1987-01 to the end of the test window.
- At each step: fit model on training data, generate one-step-ahead return prediction for all stocks in that month.
- Store predictions: `data/processed/predictions.parquet` — columns: `permno, date, model, pred_ret`.

### Evaluation (`src/evaluation/`)
- **`metrics.py`:** OOS-R² (pooled, by calendar year, by NYSE size quintile), monthly IC.
- **`portfolio.py`:** Monthly decile sorts on predicted return (value-weighted), long-short (P10 − P1) portfolio, annualised Sharpe ratio, FF3/FF5 alpha via time-series regression.

### Files created
`data/processed/predictions.parquet` · `data/processed/portfolio_returns.parquet`

### ⚠ Bottlenecks
- **Compute time:** NN training (10 seeds × architectures × ~360 rolling months) is too slow on CPU for the deadline. The project therefore keeps the full 10-seed results for NN1–NN2, and completes NN3–NN5 with a reduced 3-seed setup. Phase 4 training is now complete.
- **NN reproducibility:** Set seeds at the top of every training run: `np.random.seed(seed)`, `tf.random.set_seed(seed)`, `random.seed(seed)`. Log all seeds in a run manifest (`data/processed/run_manifest.json`).
- **GBRT memory:** Scikit-learn's `GradientBoostingRegressor` is slow for large N. Use `lightgbm.LGBMRegressor` as a drop-in replacement with identical interface.
- **Target variable:** Predict excess return (raw return minus risk-free rate), not total return. Use the 1-month T-bill rate from the FF factors file.

---

## Phase 5 — Extensions

*All three extensions are independent of each other and can be developed in parallel once Phase 4 is complete.*

---

### Extension 1: Post-2020 Out-of-Sample

**File:** `src/extensions/post2020_eval.py`

**Purpose:** Test whether model rankings and return predictability persist through unseen macro regimes (COVID crash, reflation, 2022 bear market).

#### Tasks
1. Extend the Phase 4 training loop beyond 2016-12 through `EXT_END` (the latest available WRDS month, resolved from `crsp.msf`).
2. Compute OOS-R² separately for each sub-period:

| Sub-period | Label |
|---|---|
| 2020-01 → 2020-12 | COVID crash & recovery |
| 2021-01 → 2021-12 | Reflation / growth rally |
| 2022-01 → 2022-12 | Rate-hike bear market |
| 2023-01 → EXT_END | Post-normalisation |

3. Compute annualised Sharpe ratios for each model's L/S portfolio within each sub-period.
4. Plot cumulative L/S returns for the top 3 models, overlaid with NBER recession shading and key macro events as vertical lines.

#### ⚠ Bottlenecks
- WRDS data availability lags ~6–8 weeks. Confirm the latest available `crsp.msf` date programmatically before finalising `EXT_END`. Add an assertion to `wrds_downloader.py` that logs the actual last date retrieved.
- Compustat data for recent quarters may be incomplete (firms file late). Apply a conservative extra 1-month lag for fiscal Q4 filings after 2023.

---

### Extension 2: Model Rankings Net of Transaction Costs

**File:** `src/extensions/transaction_costs.py`

**Purpose:** Show whether the gross return model ranking changes once realistic trading frictions are subtracted — an economically critical result the original paper ignores.

#### Tasks

**Step 1 — Estimate effective spreads (two methods, report both):**
- **Primary — Amihud (2002) illiquidity proxy:**
  $$\text{ILLIQ}_{i,t} = \frac{1}{D_{i,t}} \sum_{d=1}^{D_{i,t}} \frac{|r_{id}|}{DVOL_{id}}$$
  Scale to basis points using the Hasbrouck (2009) calibration (one-way spread ≈ $2\sqrt{\text{ILLIQ}}$ after unit adjustment).
- **Alternative — fixed schedule:** 5 bps one-way for stocks in NYSE size quintiles 4–5, 20 bps for quintiles 1–2 (small-cap). Use NYSE size breakpoints from the FF data library.

**Step 2 — Estimate portfolio turnover:**
  - For each model and each month $t$, compute one-way turnover:
    $$\text{TO}_{t} = \frac{1}{2} \sum_i |w_{i,t} - w_{i,t-1}|$$
  - Weights $w_{i,t}$ are rank-scaled value weights within the long and short deciles.

**Step 3 — Net return:**
  $$r_{\text{net},t} = r_{\text{gross},t} - \text{TO}_t \times s_{t}$$
  where $s_t$ is the portfolio-average one-way spread.

**Step 4 — Re-rank:** Compute net annualised Sharpe ratio for each model and report whether the ranking changes from gross.

#### ⚠ Bottlenecks
- DVOL (dollar volume) for small-cap stocks can be near zero, inflating ILLIQ to extreme values. Winsorise ILLIQ at the 99th percentile each month before any calculation.
- Turnover calculation is sensitive to how weight changes from fund-flow vs. active rebalancing are attributed. Use end-of-month weights (before rebalancing) for $w_{i,t-1}$.

---

### Extension 3: Feature Parsimony — 15–20 Characteristics vs. Full Set

**File:** `src/extensions/feature_parsimony.py`

**Purpose:** Determine whether the full 94-feature set is necessary, or whether 15–20 highly interpretable characteristics achieve comparable predictability — answering a key question for practitioners.

#### Feature Selection Methodology
Run this selection step once, after Phase 4 is complete, before retraining:

1. **Importance ranking:** Extract permutation importance from the best RF model (averaged across all 10 seeds and all months in the validation period 1975–1986). Rank all 94 features by mean importance.
2. **Interpretability filter (allow-list):** Pre-specify the following economically motivated candidates:

| Group | Candidates |
|---|---|
| Momentum | mom12m, mom1m, chmom, indmom |
| Value | bm, ep, cfp |
| Profitability | roaq, gp, roe |
| Investment | agr, invest, noa |
| Trading frictions | me, ill, idiovol, beta, retvol |

3. **Final shortlist (15–20 features):** Intersection of top-N by RF importance and the allow-list above. If fewer than 15 features survive the intersection, expand the importance threshold until 15 are reached.

#### Tasks
4. Retrain all models on the parsimonious feature set using the same rolling-window scheme and hyperparameters as Phase 4.
5. Report:
   - $\Delta R^2 = R^2_{\text{full}} - R^2_{\text{parsimonious}}$ (the parsimony cost)
   - Whether model rankings are preserved under the reduced set.
   - Feature importance stability: does the reduced model's SHAP importance align with the same features in the full model?

#### ⚠ Bottlenecks
- RF permutation importance has high variance. Average over all 10 seeds; do not use a single seed's importance for selection.
- Ensure the feature selection step uses **only validation-period data** (1975–1986). Using test-period data for selection would constitute a form of look-ahead bias.

---

## Phase 6 — Notebooks & Visualisation

**Goal:** Reproducible, report-ready figures and exploratory analysis.

| Notebook | Content |
|---|---|
| `notebooks/01_eda.ipynb` | Universe coverage (N stocks per year), return distribution, missing-value heatmap |
| `notebooks/02_feature_inspection.ipynb` | IC distribution per feature (boxplots), cross-correlation matrix, rank normalization spot checks |
| `notebooks/03_replication_results.ipynb` | OOS-R² table (Table 3 equivalent), cumulative L/S portfolio returns (Figure 3 equivalent), Sharpe bar chart |
| `notebooks/04_extensions.ipynb` | Sub-period OOS-R² (Ext 1), net-of-TC Sharpe comparison (Ext 2), parsimony R² bar chart (Ext 3) |
| `notebooks/05_final_plots.ipynb` | Publication-quality figures (300 dpi, consistent colour palette, LaTeX-style labels via `matplotlib.rc`) |

All notebooks must be **fully re-runnable from top to bottom** after Phase 4 outputs exist. No in-notebook data transformations — call functions from `src/` only.

---

## Phase 7 — Reporting

**Goal:** A structured academic write-up covering the replication and all extensions.

### Tasks
1. Create `report/main.tex` with the following sections: Introduction · Data & Sample Construction · Methodology · Replication Results · Extension 1 (Post-2020) · Extension 2 (Transaction Costs) · Extension 3 (Feature Parsimony) · Conclusion · Appendix (feature definitions).
2. Create `report/references.bib` — must include at minimum: GKX (2020), Shumway (1997), Amihud (2002), Hasbrouck (2009), Fama & French (1993, 2015), and 5+ additional empirical asset pricing references.
3. Export all final figures from `notebooks/05_final_plots.ipynb` to `report/figures/`.
4. Export all final tables from Python to `report/tables/` as `.tex` fragments using `df.to_latex(index=False, escape=False)`.
5. Set up `report/Makefile` for one-command PDF compilation: `make` → `pdflatex main.tex && bibtex main && pdflatex main.tex`.

### Key Tables to Reproduce
- **Table 3 equivalent:** OOS-R² (%) by model, full sample and by size quintile.
- **Portfolio table:** Annualised return, volatility, Sharpe ratio, FF5 alpha and t-stat for each model's L/S decile portfolio.
- **Extension tables:** Sub-period Sharpe ratios (Ext 1); gross vs. net Sharpe re-ranking (Ext 2); parsimony cost $\Delta R^2$ by model (Ext 3).

---

## Full File Map

```
src/
├── config.py                          Phase 0 — paths, dates, hyperparameter defaults
├── data/
│   ├── wrds_downloader.py             Phase 1 — WRDS query functions
│   ├── crsp_cleaner.py                Phase 2 — CRSP filtering, delisting returns
│   ├── compustat_cleaner.py           Phase 2 — Compustat cleaning, public_date
│   └── ccm_merger.py                  Phase 2 — CRSP/Compustat merge
├── features/
│   ├── momentum_features.py           Phase 3
│   ├── value_features.py              Phase 3
│   ├── profitability_features.py      Phase 3
│   ├── investment_features.py         Phase 3
│   ├── trading_friction_features.py   Phase 3
│   └── feature_assembler.py           Phase 3 — rank normalization, final panel
├── models/
│   ├── linear_models.py               Phase 4 — OLS, PCR, PLS, ENet, GLM
│   ├── tree_models.py                 Phase 4 — RF, GBRT (LightGBM)
│   ├── neural_nets.py                 Phase 4 — NN1–NN5 (TF/Keras)
│   └── train_eval.py                  Phase 4 — rolling-window training loop
├── evaluation/
│   ├── metrics.py                     Phase 4 — OOS-R², IC
│   └── portfolio.py                   Phase 4 — decile sorts, L/S returns, FF alpha
└── extensions/
    ├── post2020_eval.py               Phase 5 — Extension 1
    ├── transaction_costs.py           Phase 5 — Extension 2
    └── feature_parsimony.py           Phase 5 — Extension 3

data/
├── raw/                               Phase 1 outputs (git-ignored)
│   ├── crsp_monthly.parquet
│   ├── crsp_daily_{year}.parquet
│   ├── compustat_annual.parquet
│   ├── compustat_quarterly.parquet
│   ├── ccm_links.parquet
│   └── ff_factors.parquet
└── processed/                         Phases 2–4 outputs (git-ignored)
    ├── crsp_clean.parquet
    ├── compustat_clean.parquet
    ├── merged_panel.parquet
    ├── features_panel.parquet
    ├── predictions.parquet
    ├── portfolio_returns.parquet
    └── run_manifest.json

notebooks/
├── 01_eda.ipynb
├── 02_feature_inspection.ipynb
├── 03_replication_results.ipynb
├── 04_extensions.ipynb
└── 05_final_plots.ipynb

report/
├── main.tex
├── references.bib
├── Makefile
├── figures/
└── tables/
```

---

## Scope Boundaries & Standing Decisions

| Topic | Decision |
|---|---|
| Universe | CRSP ordinary common shares `shrcd ∈ {10, 11}`, NYSE/AMEX/NASDAQ `exchcd ∈ {1, 2, 3}` |
| Sample start | January 1957 (matching GKX) |
| Sample end | Most recent available WRDS month (resolved at runtime) |
| Options-based features | **Out of scope** — OPTIONM is complex; omit the ~6 GKX features that require it |
| Non-US equities | **Out of scope** |
| TC data source | Amihud (2002) as primary; fixed bps schedule as robustness check |
| NN framework | TensorFlow / Keras (matches GKX original implementation) |
| NN ensemble | Mixed due to deadline constraints: NN1–NN2 use 10 seeds, NN3–NN4 use 3 seeds, NN5 pending (target 3 seeds) |
| Feature parsimony shortlist | Determined algorithmically (Phase 5 methodology) — not hard-coded |
| WRDS credentials | Read from `WRDS_USER` environment variable; never commit credentials |
| Compute | Run NN training on GPU if available; use LightGBM for GBRT |

---

## Verification Checklist

Run these checks at the end of each phase before moving to the next.

- [ ] **Phase 2 — Look-ahead guard:** For all December fiscal-year Compustat rows, assert `feature_month >= datadate + 6 months`. Implement as a pytest in `tests/test_no_lookahead.py`.
- [ ] **Phase 3 — IC sanity:** Monthly average IC for `mom12m`, `bm`, `ep` should be positive and ≈ 0.02–0.05. A negative mean IC indicates a sign error in construction.
- [ ] **Phase 3 — Coverage:** Feature coverage heatmap shows no characteristic has < 50% non-missing after 1980.
- [ ] **Phase 4 — Replication:** NN3 and NN4 are now complete but materially below GKX benchmarks; run NN5 and reassess whether any deep model beyond NN2 approaches GKX's reported NN gains.
- [ ] **Extension 1 — Rolling R²:** Compute 12-month rolling OOS-R²; verify it does not go below −1% for more than 3 consecutive months for the best model.
- [ ] **Extension 2 — Turnover face-validity:** Monthly turnover for `mom1m`-weighted portfolio should be materially higher than for `bm`-weighted portfolio (momentum rebalances frequently; value does not).
- [ ] **Extension 3 — Parsimony cost:** $\Delta R^2$ (full − parsimonious) should be ≤ 30% of full-model R². If larger, revisit the feature selection methodology.
