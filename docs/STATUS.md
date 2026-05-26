# Project Status

> Last updated: 2026-05-26 (Phase 4 complete — all 13 models trained and evaluated)

---

## Current Phase: 4 Complete — Model Training & Evaluation (all 13 models done)

| Phase | Name | Status | Notes |
|-------|------|--------|-------|
| 0 | Environment setup | ✅ Done | Python 3.12, all deps installed |
| 1 | WRDS data extraction | ✅ Done | All raw tables downloaded, audited |
| 2 | Data cleaning & CCM merge | ✅ Done | merged_panel 2.5M rows × 118 cols, 11/11 tests ✅ |
| 3 | Feature engineering (94 characteristics) | ✅ Done | **94/94** characteristics implemented including `hire`, `ear` |
| 4 | Model training & evaluation | ✅ **Complete** | All 13 models evaluated (8 non-NN + NN1 + NN2 + NN3 + NN4 + NN5), 18,165,186 predictions, 1987–2016 |
| 5a | Extension — Post-2020 OOS | 🔲 Not started | — |
| 5b | Extension — Net of transaction costs | 🔲 Not started | — |
| 5c | Extension — Feature parsimony | 🔲 Not started | — |
| 6 | Notebooks & visualisation | 🔲 Not started | — |
| 7 | LaTeX report | 🔲 Not started | — |

---

## Phase 4 — Model Training & Evaluation Results (94/94 features)

> **Full comparison vs paper:** see [docs/results_comparison.md](results_comparison.md)

### Training scheme
- Expanding window: train on all months < Jan Y, predict all months in year Y
- Test window: 1987–2016 (360 months, **18,165,186** stock-month predictions)
- Features panel: 2,431,956 rows × **94 features** (all 94 GKX characteristics)
- Hyperparameter selection: train 1957–1974, validate 1975–1986
- Selected hyperparams: pcr_n=50, pls_n=10, enet_α=0.001 l1=0.1, glm_α=0.001, rf_depth=2, gbrt_lr=0.01 depth=2
- Runtime: ~84 min full non-NN run on CPU + NN1 and NN2 full 10-seed expanding-window runs + NN3 Kaggle run (23,331.6s) + NN4 local run (20,003s) + NN5 local run (38,665.7s)

### Pooled OOS R² vs GKX Table 3 (1987–2016)

| Model | Our OOS R² | GKX Table 3 | Status |
|-------|-----------|-------------|--------|
| OLS-3 | +0.025% | +0.06% | ✅ Close |
| OLS-all | +0.152% | +0.09% | ✅ Close |
| PCR | +0.169% | +0.19% | ✅ Very close |
| PLS | +0.163% | +0.25% | ✅ Close |
| ElasticNet | +0.180% | +0.22% | ✅ Close |
| GLM | +0.063% | +0.06% | ✅ **Exact match** |
| RF | −0.509% | +0.39% | ❌ Negative |
| GBRT | −0.989% | +0.34% | ⚠️ Improved, still negative |
| NN1 | +0.344% | +0.39% | ✅ Close |
| NN2 | +0.178% | +0.40% | ⚠️ Positive, below GKX and below NN1 |
| NN3 | +0.023% | +0.41% | ⚠️ Completed, far below paper |
| NN4 | +0.003% | +0.45% | ❌ Completed, near-zero OOS R² |
| NN5 | +0.007% | +0.55% | ❌ Completed, near-zero OOS R² (3-seed deadline mode) |

> **Tree finding:** The tree-fix pass validated pipeline logic. Forcing RF to search deeper trees selected `max_depth=2` but worsened RF OOS R². Restricting GBRT to `lr=0.01` improved OOS R² from `−3.80%` to `−0.99%` by reducing prediction variance.

### L/S Decile Portfolio Performance (value-weighted, 360/360 months)

> **Portfolio fix:** CRSP dates in `crsp_clean.parquet` are raw trading dates (e.g., `1987-05-29`). Predictions use calendar month-end dates (`1987-05-31`). The merge silently dropped 108 months. Fixed in `src/evaluation/portfolio.py` with `pd.offsets.MonthEnd(0)` normalization. Coverage is now **360/360**.

| Model | Monthly L/S | Annual Return | Sharpe | FF5 α | t(α) |
|-------|------------|--------------|--------|-------|------|
| PLS | **1.07%** | 12.9% | **0.85** | 13.2% | **4.76** |
| PCR | 1.10% | 13.2% | 0.81 | 14.1% | 4.42 |
| OLS-all | 1.00% | 12.0% | 0.74 | 12.5% | 3.81 |
| ElasticNet | 0.99% | 11.8% | 0.72 | 12.5% | 3.84 |
| OLS-3 | 0.86% | 10.3% | 0.70 | 9.3% | 3.35 |
| GLM | 0.75% | 9.0% | 0.56 | 9.0% | 3.57 |
| NN1 | **2.50%** | **30.0%** | **1.60** | **28.3%** | **6.21** |
| NN2 | 2.44% | 29.3% | 1.51 | 28.0% | 5.92 |
| NN3 | 0.58% | 7.0% | 0.41 | 8.4% | 2.55 |
| NN4 | 0.11% | 1.4% | 0.09 | 1.6% | 0.55 (NS) |
| NN5 | 0.11% | 1.3% | 0.10 | 2.7% | 1.05 (NS) |
| RF | 0.13% | 1.6% | 0.10 | 2.3% | 0.61 (NS) |
| GBRT | 0.06% | 0.8% | 0.04 | 1.1% | 0.30 (NS) |

### Open issues

1. **Tree models remain below paper** — both RF and GBRT OOS R² are negative and portfolio Sharpes are well below GKX; root cause unresolved; optional to investigate in Phase 5
2. **NN3–NN5 non-compliant** — trained with 3 seeds vs GKX’s 10; OOS R² near-zero; results for these models are directionally interesting but not directly comparable to GKX

---

## Phase 1 — Raw Data Audit (validated 2026-05-21)

All raw files are in `data/raw/` (gitignored).

| File | Rows | Size | Status |
|------|------|------|--------|
| `crsp_monthly.parquet` | 4,842,296 | 90 MB | ✅ All required cols present |
| `crsp_daily_YYYY.parquet` × 68 files | ~100M total | ~11 GB | ✅ 1957–2024 complete |
| `crsp_daily_index.parquet` | 17,117 | — | ✅ Market index |
| `crsp_names.parquet` | 117,830 | 3 MB | ✅ |
| `crsp_delistings.parquet` | 38,843 | 1 MB | ✅ |
| `compustat_annual.parquet` | 598,935 | 88 MB | ✅ 67 Compustat cols (incl. dcvt) |
| `compustat_quarterly.parquet` | 2,112,947 | 143 MB | ✅ Includes oancfy, capsq, txtq |
| `compustat_company.parquet` | 57,583 | 1.6 MB | ✅ SIC code — 2.3% null (use as sich fallback) |
| `compustat_security.parquet` | 76,566 | 1.8 MB | ✅ CUSIP/exchange data for CCM linker |
| `ff_factors.parquet` | 1,197 months | — | ✅ FF5 factors through 2026-03 |

> `sich` in compustat_annual has 38.9% null in the merged panel (expected — use `compustat_company.sic` as fallback in future run). Industry-adjusted features are imputed to 0 (cross-sectional median) for those firms.

---

## Phase 2 — Processed Data (validated 2026-05-21)

| File | Rows | Cols | Notes |
|------|------|------|-------|
| `crsp_clean.parquet` | 3,504,027 | 13 | Delistings imputed, excess return attached |
| `compustat_annual_clean.parquet` | 598,935 | 67 | `public_date` = fiscal year end + 6 months |
| `compustat_quarterly_clean.parquet` | 2,112,947 | 35 | `q_public_date` = quarter end + 3 months |
| `crsp_compustat_link.parquet` | 81,616 | 8 | CUSIP-based CCM links |
| `merged_panel.parquet` | 2,466,279 | 118 | 23,937 permnos, 816 months (1957-01 → 2024-12) |

Look-ahead bias: **11/11 tests pass** — no forward contamination confirmed.

### Known implementation deviations (Phase 2)

#### Quarterly fallback logic (§0.5 in results_comparison.md)

**Planned** (`docs/project_plan.md §Phase 2`):
> When a quarterly Compustat item is missing for a given month, fall back to the most-recently filed annual value for the same firm.

**Implemented:**
- `ccm_merger.py` left-joins the quarterly clean table onto the monthly panel.  Months with no matching quarterly row receive `NaN` for all `q_*` columns.
- `feature_assembler.py` applies cross-sectional rank normalisation, then calls `fillna(0.0)` — i.e., NaN is imputed to the **cross-sectional median rank (0)**, not to the corresponding annual filing value.

**Affected characteristics (quarterly-sourced):** `roaq`, `roeq`, `rdmq`, `sue`, `rs`, `stdcf`, `rsup`

**Consequence:** Firms without available quarterly data receive the median rank for those seven characteristics instead of a value anchored to their annual filing.  This introduces a mild attenuation bias for stocks with sparse Compustat quarterly coverage (primarily micro-caps and firms before ~1975).  The effect on test-window (1987–2016) results is expected to be small because quarterly coverage is near-complete for exchange-listed stocks by that period.

#### Annual Compustat attachment window (§0.4 in results_comparison.md)

**Planned:** A 12-month upper cap on annual Compustat rows — an annual filing for fiscal year ending in month M should not remain attached beyond M+18 (i.e., the next annual filing + 6-month disclosure lag).

**Implemented:** `ccm_merger.py` uses `public_date` as the lower bound for attachment but applies no upper cap.  Firms with missing subsequent filings may retain stale Compustat data indefinitely.

---

## Phase 3 — Feature Engineering

### Feature modules (`src/features/`)

| Module | Features (N) | Key characteristics |
|--------|-------------|---------------------|
| `momentum_features.py` | 11 | mom1m, mom6m, mom12m, mom36m, chmom, indmom, turn, std_turn, dolvol, std_dolvol, mve_ia |
| `value_features.py` | 20 | bm, bm_ia, cfp, cfp_ia, ep, ep_ia, sp, agr, sgr, lgr, lev, dy, pchsale_p*, cash, salecash, saleinv, salerec |
| `profitability_features.py` | 25 | gp, gma, roe, niy, roic, pm, chpm, chato, chatoia, rd_sale, rd_mve, tb, quick, curr, cashpr, roaq, roeq, rdmq, chinv, chtx, sue, rs, roavol, stdcf, ms |
| `investment_features.py` | 16 | invest, noa, chcsho, grprofits, pchcapex, cinvest, tang, realestate, acc, pctacc, absacc, convind, secured, securedind, divi, divo |
| `trading_friction_features.py` | 9 | beta, betasq, idiovol, retvol, maxret, ill, zerotrade, me, age |
| **Total** | **94** | All GKX characteristics implemented (including `hire`, `ear`) |

### Output (validated 2026-05-23)

`data/processed/features_panel.parquet`:
- **Shape**: 2,431,956 rows × 97 cols (permno + date + **94 characteristics** + ret_exc)
- **Date range**: 1957-01-31 → 2024-11-30 (815 months, 23,750 permnos)
- **ret_exc**: 0 nulls; mean = +0.82%/mo, median = −0.32%/mo, σ = 17.8%/mo (normal for full cross-section)
- **Characteristics**: 0% null (all imputed to 0 = cross-sectional median)
- `hire` and `ear` added via updated `profitability_features.py` (fixed `pd.merge_asof` crash in pandas 2.2.3)

### Phase 3 audit (2026-05-21) — fixes applied

| # | Severity | Feature | Issue | Fix |
|---|----------|---------|-------|-----|
| 1 | Critical | sue, rs | `shift(4)` / `rolling(8)` wrong in monthly panel | `shift(12)` / `rolling(24, min_periods=12)` |
| 2 | Critical | chtx | Used `q_txditcq` (deferred taxes, wrong field) | Use `q_txtq` (total income taxes, Thomas & Zhang 2011) |
| 3 | Critical | roavol, stdcf | `rolling(8)` covers only ~2 quarters in monthly panel | `rolling(24, min_periods=12)` |
| 4 | Moderate | convind | `a_dltt > 0` true for ~90% of firms (useless) | `a_dcvt > 0` (actual convertible debt field, Valta 2012) |
| 5 | Moderate | G-score g5/g6/g7 | `median(capx)/median(at)` ≠ `median(capx/at)` | Compute ratio first, then cross-sectional median |
| 6 | Minor | roeq | `shift(1)` instead of `shift(3)` for 1-quarter lag | `shift(3)` |

---

## What exists in the repo right now

```
src/config.py                            ← all paths, constants, hyperparameters
src/data/wrds_downloader.py              ← downloads all raw WRDS tables
src/data/crsp_cleaner.py                 ← Phase 2 cleaning
src/data/compustat_cleaner.py            ← Phase 2 cleaning
src/data/crsp_compustat_linker.py        ← builds CCM link table (CUSIP-based)
src/data/ccm_merger.py                   ← Phase 2: CCM merge → merged_panel.parquet
src/features/__init__.py
src/features/momentum_features.py        ← Phase 3 ✅
src/features/value_features.py           ← Phase 3 ✅
src/features/profitability_features.py   ← Phase 3 ✅
src/features/investment_features.py      ← Phase 3 ✅
src/features/trading_friction_features.py← Phase 3 ✅ (uses daily CRSP, ~15 min)
src/features/feature_assembler.py        ← Phase 3 orchestrator ✅
tests/test_no_lookahead.py               ← look-ahead bias tests (11/11 pass)
docs/project_plan.md
docs/STATUS.md                           ← this file
```

---

## How data flows

```
WRDS (PostgreSQL)
    ↓  wrds_downloader.py
data/raw/*.parquet                         ← gitignored
    ↓  crsp_cleaner.py
    ↓  compustat_cleaner.py
    ↓  crsp_compustat_linker.py
    ↓  ccm_merger.py
data/processed/merged_panel.parquet        ← 2.5M rows, 118 cols
    ↓  feature_assembler.py                ← ~15–20 min (daily rolling on 100M rows)
data/processed/features_panel.parquet     ← ~2.5M rows, 83 cols
    ↓  Phase 4: models
data/processed/predictions.parquet        ← permno, date, model, pred_ret
```

---

## Reproducing the pipeline (from scratch)

```powershell
# 1. Environment
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Set WRDS credentials (only Tobia needs this for Step 3)
$env:WRDS_USER = "tab840o02"

# 3. Download raw data from WRDS  [skip if data/raw/ is already populated]
python -m src.data.wrds_downloader

# 4. Phase 2 — clean and merge
python -m src.data.crsp_cleaner
python -m src.data.compustat_cleaner
python -m src.data.crsp_compustat_linker
python -m src.data.ccm_merger

# 5. Verify no look-ahead bias
python -m pytest tests/test_no_lookahead.py -v

# 6. Phase 3 — feature engineering  (~15–20 min)
python -m src.features.feature_assembler

# 7. Phase 4 — model training (~77 min for 8 non-NN models)
python -m src.models.train_eval

# 8. Evaluate results
python -W ignore -m src.evaluation.eval_summary
```

---

## WRDS access

Only **Tobia** has a WRDS account. The raw Parquet files (~12 GB total) are on his machine and OneDrive. No other team members need a WRDS account — copy the `data/raw/` folder to run Phases 2–3 onwards.

---

## Immediate next steps

1. ✅ Phase 3 complete — `features_panel.parquet` validated
2. ✅ Phase 4 partial complete — 12 models evaluated (8 non-NN + NN1 + NN2 + NN3 + NN4)
3. ⬜ Phase 4 NN5 — run `python -m src.models.train_eval --models nn5 --append --resume` (deadline-mode 3 seeds)
4. ⬜ Phase 5a — post-2020 OOS extension (`src/extensions/post2020_eval.py`)
5. ⬜ Phase 5b — net of transaction costs (`src/extensions/transaction_costs.py`)
6. ⬜ Phase 5c — feature parsimony (`src/extensions/feature_parsimony.py`)
7. ⬜ Notebooks 03, 04 — figures and tables for report
8. ⬜ Finalize report tables/figures after NN5 evaluation
