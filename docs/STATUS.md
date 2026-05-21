# Project Status

> Last updated: 2026-05-21 (post-audit fixes applied)

---

## Current Phase: 3 — Feature Engineering (complete, audited)

| Phase | Name | Status | Notes |
|-------|------|--------|-------|
| 0 | Environment setup | ✅ Done | Python 3.12, all deps installed |
| 1 | WRDS data extraction | ✅ Done | All raw tables downloaded, audited |
| 2 | Data cleaning & CCM merge | ✅ Done | merged_panel 2.5M rows × 118 cols, 11/11 tests ✅ |
| 3 | Feature engineering (94 characteristics) | ✅ Done | 81/94 chars; audited & fixed; features_panel.parquet rebuilt (8.2 min) |
| 4 | Model training & evaluation | 🔲 Not started | — |
| 5a | Extension — Post-2020 OOS | 🔲 Not started | — |
| 5b | Extension — Net of transaction costs | 🔲 Not started | — |
| 5c | Extension — Feature parsimony | 🔲 Not started | — |
| 6 | Notebooks & visualisation | 🔲 Not started | — |
| 7 | LaTeX report | 🔲 Not started | — |

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
| **Total** | **81** | GKX has 94; 13 features TBD (depr, hire, herf, etc.) |

### Output (validated 2026-05-21)

`data/processed/features_panel.parquet`:
- **Shape**: 2,431,956 rows × 84 cols (permno + date + 81 characteristics + ret_exc)
- **Date range**: 1957-01-31 → 2024-11-30 (815 months, 23,750 permnos)
- **ret_exc**: 0 nulls; mean = +0.82%/mo, median = −0.32%/mo, σ = 17.8%/mo (normal for full cross-section)
- **Characteristics**: 0% null (all imputed to 0 = cross-sectional median)
- **Runtime**: 8.2 minutes

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

# 7. (Phase 4 — coming next)
# python -m src.models.train
```

---

## WRDS access

Only **Tobia** has a WRDS account. The raw Parquet files (~12 GB total) are on his machine and OneDrive. No other team members need a WRDS account — copy the `data/raw/` folder to run Phases 2–3 onwards.

---

## Immediate next steps

1. ✅ Phase 3 complete — `features_panel.parquet` validated
2. ⬜ Commit Phase 3 feature engineering code
3. ⬜ Start Phase 4: baseline models (OLS, Ridge, ElasticNet, RF, GBRT, NN1–NN5)
4. ⬜ Add missing ~13 GKX characteristics (depr, hire, herf, orgcap, etc.) in a follow-up PR
