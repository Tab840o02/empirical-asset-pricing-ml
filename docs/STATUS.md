# Project Status

> Last updated: 2026-05-21

---

## Current Phase: 1 → 2 (data downloading, Phase 2 code ready)

| Phase | Name | Status | Owner |
|-------|------|--------|-------|
| 0 | Environment setup | ✅ Done | All |
| 1 | WRDS data extraction — **code** | ✅ Done | — |
| 1 | WRDS data extraction — **download running** | ⏳ In progress | Tobia |
| 2 | Data cleaning & CRSP–Compustat merge | ✅ Code ready, waiting for data | — |
| 3 | Feature engineering (94 characteristics) | 🔲 Not started | — |
| 4 | Model training & evaluation | 🔲 Not started | — |
| 5a | Extension — Post-2020 OOS | 🔲 Not started | — |
| 5b | Extension — Net of transaction costs | 🔲 Not started | — |
| 5c | Extension — Feature parsimony | 🔲 Not started | — |
| 6 | Notebooks & visualisation | 🔲 Not started | — |
| 7 | LaTeX report | 🔲 Not started | — |

---

## What exists in the repo right now

```
src/config.py                        ← all paths, constants, hyperparameters
src/data/wrds_downloader.py          ← downloads all raw WRDS tables
src/data/crsp_compustat_linker.py    ← builds CRSP–Compustat link (CUSIP-based)
src/data/crsp_cleaner.py             ← Phase 2: CRSP cleaning
src/data/compustat_cleaner.py        ← Phase 2: Compustat cleaning
src/data/ccm_merger.py               ← Phase 2: merge into monthly panel
tests/test_no_lookahead.py           ← look-ahead bias unit tests
docs/project_plan.md                 ← full 7-phase plan
docs/STATUS.md                       ← this file
```

Skeleton `__init__.py` files exist for `features/`, `models/`, `evaluation/`, `extensions/`.

---

## How data flows

```
WRDS (PostgreSQL)
    ↓  wrds_downloader.py
data/raw/*.parquet          ← gitignored, only on Tobia's machine
    ↓  crsp_cleaner.py
    ↓  compustat_cleaner.py
data/processed/*_clean.parquet
    ↓  crsp_compustat_linker.py  (builds link table from CUSIPs)
    ↓  ccm_merger.py
data/processed/merged_panel.parquet
    ↓  Phase 3: feature_assembler.py
data/processed/features_panel.parquet
    ↓  Phase 4: models
data/processed/predictions.parquet
```

---

## WRDS — who needs to do what

| Who | What | When |
|-----|------|-------|
| **Tobia** | Run `python -m src.data.wrds_downloader` (downloads ~3–5 GB of raw Parquet files) | **Now — already running** |
| **Tobia** | Share `data/raw/*.parquet` via OneDrive / network drive / USB | After download finishes |
| **Everyone else** | Clone the repo, create `.venv` with Python 3.12, install `requirements.txt` | Any time |
| **Everyone else** | Copy the shared `data/raw/` folder into their local `data/raw/` | After Tobia shares the files |
| **Everyone else** | **No WRDS account needed** — just the Parquet files | — |

> Raw data is gitignored and never committed. Only **one person** needs to download it.

---

## How to set up your machine (team members)

```powershell
# 1. Clone or pull the repo
git clone <repo-url>
cd "Group project"

# 2. Create venv with Python 3.12  (NOT 3.14 — it breaks pandas on Windows)
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy the shared data/raw/ folder from Tobia into your data/raw/

# 5. Run Phase 2 (once raw data is available)
python -m src.data.crsp_cleaner
python -m src.data.compustat_cleaner
python -m src.data.crsp_compustat_linker
python -m src.data.ccm_merger
```

---

## Immediate next steps

1. ⏳ Wait for `wrds_downloader` to finish (30–90 min)
2. Share `data/raw/` with the team
3. `python -m src.data.crsp_cleaner`
4. `python -m src.data.compustat_cleaner`
5. `python -m src.data.crsp_compustat_linker`
6. `python -m src.data.ccm_merger`
7. `pytest tests/test_no_lookahead.py -v`  ← must all pass before Phase 3
8. Start Phase 3 — feature engineering
