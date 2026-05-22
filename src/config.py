"""
src/config.py
=============
Central configuration for the GKX (2020) replication project.

All paths, date constants, universe filters, and model hyperparameter
defaults live here.  Import this module everywhere — never hard-code a
path or date in any other file.

WRDS credentials
----------------
Set the environment variable WRDS_USER to your WRDS username before
running any download script.  Passwords are cached by the wrds package
in ~/.pgpass (Linux/Mac) or %APPDATA%\\postgresql\\pgpass.conf (Windows)
after the first interactive login.

    PowerShell:   $env:WRDS_USER = "your_wrds_username"
    bash/zsh:     export WRDS_USER="your_wrds_username"
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Repository root — two levels up from this file: src/config.py → src/ → root
ROOT: Path = Path(__file__).resolve().parent.parent

RAW_DIR: Path = ROOT / "data" / "raw"
PROCESSED_DIR: Path = ROOT / "data" / "processed"

# Create directories if they do not exist (safe to call repeatedly)
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# WRDS credentials — read from environment, NEVER hard-coded
# ---------------------------------------------------------------------------

WRDS_USER: str | None = os.environ.get("WRDS_USER")

# ---------------------------------------------------------------------------
# Sample period
# ---------------------------------------------------------------------------

SAMPLE_START: str = "1957-01-01"   # First month included in the panel

# Rolling-window split points (matching GKX §3)
TRAIN_END: str = "1974-12-31"          # End of initial training window
VAL_START: str = "1975-01-01"          # Start of validation / hyperparameter tuning window
VAL_END: str = "1986-12-31"            # End of validation / hyperparameter tuning window
REPLICATE_TEST_END: str = "2016-12-31" # End of the original GKX test window

# Extension end-date: resolved at runtime from the latest available CRSP month.
# Scripts that need it should call src.data.wrds_downloader.get_latest_crsp_date().
EXT_END: str | None = None

# Compustat lookback: pull fundamentals starting this far back to cover
# fiscal years that feed into the 1957 panel start.
COMPUSTAT_START: str = "1950-01-01"

# ---------------------------------------------------------------------------
# Universe filters (applied during CRSP cleaning — Phase 2)
# ---------------------------------------------------------------------------

VALID_SHRCDS: tuple[int, ...] = (10, 11)   # Ordinary common shares only
VALID_EXCHCDS: tuple[int, ...] = (1, 2, 3) # NYSE (1), AMEX (2), NASDAQ (3)

# ---------------------------------------------------------------------------
# Look-ahead bias lags (Phase 2 / feature engineering)
# ---------------------------------------------------------------------------

# Compustat annual: earliest month a fiscal-year filing may appear in the panel
ANNUAL_LAG_MONTHS: int = 6

# Compustat quarterly: earliest month a quarterly filing may appear in the panel
QUARTERLY_LAG_MONTHS: int = 3

# ---------------------------------------------------------------------------
# CRSP–Compustat linking (Phase 1 — professor's CUSIP approach)
# ---------------------------------------------------------------------------

# Primary key: 8-character CUSIP (crsp.msenames.ncusip == comp.security.cusip[:8])
CUSIP_LENGTH: int = 8

# Minimum overlap (days) required for a valid point-in-time link
MIN_LINK_OVERLAP_DAYS: int = 1

# Sentinel date used when an identifier's end date is missing (i.e. still active)
ACTIVE_END_DATE: str = "2099-12-31"

# ---------------------------------------------------------------------------
# Feature engineering (Phase 3)
# ---------------------------------------------------------------------------

RANK_NORM_LOWER: float = -1.0
RANK_NORM_UPPER: float = 1.0

# Rolling windows for daily-return-based features (in trading days)
BETA_WINDOW_DAYS: int = 252    # ~12 months
IDIOVOL_WINDOW_DAYS: int = 252
RETVOL_WINDOW_DAYS: int = 21   # ~1 month
MAXRET_WINDOW_DAYS: int = 21
ILLIQ_WINDOW_DAYS: int = 252

# ---------------------------------------------------------------------------
# Model hyperparameter defaults (Phase 4)
# ---------------------------------------------------------------------------

NN_SEEDS: list[int] = list(range(10))  # 10 ensemble seeds (0–9), averaged before eval

NN_PARAMS: dict = {
    "hidden_units": 32,
    "dropout_rate": 0.50,   # GKX internet appendix Table I
    "l1_penalty": 1e-5,
    "learning_rate": 1e-3,
    "batch_size": 10_000,
    "max_epochs": 100,
    "early_stopping_patience": 5,
}

RF_PARAMS: dict = {
    "n_estimators": 300,
    "max_depth": 2,
    "min_samples_leaf": 1000,
    "n_jobs": -1,
    "random_state": 42,
}

GBRT_PARAMS: dict = {
    "n_estimators": 300,
    "learning_rate": 0.01,
    "max_depth": 2,
    "subsample": 0.5,
    "n_jobs": -1,
    "random_state": 42,
}

# ---------------------------------------------------------------------------
# Processed file paths — single source of truth for all downstream code
# ---------------------------------------------------------------------------

CRSP_CLEAN_PATH = PROCESSED_DIR / "crsp_clean.parquet"
COMPUSTAT_ANNUAL_CLEAN_PATH = PROCESSED_DIR / "compustat_annual_clean.parquet"
COMPUSTAT_QUARTERLY_CLEAN_PATH = PROCESSED_DIR / "compustat_quarterly_clean.parquet"
MERGED_PANEL_PATH = PROCESSED_DIR / "merged_panel.parquet"
FEATURES_PANEL_PATH = PROCESSED_DIR / "features_panel.parquet"
PREDICTIONS_PATH = PROCESSED_DIR / "predictions.parquet"
PORTFOLIO_RETURNS_PATH = PROCESSED_DIR / "portfolio_returns.parquet"
LINK_TABLE_PATH = PROCESSED_DIR / "crsp_compustat_link.parquet"
RUN_MANIFEST_PATH = PROCESSED_DIR / "run_manifest.json"

# ---------------------------------------------------------------------------
# Backward-compat aliases (matching Phase 0 naming convention)
# ---------------------------------------------------------------------------
PROJECT_ROOT = ROOT
RAW_DATA_DIR = RAW_DIR
PROCESSED_DATA_DIR = PROCESSED_DIR

# ---------------------------------------------------------------------------
# Hyperparameter search grids (Phase 4)
# ---------------------------------------------------------------------------
HYPERPARAMS: dict = {
    "pcr":  {"n_components": [3, 5, 10, 20, 50]},
    "pls":  {"n_components": [3, 5, 10, 20, 50]},
    "enet": {"alpha": [0.001, 0.01, 0.1], "l1_ratio": [0.1, 0.5, 0.9]},
    "rf":   {"max_depth": [1, 2, 4], "n_estimators": 300, "min_samples_leaf": 1000},
    "gbrt": {"learning_rate": [0.01, 0.1], "max_depth": [1, 2], "subsample": 0.5},
    "nn":   {"hidden_units": 32, "dropout": 0.50, "lr": 1e-3, "n_seeds": 10},  # GKX IA Table I
}


# ---------------------------------------------------------------------------
# Utility: open WRDS connection
# ---------------------------------------------------------------------------
def connect_wrds():
    """Open and return a wrds.Connection(). First call caches credentials interactively."""
    import wrds
    return wrds.Connection(wrds_username=WRDS_USER)
