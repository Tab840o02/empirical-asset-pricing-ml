"""
src/data/compustat_cleaner.py
==============================
Phase 2 — Compustat cleaning (annual and quarterly).

Reads raw Parquet files from ``data/raw/`` and produces:
  - ``data/processed/compustat_annual_clean.parquet``
  - ``data/processed/compustat_quarterly_clean.parquet``

Cleaning steps (annual)
-----------------------
1. Drop rows with missing ``gvkey`` or ``datadate``.
2. Deduplicate on ``(gvkey, datadate)`` — the four-format filter in
   ``wrds_downloader.py`` should already guarantee uniqueness, but a final
   dedup guard is kept here for safety.
3. Compute ``public_date = datadate + 6 months``.  This is the earliest
   calendar month in which any characteristic derived from this annual
   filing may appear in the prediction panel (look-ahead bias guard).
4. For fiscal years ending in December (the majority), the 6-month lag
   means the filing first appears in July of the *following* calendar year,
   consistent with GKX §2.3 and the project-plan's look-ahead invariant.

Cleaning steps (quarterly)
--------------------------
Same as annual, but with a 3-month lag:
  ``public_date = datadate + 3 months``.

Run
---
    python -m src.data.compustat_cleaner
"""

from __future__ import annotations

import logging

import pandas as pd
from pandas.tseries.offsets import DateOffset

from src.config import (
    ANNUAL_LAG_MONTHS,
    COMPUSTAT_ANNUAL_CLEAN_PATH,
    COMPUSTAT_QUARTERLY_CLEAN_PATH,
    QUARTERLY_LAG_MONTHS,
    RAW_DIR,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Annual
# ---------------------------------------------------------------------------

def clean_compustat_annual(
    raw_dir=RAW_DIR,
    output_path=COMPUSTAT_ANNUAL_CLEAN_PATH,
) -> pd.DataFrame:
    """
    Clean Compustat annual fundamentals and write to Parquet.

    Key output column: ``public_date`` — the earliest month the filing
    may be used as a predictor (datadate + 6 months, end-of-month).
    """
    path = raw_dir / "compustat_annual.parquet"
    log.info("Loading %s …", path.name)
    df = pd.read_parquet(path)
    df["datadate"] = pd.to_datetime(df["datadate"])

    n_raw = len(df)

    # 1. Drop rows with missing identifiers
    df = df.dropna(subset=["gvkey", "datadate"])

    # 2. Deduplicate — keep the last row per (gvkey, datadate) as a safety net
    df = df.sort_values(["gvkey", "datadate"]).drop_duplicates(
        subset=["gvkey", "datadate"], keep="last"
    )

    # 3. Compute public_date — shifted to end-of-month for clean period joins
    df["public_date"] = (
        df["datadate"] + DateOffset(months=ANNUAL_LAG_MONTHS)
    ).dt.to_period("M").dt.to_timestamp("M")

    log.info(
        "Annual: %d raw → %d clean rows (%d unique gvkeys)",
        n_raw, len(df), df["gvkey"].nunique(),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    log.info("Wrote %s", output_path.name)
    return df


# ---------------------------------------------------------------------------
# Quarterly
# ---------------------------------------------------------------------------

def clean_compustat_quarterly(
    raw_dir=RAW_DIR,
    output_path=COMPUSTAT_QUARTERLY_CLEAN_PATH,
) -> pd.DataFrame:
    """
    Clean Compustat quarterly fundamentals and write to Parquet.

    Key output column: ``public_date`` — datadate + 3 months (end-of-month).
    """
    path = raw_dir / "compustat_quarterly.parquet"
    log.info("Loading %s …", path.name)
    df = pd.read_parquet(path)
    df["datadate"] = pd.to_datetime(df["datadate"])

    n_raw = len(df)

    # 1. Drop rows with missing identifiers
    df = df.dropna(subset=["gvkey", "datadate"])

    # 2. Deduplicate on (gvkey, datadate, fqtr) — keep last row
    dedup_cols = ["gvkey", "datadate"]
    if "fqtr" in df.columns:
        dedup_cols.append("fqtr")
    df = df.sort_values(dedup_cols).drop_duplicates(subset=dedup_cols, keep="last")

    # 3. Compute public_date
    df["public_date"] = (
        df["datadate"] + DateOffset(months=QUARTERLY_LAG_MONTHS)
    ).dt.to_period("M").dt.to_timestamp("M")

    log.info(
        "Quarterly: %d raw → %d clean rows (%d unique gvkeys)",
        n_raw, len(df), df["gvkey"].nunique(),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    log.info("Wrote %s", output_path.name)
    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def clean_all(raw_dir=RAW_DIR):
    """Run both annual and quarterly cleaning."""
    clean_compustat_annual(raw_dir)
    clean_compustat_quarterly(raw_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    clean_all()
