"""
src/data/crsp_cleaner.py
========================
Phase 2 — CRSP cleaning.

Reads raw Parquet files from ``data/raw/`` and produces
``data/processed/crsp_clean.parquet``.

Cleaning steps
--------------
1. Restrict to ordinary common shares: ``shrcd ∈ {10, 11}``.
2. Restrict to major US exchanges: ``exchcd ∈ {1, 2, 3}`` (NYSE / AMEX / NASDAQ).
3. Delisting-return adjustment (Shumway 1997):
   - For performance-related delistings (``dlstcd 500–584``) with a missing
     ``dlret``, impute −0.30.
   - Compound the delist return into the month's regular return.
4. Compute market cap ``me = |prc| × shrout / 1000`` (in $thousands).
5. Compute ``me_lag1`` — market cap lagged one month (used for value-weighting).
6. Adjust closing price for splits: ``prc_adj = prc / cfacshr``.

Run
---
    python -m src.data.crsp_cleaner
"""

from __future__ import annotations

import logging

import pandas as pd

from src.config import (
    ACTIVE_END_DATE,
    CRSP_CLEAN_PATH,
    RAW_DIR,
    VALID_EXCHCDS,
    VALID_SHRCDS,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_monthly(raw_dir=RAW_DIR) -> pd.DataFrame:
    path = raw_dir / "crsp_monthly.parquet"
    log.info("Loading %s …", path.name)
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _load_names(raw_dir=RAW_DIR) -> pd.DataFrame:
    path = raw_dir / "crsp_names.parquet"
    log.info("Loading %s …", path.name)
    df = pd.read_parquet(path)
    df["namedt"] = pd.to_datetime(df["namedt"])
    df["nameendt"] = pd.to_datetime(df["nameendt"])
    # Fill missing end dates — still active
    df["nameendt"] = df["nameendt"].fillna(pd.Timestamp(ACTIVE_END_DATE))
    return df


def _load_delistings(raw_dir=RAW_DIR) -> pd.DataFrame:
    path = raw_dir / "crsp_delistings.parquet"
    log.info("Loading %s …", path.name)
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------------------------
# Step 1 & 2 — Universe filter
# ---------------------------------------------------------------------------

def _apply_universe_filter(monthly: pd.DataFrame, names: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only stocks that were ordinary common shares listed on a major
    exchange *on the observation date*.

    The share/exchange codes are time-varying (stored in ``msenames`` with
    validity ranges ``[namedt, nameendt]``).  We do a point-in-time join:
    attach the applicable name row to each monthly observation, then filter.
    """
    log.info("Applying universe filter (shrcd / exchcd) …")

    # Keep only the columns needed for filtering
    names_filt = names[["permno", "shrcd", "exchcd", "namedt", "nameendt"]].copy()

    # Merge monthly obs with the name record that was valid on that date
    # Using a merge on permno, then filter by date range
    merged = monthly.merge(names_filt, on="permno", how="left")
    in_range = (merged["date"] >= merged["namedt"]) & (merged["date"] <= merged["nameendt"])
    merged = merged[in_range].copy()

    # If a permno has multiple valid name rows on the same date (rare), keep the latest
    merged = (
        merged
        .sort_values(["permno", "date", "namedt"])
        .drop_duplicates(subset=["permno", "date"], keep="last")
    )

    # Apply share-class and exchange filter
    n_before = len(merged)
    merged = merged[
        merged["shrcd"].isin(VALID_SHRCDS) & merged["exchcd"].isin(VALID_EXCHCDS)
    ]
    log.info(
        "  Universe filter: %d → %d rows (removed %d)",
        n_before, len(merged), n_before - len(merged),
    )
    return merged


# ---------------------------------------------------------------------------
# Step 3 — Delisting returns
# ---------------------------------------------------------------------------

def _merge_delistings(df: pd.DataFrame, delistings: pd.DataFrame) -> pd.DataFrame:
    """
    Adjust monthly returns for delisting.

    For performance-related delistings (dlstcd 500–584) with a missing
    dlret, impute −0.30 (Shumway 1997 / GKX convention).

    The delist return is compounded into the last regular monthly return:
        r_adj = (1 + ret) * (1 + dlret) − 1

    If ``ret`` is also missing in the final month, use only ``dlret``.
    """
    log.info("Merging delisting returns …")

    # Normalise delistings to year-month for joining
    delist = delistings[["permno", "date", "dlstcd", "dlret"]].copy()
    delist["ym"] = delist["date"].dt.to_period("M")

    # Impute −30% for performance-related delistings with missing dlret
    perf_delist = delist["dlstcd"].between(500, 584)
    delist.loc[perf_delist & delist["dlret"].isna(), "dlret"] = -0.30

    # Keep only rows that actually carry a delist return
    delist = delist[delist["dlret"].notna()][["permno", "ym", "dlret"]]

    df["ym"] = df["date"].dt.to_period("M")
    df = df.merge(delist, on=["permno", "ym"], how="left")

    # Compound delist return into regular return
    has_delist = df["dlret"].notna()
    df.loc[has_delist & df["ret"].notna(), "ret"] = (
        (1 + df.loc[has_delist & df["ret"].notna(), "ret"])
        * (1 + df.loc[has_delist & df["ret"].notna(), "dlret"])
        - 1
    )
    # If regular ret is missing but delist ret exists, use delist ret alone
    df.loc[has_delist & df["ret"].isna(), "ret"] = df.loc[
        has_delist & df["ret"].isna(), "dlret"
    ]

    df = df.drop(columns=["dlret", "ym"])
    log.info("  Delisting returns merged.")
    return df


# ---------------------------------------------------------------------------
# Steps 4–6 — Market cap and price adjustment
# ---------------------------------------------------------------------------

def _compute_market_cap(df: pd.DataFrame) -> pd.DataFrame:
    """me = |prc| × shrout / 1000  ($thousands)."""
    log.info("Computing market cap …")
    df["me"] = df["prc"].abs() * df["shrout"] / 1_000
    return df


def _adjust_price(df: pd.DataFrame) -> pd.DataFrame:
    """Split-adjusted price: prc_adj = prc / cfacshr."""
    log.info("Adjusting prices for splits …")
    df["prc_adj"] = df["prc"].abs() / df["cfacshr"].replace(0, float("nan"))
    return df


def _compute_lagged_me(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lag market cap by one month within each permno.

    ``me_lag1`` is used as the portfolio weight in value-weighted returns
    and as a size characteristic in the feature panel.
    """
    log.info("Computing lagged market cap (me_lag1) …")
    df = df.sort_values(["permno", "date"])
    df["me_lag1"] = df.groupby("permno")["me"].shift(1)
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_crsp(raw_dir=RAW_DIR, output_path=CRSP_CLEAN_PATH) -> pd.DataFrame:
    """
    Run the full CRSP cleaning pipeline.

    Returns the cleaned DataFrame and writes it to ``output_path``.
    """
    monthly = _load_monthly(raw_dir)
    names = _load_names(raw_dir)
    delistings = _load_delistings(raw_dir)

    df = _apply_universe_filter(monthly, names)
    df = _merge_delistings(df, delistings)
    df = _compute_market_cap(df)
    df = _adjust_price(df)
    df = _compute_lagged_me(df)

    # Keep only the columns needed downstream
    keep_cols = [
        "permno", "date", "ret", "retx",
        "prc", "prc_adj", "shrout", "vol", "cfacshr",
        "me", "me_lag1",
        "shrcd", "exchcd",
    ]
    df = df[[c for c in keep_cols if c in df.columns]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    log.info("Wrote %s  (%d rows, %d stocks)", output_path.name, len(df), df["permno"].nunique())
    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    clean_crsp()
