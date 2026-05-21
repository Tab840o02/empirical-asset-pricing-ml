"""
src/features/feature_assembler.py
====================================
Orchestrates all feature modules, applies cross-sectional rank normalisation,
imputes missing values, attaches the target excess return, and writes
``data/processed/features_panel.parquet``.

Schema of output file
---------------------
permno      int64
date        datetime64[ns]   (month-end)
<94 characteristics>         float32 (rank-normalised to [-1, +1])
ret_exc     float32          excess return (next-month ret minus RF)

Cross-sectional rank normalisation (GKX §2.2)
----------------------------------------------
For each characteristic c at each month t:

    c̃_{i,t} = 2 * (rank(c_{i,t}) - 1) / (N_t - 1) - 1

Missing values are set to 0 after normalisation (= cross-sectional median).
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from src.config import (
    MERGED_PANEL_PATH,
    FEATURES_PANEL_PATH,
    PROCESSED_DIR,
    RAW_DIR,
    RANK_NORM_LOWER,
    RANK_NORM_UPPER,
)
from src.features import (
    momentum_features,
    value_features,
    profitability_features,
    investment_features,
    trading_friction_features,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rank normalisation
# ---------------------------------------------------------------------------

def _rank_normalise_month(s: pd.Series) -> pd.Series:
    """
    Rank-normalise a cross-section to [–1, +1] as in GKX §2.2.
    Missing values remain NaN (they are imputed to 0 separately).
    """
    r = s.rank(method="average", na_option="keep")
    N = r.notna().sum()
    if N < 2:
        return s * np.nan
    return 2.0 * (r - 1.0) / (N - 1.0) - 1.0


def _rank_normalise(features: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """Apply monthly rank normalisation to all characteristic columns."""
    log.info("Applying cross-sectional rank normalisation (%d characteristics) ...",
             len(feat_cols))
    # Operate on a copy; groupby.transform is vectorised enough for this size
    for col in feat_cols:
        features[col] = (
            features.groupby("date")[col]
            .transform(_rank_normalise_month)
            .astype("float32")
        )
    return features


# ---------------------------------------------------------------------------
# Target variable
# ---------------------------------------------------------------------------

def _attach_excess_return(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Attach next-month excess return as the prediction target.

    ret_exc_{t} = ret_{t+1} - RF_{t+1}

    RF comes from the FF factors file (column rf, monthly frequency).
    """
    ff_path = RAW_DIR / "ff_factors.parquet"
    # date is the index in the FF factors file; RF column is uppercase
    ff = pd.read_parquet(ff_path, columns=["RF"]).reset_index()
    ff["date"] = pd.to_datetime(ff["date"]).dt.to_period("M").dt.to_timestamp("M")
    ff = ff.rename(columns={"RF": "rf"})

    # Next-month return (shift -1 within permno, sorted by date)
    panel = panel.sort_values(["permno", "date"])
    panel["ret_next"] = panel.groupby("permno")["ret"].shift(-1)

    # Join RF for next month
    panel["date_next"] = panel["date"].dt.to_period("M").dt.to_timestamp("M") + \
                         pd.DateOffset(months=1)
    panel["date_next"] = panel["date_next"].dt.to_period("M").dt.to_timestamp("M")
    panel = panel.merge(ff, left_on="date_next", right_on="date",
                        how="left", suffixes=("", "_ff"))
    panel["ret_exc"] = (panel["ret_next"] - panel["rf"]).astype("float32")
    panel = panel.drop(columns=["ret_next", "date_next", "rf", "date_ff"],
                       errors="ignore")
    return panel


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_features_panel(output_path=FEATURES_PANEL_PATH) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    1. Load merged panel.
    2. Compute all feature groups.
    3. Merge feature DataFrames on (permno, date).
    4. Rank-normalise each characteristic cross-sectionally.
    5. Impute remaining NaN to 0.
    6. Attach next-month excess return target.
    7. Write features_panel.parquet and return the DataFrame.
    """
    t0 = time.time()

    # ------------------------------------------------------------------
    # 1. Load merged panel
    # ------------------------------------------------------------------
    log.info("Loading merged panel ...")
    panel = pd.read_parquet(MERGED_PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["permno", "date"]).reset_index(drop=True)
    log.info("  %s rows, %d permnos, %d months",
             f"{len(panel):,}", panel["permno"].nunique(), panel["date"].nunique())

    # Normalise date to month-end for joining
    panel["date"] = panel["date"].dt.to_period("M").dt.to_timestamp("M")

    # ------------------------------------------------------------------
    # 2. Compute feature groups
    # ------------------------------------------------------------------
    log.info("=== Computing momentum features ===")
    mom = momentum_features.compute(panel)

    log.info("=== Computing value features ===")
    val = value_features.compute(panel)

    log.info("=== Computing profitability features ===")
    prof = profitability_features.compute(panel)

    log.info("=== Computing investment features ===")
    inv = investment_features.compute(panel)

    log.info("=== Computing trading friction features (daily data) ===")
    fric = trading_friction_features.compute(panel)

    # ------------------------------------------------------------------
    # 3. Merge all feature groups
    # ------------------------------------------------------------------
    log.info("Merging feature groups ...")
    base = panel[["permno", "date", "ret"]].copy()

    for df in [mom, val, prof, inv]:
        df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp("M")
        base = base.merge(df, on=["permno", "date"], how="left")

    # Friction features already have month-end dates
    fric["date"] = pd.to_datetime(fric["date"]).dt.to_period("M").dt.to_timestamp("M")
    base = base.merge(fric, on=["permno", "date"], how="left")

    # ------------------------------------------------------------------
    # 4. Identify all characteristic columns (everything except identifiers
    #    and raw ret)
    # ------------------------------------------------------------------
    id_cols = {"permno", "date", "ret"}
    feat_cols = [c for c in base.columns if c not in id_cols]
    log.info("Total characteristics before normalisation: %d", len(feat_cols))

    # ------------------------------------------------------------------
    # 5. Rank normalisation (cross-sectional, per month)
    # ------------------------------------------------------------------
    base = _rank_normalise(base, feat_cols)

    # ------------------------------------------------------------------
    # 6. Impute missing → 0  (= cross-sectional median after normalisation)
    # ------------------------------------------------------------------
    log.info("Imputing missing values → 0 ...")
    base[feat_cols] = base[feat_cols].fillna(0.0)

    # ------------------------------------------------------------------
    # 7. Attach next-month excess return
    # ------------------------------------------------------------------
    log.info("Attaching excess return target ...")
    base = _attach_excess_return(base)

    # Drop rows with no return target (last month in sample has no next return)
    base = base.dropna(subset=["ret_exc"])

    # Drop raw ret (it's an input, not a target; ret_exc is the target)
    base = base.drop(columns=["ret"], errors="ignore")

    # ------------------------------------------------------------------
    # 8. Write output
    # ------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base.to_parquet(output_path, index=False)

    elapsed = time.time() - t0
    log.info(
        "features_panel.parquet written: %s rows, %d permnos, %d months, "
        "%d characteristics  (%.1f min)",
        f"{len(base):,}",
        base["permno"].nunique(),
        base["date"].nunique(),
        len(feat_cols),
        elapsed / 60,
    )
    return base


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    build_features_panel()
