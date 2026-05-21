"""
src/data/ccm_merger.py
=======================
Phase 2 — Merge CRSP and Compustat into a single monthly panel.

Uses the custom CUSIP-based link table built in Phase 1
(``data/processed/crsp_compustat_link.parquet``) rather than the WRDS
CCM link table, as recommended by professor Afonso.

Merge logic
-----------
For each ``(permno, crsp_month)`` observation in the clean CRSP panel:

  1. Find the applicable ``gvkey`` from the link table where
     ``link_start <= crsp_month <= link_end``.

  2. Attach the most recent Compustat *annual* filing where
     ``public_date <= crsp_month``.
     (The 6-month lag is already baked into ``public_date`` in
     ``compustat_annual_clean.parquet``.)

  3. Attach the most recent Compustat *quarterly* filing where
     ``public_date <= crsp_month``.

  4. If a stock has an annual filing but no quarterly filing for a given
     month, quarterly columns are left as NaN (handled in feature
     engineering).

Output
------
``data/processed/merged_panel.parquet``
  One row per ``(permno, date)`` with all raw CRSP and Compustat fields.

Run
---
    python -m src.data.ccm_merger
"""

from __future__ import annotations

import logging

import pandas as pd

from src.config import (
    COMPUSTAT_ANNUAL_CLEAN_PATH,
    COMPUSTAT_QUARTERLY_CLEAN_PATH,
    CRSP_CLEAN_PATH,
    LINK_TABLE_PATH,
    MERGED_PANEL_PATH,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_crsp() -> pd.DataFrame:
    log.info("Loading CRSP clean …")
    df = pd.read_parquet(CRSP_CLEAN_PATH)
    df["date"] = pd.to_datetime(df["date"])
    # Period key for joins
    df["ym"] = df["date"].dt.to_period("M")
    return df


def _load_link_table() -> pd.DataFrame:
    log.info("Loading link table …")
    df = pd.read_parquet(LINK_TABLE_PATH)
    df["link_start"] = pd.to_datetime(df["link_start"]).dt.to_period("M")
    df["link_end"] = pd.to_datetime(df["link_end"]).dt.to_period("M")
    return df[["permno", "gvkey", "iid", "link_start", "link_end", "link_method"]]


def _load_annual() -> pd.DataFrame:
    log.info("Loading Compustat annual clean …")
    df = pd.read_parquet(COMPUSTAT_ANNUAL_CLEAN_PATH)
    df["public_date"] = pd.to_datetime(df["public_date"]).dt.to_period("M")
    return df


def _load_quarterly() -> pd.DataFrame:
    log.info("Loading Compustat quarterly clean …")
    df = pd.read_parquet(COMPUSTAT_QUARTERLY_CLEAN_PATH)
    df["public_date"] = pd.to_datetime(df["public_date"]).dt.to_period("M")
    return df


# ---------------------------------------------------------------------------
# Step 1 — Attach gvkey via the link table
# ---------------------------------------------------------------------------

def _attach_gvkey(crsp: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """
    Point-in-time join: for each (permno, ym), find the active link.

    When a permno has multiple simultaneous links (rare merger edge case),
    prefer the CUSIP-matched link over the ticker-fallback link.
    """
    log.info("Attaching gvkey via link table …")

    # Merge on permno, then filter to valid date range
    merged = crsp.merge(links, on="permno", how="left")
    valid = (merged["ym"] >= merged["link_start"]) & (merged["ym"] <= merged["link_end"])
    merged = merged[valid].copy()

    # Resolve duplicates: cusip > ticker, then longer link (arbitrary tiebreak)
    method_order = {"cusip": 0, "ticker": 1}
    merged["_method_rank"] = merged["link_method"].map(method_order).fillna(9)
    merged["_link_len"] = (merged["link_end"] - merged["link_start"]).apply(
        lambda x: x.n if hasattr(x, "n") else 0
    )
    merged = (
        merged
        .sort_values(["permno", "ym", "_method_rank", "_link_len"],
                     ascending=[True, True, True, False])
        .drop_duplicates(subset=["permno", "ym"], keep="first")
        .drop(columns=["_method_rank", "_link_len"])
    )

    n_linked = merged["gvkey"].notna().sum()
    n_total = len(crsp)
    log.info(
        "  gvkey attached: %d / %d obs (%.1f%%)",
        n_linked, n_total, 100 * n_linked / n_total,
    )
    return merged


# ---------------------------------------------------------------------------
# Step 2 — Attach most-recent annual filing (as-of-date join)
# ---------------------------------------------------------------------------

def _attach_annual(panel: pd.DataFrame, annual: pd.DataFrame) -> pd.DataFrame:
    """
    For each (permno, ym) that has a gvkey, attach the most recent annual
    filing where ``public_date <= ym``.

    Uses a sort-merge / backward-merge approach (pandas merge_asof) to
    avoid exploding the DataFrame with a cross-join.
    """
    log.info("Attaching Compustat annual data …")

    # Work only with rows that have a gvkey
    has_gvkey = panel["gvkey"].notna()
    panel_linked = panel[has_gvkey].copy()

    # Prefix annual columns to avoid clashes with CRSP columns
    ann = annual.copy()
    ann_id_cols = {"gvkey", "datadate", "public_date", "fyear",
                   "indfmt", "datafmt", "popsrc", "consol"}
    rename_map = {
        c: f"a_{c}" for c in ann.columns if c not in ann_id_cols
    }
    ann = ann.rename(columns=rename_map)

    # merge_asof requires the 'on' key to be globally monotone (not just
    # within groups), so sort only by the join key, not by (gvkey, key).
    panel_linked = panel_linked.sort_values("ym")
    ann = ann.sort_values("public_date")

    merged = pd.merge_asof(
        panel_linked,
        ann,
        left_on="ym",
        right_on="public_date",
        by="gvkey",
        direction="backward",  # most recent filing where public_date <= ym
    )

    # Recombine with unlinked rows (no gvkey → all annual cols are NaN)
    unlinked = panel[~has_gvkey].copy()
    for col in merged.columns:
        if col not in unlinked.columns:
            unlinked[col] = pd.NA

    result = pd.concat([merged, unlinked], ignore_index=True)
    log.info("  Annual data attached.")
    return result


# ---------------------------------------------------------------------------
# Step 3 — Attach most-recent quarterly filing (as-of-date join)
# ---------------------------------------------------------------------------

def _attach_quarterly(panel: pd.DataFrame, quarterly: pd.DataFrame) -> pd.DataFrame:
    """
    Attach the most recent quarterly filing where ``q_public_date <= ym``.

    All quarterly columns are prefixed with ``q_`` (except ``gvkey``, the
    merge key).  This avoids name collisions with annual columns already in
    the panel (``datadate``, ``public_date``, ``indfmt``, ``datafmt``,
    ``popsrc``, ``consol``) which would otherwise produce ``_x``/``_y``
    suffixed duplicates and silently break the look-ahead-bias tests.
    """
    log.info("Attaching Compustat quarterly data …")

    has_gvkey = panel["gvkey"].notna()
    panel_linked = panel[has_gvkey].copy()

    qtr = quarterly.copy()
    # Only gvkey is kept as-is (it's the merge key via `by=`).  Every other
    # column — including datadate, public_date, fyearq, fqtr, indfmt, etc. —
    # is prefixed with "q_" to prevent conflicts with annual data in the panel.
    rename_map = {c: f"q_{c}" for c in qtr.columns if c != "gvkey"}
    qtr = qtr.rename(columns=rename_map)

    panel_linked = panel_linked.sort_values("ym")
    qtr = qtr.sort_values("q_public_date")

    merged = pd.merge_asof(
        panel_linked,
        qtr,
        left_on="ym",
        right_on="q_public_date",
        by="gvkey",
        direction="backward",
    )

    unlinked = panel[~has_gvkey].copy()
    for col in merged.columns:
        if col not in unlinked.columns:
            unlinked[col] = pd.NA

    result = pd.concat([merged, unlinked], ignore_index=True)
    log.info("  Quarterly data attached.")
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_merged_panel(output_path=MERGED_PANEL_PATH) -> pd.DataFrame:
    """
    Run the full CRSP–Compustat merge pipeline.

    Returns the merged panel DataFrame and writes it to ``output_path``.
    """
    crsp = _load_crsp()
    links = _load_link_table()
    annual = _load_annual()
    quarterly = _load_quarterly()

    panel = _attach_gvkey(crsp, links)
    panel = _attach_annual(panel, annual)
    panel = _attach_quarterly(panel, quarterly)

    # Drop the period helper column before saving
    panel = panel.drop(columns=["ym"], errors="ignore")

    # Convert any Period columns to datetime64 so Parquet stores them correctly
    # and the look-ahead tests can compare directly to the CRSP date column.
    for col in ["public_date", "q_public_date"]:
        if col in panel.columns and isinstance(panel[col].dtype, pd.PeriodDtype):
            panel[col] = panel[col].dt.to_timestamp()

    # Sort final panel
    panel = panel.sort_values(["permno", "date"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output_path, index=False)
    log.info(
        "Wrote %s  (%d rows, %d permno, %d months)",
        output_path.name,
        len(panel),
        panel["permno"].nunique(),
        panel["date"].nunique(),
    )
    return panel


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    build_merged_panel()
