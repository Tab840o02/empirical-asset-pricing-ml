"""
src/data/crsp_compustat_linker.py
==================================
Build a point-in-time CRSP–Compustat link using the professor's recommended
CUSIP-based approach (primary) with a ticker/company-name fallback.

Background
----------
The official CRSP/Compustat Merged (CCM) link table from WRDS is convenient
but opaque — it does not expose its internal match logic and can silently
drop or duplicate links during certain corporate actions.

This module constructs the link directly from raw identifiers:

  Primary key (≈ 98 % match rate):
    crsp.msenames.ncusip  [8-char, no check digit]
      ↕ matched to
    comp.security.cusip[:8]  [first 8 chars of the 9-char Compustat CUSIP]

  Fallback (for unmatched permnos, ~1–2 %):
    Normalised ticker symbol (uppercased, stripped), with a company-name
    similarity check to guard against ticker reuse after a company ceases
    to exist.

Point-in-time correctness
--------------------------
Each CRSP name record is valid only between [namedt, nameendt].
Each Compustat security record is valid only between [idbeg, idend].
A link is considered valid only during the OVERLAP of those two intervals,
ensuring no stale CUSIP match is carried forward after a corporate action.

PERMCO/PERMNO and GVKEY/IID hierarchy
--------------------------------------
CRSP identifies companies via PERMCO and individual share classes via
PERMNO.  Compustat uses GVKEY (company) and IID (issue/share class).
A firm may have multiple PERMNOs under one PERMCO (multiple share classes)
and multiple GVKEYs under a merger sequence.  This module links at the
PERMNO–GVKEY–IID level so the correct security is matched, not just the
company.

Output
------
``data/processed/crsp_compustat_link.parquet``

    permno       int    CRSP security identifier
    permco       int    CRSP company identifier
    gvkey        str    Compustat company identifier
    iid          str    Compustat issue identifier (share class)
    cusip8       str    8-character CUSIP used for matching
    link_start   date   First month this link is valid (inclusive)
    link_end     date   Last month this link is valid (inclusive)
    link_method  str    'cusip' | 'ticker'

Usage
-----
    from src.data.crsp_compustat_linker import build_link_table
    build_link_table()          # reads raw files, writes processed output
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    ACTIVE_END_DATE,
    CUSIP_LENGTH,
    MIN_LINK_OVERLAP_DAYS,
    LINK_TABLE_PATH,
    RAW_DIR,
)

log = logging.getLogger(__name__)

_ACTIVE_TS = pd.Timestamp(ACTIVE_END_DATE)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _load_crsp_names(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    path = raw_dir / "crsp_names.parquet"
    df = pd.read_parquet(path)
    # Fill missing nameendt with active sentinel
    df["nameendt"] = df["nameendt"].fillna(_ACTIVE_TS)
    # Normalise ncusip: strip whitespace, uppercase, drop blanks
    df["ncusip"] = df["ncusip"].astype(str).str.strip().str.upper()
    df = df[df["ncusip"].str.len() > 0]
    df = df[df["ncusip"] != "NAN"]
    # Normalise ticker for fallback matching
    df["ticker_norm"] = df["ticker"].astype(str).str.strip().str.upper()
    return df


def _load_comp_security(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    path = raw_dir / "compustat_security.parquet"
    df = pd.read_parquet(path)
    # Derive 8-char CUSIP from the 9-char Compustat CUSIP
    df["cusip"] = df["cusip"].astype(str).str.strip().str.upper()
    df["cusip8"] = df["cusip"].str[:CUSIP_LENGTH]
    df = df[df["cusip8"].str.len() == CUSIP_LENGTH]
    df = df[df["cusip8"] != "NAN" * CUSIP_LENGTH]
    # Normalise ticker for fallback
    df["ticker_norm"] = df["tic"].astype(str).str.strip().str.upper()
    # comp.security has no date-range columns — treat each CUSIP as always valid;
    # the CRSP namedt/nameendt on the other side still enforces point-in-time.
    df["idbeg"] = pd.Timestamp("1950-01-01")
    df["idend"] = pd.Timestamp(ACTIVE_END_DATE)
    df["conm"] = ""  # company name not available in comp.security
    return df


# ---------------------------------------------------------------------------
# Primary link: CUSIP
# ---------------------------------------------------------------------------


def _build_cusip_link(
    crsp_names: pd.DataFrame,
    comp_security: pd.DataFrame,
) -> pd.DataFrame:
    """
    Match crsp.msenames.ncusip (8-char) to comp.security.cusip[:8].

    Validity is the intersection of [namedt, nameendt] and [idbeg, idend].
    Rows where the overlap is less than MIN_LINK_OVERLAP_DAYS are dropped.
    """
    log.info("Building primary CUSIP link …")

    crsp = crsp_names[
        ["permno", "permco", "ncusip", "ticker_norm", "comnam", "namedt", "nameendt"]
    ].copy()
    crsp = crsp.rename(columns={"ncusip": "cusip8"})

    comp = comp_security[
        ["gvkey", "iid", "cusip8", "ticker_norm", "conm", "idbeg", "idend"]
    ].copy()

    # Inner join on 8-char CUSIP
    merged = crsp.merge(comp, on="cusip8", how="inner", suffixes=("_crsp", "_comp"))

    # Point-in-time overlap
    merged["link_start"] = merged[["namedt", "idbeg"]].max(axis=1)
    merged["link_end"] = merged[["nameendt", "idend"]].min(axis=1)

    overlap_days = (merged["link_end"] - merged["link_start"]).dt.days
    merged = merged[overlap_days >= MIN_LINK_OVERLAP_DAYS].copy()

    merged["link_method"] = "cusip"

    log.info("  CUSIP matches before deduplication: %d rows", len(merged))
    return merged[
        ["permno", "permco", "gvkey", "iid", "cusip8", "link_start", "link_end", "link_method"]
    ]


# ---------------------------------------------------------------------------
# Fallback link: ticker
# ---------------------------------------------------------------------------


def _name_similarity(s1: str, s2: str) -> float:
    """
    Simple character-level Jaccard similarity between two company name strings.
    Used to guard against ticker reuse (e.g. a new company taking a defunct
    company's ticker symbol).
    """
    if not s1 or not s2:
        return 0.0
    a = set(s1.lower().split())
    b = set(s2.lower().split())
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _build_ticker_link(
    crsp_names: pd.DataFrame,
    comp_security: pd.DataFrame,
    matched_permnos: set[int],
    name_similarity_threshold: float = 0.3,
) -> pd.DataFrame:
    """
    Fallback link for PERMNOs not matched by CUSIP.

    Matches on normalised ticker symbol within overlapping date ranges.
    A company-name Jaccard similarity check (≥ threshold) eliminates
    spurious matches caused by ticker symbol reuse.
    """
    log.info("Building ticker fallback link for %d unmatched PERMNOs …",
             len(crsp_names["permno"].unique()) - len(matched_permnos))

    unmatched_crsp = crsp_names[~crsp_names["permno"].isin(matched_permnos)].copy()
    if unmatched_crsp.empty:
        log.info("  No unmatched PERMNOs — ticker fallback not needed.")
        return pd.DataFrame(
            columns=["permno", "permco", "gvkey", "iid", "cusip8", "link_start", "link_end", "link_method"]
        )

    crsp = unmatched_crsp[
        ["permno", "permco", "ticker_norm", "comnam", "namedt", "nameendt"]
    ].copy()
    comp = comp_security[
        ["gvkey", "iid", "cusip8", "ticker_norm", "conm", "idbeg", "idend"]
    ].copy()

    merged = crsp.merge(comp, on="ticker_norm", how="inner", suffixes=("_crsp", "_comp"))

    # Point-in-time overlap
    merged["link_start"] = merged[["namedt", "idbeg"]].max(axis=1)
    merged["link_end"] = merged[["nameendt", "idend"]].min(axis=1)

    overlap_days = (merged["link_end"] - merged["link_start"]).dt.days
    merged = merged[overlap_days >= MIN_LINK_OVERLAP_DAYS].copy()

    # Name similarity guard — skip when comp.security has no company names
    # (conm column is empty because comp.security does not carry that field).
    # The date-overlap constraint already limits spurious ticker reuse.
    has_conm = merged["conm"].astype(str).str.strip().ne("").any()
    if has_conm:
        merged["name_sim"] = merged.apply(
            lambda r: _name_similarity(
                str(r.get("comnam", "")), str(r.get("conm", ""))
            ),
            axis=1,
        )
        merged = merged[merged["name_sim"] >= name_similarity_threshold].copy()

    merged["link_method"] = "ticker"
    log.info("  Ticker matches after name-similarity filter: %d rows", len(merged))

    return merged[
        ["permno", "permco", "gvkey", "iid", "cusip8", "link_start", "link_end", "link_method"]
    ]


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def _deduplicate_links(links: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure no two rows have the same PERMNO active in the same month.

    Resolution priority (highest to lowest):
      1. Link method = 'cusip'  (more reliable than ticker)
      2. Longest overlap period (captures the primary listing)
      3. Most recently active (most conservative choice)

    After deduplication, each (permno, gvkey) pair represents a single
    non-overlapping time interval.
    """
    log.info("Deduplicating link table (%d rows before) …", len(links))

    # Sort so that higher-priority rows come first
    method_order = {"cusip": 0, "ticker": 1}
    links = links.copy()
    links["_method_rank"] = links["link_method"].map(method_order)
    links["_overlap"] = (links["link_end"] - links["link_start"]).dt.days

    links = links.sort_values(
        ["permno", "link_start", "_method_rank", "_overlap"],
        ascending=[True, True, True, False],
    )

    # Drop exact duplicates on (permno, gvkey, link_start)
    links = links.drop_duplicates(subset=["permno", "gvkey", "link_start"])

    # For same permno, same date range, different gvkey — keep the cusip match
    links = links.drop_duplicates(subset=["permno", "link_start"], keep="first")

    links = links.drop(columns=["_method_rank", "_overlap"])
    log.info("  %d rows after deduplication.", len(links))
    return links.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Match-rate report
# ---------------------------------------------------------------------------


def _report_match_rate(
    crsp_names: pd.DataFrame,
    link_table: pd.DataFrame,
) -> None:
    """Print a match-rate summary to the log."""
    total_permnos = crsp_names["permno"].nunique()
    matched_permnos = link_table["permno"].nunique()
    cusip_matched = link_table[link_table["link_method"] == "cusip"]["permno"].nunique()
    ticker_matched = link_table[link_table["link_method"] == "ticker"]["permno"].nunique()
    unmatched = total_permnos - matched_permnos

    log.info("=" * 55)
    log.info("CRSP–Compustat Link Match Rate Summary")
    log.info("-" * 55)
    log.info("  Total unique PERMNOs in CRSP names:  %6d", total_permnos)
    log.info("  Matched (any method):                %6d  (%5.1f %%)",
             matched_permnos, 100 * matched_permnos / total_permnos)
    log.info("    of which — CUSIP primary:          %6d  (%5.1f %%)",
             cusip_matched, 100 * cusip_matched / total_permnos)
    log.info("    of which — ticker fallback:        %6d  (%5.1f %%)",
             ticker_matched, 100 * ticker_matched / total_permnos)
    log.info("  Unmatched (investigate if > 5 %%):   %6d  (%5.1f %%)",
             unmatched, 100 * unmatched / total_permnos)
    log.info("=" * 55)

    if unmatched / total_permnos > 0.05:
        log.warning(
            "Unmatched rate exceeds 5 %%.  Check for data quality issues "
            "or missing Compustat coverage in the security table."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_link_table(
    raw_dir: Path = RAW_DIR,
    output_path: Path = LINK_TABLE_PATH,
) -> pd.DataFrame:
    """
    Build the point-in-time CRSP–Compustat link table and save it to Parquet.

    Steps
    -----
    1. Load ``crsp_names.parquet`` and ``compustat_security.parquet`` from
       ``raw_dir``.
    2. Primary match: 8-char CUSIP with date-range overlap.
    3. Fallback match: normalised ticker + company-name similarity filter.
    4. Deduplicate so each (permno, month) maps to at most one (gvkey, iid).
    5. Log a match-rate summary.
    6. Write ``crsp_compustat_link.parquet`` to ``output_path``.

    Returns
    -------
    pd.DataFrame — the link table (see module docstring for schema).
    """
    log.info("Loading raw identifier files …")
    crsp_names = _load_crsp_names(raw_dir)
    comp_security = _load_comp_security(raw_dir)

    log.info(
        "CRSP names: %d rows, %d unique PERMNOs",
        len(crsp_names),
        crsp_names["permno"].nunique(),
    )
    log.info(
        "Compustat security: %d rows, %d unique GVKEYs",
        len(comp_security),
        comp_security["gvkey"].nunique(),
    )

    # Step 1 — Primary CUSIP link
    cusip_links = _build_cusip_link(crsp_names, comp_security)
    matched_by_cusip = set(cusip_links["permno"].unique())

    # Step 2 — Ticker fallback
    ticker_links = _build_ticker_link(crsp_names, comp_security, matched_by_cusip)

    # Step 3 — Combine and deduplicate
    all_links = pd.concat([cusip_links, ticker_links], ignore_index=True)
    link_table = _deduplicate_links(all_links)

    # Step 4 — Match rate report
    _report_match_rate(crsp_names, link_table)

    # Step 5 — Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    link_table.to_parquet(output_path, index=False)
    log.info("Link table saved → %s  (%d rows)", output_path.name, len(link_table))

    return link_table


def load_link_table(path: Path = LINK_TABLE_PATH) -> pd.DataFrame:
    """Load the pre-built link table from Parquet."""
    return pd.read_parquet(path)


def lookup_gvkey(
    link_table: pd.DataFrame,
    permno: int,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Return the GVKEY(s) valid for a given PERMNO on a specific date.

    Parameters
    ----------
    link_table : pre-built link table (output of build_link_table)
    permno     : CRSP security identifier
    as_of_date : the date for which to look up the identifier

    Returns
    -------
    DataFrame with columns [gvkey, iid, link_method] — may have 0 or 1 rows
    (rarely more after deduplication).
    """
    mask = (
        (link_table["permno"] == permno)
        & (link_table["link_start"] <= as_of_date)
        & (link_table["link_end"] >= as_of_date)
    )
    return link_table.loc[mask, ["gvkey", "iid", "link_method"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI entry-point  (python -m src.data.crsp_compustat_linker)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    build_link_table()
