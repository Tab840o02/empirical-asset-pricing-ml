"""
scripts/audit_linkage_delta.py
================================
Quantify the discrepancy between this replication's custom CRSP–Compustat
link table and the canonical WRDS CCM link table (comp.ccmxpf_lnkhist).

Why this matters
----------------
GKX (2020) use the WRDS CCM link table as the standard PERMNO–GVKEY bridge.
This replication uses a custom CUSIP-primary / Jaccard name-similarity
fallback linker (src/data/crsp_compustat_linker.py).  Any difference in
the set of matched firms changes the universe of stocks in the test panel
and may introduce a systematic bias in all model results.

This script:
  1. Characterises the custom link table in full (coverage, method breakdown,
     multi-gvkey firms, test-window universe).
  2. If WRDS is accessible, queries comp.ccmxpf_lnkhist using the standard
     primary-link filter (linktype in LC/LU/LS, linkprim in P/C) and computes:
       - permno overlap / delta
       - gvkey overlap / delta
       - firm-months covered in the 1987–2016 test window per linker
  3. Falls back gracefully if WRDS is not accessible — reports custom-only
     stats with a clear note that the CCM comparison is unquantified.
  4. Writes data/processed/linkage_audit.json with all findings.

Run:
    python scripts/audit_linkage_delta.py

Environment:
    $env:WRDS_USER = "your_wrds_username"   # required for CCM comparison
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

LINK_PATH     = PROCESSED / "crsp_compustat_link.parquet"
FEATURES_PATH = PROCESSED / "features_panel.parquet"
CRSP_PATH     = PROCESSED / "crsp_clean.parquet"
OUTPUT_PATH   = PROCESSED / "linkage_audit.json"

TEST_START = "1987-01-01"
TEST_END   = "2016-12-31"

# Canonical CCM filter (standard in the literature; matches GKX universe)
CCM_LINKTYPES = ("LC", "LU", "LS")
CCM_LINKPRIMS = ("P", "C")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pct(a: int, b: int) -> float:
    """Return a/b as a percentage, or 0.0 if b == 0."""
    return round(a / b * 100, 2) if b else 0.0


def characterise_custom_link(link: pd.DataFrame, crsp: pd.DataFrame, fp: pd.DataFrame) -> dict:
    """Return a dict of statistics about the custom link table."""
    total_records    = len(link)
    method_counts    = link["link_method"].value_counts().to_dict()
    unique_permnos   = link["permno"].nunique()
    unique_gvkeys    = link["gvkey"].nunique()

    # Unique gvkey counts per permno
    gvkey_per_permno = link.groupby("permno")["gvkey"].nunique()
    permnos_1_gvkey  = int((gvkey_per_permno == 1).sum())
    permnos_multi    = int((gvkey_per_permno > 1).sum())

    # CRSP universe match rate
    crsp_permnos     = set(crsp["permno"].unique())
    link_permnos     = set(link["permno"].unique())
    crsp_matched     = len(crsp_permnos & link_permnos)
    crsp_unmatched   = len(crsp_permnos - link_permnos)

    # Test-window universe
    fp_test = fp[(fp["date"] >= TEST_START) & (fp["date"] <= TEST_END)]
    test_permnos  = set(fp_test["permno"].unique())
    test_pm_rows  = len(fp_test)                                # permno-months
    test_linked   = len(test_permnos & link_permnos)
    test_unlinked = len(test_permnos - link_permnos)

    return {
        "total_link_records":            total_records,
        "link_method_breakdown":         method_counts,
        "cusip_match_pct":               pct(method_counts.get("cusip", 0), total_records),
        "ticker_match_pct":              pct(method_counts.get("ticker", 0), total_records),
        "unique_permnos":                unique_permnos,
        "unique_gvkeys":                 unique_gvkeys,
        "permnos_with_1_gvkey":          permnos_1_gvkey,
        "permnos_with_multi_gvkey":      permnos_multi,
        "crsp_universe_permnos":         len(crsp_permnos),
        "crsp_permnos_matched":          crsp_matched,
        "crsp_permnos_unmatched":        crsp_unmatched,
        "crsp_match_rate_pct":           pct(crsp_matched, len(crsp_permnos)),
        "test_window":                   f"{TEST_START} to {TEST_END}",
        "test_window_permnos":           len(test_permnos),
        "test_window_permno_months":     test_pm_rows,
        "test_permnos_with_link":        test_linked,
        "test_permnos_without_link":     test_unlinked,
        "test_link_coverage_pct":        pct(test_linked, len(test_permnos)),
    }


def build_ccm_universe(db) -> tuple[set[int], set[str]] | None:
    """
    Query WRDS for canonical CCM primary links.  Returns (permno_set, gvkey_set)
    or None if the query fails.
    """
    query = f"""
        SELECT lpermno AS permno, gvkey
        FROM   comp.ccmxpf_lnkhist
        WHERE  linktype IN {CCM_LINKTYPES!r}
          AND  linkprim IN {CCM_LINKPRIMS!r}
    """
    try:
        df = db.raw_sql(query, date_cols=[])
        return set(df["permno"].dropna().astype(int)), set(df["gvkey"].dropna().astype(str))
    except Exception as exc:
        print(f"  [warn] CCM query failed: {exc}")
        return None


def build_ccm_test_universe(db) -> tuple[set[int], int] | None:
    """
    Query WRDS for CCM links active during the 1987–2016 test window.
    Returns (permno_set, total_ccm_link_records_in_window) or None.
    """
    query = f"""
        SELECT lpermno AS permno, gvkey, linkdt, linkenddt
        FROM   comp.ccmxpf_lnkhist
        WHERE  linktype IN {CCM_LINKTYPES!r}
          AND  linkprim IN {CCM_LINKPRIMS!r}
          AND  linkdt   <= '{TEST_END}'
          AND  (linkenddt >= '{TEST_START}' OR linkenddt IS NULL)
    """
    try:
        df = db.raw_sql(query, date_cols=["linkdt", "linkenddt"])
        permnos = set(df["permno"].dropna().astype(int))
        return permnos, len(df)
    except Exception as exc:
        print(f"  [warn] CCM test-window query failed: {exc}")
        return None


def compare_with_ccm(
    custom_all_permnos: set[int],
    custom_all_gvkeys: set[str],
    custom_test_permnos: set[int],
    ccm_all_permnos: set[int],
    ccm_all_gvkeys: set[str],
    ccm_test_permnos: set[int],
    ccm_test_records: int,
) -> dict:
    """Compute symmetric and asymmetric delta statistics between the two linkers."""
    # Full-history comparison
    only_custom    = custom_all_permnos - ccm_all_permnos
    only_ccm       = ccm_all_permnos - custom_all_permnos
    both           = custom_all_permnos & ccm_all_permnos
    union          = custom_all_permnos | ccm_all_permnos

    gvkey_only_custom = custom_all_gvkeys - ccm_all_gvkeys
    gvkey_only_ccm    = ccm_all_gvkeys - custom_all_gvkeys

    # Test-window comparison
    test_only_custom = custom_test_permnos - ccm_test_permnos
    test_only_ccm    = ccm_test_permnos - custom_test_permnos
    test_both        = custom_test_permnos & ccm_test_permnos

    jaccard = round(len(both) / len(union) * 100, 2) if union else 0.0

    return {
        "ccm_source":                    "comp.ccmxpf_lnkhist",
        "ccm_filter":                    f"linktype in {CCM_LINKTYPES}, linkprim in {CCM_LINKPRIMS}",
        # Full history
        "ccm_total_permnos":             len(ccm_all_permnos),
        "ccm_total_gvkeys":              len(ccm_all_gvkeys),
        "custom_total_permnos":          len(custom_all_permnos),
        "custom_total_gvkeys":           len(custom_all_gvkeys),
        "permnos_in_both":               len(both),
        "permnos_only_in_custom":        len(only_custom),
        "permnos_only_in_ccm":           len(only_ccm),
        "permno_jaccard_similarity_pct": jaccard,
        "gvkeys_only_in_custom":         len(gvkey_only_custom),
        "gvkeys_only_in_ccm":            len(gvkey_only_ccm),
        # Test window
        "test_window":                   f"{TEST_START} to {TEST_END}",
        "ccm_test_permnos":              len(ccm_test_permnos),
        "ccm_test_link_records":         ccm_test_records,
        "custom_test_permnos":           len(custom_test_permnos),
        "test_permnos_in_both":          len(test_both),
        "test_permnos_only_custom":      len(test_only_custom),
        "test_permnos_only_ccm":         len(test_only_ccm),
        "test_permno_match_rate_pct":    pct(len(test_both), len(ccm_test_permnos)),
        "interpretation": (
            f"The custom linker covers {pct(len(test_both), len(ccm_test_permnos)):.1f}% "
            f"of CCM test-window permnos. "
            f"{len(test_only_ccm)} permnos are in CCM but not the custom linker "
            f"(potential missed firms). "
            f"{len(test_only_custom)} permnos are in the custom linker but not CCM "
            f"(potential spurious matches or universe extension)."
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("audit_linkage_delta.py")
    print("=" * 65)

    # --- Load local artefacts ---
    print("\n[1] Loading custom link table and panel files ...")
    if not LINK_PATH.exists():
        print(f"  ERROR: {LINK_PATH} not found. Run Phase 2 first.", file=sys.stderr)
        sys.exit(1)

    link = pd.read_parquet(LINK_PATH)
    crsp = pd.read_parquet(CRSP_PATH, columns=["permno", "date"])
    fp   = pd.read_parquet(FEATURES_PATH, columns=["permno", "date"])

    # --- Characterise custom link ---
    print("[2] Characterising custom link table ...")
    custom_stats = characterise_custom_link(link, crsp, fp)

    custom_all_permnos  = set(link["permno"].unique())
    custom_all_gvkeys   = set(link["gvkey"].unique())
    fp_test             = fp[(fp["date"] >= TEST_START) & (fp["date"] <= TEST_END)]
    custom_test_permnos = set(fp_test["permno"].unique()) & custom_all_permnos

    # --- Print custom-only summary ---
    print(f"\n  Custom link table summary:")
    print(f"    Total link records:          {custom_stats['total_link_records']:>8,}")
    print(f"    Unique PERMNOs:              {custom_stats['unique_permnos']:>8,}")
    print(f"    Unique GVKEYs:               {custom_stats['unique_gvkeys']:>8,}")
    print(f"    CUSIP match rate:            {custom_stats['cusip_match_pct']:>7.2f}%")
    print(f"    Ticker match rate:           {custom_stats['ticker_match_pct']:>7.2f}%")
    print(f"    CRSP universe match rate:    {custom_stats['crsp_match_rate_pct']:>7.2f}%")
    print(f"    Test-window link coverage:   {custom_stats['test_link_coverage_pct']:>7.2f}%")
    print(f"    Test-window permno-months:   {custom_stats['test_window_permno_months']:>8,}")

    # --- Attempt WRDS comparison ---
    ccm_stats: dict | None = None
    wrds_user = os.environ.get("WRDS_USER")

    if not wrds_user:
        print("\n[3] WRDS_USER not set — skipping CCM comparison.")
        print("    Set $env:WRDS_USER and re-run for a full delta report.")
        ccm_available = False
    else:
        print(f"\n[3] Connecting to WRDS as '{wrds_user}' ...")
        try:
            import wrds  # type: ignore[import]
            db = wrds.Connection(wrds_username=wrds_user)

            print("  Querying comp.ccmxpf_lnkhist (full history) ...")
            result_all = build_ccm_universe(db)

            print("  Querying comp.ccmxpf_lnkhist (test window 1987–2016) ...")
            result_test = build_ccm_test_universe(db)

            db.close()

            if result_all and result_test:
                ccm_all_permnos, ccm_all_gvkeys = result_all
                ccm_test_permnos, ccm_test_records = result_test

                ccm_stats = compare_with_ccm(
                    custom_all_permnos, custom_all_gvkeys, custom_test_permnos,
                    ccm_all_permnos,    ccm_all_gvkeys,    ccm_test_permnos,
                    ccm_test_records,
                )
                ccm_available = True

                print(f"\n  CCM comparison (test window):")
                print(f"    CCM test-window PERMNOs:     {ccm_stats['ccm_test_permnos']:>8,}")
                print(f"    Custom test-window PERMNOs:  {ccm_stats['custom_test_permnos']:>8,}")
                print(f"    PERMNOs in both:             {ccm_stats['test_permnos_in_both']:>8,}")
                print(f"    PERMNOs only in custom:      {ccm_stats['test_permnos_only_custom']:>8,}")
                print(f"    PERMNOs only in CCM:         {ccm_stats['test_permnos_only_ccm']:>8,}")
                print(f"    Test match rate vs CCM:      {ccm_stats['test_permno_match_rate_pct']:>7.2f}%")
                print(f"\n  {ccm_stats['interpretation']}")
            else:
                ccm_available = False

        except ImportError:
            print("  [skip] wrds package not installed.")
            ccm_available = False
        except Exception as exc:
            print(f"  [error] WRDS connection failed: {exc}")
            ccm_available = False

    # --- Assemble output ---
    output = {
        "generated_at":       datetime.now(timezone.utc).isoformat(),
        "ccm_comparison_available": ccm_available,
        "custom_link_stats":  custom_stats,
        "ccm_delta":          ccm_stats,
        "unquantified_gap_note": (
            None if ccm_available else
            "CCM comparison was not run (WRDS_USER not set or connection failed). "
            "The impact of using a custom CUSIP linker instead of the canonical CCM "
            "link table on universe composition remains unquantified. "
            "Re-run with WRDS credentials to obtain the full delta report."
        ),
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[4] Written: {OUTPUT_PATH.relative_to(ROOT)}")
    print("=" * 65)


if __name__ == "__main__":
    main()
