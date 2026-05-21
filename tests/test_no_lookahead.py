"""
tests/test_no_lookahead.py
===========================
Guard tests for look-ahead bias in the CRSP–Compustat merge.

These tests enforce the single most important invariant in the codebase:
Compustat accounting data for a fiscal year ending in month M may NOT
appear as a predictor in the feature panel before month M + 6 (annual)
or M + 3 (quarterly).

Run with:
    pytest tests/test_no_lookahead.py -v
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.config import (
    ANNUAL_LAG_MONTHS,
    PROCESSED_DIR,
    QUARTERLY_LAG_MONTHS,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _months_delta(later: pd.Timestamp, earlier: pd.Timestamp) -> float:
    """Return the approximate number of calendar months between two timestamps."""
    return (later.year - earlier.year) * 12 + (later.month - earlier.month)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def merged_panel() -> pd.DataFrame:
    """
    Load the merged panel from data/processed/merged_panel.parquet.
    Skip all tests in this module if the file does not yet exist (i.e.
    Phase 2 has not been run yet).
    """
    path = PROCESSED_DIR / "merged_panel.parquet"
    if not path.exists():
        pytest.skip(
            "merged_panel.parquet not found — run Phase 2 (ccm_merger.py) first."
        )
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def compustat_clean_annual() -> pd.DataFrame:
    """Load cleaned Compustat annual from data/processed/."""
    path = PROCESSED_DIR / "compustat_annual_clean.parquet"
    if not path.exists():
        pytest.skip("compustat_annual_clean.parquet not found — run Phase 2 first.")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def compustat_clean_quarterly() -> pd.DataFrame:
    """Load cleaned Compustat quarterly from data/processed/."""
    path = PROCESSED_DIR / "compustat_quarterly_clean.parquet"
    if not path.exists():
        pytest.skip("compustat_quarterly_clean.parquet not found — run Phase 2 first.")
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Test: public_date construction (annual)
# ---------------------------------------------------------------------------

class TestAnnualPublicDate:
    """Verify that public_date = datadate + ANNUAL_LAG_MONTHS for annual data."""

    def test_public_date_column_exists(self, compustat_clean_annual):
        assert "public_date" in compustat_clean_annual.columns, (
            "compustat_annual_clean.parquet is missing the 'public_date' column. "
            "Check compustat_cleaner.py."
        )

    def test_public_date_is_at_least_lag_months_after_datadate(self, compustat_clean_annual):
        """
        For every row, public_date must be >= datadate + ANNUAL_LAG_MONTHS months.
        """
        df = compustat_clean_annual[["datadate", "public_date"]].dropna()
        actual_lag = df.apply(
            lambda r: _months_delta(r["public_date"], r["datadate"]), axis=1
        )
        violations = (actual_lag < ANNUAL_LAG_MONTHS).sum()
        assert violations == 0, (
            f"{violations} rows in compustat_annual_clean have public_date "
            f"less than {ANNUAL_LAG_MONTHS} months after datadate.  "
            "This would introduce look-ahead bias."
        )

    def test_december_fiscal_year_not_available_before_june_next_year(
        self, compustat_clean_annual
    ):
        """
        The classic look-ahead case: a December fiscal year-end (e.g. 2005-12-31)
        must not become available before June of the following year (2006-06-30).
        With a 6-month lag, June is the first acceptable month — months 1–5 of
        the next calendar year are violations.
        """
        dec_filings = compustat_clean_annual[
            compustat_clean_annual["datadate"].dt.month == 12
        ][["datadate", "public_date"]].dropna()

        if dec_filings.empty:
            pytest.skip("No December fiscal year-end rows found.")

        # public_date must be in the FOLLOWING year, on or after July
        violations = dec_filings[
            (dec_filings["public_date"].dt.year == dec_filings["datadate"].dt.year)
            | (
                (dec_filings["public_date"].dt.year == dec_filings["datadate"].dt.year + 1)
                & (dec_filings["public_date"].dt.month < 6)
            )
        ]
        assert len(violations) == 0, (
            f"{len(violations)} December fiscal year-end rows have a public_date "
            f"before June of the following year.  Sample:\n"
            f"{violations.head(5).to_string(index=False)}"
        )


# ---------------------------------------------------------------------------
# Test: public_date construction (quarterly)
# ---------------------------------------------------------------------------

class TestQuarterlyPublicDate:
    """Verify that public_date_q = datadate_q + QUARTERLY_LAG_MONTHS for quarterly data."""

    def test_quarterly_public_date_column_exists(self, compustat_clean_quarterly):
        assert "public_date" in compustat_clean_quarterly.columns, (
            "compustat_quarterly_clean.parquet is missing the 'public_date' column."
        )

    def test_quarterly_public_date_lag(self, compustat_clean_quarterly):
        df = compustat_clean_quarterly[["datadate", "public_date"]].dropna()
        actual_lag = df.apply(
            lambda r: _months_delta(r["public_date"], r["datadate"]), axis=1
        )
        violations = (actual_lag < QUARTERLY_LAG_MONTHS).sum()
        assert violations == 0, (
            f"{violations} rows in compustat_quarterly_clean have public_date "
            f"less than {QUARTERLY_LAG_MONTHS} months after datadate."
        )


# ---------------------------------------------------------------------------
# Test: merged panel — accounting data not from the future
# ---------------------------------------------------------------------------

class TestMergedPanelNoLookahead:
    """
    End-to-end check on the merged panel that the accounting data attached
    to a given (permno, date) row was actually available on that date.
    """

    def test_merged_panel_has_required_columns(self, merged_panel):
        required = {"permno", "date", "datadate", "public_date"}
        missing = required - set(merged_panel.columns)
        assert not missing, (
            f"merged_panel.parquet is missing columns: {missing}.  "
            "Check ccm_merger.py."
        )

    def test_accounting_data_not_from_future(self, merged_panel):
        """
        For every row in the merged panel, the accounting filing's public_date
        must be on or before the panel date (CRSP month).
        Violations mean the merger joined a filing that had not yet been
        released — a look-ahead bias error.
        """
        df = merged_panel[["permno", "date", "public_date"]].dropna()
        violations = df[df["public_date"] > df["date"]]
        assert len(violations) == 0, (
            f"{len(violations)} rows in merged_panel have public_date > panel date. "
            f"Sample:\n{violations.head(5).to_string(index=False)}"
        )

    def test_no_same_year_december_filings_in_january(self, merged_panel):
        """
        A December 2005 filing (datadate = 2005-12-31) must not appear in any
        panel row dated before 2006-06-01.  This is the canonical look-ahead
        trap for calendar fiscal-year companies.
        """
        if "datadate" not in merged_panel.columns:
            pytest.skip("datadate column not in merged_panel.")

        df = merged_panel[merged_panel["datadate"].dt.month == 12].copy()
        df = df.dropna(subset=["datadate", "date"])

        # Flag rows where the panel date is in the same year as the fiscal year end
        violations = df[df["date"].dt.year == df["datadate"].dt.year]
        assert len(violations) == 0, (
            f"{len(violations)} rows have a December fiscal year-end appearing "
            f"in the same calendar year as the panel month.  "
            f"This is a look-ahead bias violation.\n"
            f"{violations[['permno', 'date', 'datadate']].head(5).to_string(index=False)}"
        )


# ---------------------------------------------------------------------------
# Test: link table validity
# ---------------------------------------------------------------------------

class TestLinkTableValidity:
    """Sanity checks on the CRSP–Compustat link table."""

    @pytest.fixture(scope="class")
    def link_table(self) -> pd.DataFrame:
        path = PROCESSED_DIR / "crsp_compustat_link.parquet"
        if not path.exists():
            pytest.skip("crsp_compustat_link.parquet not found — run Phase 1 linker first.")
        return pd.read_parquet(path)

    def test_link_start_before_link_end(self, link_table):
        bad = link_table[link_table["link_start"] > link_table["link_end"]]
        assert len(bad) == 0, f"{len(bad)} link rows have link_start > link_end."

    def test_no_null_permno_or_gvkey(self, link_table):
        assert link_table["permno"].isna().sum() == 0, "NULL permno in link table."
        assert link_table["gvkey"].isna().sum() == 0, "NULL gvkey in link table."

    def test_match_rate_above_90_percent(self, link_table):
        """
        The CUSIP+ticker approach should yield > 90 % coverage.
        The professor's experience suggests ~98 %; flag anything below 90 %.
        """
        # Need the CRSP names to compute the denominator
        names_path = PROCESSED_DIR.parent / "raw" / "crsp_names.parquet"
        if not names_path.exists():
            pytest.skip("crsp_names.parquet not found in data/raw/.")
        crsp_names = pd.read_parquet(names_path)
        total = crsp_names["permno"].nunique()
        matched = link_table["permno"].nunique()
        rate = matched / total
        assert rate >= 0.90, (
            f"Match rate is {rate:.1%} — below the 90 % threshold.  "
            "Investigate unmatched PERMNOs."
        )
