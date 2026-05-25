"""
tests/test_portfolio_coverage.py
==================================
Strict regression tests for the portfolio date-normalization fix and
full 360-month coverage.

Background
----------
A date-format mismatch between crsp_clean.parquet (which stores raw
last-trading-day-of-month dates such as 1987-01-30 or 1987-05-29) and
predictions.parquet (which uses calendar month-end dates such as 1987-01-31
or 1987-05-31) caused the portfolio merge to silently drop ~108 of 360 test
months before the fix was applied.

The fix (pd.offsets.MonthEnd(0) in src/evaluation/portfolio.py) normalizes
CRSP dates to calendar month-end before the merge, restoring full coverage.

These tests:
  1. Prove that predictions.parquet has exactly 360 months per model.
  2. Prove that crsp_clean.parquet contains raw non-month-end trading dates
     (documents the pre-fix reality that made the bug possible).
  3. Prove that merging WITHOUT MonthEnd normalization drops months
     (regression test: documents the bug and proves the fix is necessary).
  4. Prove that build_portfolios() now produces exactly 360 months per model.
  5. Prove that portfolio decile returns (P1, P10) contain no NaN values.

Run with:
    pytest tests/test_portfolio_coverage.py -v
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config import PROCESSED_DIR, PREDICTIONS_PATH
from src.evaluation.portfolio import build_portfolios, ls_returns

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPECTED_MONTHS = 360
TEST_START = pd.Timestamp("1987-01-31")
TEST_END   = pd.Timestamp("2016-12-31")

# Models that must have full coverage (NN5 was not run, so excluded)
TRAINED_MODELS = {
    "ols3", "ols_all", "pcr", "pls", "enet", "glm",
    "rf", "gbrt", "nn1", "nn2", "nn3", "nn4",
}

# One model used for live portfolio construction tests (cheapest to compute)
PROBE_MODEL = "ols3"

# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def predictions() -> pd.DataFrame:
    if not PREDICTIONS_PATH.exists():
        pytest.skip(f"predictions.parquet not found at {PREDICTIONS_PATH}")
    return pd.read_parquet(PREDICTIONS_PATH)


@pytest.fixture(scope="module")
def crsp_me_raw() -> pd.DataFrame:
    """Raw crsp_clean lagged market cap — dates NOT yet normalized to month-end."""
    path = PROCESSED_DIR / "crsp_clean.parquet"
    if not path.exists():
        pytest.skip(f"crsp_clean.parquet not found at {path}")
    return pd.read_parquet(path, columns=["permno", "date", "me_lag1"])


@pytest.fixture(scope="module")
def features_ret() -> pd.DataFrame:
    """Excess returns column from features_panel.parquet."""
    path = PROCESSED_DIR / "features_panel.parquet"
    if not path.exists():
        pytest.skip(f"features_panel.parquet not found at {path}")
    return pd.read_parquet(path, columns=["permno", "date", "ret_exc"])


@pytest.fixture(scope="module")
def probe_preds(predictions: pd.DataFrame) -> pd.DataFrame:
    """Single-model predictions for PROBE_MODEL, merged with ret_exc."""
    return predictions[predictions["model"] == PROBE_MODEL].copy()


# ---------------------------------------------------------------------------
# 1. Predictions coverage — fast file-level assertions
# ---------------------------------------------------------------------------

class TestPredictionsCoverage:
    """Verify predictions.parquet has correct model set and 360-month coverage."""

    def test_all_trained_models_present(self, predictions: pd.DataFrame) -> None:
        """Every model in TRAINED_MODELS has predictions in predictions.parquet."""
        found = set(predictions["model"].unique())
        missing = TRAINED_MODELS - found
        assert not missing, (
            f"Models missing from predictions.parquet: {missing}"
        )

    def test_exactly_360_months_per_model(self, predictions: pd.DataFrame) -> None:
        """Each trained model has predictions for exactly 360 test months."""
        month_counts = predictions.groupby("model")["date"].nunique()
        wrong = month_counts[month_counts != EXPECTED_MONTHS]
        assert wrong.empty, (
            f"Models with wrong month count (expected {EXPECTED_MONTHS}):\n{wrong}"
        )

    def test_date_range_is_1987_01_to_2016_12(self, predictions: pd.DataFrame) -> None:
        """Prediction dates span exactly 1987-01-31 through 2016-12-31."""
        actual_min = predictions["date"].min()
        actual_max = predictions["date"].max()
        assert actual_min == TEST_START, (
            f"Earliest prediction date is {actual_min}, expected {TEST_START}"
        )
        assert actual_max == TEST_END, (
            f"Latest prediction date is {actual_max}, expected {TEST_END}"
        )

    def test_prediction_dates_are_month_end(self, predictions: pd.DataFrame) -> None:
        """All prediction dates are calendar month-end timestamps."""
        unique_dates = predictions["date"].drop_duplicates()
        normalized   = unique_dates + pd.offsets.MonthEnd(0)
        non_month_end = unique_dates[unique_dates != normalized]
        assert non_month_end.empty, (
            f"Found {len(non_month_end)} non-month-end prediction dates: "
            f"{non_month_end.head(5).tolist()}"
        )

    def test_no_nan_pred_ret(self, predictions: pd.DataFrame) -> None:
        """pred_ret contains no NaN values."""
        n_nan = predictions["pred_ret"].isna().sum()
        assert n_nan == 0, f"Found {n_nan} NaN values in pred_ret"


# ---------------------------------------------------------------------------
# 2. CRSP raw-date documentation — confirms pre-fix reality
# ---------------------------------------------------------------------------

class TestCrspRawDates:
    """
    Document that crsp_clean.parquet stores raw last-trading-day-of-month
    dates, NOT calendar month-end dates.  These tests confirm the pre-fix
    state that made the 108-month portfolio drop possible.
    """

    def test_crsp_dates_include_non_month_end_entries(
        self, crsp_me_raw: pd.DataFrame
    ) -> None:
        """
        At least some CRSP dates in the test window are NOT calendar
        month-end (e.g., 1987-01-30, 1987-05-29).  This is the raw
        last-trading-day date stored by CRSP.
        """
        test_window = crsp_me_raw[
            (crsp_me_raw["date"] >= TEST_START)
            & (crsp_me_raw["date"] <= TEST_END)
        ]
        unique_dates = test_window["date"].drop_duplicates()
        normalized   = unique_dates + pd.offsets.MonthEnd(0)
        n_non_month_end = (unique_dates != normalized).sum()
        assert n_non_month_end > 0, (
            "Expected crsp_clean.parquet to contain raw trading dates "
            "(not all month-end), but all dates are already month-end. "
            "If CRSP dates were pre-normalized in crsp_cleaner.py, "
            "re-evaluate whether the MonthEnd fix in portfolio.py is still needed."
        )

    def test_crsp_has_known_trading_date_examples(
        self, crsp_me_raw: pd.DataFrame
    ) -> None:
        """
        Spot-check: 1987-01-30 (not 01-31) and 1987-05-29 (not 05-31) exist
        in crsp_clean.parquet, confirming these are business-day dates.
        """
        known_trading_dates = [
            pd.Timestamp("1987-01-30"),
            pd.Timestamp("1987-05-29"),
        ]
        present = crsp_me_raw["date"].unique()
        for td in known_trading_dates:
            assert td in present, (
                f"Expected raw trading date {td.date()} in crsp_clean.parquet "
                f"but it was not found.  "
                f"(The corresponding calendar month-end is {(td + pd.offsets.MonthEnd(0)).date()}.)"
            )


# ---------------------------------------------------------------------------
# 3. Regression: without MonthEnd fix, months are dropped
# ---------------------------------------------------------------------------

class TestMonthEndRegressionBug:
    """
    Prove that the pre-fix behavior (no MonthEnd normalization) would drop
    months from the portfolio.  This is an explicit regression test: if the
    normalization step is removed from portfolio.py, these tests will fail.
    """

    def test_merge_without_fix_drops_months(
        self,
        probe_preds: pd.DataFrame,
        crsp_me_raw: pd.DataFrame,
        features_ret: pd.DataFrame,
    ) -> None:
        """
        Merging 1987 predictions against raw (un-normalized) CRSP dates
        yields < 12 months in the portfolio — because months where the last
        trading day ≠ calendar month-end produce all-NaN me_lag1 rows which
        are then dropped by build_portfolios.
        """
        # Restrict to 1987 for speed
        preds_1987 = probe_preds[probe_preds["date"].dt.year == 1987].copy()
        preds_1987 = preds_1987.merge(
            features_ret[["permno", "date", "ret_exc"]], on=["permno", "date"], how="left"
        )
        crsp_1987 = crsp_me_raw[crsp_me_raw["date"].dt.year == 1987].copy()

        # Simulate pre-fix: do NOT apply MonthEnd(0) to CRSP dates
        merged_no_fix = preds_1987.merge(
            crsp_1987[["permno", "date", "me_lag1"]],
            on=["permno", "date"],
            how="left",
        )
        # Rows with me_lag1 available (i.e., where CRSP date matched prediction date)
        matched = merged_no_fix[merged_no_fix["me_lag1"].notna()]
        months_with_weights = matched["date"].nunique()

        assert months_with_weights < 12, (
            f"Expected < 12 months of weight coverage in 1987 without MonthEnd fix, "
            f"got {months_with_weights}.  "
            f"If this assertion fails, crsp_clean.parquet may already store month-end dates "
            f"(check crsp_cleaner.py) and the fix in portfolio.py may be redundant."
        )

    def test_merge_with_fix_restores_full_coverage(
        self,
        probe_preds: pd.DataFrame,
        crsp_me_raw: pd.DataFrame,
        features_ret: pd.DataFrame,
    ) -> None:
        """
        After applying MonthEnd(0) normalization to CRSP dates (the fix),
        all 12 months of 1987 have full weight coverage.
        """
        preds_1987 = probe_preds[probe_preds["date"].dt.year == 1987].copy()
        preds_1987 = preds_1987.merge(
            features_ret[["permno", "date", "ret_exc"]], on=["permno", "date"], how="left"
        )
        crsp_1987 = crsp_me_raw[crsp_me_raw["date"].dt.year == 1987].copy()

        # Apply the fix: normalize CRSP dates to month-end
        crsp_1987["date"] = crsp_1987["date"] + pd.offsets.MonthEnd(0)

        merged_with_fix = preds_1987.merge(
            crsp_1987[["permno", "date", "me_lag1"]],
            on=["permno", "date"],
            how="left",
        )
        matched = merged_with_fix[merged_with_fix["me_lag1"].notna()]
        months_with_weights = matched["date"].nunique()

        assert months_with_weights == 12, (
            f"Expected 12 months of weight coverage in 1987 with MonthEnd fix, "
            f"got {months_with_weights}."
        )


# ---------------------------------------------------------------------------
# 4. Live portfolio construction — 360-month coverage
# ---------------------------------------------------------------------------

class TestPortfolioBuildCoverage:
    """
    Run build_portfolios() and verify full coverage and data quality.
    Restricted to PROBE_MODEL × full test window for reasonable test runtime.
    """

    @pytest.fixture(scope="class")
    def built_portfolio(
        self,
        probe_preds: pd.DataFrame,
        crsp_me_raw: pd.DataFrame,
        features_ret: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build the full decile portfolio for PROBE_MODEL."""
        preds_with_ret = probe_preds.merge(
            features_ret[["permno", "date", "ret_exc"]],
            on=["permno", "date"],
            how="left",
        )
        return build_portfolios(preds_with_ret, crsp_me_raw)

    def test_portfolio_has_360_months(self, built_portfolio: pd.DataFrame) -> None:
        """build_portfolios() produces exactly 360 monthly observations."""
        n_months = built_portfolio["date"].nunique()
        assert n_months == EXPECTED_MONTHS, (
            f"Portfolio has {n_months} months, expected {EXPECTED_MONTHS}. "
            f"The 108-month drop bug may have regressed."
        )

    def test_portfolio_date_range(self, built_portfolio: pd.DataFrame) -> None:
        """Portfolio date range is exactly 1987-01-31 through 2016-12-31."""
        actual_min = built_portfolio["date"].min()
        actual_max = built_portfolio["date"].max()
        assert actual_min == TEST_START, (
            f"Earliest portfolio month is {actual_min}, expected {TEST_START}"
        )
        assert actual_max == TEST_END, (
            f"Latest portfolio month is {actual_max}, expected {TEST_END}"
        )

    def test_portfolio_has_10_deciles_every_month(
        self, built_portfolio: pd.DataFrame
    ) -> None:
        """Every month has all 10 deciles populated."""
        decile_counts = built_portfolio.groupby("date")["decile"].nunique()
        incomplete = decile_counts[decile_counts != 10]
        assert incomplete.empty, (
            f"Found {len(incomplete)} months with fewer than 10 deciles:\n"
            f"{incomplete.head(10)}"
        )

    def test_no_nan_port_ret_decile_1(self, built_portfolio: pd.DataFrame) -> None:
        """Bottom decile (P1 short leg) has no NaN port_ret values."""
        p1 = built_portfolio[built_portfolio["decile"] == 1]
        n_nan = p1["port_ret"].isna().sum()
        assert n_nan == 0, (
            f"Found {n_nan} NaN values in port_ret for P1 (short leg)"
        )

    def test_no_nan_port_ret_decile_10(self, built_portfolio: pd.DataFrame) -> None:
        """Top decile (P10 long leg) has no NaN port_ret values."""
        p10 = built_portfolio[built_portfolio["decile"] == 10]
        n_nan = p10["port_ret"].isna().sum()
        assert n_nan == 0, (
            f"Found {n_nan} NaN values in port_ret for P10 (long leg)"
        )

    def test_ls_returns_have_360_months(self, built_portfolio: pd.DataFrame) -> None:
        """ls_returns() produces exactly 360 monthly L/S observations."""
        ls = ls_returns(built_portfolio)
        n_months = len(ls)
        assert n_months == EXPECTED_MONTHS, (
            f"ls_returns() has {n_months} rows, expected {EXPECTED_MONTHS}"
        )

    def test_no_nan_ls_ret(self, built_portfolio: pd.DataFrame) -> None:
        """L/S return series contains no NaN values."""
        ls = ls_returns(built_portfolio)
        n_nan = ls["ls_ret"].isna().sum()
        assert n_nan == 0, (
            f"Found {n_nan} NaN values in ls_ret"
        )


# ---------------------------------------------------------------------------
# 5. Pre-saved portfolio metrics cross-check (fast)
# ---------------------------------------------------------------------------

class TestSavedPortfolioMetrics:
    """
    Cross-check the pre-saved portfolio performance CSV against the
    expected 360-month coverage for all models.  Fast — no recomputation.
    """

    @pytest.fixture(scope="class")
    def portfolio_perf(self) -> pd.DataFrame:
        path = PROCESSED_DIR / "eval_portfolio_perf_latest.csv"
        if not path.exists():
            pytest.skip(f"eval_portfolio_perf_latest.csv not found at {path}")
        return pd.read_csv(path)

    def test_all_models_have_360_portfolio_months(
        self, portfolio_perf: pd.DataFrame
    ) -> None:
        """
        eval_portfolio_perf_latest.csv reports n_months=360 for every
        trained model — confirming the date-normalization fix was applied
        when these metrics were computed.
        """
        wrong = portfolio_perf[portfolio_perf["n_months"] != EXPECTED_MONTHS]
        assert wrong.empty, (
            f"Models with n_months ≠ {EXPECTED_MONTHS} in saved portfolio CSV:\n"
            f"{wrong[['model', 'n_months']]}"
        )

    def test_all_trained_models_in_portfolio_csv(
        self, portfolio_perf: pd.DataFrame
    ) -> None:
        """All trained models appear in the saved portfolio performance CSV."""
        found   = set(portfolio_perf["model"].unique())
        missing = TRAINED_MODELS - found
        assert not missing, (
            f"Models missing from eval_portfolio_perf_latest.csv: {missing}"
        )
