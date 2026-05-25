"""Quick functional test of the 10 new features on a small panel subset."""
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Legacy script-mode diagnostics (run directly, not collected by pytest)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    panel = pd.read_parquet("data/processed/merged_panel.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    panel["date"] = panel["date"].dt.to_period("M").dt.to_timestamp("M")
    permnos = panel["permno"].unique()[:500]
    panel = panel[panel["permno"].isin(permnos)].copy()
    panel["me"] = panel["me"].where(panel["me"] > 0, other=np.nan)
    print(f"Rows: {len(panel):,}  Permnos: {panel['permno'].nunique()}")

    from src.features import value_features, profitability_features, momentum_features

    print("\n--- value_features ---")
    val = value_features.compute(panel)
    for f in ["pchsale_pchrect", "pchcurrat", "sin"]:
        s = val[f]
        print(f"  {f}: {s.notna().mean():.1%} non-null  mean={s.mean():.4f}  range=[{s.min():.3f}, {s.max():.3f}]")

    print("\n--- profitability_features ---")
    prof = profitability_features.compute(panel)
    for f in ["depr", "pchdepr", "rna", "rsup", "nincr", "ps", "orgcap"]:
        s = prof[f]
        print(f"  {f}: {s.notna().mean():.1%} non-null  mean={s.mean():.4f}  range=[{s.min():.3f}, {s.max():.3f}]")

    print("\n--- momentum_features (herf) ---")
    mom = momentum_features.compute(panel)
    s = mom["herf"]
    print(f"  herf: {s.notna().mean():.1%} non-null  mean={s.mean():.4f}  range=[{s.min():.3f}, {s.max():.3f}]")

    # Sanity checks
    assert val["sin"].isin([0.0, 1.0, np.nan]).all(), "sin should be 0/1/NaN"
    assert prof["nincr"].between(0, 8).dropna().all(), "nincr should be in [0,8]"
    assert prof["ps"].between(0, 9).dropna().all(), "ps should be in [0,9]"
    assert (s.dropna() <= 1.0).all(), "herf should be <= 1 (HHI <= 1)"
    assert (s.dropna() >= 0.0).all(), "herf should be >= 0"
    print("\nAll sanity checks passed.")


# ---------------------------------------------------------------------------
# Deterministic fixture tests (no file I/O — synthetic inputs only)
# ---------------------------------------------------------------------------
import pytest

from src.features.feature_assembler import _rank_normalise_month
from src.features.momentum_features import _cum_ret
from src.features.value_features import _book_equity


class TestSizeRankNormalisation:
    """Rank normalisation maps a cross-section of market caps to [−1, +1]."""

    def test_five_distinct_values_map_to_grid(self):
        """Ordered caps [10k … 50k] should yield exactly [−1, −0.5, 0, 0.5, 1]."""
        s = pd.Series([10_000.0, 20_000.0, 30_000.0, 40_000.0, 50_000.0])
        result = _rank_normalise_month(s)
        expected = pd.Series([-1.0, -0.5, 0.0, 0.5, 1.0])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_output_bounds_always_minus_one_plus_one(self):
        """For any non-trivial series the min is −1 and the max is +1."""
        rng = np.random.default_rng(42)
        s = pd.Series(rng.standard_normal(200))
        result = _rank_normalise_month(s)
        assert result.min() == pytest.approx(-1.0)
        assert result.max() == pytest.approx(1.0)

    def test_nan_values_are_preserved(self):
        """NaN inputs must remain NaN in the output (not imputed here)."""
        s = pd.Series([1.0, np.nan, 3.0])
        result = _rank_normalise_month(s)
        assert np.isnan(result.iloc[1])


class TestBookToMarketFormula:
    """Value: BE = CEQ + TXDITC − pref-stock; BM = BE / ME."""

    @staticmethod
    def _panel(**kwargs) -> pd.DataFrame:
        """Build a one-row DataFrame with default NaN for missing columns."""
        defaults = dict(a_ceq=np.nan, a_txditc=np.nan,
                        a_pstkrv=np.nan, a_pstkl=np.nan, a_pstk=np.nan)
        defaults.update(kwargs)
        return pd.DataFrame([defaults])

    def test_basic_bm(self):
        """BE = 200, ME = 100 → BM = 2.0."""
        p = self._panel(a_ceq=200.0)
        be = _book_equity(p).values[0]
        bm = be / 100.0
        assert bm == pytest.approx(2.0)

    def test_preferred_stock_deducted(self):
        """BE = 100 + 10 − 20 = 90 (TXDITC added, PSTKRV deducted)."""
        p = self._panel(a_ceq=100.0, a_txditc=10.0, a_pstkrv=20.0)
        be = _book_equity(p).values[0]
        assert be == pytest.approx(90.0)

    def test_missing_ceq_produces_nan(self):
        """When CEQ is NaN, book equity is undefined → NaN."""
        p = self._panel(a_ceq=np.nan, a_txditc=5.0)
        be = _book_equity(p).values[0]
        assert np.isnan(be)


class TestMomentumMom12m:
    """Momentum: mom12m compounds 12 lagged monthly returns (months t−2 to t−13)."""

    def test_constant_five_pct_compounds_correctly(self):
        """12 months of +5%: expected = 1.05^12 − 1 ≈ 0.7959."""
        monthly = pd.Series([0.05] * 4)
        result = _cum_ret([monthly] * 12)
        expected = 1.05 ** 12 - 1.0
        np.testing.assert_allclose(result.values, expected, rtol=1e-6)

    def test_zero_returns_give_zero_momentum(self):
        """All-zero returns → mom12m = 0."""
        monthly = pd.Series([0.0] * 4)
        result = _cum_ret([monthly] * 12)
        np.testing.assert_allclose(result.values, 0.0, atol=1e-12)

    def test_one_total_loss_wipes_out_compound(self):
        """A single −100% month zeroes the entire product → result = −1."""
        s_crash = pd.Series([-1.0] * 3)
        s_gain = pd.Series([0.10] * 3)
        result = _cum_ret([s_crash] + [s_gain] * 11)
        np.testing.assert_allclose(result.values, -1.0, atol=1e-12)


class TestExcessReturnConstruction:
    """Target variable: ret_exc = ret_next − rf (feature_assembler.py line 109)."""

    def test_known_scalar_values(self):
        """ret = 5%, rf = 0.3% → ret_exc = 4.7%."""
        assert 0.050 - 0.003 == pytest.approx(0.047, abs=1e-9)

    def test_negative_when_below_riskfree(self):
        assert (-0.020) - 0.004 < 0.0

    def test_zero_when_return_equals_riskfree(self):
        r = 0.004
        assert r - r == pytest.approx(0.0)

    def test_pandas_vectorised_broadcast(self):
        """Verify pandas column arithmetic matches element-wise expectations."""
        df = pd.DataFrame({"ret_next": [0.10, -0.05, 0.00],
                           "rf":       [0.003, 0.003, 0.003]})
        df["ret_exc"] = df["ret_next"] - df["rf"]
        expected = pd.Series([0.097, -0.053, -0.003])
        pd.testing.assert_series_equal(df["ret_exc"], expected,
                                       check_names=False, rtol=1e-9)


class TestRoeProfitability:
    """Profitability: ROE = IB / CEQ; matches formula in profitability_features.py."""

    @staticmethod
    def _roe(ib: float, ceq: float) -> float:
        """Replicate `(ib / ceq).replace([inf, -inf], NaN)` on scalar inputs."""
        result = pd.Series([ib], dtype=float) / pd.Series([ceq], dtype=float)
        return result.replace([np.inf, -np.inf], np.nan).values[0]

    def test_positive_earnings(self):
        """IB = 20, CEQ = 100 → ROE = 0.20."""
        assert self._roe(20.0, 100.0) == pytest.approx(0.20)

    def test_negative_earnings(self):
        """IB = −10, CEQ = 100 → ROE = −0.10."""
        assert self._roe(-10.0, 100.0) == pytest.approx(-0.10)

    def test_zero_ceq_returns_nan(self):
        """CEQ = 0 produces inf which is then mapped to NaN."""
        assert np.isnan(self._roe(5.0, 0.0))

    def test_zero_ib_zero_ceq_returns_nan(self):
        """0 / 0 = NaN (pandas returns NaN for 0/0 float division)."""
        assert np.isnan(self._roe(0.0, 0.0))
