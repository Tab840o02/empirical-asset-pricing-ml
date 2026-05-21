"""
src/features/value_features.py
================================
Value and price-level characteristics from GKX (2020) Table A.1.

All functions return raw (un-normalised) values.

Features implemented
--------------------
bm          – Book-to-market equity ratio
bm_ia       – B/M minus within-industry mean
cfp         – Cash flow to price (earnings + D&A) / ME
cfp_ia      – CF/P minus within-industry mean
ep          – Earnings-to-price (IB / ME)
ep_ia       – E/P minus within-industry mean
sp          – Sales-to-price (SALE / ME)
agr         – Annual asset growth rate (AT/lag_AT - 1)
sgr         – Annual sales growth rate (SALE/lag_SALE - 1)
lgr         – Long-term debt growth rate
lev         – Leverage (LT / AT, or (DLTT+DLC) / AT)
dy          – Dividend yield (DV / ME)
pchsale_pchinvt – %ΔSales - %ΔInventory
pchsale_pchxsga – %ΔSales - %ΔSGA
pchsale_pchgm   – %ΔSales - %ΔGross margin
pchcapx_ia      – %ΔCapex minus within-industry mean
cash        – Cash / total assets
salecash    – Sales / cash
saleinv     – Sales / inventory
salerec     – Sales / accounts receivable
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _book_equity(p: pd.DataFrame) -> pd.Series:
    """
    Book equity (Compustat annual):
      BE = CEQ + TXDITC (deferred taxes) - PSTKRV (preferred stock at redemption value)
      Fallback order: PSTKRV → PSTKL → PSTK → 0
    """
    pstk = (
        p["a_pstkrv"]
        .fillna(p["a_pstkl"])
        .fillna(p["a_pstk"])
        .fillna(0.0)
    )
    txditc = p["a_txditc"].fillna(0.0)
    be = p["a_ceq"].fillna(0.0) + txditc - pstk
    return be.where(p["a_ceq"].notna(), other=np.nan)


def _sic2(p: pd.DataFrame) -> pd.Series:
    sich = p["a_sich"].fillna(-1).astype(int)
    return (sich // 10).clip(lower=0)


def _ind_adjust(series: pd.Series, dates: pd.Series, sic2: pd.Series) -> pd.Series:
    """Subtract the within-industry, within-month mean from series."""
    tmp = pd.DataFrame({"val": series.values, "date": dates.values, "sic2": sic2.values})
    valid = sic2 > 0
    mean_ = tmp[valid].groupby(["date", "sic2"])["val"].transform("mean")
    result = series.copy()
    result[valid] = series[valid] - mean_
    return result


def _pct_chg(curr: pd.Series, prev: pd.Series) -> pd.Series:
    """Safe percentage change: (curr - prev) / |prev|."""
    return (curr - prev) / prev.abs().replace(0, np.nan)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute value and size characteristics.

    Parameters
    ----------
    panel : pd.DataFrame
        Merged monthly panel with `a_` prefixed Compustat annual columns.

    Returns
    -------
    pd.DataFrame   Columns: permno, date, <features>.
    """
    panel = panel.sort_values(["permno", "date"]).reset_index(drop=True)
    out = panel[["permno", "date"]].copy()

    sic2 = _sic2(panel)
    me = panel["me"].replace(0, np.nan)

    # ------------------------------------------------------------------
    # Book equity
    # ------------------------------------------------------------------
    be = _book_equity(panel)

    # ------------------------------------------------------------------
    # Book-to-market
    # ------------------------------------------------------------------
    out["bm"] = (be / me).replace([np.inf, -np.inf], np.nan)
    out["bm_ia"] = _ind_adjust(out["bm"], panel["date"], sic2)

    # ------------------------------------------------------------------
    # Cash flow to price: CF = IB + DP (earnings before extraordinary + D&A)
    # ------------------------------------------------------------------
    cf = panel["a_ib"].fillna(0.0) + panel["a_dp"].fillna(0.0)
    out["cfp"] = (cf / me).replace([np.inf, -np.inf], np.nan)
    out["cfp_ia"] = _ind_adjust(out["cfp"], panel["date"], sic2)

    # ------------------------------------------------------------------
    # Earnings to price
    # ------------------------------------------------------------------
    out["ep"] = (panel["a_ib"] / me).replace([np.inf, -np.inf], np.nan)
    out["ep_ia"] = _ind_adjust(out["ep"], panel["date"], sic2)

    # ------------------------------------------------------------------
    # Sales to price
    # ------------------------------------------------------------------
    out["sp"] = (panel["a_sale"] / me).replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # Dividend yield
    # ------------------------------------------------------------------
    dv = panel["a_dvt"].fillna(panel["a_dv"]).fillna(0.0)
    out["dy"] = (dv / me).replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # Leverage: (DLTT + DLC) / AT
    # ------------------------------------------------------------------
    debt = panel["a_dltt"].fillna(0.0) + panel["a_dlc"].fillna(0.0)
    out["lev"] = (debt / panel["a_at"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # Lagged annual accounting variables (1-year lag within permno)
    # Note: public_date already enforces the no-look-ahead lag;
    # these lags are for computing growth rates.
    # ------------------------------------------------------------------
    grp = panel.groupby("permno", sort=False)

    at_lag = grp["a_at"].shift(12)    # 12 monthly rows ≈ 1 year back
    sale_lag = grp["a_sale"].shift(12)
    dltt_lag = grp["a_dltt"].shift(12)
    capx_lag = grp["a_capx"].shift(12)
    invt_lag = grp["a_invt"].shift(12)
    xsga_lag = grp["a_xsga"].shift(12)
    gp_curr = panel["a_sale"] - panel["a_cogs"]
    gp_lag = grp["a_sale"].shift(12) - grp["a_cogs"].shift(12)

    # ------------------------------------------------------------------
    # Asset growth (Cooper et al. 2008)
    # ------------------------------------------------------------------
    out["agr"] = _pct_chg(panel["a_at"], at_lag)

    # ------------------------------------------------------------------
    # Sales growth
    # ------------------------------------------------------------------
    out["sgr"] = _pct_chg(panel["a_sale"], sale_lag)

    # ------------------------------------------------------------------
    # Long-term debt growth
    # ------------------------------------------------------------------
    out["lgr"] = _pct_chg(
        panel["a_dltt"].fillna(0.0),
        dltt_lag.fillna(0.0)
    ).replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # %ΔSales - %ΔInventory  (Abarbanell & Bushee 1998)
    # ------------------------------------------------------------------
    pch_sale = _pct_chg(panel["a_sale"], sale_lag)
    pch_invt = _pct_chg(panel["a_invt"], invt_lag)
    out["pchsale_pchinvt"] = pch_sale - pch_invt

    # ------------------------------------------------------------------
    # %ΔSales - %ΔSGA
    # ------------------------------------------------------------------
    pch_xsga = _pct_chg(panel["a_xsga"].fillna(0.0), xsga_lag.fillna(0.0))
    out["pchsale_pchxsga"] = pch_sale - pch_xsga

    # ------------------------------------------------------------------
    # %ΔSales - %ΔGross margin
    # ------------------------------------------------------------------
    pch_gm = _pct_chg(gp_curr, gp_lag)
    out["pchsale_pchgm"] = pch_sale - pch_gm

    # ------------------------------------------------------------------
    # %ΔCapex minus industry mean (Titman, Wei, Xie 2004)
    # ------------------------------------------------------------------
    pch_capx = _pct_chg(panel["a_capx"].fillna(0.0), capx_lag.fillna(0.0))
    out["pchcapx_ia"] = _ind_adjust(pch_capx, panel["date"], sic2)

    # ------------------------------------------------------------------
    # Cash / total assets
    # ------------------------------------------------------------------
    out["cash"] = (panel["a_che"] / panel["a_at"].replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )

    # ------------------------------------------------------------------
    # Sales / cash
    # ------------------------------------------------------------------
    out["salecash"] = (panel["a_sale"] / panel["a_che"].replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )

    # ------------------------------------------------------------------
    # Sales / inventory
    # ------------------------------------------------------------------
    out["saleinv"] = (panel["a_sale"] / panel["a_invt"].replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )

    # ------------------------------------------------------------------
    # Sales / receivables
    # ------------------------------------------------------------------
    out["salerec"] = (panel["a_sale"] / panel["a_rect"].replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )

    log.info("Value features computed: %d rows, %d features",
             len(out), out.shape[1] - 2)
    return out
