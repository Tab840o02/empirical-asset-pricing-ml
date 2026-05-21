"""
src/features/momentum_features.py
==================================
Price-trend and trading-activity characteristics from GKX (2020) Table A.1.

All functions take the merged monthly panel (sorted by permno, date) and
return raw (un-normalised) feature values.  Cross-sectional rank normalisation
is applied later in ``feature_assembler.py``.

Features implemented
--------------------
mom1m       – 1-month return (short-term reversal)
mom6m       – Cumulative return months t-7 to t-2 (skip most recent)
mom12m      – Cumulative return months t-13 to t-2
mom36m      – Cumulative return months t-37 to t-13
chmom       – mom6m minus mom6m lagged 6 months
indmom      – Equal-weighted industry return over past 6 months
turn        – Average monthly turnover (vol / shrout) past 3 months
std_turn    – Std-dev of monthly turnover past 3 months
dolvol      – Log average daily dollar volume past 3 months
std_dolvol  – Std-dev of log monthly dollar volume past 3 months
mve_ia      – Log(ME) minus within-industry average log(ME)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _cum_ret(shifted_returns: list[pd.Series]) -> pd.Series:
    """Compound a list of lagged return Series into a cumulative return."""
    prod = pd.Series(np.ones(len(shifted_returns[0])), index=shifted_returns[0].index,
                     dtype="float64")
    for s in shifted_returns:
        prod = prod * (1.0 + s.fillna(0.0))
    return prod - 1.0


def _sic2_industry(panel: pd.DataFrame) -> pd.Series:
    """Return a 2-digit SIC code series (used as rough industry proxy)."""
    sich = panel["a_sich"].fillna(-1).astype(int)
    return (sich // 10).clip(lower=0)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute momentum and price-trend features.

    Parameters
    ----------
    panel : pd.DataFrame
        Merged CRSP–Compustat monthly panel, sorted by (permno, date).
        Must contain at minimum: permno, date, ret, vol, shrout, prc, me.

    Returns
    -------
    pd.DataFrame
        Columns: permno, date, <features>.
    """
    panel = panel.sort_values(["permno", "date"]).reset_index(drop=True)

    out = panel[["permno", "date"]].copy()

    grp = panel.groupby("permno", sort=False)

    ret = grp["ret"]

    # ------------------------------------------------------------------
    # Shift return series: shift(k) = return k months ago
    # ------------------------------------------------------------------
    shifts = {k: ret.shift(k) for k in range(1, 38)}

    # mom1m — past 1-month return
    out["mom1m"] = shifts[1]

    # mom6m — cumulative return months 2–7 ago (skip t-1)
    out["mom6m"] = _cum_ret([shifts[k] for k in range(2, 8)])

    # mom12m — cumulative return months 2–13 ago
    out["mom12m"] = _cum_ret([shifts[k] for k in range(2, 14)])

    # mom36m — cumulative return months 13–37 ago
    out["mom36m"] = _cum_ret([shifts[k] for k in range(13, 38)])

    # chmom — change in 6-month momentum (this month's mom6m minus 6 months ago mom6m)
    # mom6m_{t-6} is the prod of (1+r) over months 8–13 ago
    mom6m_lag6 = _cum_ret([shifts[k] for k in range(8, 14)])
    out["chmom"] = out["mom6m"] - mom6m_lag6

    # ------------------------------------------------------------------
    # Industry momentum (indmom)
    # Equal-weighted average mom6m of stocks in the same 2-digit SIC industry
    # ------------------------------------------------------------------
    panel_tmp = panel.copy()
    panel_tmp["_mom6m"] = out["mom6m"].values
    sic2 = _sic2_industry(panel)
    # Use NaN for invalid SIC so groupby excludes those stocks
    panel_tmp["_ind"] = sic2.where(sic2 > 0, other=np.nan)
    out["indmom"] = (
        panel_tmp.groupby(["date", "_ind"])["_mom6m"]
        .transform("mean")
    ).values

    # ------------------------------------------------------------------
    # Turnover features
    # CRSP monthly vol is in shares.  shrout is in thousands of shares.
    # Turnover = monthly vol / (shrout * 1000)
    # ------------------------------------------------------------------
    raw_turn = panel["vol"] / (panel["shrout"] * 1_000.0)
    raw_turn = raw_turn.replace([np.inf, -np.inf], np.nan)
    panel_tmp["_turn"] = raw_turn.values

    turn_g = panel_tmp.groupby("permno", sort=False)["_turn"]
    out["turn"] = turn_g.rolling(3, min_periods=2).mean().reset_index(level=0, drop=True)
    out["std_turn"] = turn_g.rolling(3, min_periods=2).std().reset_index(level=0, drop=True)

    # ------------------------------------------------------------------
    # Dollar volume features
    # dolvol = log of (|prc| * vol) averaged over past 3 months
    # ------------------------------------------------------------------
    raw_dvol = np.log((panel["prc"].abs() * panel["vol"]).clip(lower=1e-6))
    panel_tmp["_dvol"] = raw_dvol.values
    dvol_g = panel_tmp.groupby("permno", sort=False)["_dvol"]
    out["dolvol"] = dvol_g.rolling(3, min_periods=2).mean().reset_index(level=0, drop=True)
    out["std_dolvol"] = dvol_g.rolling(3, min_periods=2).std().reset_index(level=0, drop=True)

    # ------------------------------------------------------------------
    # mve_ia — log market cap minus within-industry average
    # ------------------------------------------------------------------
    log_me = np.log(panel["me"].clip(lower=1e-6))
    panel_tmp["_logme"] = log_me.values
    ind_avg_logme = (
        panel_tmp.groupby(["date", "_ind"])["_logme"]
        .transform("mean")
    )
    out["mve_ia"] = (log_me.values - ind_avg_logme.values)

    log.info("Momentum features computed: %d rows, %d features",
             len(out), out.shape[1] - 2)
    return out
