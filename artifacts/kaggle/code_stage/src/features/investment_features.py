"""
src/features/investment_features.py
=====================================
Investment, financing and accrual characteristics from GKX (2020) Table A.1.

Features implemented
--------------------
invest      – Total investment / assets: (ΔPPENT + ΔINVT) / lag(AT)
noa         – Net operating assets: (AT - CHE - (AT - LCT - DLTT - DLC - MKVALT - PSTK)) / AT
                 simplified: (operating assets - operating liabilities) / lag(AT)
chcsho      – Change in shares outstanding: CSHO / lag(CSHO) - 1
grprofits   – Growth in gross profit: Δ(SALE - COGS) / lag(AT)
pchcapex    – Change in capex: CAPX / lag(CAPX) - 1
cinvest     – Corporate investment: (CAPX - mean(CAPX over 3 years)) / SALE
tang        – Tangibility: (CHE + 0.715*RECT + 0.547*INVT + 0.535*PPENT) / AT
realestate  – Real estate / total assets: PPEGT / AT (proxy; no separate RE line)
acc         – Accruals: (ΔACT - ΔCHE - ΔLCT + ΔDLC + ΔTXP - DP) / AT
pctacc      – Percent accruals: ACC / |IB| (if IB ≠ 0)
absacc      – Absolute value of accruals
lgr         – (already in value_features; included here for completeness if needed)
convind     – Convertible debt indicator: 1 if DCVT > 0 (Valta 2012)
secured     – Secured debt / total debt (proxy: DLTT / (DLTT + DLC))
securedind  – 1 if any secured debt exists (proxy: DLTT > 0)
divi        – Dividend initiator: 1 if DV > 0 and lag(DV) == 0
divo        – Dividend omitter: 1 if DV == 0 and lag(DV) > 0
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute investment and accrual features.

    Parameters
    ----------
    panel : pd.DataFrame
        Merged monthly panel with `a_` prefixed Compustat annual columns.

    Returns
    -------
    pd.DataFrame  Columns: permno, date, <features>.
    """
    panel = panel.sort_values(["permno", "date"]).reset_index(drop=True)
    out = panel[["permno", "date"]].copy()

    grp = panel.groupby("permno", sort=False)

    at = panel["a_at"]
    at_lag12 = grp["a_at"].shift(12).replace(0, np.nan)

    # ------------------------------------------------------------------
    # Total investment: (ΔPPENT + ΔINVT) / lag(AT)  (Titman, Wei, Xie 2004)
    # ------------------------------------------------------------------
    ppent_lag12 = grp["a_ppent"].shift(12)
    invt_lag12 = grp["a_invt"].shift(12)
    delta_ppent = panel["a_ppent"].fillna(0.0) - ppent_lag12.fillna(0.0)
    delta_invt = panel["a_invt"].fillna(0.0) - invt_lag12.fillna(0.0)
    out["invest"] = ((delta_ppent + delta_invt) / at_lag12).replace(
        [np.inf, -np.inf], np.nan
    )

    # ------------------------------------------------------------------
    # Net operating assets (Hirshleifer et al. 2004)
    # Operating assets = AT - CHE - IVAO
    # Operating liabilities = AT - DLC - DLTT - MKVALT (book equity proxy) - PSTK
    # NOA = (OA - OL) / lag(AT)
    # ------------------------------------------------------------------
    oa = (at.fillna(0.0)
          - panel["a_che"].fillna(0.0)
          - panel["a_ivao"].fillna(0.0))
    pstk = (panel["a_pstkrv"].fillna(panel["a_pstkl"]).fillna(panel["a_pstk"]).fillna(0.0))
    ol = (at.fillna(0.0)
          - panel["a_dlc"].fillna(0.0)
          - panel["a_dltt"].fillna(0.0)
          - panel["a_ceq"].fillna(0.0)
          - panel["a_txditc"].fillna(0.0)
          - pstk)
    out["noa"] = ((oa - ol) / at_lag12).replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # Change in shares outstanding (Pontiff & Woodgate 2008)
    # ------------------------------------------------------------------
    csho_lag12 = grp["a_csho"].shift(12).replace(0, np.nan)
    out["chcsho"] = (panel["a_csho"] / csho_lag12 - 1.0).replace(
        [np.inf, -np.inf], np.nan
    )

    # ------------------------------------------------------------------
    # Growth in gross profits: Δ(SALE - COGS) / lag(AT)
    # ------------------------------------------------------------------
    gp_curr = panel["a_sale"].fillna(0.0) - panel["a_cogs"].fillna(0.0)
    gp_lag12 = grp["a_sale"].shift(12).fillna(0.0) - grp["a_cogs"].shift(12).fillna(0.0)
    out["grprofits"] = ((gp_curr - gp_lag12) / at_lag12).replace(
        [np.inf, -np.inf], np.nan
    )

    # ------------------------------------------------------------------
    # Change in capital expenditures: CAPX / lag(CAPX) - 1
    # ------------------------------------------------------------------
    capx_lag12 = grp["a_capx"].shift(12).replace(0, np.nan)
    out["pchcapex"] = (panel["a_capx"].fillna(0.0) / capx_lag12 - 1.0).replace(
        [np.inf, -np.inf], np.nan
    )

    # ------------------------------------------------------------------
    # Corporate investment (Titman et al. 2004 alternative)
    # cinvest = (CAPX - 3yr_avg_CAPX) / SALE
    # ------------------------------------------------------------------
    capx = panel["a_capx"].fillna(0.0)
    capx_3yr_avg = (grp["a_capx"].shift(12).fillna(0.0)
                    + grp["a_capx"].shift(24).fillna(0.0)
                    + grp["a_capx"].shift(36).fillna(0.0)) / 3.0
    sale_pos = panel["a_sale"].replace(0, np.nan)
    out["cinvest"] = ((capx - capx_3yr_avg) / sale_pos).replace(
        [np.inf, -np.inf], np.nan
    )

    # ------------------------------------------------------------------
    # Tangibility (Almeida & Campello 2007)
    # TANG = (CHE + 0.715*RECT + 0.547*INVT + 0.535*PPENT) / AT
    # ------------------------------------------------------------------
    at_pos = at.replace(0, np.nan)
    out["tang"] = (
        (panel["a_che"].fillna(0.0)
         + 0.715 * panel["a_rect"].fillna(0.0)
         + 0.547 * panel["a_invt"].fillna(0.0)
         + 0.535 * panel["a_ppent"].fillna(0.0))
        / at_pos
    ).replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # Real estate proxy: PPEGT / AT (total PP&E including leases)
    # (True real estate holdings not separately reported in Compustat)
    # ------------------------------------------------------------------
    out["realestate"] = (panel["a_ppegt"].fillna(0.0) / at_pos).replace(
        [np.inf, -np.inf], np.nan
    )

    # ------------------------------------------------------------------
    # Accruals (Sloan 1996)
    # ACC = (ΔACT - ΔCHE) - (ΔLCT - ΔDLC - ΔTXP) - DP
    # Scaled by average total assets
    # ------------------------------------------------------------------
    act_lag12 = grp["a_act"].shift(12)
    che_lag12 = grp["a_che"].shift(12)
    lct_lag12 = grp["a_lct"].shift(12)
    dlc_lag12 = grp["a_dlc"].shift(12)
    txp_lag12 = grp["a_txp"].shift(12)

    delta_act = panel["a_act"].fillna(0.0) - act_lag12.fillna(0.0)
    delta_che = panel["a_che"].fillna(0.0) - che_lag12.fillna(0.0)
    delta_lct = panel["a_lct"].fillna(0.0) - lct_lag12.fillna(0.0)
    delta_dlc = panel["a_dlc"].fillna(0.0) - dlc_lag12.fillna(0.0)
    delta_txp = panel["a_txp"].fillna(0.0) - txp_lag12.fillna(0.0)
    dp = panel["a_dp"].fillna(0.0)

    acc_raw = (delta_act - delta_che) - (delta_lct - delta_dlc - delta_txp) - dp
    avg_at = (at.fillna(0.0) + at_lag12.fillna(0.0)) / 2.0
    avg_at = avg_at.replace(0, np.nan)
    out["acc"] = (acc_raw / avg_at).replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # Percent accruals: ACC / |IB|
    # ------------------------------------------------------------------
    ib_abs = panel["a_ib"].abs().replace(0, np.nan)
    out["pctacc"] = (acc_raw / ib_abs).replace([np.inf, -np.inf], np.nan)
    out["absacc"] = out["acc"].abs()

    # ------------------------------------------------------------------
    # Convertible debt indicator (Valta 2012)
    # convind = 1 if DCVT (convertible debt) > 0
    # ------------------------------------------------------------------
    out["convind"] = (panel["a_dcvt"].fillna(0.0) > 0).astype(float)

    # ------------------------------------------------------------------
    # Secured debt proxy: DLTT / (DLTT + DLC)
    # Note: true secured debt (Valta 2012) requires hand-collected data
    # not available in standard Compustat.  DLTT/(DLTT+DLC) measures the
    # long-term debt fraction and is the standard proxy in GKX replications.
    # ------------------------------------------------------------------
    dltt = panel["a_dltt"].fillna(0.0)
    dlc = panel["a_dlc"].fillna(0.0)
    total_debt = dltt + dlc
    out["secured"] = (dltt / total_debt.replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )
    out["securedind"] = (dltt > 0).astype(float)

    # ------------------------------------------------------------------
    # Dividend initiator / omitter
    # ------------------------------------------------------------------
    dv_curr = panel["a_dvt"].fillna(panel["a_dv"]).fillna(0.0)
    dv_lag12 = grp["a_dvt"].shift(12).fillna(grp["a_dv"].shift(12)).fillna(0.0)
    out["divi"] = ((dv_curr > 0) & (dv_lag12 == 0)).astype(float)
    out["divo"] = ((dv_curr == 0) & (dv_lag12 > 0)).astype(float)

    # ------------------------------------------------------------------
    # Employment growth: hire  (Belo, Lin & Bazdresch 2014)
    # hire = (EMP_t - EMP_{t-1}) / (0.5 * (EMP_t + EMP_{t-1}))
    # Davis–Haltiwanger–Schuh symmetric growth rate; bounded in (−2, +2).
    # Annual employee count (a_emp) is lagged 12 months for the prior fiscal year.
    # ------------------------------------------------------------------
    emp = panel["a_emp"]
    emp_lag12 = grp["a_emp"].shift(12)
    avg_emp = (emp + emp_lag12) / 2.0
    avg_emp = avg_emp.replace(0, np.nan)
    out["hire"] = ((emp - emp_lag12) / avg_emp).replace([np.inf, -np.inf], np.nan)

    log.info("Investment features computed: %d rows, %d features",
             len(out), out.shape[1] - 2)
    return out
