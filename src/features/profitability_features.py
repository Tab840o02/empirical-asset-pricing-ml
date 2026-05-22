"""
src/features/profitability_features.py
========================================
Profitability characteristics from GKX (2020) Table A.1.

Features implemented
--------------------
gp          – Gross profitability: (SALE - COGS) / AT (Novy-Marx 2013)
gma         – Gross margins: (SALE - COGS) / lag(AT)
roe         – Return on equity: IB / CEQ
niy         – Net income / total assets
roic        – Return on invested capital: IB / (CEQ + DLTT + DLC)
pm          – Profit margin: IB / SALE
chpm        – Change in profit margin (vs. 12 months ago)
chato       – Change in asset turnover: (SALE/AT) - lag(SALE/lag_AT)
chatoia     – Industry-adjusted change in asset turnover
roaq        – Quarterly ROA: IBQ / lag(ATQ)
roeq        – Quarterly ROE: IBQ / CEQQ
rdmq        – R&D to market cap (quarterly): XRDQ / ME
rd_sale     – R&D / sales: XRD / SALE
rd_mve      – R&D / market equity: XRD / ME
depr        – Depreciation / PP&E: DP / PPENT  (Holthausen & Larcker 1992)
pchdepr     – % change in DP/PPENT  (Holthausen & Larcker 1992)
rna         – Return on net assets: IB / avg(NOA) (Soliman 2008)
ms          – Mohanram's G-score (8 binary signals)
chinv       – Quarterly change in inventory: ΔINVTQ / ATQ
chtx        – Quarterly change in taxes (TXTQ or TXDIQ)
sue         – Standardized unexpected earnings (EPS surprise)
rs          – Revenue surprise (quarterly sales surprise)
rsup        – Quarterly revenue surprise: ΔsaleQ_{t,t-4} / ME  (Kama 2009)
nincr       – Consecutive quarterly EPS increases  (Barth et al. 1999)
ps          – Piotroski F-score: sum of 9 binary profitability signals (Piotroski 2000)
orgcap      – Organisational capital via perpetual inventory of SG&A  (Eisfeldt & Papanikolaou 2013)
roavol      – Standard deviation of quarterly ROA over 8 quarters
stdcf       – Standard deviation of quarterly cash flow scaled by assets
cashpr      – Cash productivity: (ME + DLTT - AT) / CHE
tb          – Tax burden: IB / PI (net income / pre-tax income)
quick       – Quick ratio: (ACT - INVT) / LCT
curr        – Current ratio: ACT / LCT
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import RAW_DIR

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_ear(panel: pd.DataFrame) -> pd.Series:
    """
    Earnings announcement return (ear) — Kishore, Brockman, Altieri & Bethke (2008).

    For each row in the panel, the feature is the 3-day cumulative CRSP return
    in the window [rdq − 1 trading day, rdq, rdq + 1 trading day], where rdq is
    the most recent quarterly earnings announcement date (column ``q_rdq``).

    Returns a Series aligned to panel.index.
    """
    if "q_rdq" not in panel.columns:
        log.warning("q_rdq not in panel — ear will be all NaN")
        return pd.Series(np.nan, index=panel.index, name="ear")

    rdq_col = pd.to_datetime(panel["q_rdq"], errors="coerce")
    valid = rdq_col.notna()
    if valid.sum() == 0:
        return pd.Series(np.nan, index=panel.index, name="ear")

    # Unique (permno, rdq) pairs — these are what we need daily returns for
    ann = (
        panel.loc[valid, ["permno"]]
        .assign(rdq=rdq_col[valid])
        .drop_duplicates()
        .sort_values(["permno", "rdq"])
        .reset_index(drop=True)
    )
    ann["permno"] = ann["permno"].astype("int64")

    # Load CRSP daily files for the relevant years (±1 year buffer for edge days)
    yr_min = int(ann["rdq"].dt.year.min())
    yr_max = int(ann["rdq"].dt.year.max())
    parts = []
    for yr in range(max(yr_min - 1, 1925), yr_max + 2):
        path = RAW_DIR / f"crsp_daily_{yr}.parquet"
        if path.exists():
            df = pd.read_parquet(path, columns=["permno", "date", "ret"])
            df["permno"] = df["permno"].astype("int64")
            df["date"] = pd.to_datetime(df["date"])
            df["ret"] = pd.to_numeric(df["ret"], errors="coerce").astype("float64")
            parts.append(df)

    if not parts:
        log.warning("No CRSP daily files found — ear will be all NaN")
        return pd.Series(np.nan, index=panel.index, name="ear")

    daily = (
        pd.concat(parts, ignore_index=True)
        .sort_values(["permno", "date"])
        .reset_index(drop=True)
    )
    # Sequential trading-day index within each permno (used as a positional key)
    daily["_didx"] = daily.groupby("permno").cumcount()

    # Find the nearest trading day on-or-after each announcement date
    matched = pd.merge_asof(
        ann.sort_values(["permno", "rdq"]),
        daily[["permno", "date", "_didx"]].sort_values(["permno", "date"]),
        left_on="rdq",
        right_on="date",
        by="permno",
        direction="forward",  # closest trading day >= rdq
    ).rename(columns={"_didx": "t_idx"})

    # For each announcement, sum returns over [t-1, t, t+1] trading days
    ret_lookup = daily[["permno", "_didx", "ret"]].rename(
        columns={"_didx": "_lookup_idx"}
    )
    window_parts = []
    for offset in (-1, 0, 1):
        tmp = matched.loc[matched["t_idx"].notna(), ["permno", "rdq", "t_idx"]].copy()
        tmp["t_idx"] = tmp["t_idx"].astype("int64")
        tmp["_lookup_idx"] = tmp["t_idx"] + offset
        tmp = tmp.merge(ret_lookup, on=["permno", "_lookup_idx"], how="left")
        window_parts.append(tmp[["permno", "rdq", "ret"]])

    ear_df = (
        pd.concat(window_parts, ignore_index=True)
        .groupby(["permno", "rdq"], as_index=False)["ret"]
        .sum(min_count=1)   # NaN if all three days are missing
        .rename(columns={"ret": "ear"})
    )

    # Merge ear back to panel rows via (permno, q_rdq)
    result = pd.Series(np.nan, index=panel.index, name="ear")
    merge_key = pd.DataFrame(
        {"permno": panel["permno"].astype("int64"), "rdq": rdq_col},
        index=panel.index,
    )
    merged_back = merge_key.merge(ear_df, on=["permno", "rdq"], how="left")
    merged_back.index = panel.index
    result.loc[valid] = merged_back.loc[valid, "ear"]
    return result


def _sic2(p: pd.DataFrame) -> pd.Series:
    sich = p["a_sich"].fillna(-1).astype(int)
    return (sich // 10).clip(lower=0)


def _ind_adjust(series: pd.Series, dates: pd.Series, sic2: pd.Series) -> pd.Series:
    tmp = pd.DataFrame({"val": series.values, "date": dates.values, "sic2": sic2.values})
    valid = sic2 > 0
    mean_ = tmp[valid].groupby(["date", "sic2"])["val"].transform("mean")
    result = series.copy()
    result[valid] = series[valid] - mean_
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute profitability features.

    Parameters
    ----------
    panel : pd.DataFrame
        Merged monthly panel.

    Returns
    -------
    pd.DataFrame  Columns: permno, date, <features>.
    """
    panel = panel.sort_values(["permno", "date"]).reset_index(drop=True)
    out = panel[["permno", "date"]].copy()

    sic2 = _sic2(panel)
    me = panel["me"].replace(0, np.nan)
    grp = panel.groupby("permno", sort=False)

    at = panel["a_at"].replace(0, np.nan)
    ceq = panel["a_ceq"].replace(0, np.nan)
    sale = panel["a_sale"].replace(0, np.nan)
    ib = panel["a_ib"]
    cogs = panel["a_cogs"].fillna(0.0)
    gp_curr = panel["a_sale"] - cogs

    at_lag12 = grp["a_at"].shift(12).replace(0, np.nan)
    sale_lag12 = grp["a_sale"].shift(12)

    # ------------------------------------------------------------------
    # Gross profitability (Novy-Marx 2013)
    # ------------------------------------------------------------------
    out["gp"] = (gp_curr / at).replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # Gross margins / lagged assets
    # ------------------------------------------------------------------
    out["gma"] = (gp_curr / at_lag12).replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # Return on equity (annual)
    # ------------------------------------------------------------------
    out["roe"] = (ib / ceq).replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # Net income / total assets
    # ------------------------------------------------------------------
    out["niy"] = (panel["a_ni"] / at).replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # Return on invested capital
    # ROIC = IB / (CEQ + DLTT + DLC)
    # ------------------------------------------------------------------
    debt = panel["a_dltt"].fillna(0.0) + panel["a_dlc"].fillna(0.0)
    ic = (ceq.fillna(0.0) + debt).replace(0, np.nan)
    out["roic"] = (ib / ic).replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # Profit margin
    # ------------------------------------------------------------------
    out["pm"] = (ib / sale).replace([np.inf, -np.inf], np.nan)

    # Change in profit margin vs 12 months ago
    pm_lag12 = grp["a_ib"].shift(12) / grp["a_sale"].shift(12).replace(0, np.nan)
    out["chpm"] = out["pm"] - pm_lag12

    # ------------------------------------------------------------------
    # Asset turnover and its change
    # ------------------------------------------------------------------
    ato = (sale / at).replace([np.inf, -np.inf], np.nan)
    ato_lag12 = (sale_lag12 / at_lag12).replace([np.inf, -np.inf], np.nan)
    out["chato"] = ato - ato_lag12
    out["chatoia"] = _ind_adjust(out["chato"], panel["date"], sic2)

    # ------------------------------------------------------------------
    # R&D ratios (annual)
    # ------------------------------------------------------------------
    xrd = panel["a_xrd"].fillna(0.0)
    out["rd_sale"] = (xrd / sale).replace([np.inf, -np.inf], np.nan)
    out["rd_mve"] = (xrd / me).replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # Tax burden: IB / PI (if PI > 0)
    # ------------------------------------------------------------------
    pi = panel["a_pi"].replace(0, np.nan)
    out["tb"] = (ib / pi).replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # Quick ratio: (ACT - INVT) / LCT
    # ------------------------------------------------------------------
    lct = panel["a_lct"].replace(0, np.nan)
    out["quick"] = ((panel["a_act"] - panel["a_invt"].fillna(0.0)) / lct).replace(
        [np.inf, -np.inf], np.nan
    )

    # ------------------------------------------------------------------
    # Current ratio: ACT / LCT
    # ------------------------------------------------------------------
    out["curr"] = (panel["a_act"] / lct).replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # Cash productivity: (ME + DLTT - AT) / CHE
    # ------------------------------------------------------------------
    che = panel["a_che"].replace(0, np.nan)
    out["cashpr"] = ((me + panel["a_dltt"].fillna(0.0) - panel["a_at"].fillna(0.0)) / che).replace(
        [np.inf, -np.inf], np.nan
    )

    # ------------------------------------------------------------------
    # Quarterly ROA: IBQ / lag(ATQ)
    # ------------------------------------------------------------------
    atq_lag3 = grp["q_atq"].shift(3).replace(0, np.nan)
    out["roaq"] = (panel["q_ibq"] / atq_lag3).replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # Quarterly ROE: IBQ / lag(CEQQ)  — lag 1 quarter = ~3 monthly rows
    # ------------------------------------------------------------------
    ceqq = grp["q_ceqq"].shift(3).replace(0, np.nan)
    out["roeq"] = (panel["q_ibq"] / ceqq).replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # R&D / market (quarterly)
    # ------------------------------------------------------------------
    out["rdmq"] = (panel["q_xrdq"].fillna(0.0) / me).replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # Quarterly change in inventory: ΔINVTQ / ATQ
    # ------------------------------------------------------------------
    invtq_lag3 = grp["q_invtq"].shift(3)
    atq = panel["q_atq"].replace(0, np.nan)
    out["chinv"] = ((panel["q_invtq"] - invtq_lag3) / atq).replace(
        [np.inf, -np.inf], np.nan
    )

    # ------------------------------------------------------------------
    # Quarterly change in taxes: ΔTXTQ / ATQ  (Thomas & Zhang 2011)
    # txtq = total income taxes; lag 3 monthly rows ≈ 1 quarter
    # ------------------------------------------------------------------
    txq_lag3 = grp["q_txtq"].shift(3)
    out["chtx"] = (
        (panel["q_txtq"].fillna(0.0) - txq_lag3.fillna(0.0)) / atq
    ).replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # Standardized unexpected earnings (SUE): (EPS - lag EPS) / std(EPS)
    # Use EPSPXQ.  In the monthly panel quarterly data repeats ~3 months,
    # so "same quarter last year" = shift(12) and "8 quarters" = rolling(24).
    # ------------------------------------------------------------------
    eps = panel["q_epspxq"]
    eps_lag4 = grp["q_epspxq"].shift(12)   # 4 quarters back ≈ 12 monthly rows
    eps_diff = eps - eps_lag4
    eps_std8 = grp["q_epspxq"].rolling(24, min_periods=12).std().reset_index(
        level=0, drop=True
    ).replace(0, np.nan)
    out["sue"] = (eps_diff / eps_std8).replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # Revenue surprise (RS): quarterly sales growth standardized
    # Same timing correction: shift(12) for year-over-year, rolling(24)
    # ------------------------------------------------------------------
    saleq = panel["q_saleq"]
    saleq_lag4 = grp["q_saleq"].shift(12)   # 4 quarters back
    saleq_diff = (saleq - saleq_lag4) / saleq_lag4.abs().replace(0, np.nan)
    saleq_std8 = grp["q_saleq"].rolling(24, min_periods=12).std().reset_index(
        level=0, drop=True
    ).replace(0, np.nan)
    out["rs"] = (saleq_diff / saleq_std8).replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # ROA volatility: std of quarterly ROA over past 8 quarters (~24 months)
    # ------------------------------------------------------------------
    roaq_q = panel["q_ibq"] / panel["q_atq"].replace(0, np.nan)
    panel_tmp = panel.copy()
    panel_tmp["_roaq_q"] = roaq_q.values
    out["roavol"] = (
        panel_tmp.groupby("permno", sort=False)["_roaq_q"]
        .rolling(24, min_periods=12)
        .std()
        .reset_index(level=0, drop=True)
    )

    # ------------------------------------------------------------------
    # Standard deviation of cash flow (OANCFY / ATQ), rolling 8 qtrs (~24 months)
    # ------------------------------------------------------------------
    cf_q = panel["q_oancfy"] / panel["q_atq"].replace(0, np.nan)
    panel_tmp["_cf_q"] = cf_q.values
    out["stdcf"] = (
        panel_tmp.groupby("permno", sort=False)["_cf_q"]
        .rolling(24, min_periods=12)
        .std()
        .reset_index(level=0, drop=True)
    )

    # ------------------------------------------------------------------
    # Mohanram's G-score (8 signals: positive ROA, cash flow > income,
    # accruals < median, ROA trend positive, cash flow trend positive,
    # advertising intensity > median, capital intensity > median,
    # R&D intensity > median). Simplified using available columns.
    # ------------------------------------------------------------------
    # Each signal is 1 or 0. Use .to_numpy(float, na_value=nan) to avoid
    # pandas nullable-NA boolean ambiguity.
    def _f(s: pd.Series) -> np.ndarray:
        return s.to_numpy(dtype=float, na_value=np.nan)

    roa_curr = _f(ib / at)
    roa_lag1 = _f(grp["a_ib"].shift(12) / at_lag12)
    cf_ann   = _f(panel["a_oancf"].fillna(0.0))
    ib_np    = _f(ib.fillna(0.0))
    at_np    = _f(at)

    g1 = np.where(np.isnan(roa_curr), 0.0, (roa_curr > 0).astype(float))
    g2 = np.where(np.isnan(cf_ann), 0.0, (cf_ann > ib_np).astype(float))
    g3_ok = ~(np.isnan(roa_curr) | np.isnan(roa_lag1))
    g3 = np.where(g3_ok, (roa_curr > roa_lag1).astype(float), 0.0)
    cf_roa = _f(panel["a_oancf"].fillna(0.0) / at)
    g4 = np.where(np.isnan(cf_roa), 0.0, (cf_roa > roa_curr).astype(float))

    # G-score signals 5-7: compare firm ratio to cross-sectional median of that ratio
    # (median of ratios, not ratio of medians)
    capx_at = _f(panel["a_capx"].fillna(0.0) / at)
    panel_ms = panel.copy()
    panel_ms["_capx_at"] = panel["a_capx"].fillna(0.0) / at
    capx_med = _f(panel_ms.groupby("date")["_capx_at"].transform("median"))
    g5 = np.where(np.isnan(capx_at) | np.isnan(capx_med), 0.0,
                  (capx_at > capx_med).astype(float))

    xrd_at = _f(xrd / at)
    panel_ms["_xrd_at"] = xrd / at
    xrd_med = _f(panel_ms.groupby("date")["_xrd_at"].transform("median"))
    g6 = np.where(np.isnan(xrd_at) | np.isnan(xrd_med), 0.0,
                  (xrd_at > xrd_med).astype(float))

    xsga_at = _f(panel["a_xsga"].fillna(0.0) / at)
    panel_ms["_xsga_at"] = panel["a_xsga"].fillna(0.0) / at
    xsga_med = _f(panel_ms.groupby("date")["_xsga_at"].transform("median"))
    g7 = np.where(np.isnan(xsga_at) | np.isnan(xsga_med), 0.0,
                  (xsga_at > xsga_med).astype(float))

    csho_curr = _f(panel["a_csho"])
    csho_lag  = _f(grp["a_csho"].shift(12))
    g8 = np.where(np.isnan(csho_curr) | np.isnan(csho_lag), 0.0,
                  (csho_curr <= csho_lag).astype(float))

    out["ms"] = g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8

    # ------------------------------------------------------------------
    # Depreciation / PP&E  (Holthausen & Larcker 1992)
    # ------------------------------------------------------------------
    ppent = panel["a_ppent"].replace(0, np.nan)
    dp = panel["a_dp"].fillna(0.0)
    depr_rate = (dp / ppent).replace([np.inf, -np.inf], np.nan)
    out["depr"] = depr_rate

    # ------------------------------------------------------------------
    # % change in DP/PPENT  (Holthausen & Larcker 1992)
    # ------------------------------------------------------------------
    depr_lag12 = (
        grp["a_dp"].shift(12).fillna(0.0)
        / grp["a_ppent"].shift(12).replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    out["pchdepr"] = ((depr_rate - depr_lag12) / depr_lag12.abs().replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )

    # ------------------------------------------------------------------
    # Return on net assets  (Soliman 2008)
    # NOA = AT - CHE - IVAO  (total assets minus financial assets)
    # rna = IB / avg(NOA_t, NOA_{t-12})
    # ------------------------------------------------------------------
    che = panel["a_che"].fillna(0.0)
    ivao = panel["a_ivao"].fillna(0.0)
    noa = panel["a_at"].fillna(0.0) - che - ivao
    noa_lag12 = (
        grp["a_at"].shift(12).fillna(0.0)
        - grp["a_che"].shift(12).fillna(0.0)
        - grp["a_ivao"].shift(12).fillna(0.0)
    )
    noa_avg = ((noa + noa_lag12) / 2.0).replace(0, np.nan)
    out["rna"] = (panel["a_ib"] / noa_avg).replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # Quarterly revenue surprise  (Kama 2009)
    # rsup = (SALEQ_t - SALEQ_{t-4}) / |ME|
    # where SALEQ_{t-4} is the same quarter last year (12 monthly lags)
    # ------------------------------------------------------------------
    saleq = panel["q_saleq"]
    saleq_lag4q = grp["q_saleq"].shift(12)   # 4 quarters ≈ 12 monthly lags
    out["rsup"] = ((saleq - saleq_lag4q) / me.abs().replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )

    # ------------------------------------------------------------------
    # Number of consecutive quarterly earnings increases  (Barth et al. 1999)
    # Compares each quarter's EPS to the same quarter last year (Y-o-Y).
    # Look back up to 8 quarters.
    # ------------------------------------------------------------------
    eps = panel["q_epspxq"]
    # Compute 8 Y-o-Y change indicators sampled at 3-monthly intervals
    inc_flags = []
    for q in range(8):
        lag_curr = grp["q_epspxq"].shift(q * 3)
        lag_year = grp["q_epspxq"].shift(q * 3 + 12)
        inc = (lag_curr - lag_year).fillna(0.0) > 0
        inc_flags.append(inc.astype(float))

    # Vectorised consecutive count: running product × contribution
    running = inc_flags[0].copy()
    nincr_val = running.copy()
    for i in range(1, 8):
        running = running * inc_flags[i]
        nincr_val = nincr_val + running
    out["nincr"] = nincr_val

    # ------------------------------------------------------------------
    # Piotroski F-score  (Piotroski 2000) — 9 binary profitability signals
    # ------------------------------------------------------------------
    at_p = panel["a_at"].replace(0, np.nan)
    at_lag_p = grp["a_at"].shift(12).replace(0, np.nan)
    ib_p = panel["a_ib"].fillna(0.0)
    oancf_p = panel["a_oancf"].fillna(0.0)
    dltt_p = panel["a_dltt"].fillna(0.0)

    roa_p = ib_p / at_lag_p
    cfo_p = oancf_p / at_p
    roa_lag_p = grp["a_ib"].shift(12).fillna(0.0) / grp["a_at"].shift(24).replace(0, np.nan)

    F1 = (roa_p > 0).astype(float)
    F2 = (cfo_p > 0).astype(float)
    F3 = (roa_p > roa_lag_p).astype(float)
    F4 = (cfo_p > roa_p).astype(float)   # accrual quality: CFO > ROA

    lev_p = dltt_p / at_p.fillna(np.nan)
    lev_lag_p = grp["a_dltt"].shift(12).fillna(0.0) / at_lag_p
    F5 = (lev_p < lev_lag_p).astype(float)  # leverage decreased

    lct_p = panel["a_lct"].replace(0, np.nan)
    curr_p = panel["a_act"] / lct_p
    curr_lag_p = grp["a_act"].shift(12) / grp["a_lct"].shift(12).replace(0, np.nan)
    F6 = (curr_p > curr_lag_p).astype(float)  # liquidity improved

    csho_p = panel["a_csho"].fillna(0.0)
    csho_lag_p = grp["a_csho"].shift(12).fillna(0.0)
    F7 = (csho_p <= csho_lag_p).astype(float)  # no dilution

    sale_p = panel["a_sale"].replace(0, np.nan)
    sale_lag_p = grp["a_sale"].shift(12).replace(0, np.nan)
    cogs_p = panel["a_cogs"].fillna(0.0)
    cogs_lag_p = grp["a_cogs"].shift(12).fillna(0.0)
    margin_p = (sale_p - cogs_p) / sale_p
    margin_lag_p = (sale_lag_p - cogs_lag_p) / sale_lag_p
    F8 = (margin_p > margin_lag_p).astype(float)

    ato_p = panel["a_sale"] / at_p
    ato_lag_p = grp["a_sale"].shift(12) / at_lag_p
    F9 = (ato_p > ato_lag_p).astype(float)

    for f in [F1, F2, F3, F4, F5, F6, F7, F8, F9]:
        f.fillna(0.0, inplace=True)
    out["ps"] = F1 + F2 + F3 + F4 + F5 + F6 + F7 + F8 + F9

    # ------------------------------------------------------------------
    # Organisational capital  (Eisfeldt & Papanikolaou 2013)
    # Perpetual inventory of SG&A using δ = 0.15, initial = XSGA / 0.25
    # orgcap = OC_t / AT_t   (nominal, cross-sectionally rank-normalised)
    # OC_t ≈ Σ_{k=0}^{14} 0.85^k × XSGA_{t - k×12}
    # ------------------------------------------------------------------
    xsga = panel["a_xsga"].fillna(0.0)
    oc = xsga.copy()   # k=0 contribution (0.85^0 = 1)
    for k in range(1, 15):
        lag_xsga = grp["a_xsga"].shift(k * 12).fillna(0.0)
        oc = oc + (0.85 ** k) * lag_xsga
    out["orgcap"] = (oc / at.fillna(np.nan)).replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # Earnings announcement return  (Kishore, Brockman, Altieri & Bethke 2008)
    # 3-day cumulative return centred on the quarterly earnings announcement date.
    # Requires CRSP daily files — gracefully returns NaN if files are absent.
    # ------------------------------------------------------------------
    log.info("Computing earnings announcement return (ear) from daily CRSP data …")
    out["ear"] = _compute_ear(panel).values

    log.info("Profitability features computed: %d rows, %d features",
             len(out), out.shape[1] - 2)
    return out
