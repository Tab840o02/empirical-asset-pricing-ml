"""
src/evaluation/portfolio.py
============================
Long-short decile portfolio construction and performance evaluation.

All portfolio returns are value-weighted using lagged market cap (me_lag1)
from crsp_clean.parquet, which avoids using contemporaneous size in weights.

The long-short (LS) portfolio is Long P10 (top decile) – Short P1 (bottom
decile) of predicted return, rebalanced monthly.

Functions
---------
build_portfolios     → monthly decile returns for each model
ls_returns           → long P10, short P1, and LS returns
ff_alpha             → FF3 or FF5 alpha/t-stat via OLS
performance_table    → full model comparison table

Usage
-----
    import pandas as pd
    from src.evaluation.portfolio import build_portfolios, ls_returns, performance_table
    from src.config import PREDICTIONS_PATH, PROCESSED_DIR, RAW_DIR

    preds  = pd.read_parquet(PREDICTIONS_PATH)
    crsp   = pd.read_parquet(PROCESSED_DIR / "crsp_clean.parquet")[["permno","date","me_lag1"]]
    ff     = pd.read_parquet(RAW_DIR / "ff_factors.parquet").reset_index()
    panel  = pd.read_parquet(PROCESSED_DIR / "features_panel.parquet")[["permno","date","ret_exc"]]

    preds = preds.merge(panel, on=["permno","date"], how="left")
    deciles = build_portfolios(preds, crsp)
    ls = ls_returns(deciles)
    print(performance_table(ls, ff))
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

log = logging.getLogger(__name__)

N_DECILES = 10


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------

def _assign_deciles(grp: pd.DataFrame) -> pd.Series:
    """
    Assign decile labels 1–10 within a (model, date) group.
    Ties are handled with 'first' to ensure equal bin sizes where possible.
    """
    return pd.qcut(
        grp["pred_ret"].rank(method="first"),
        q=N_DECILES,
        labels=range(1, N_DECILES + 1),
    ).astype(int)


def build_portfolios(
    preds_df: pd.DataFrame,
    crsp_me: pd.DataFrame,
    n_deciles: int = N_DECILES,
) -> pd.DataFrame:
    """
    Build monthly value-weighted decile portfolio returns.

    Parameters
    ----------
    preds_df : DataFrame with columns permno, date, model, pred_ret, ret_exc
    crsp_me  : DataFrame with columns permno, date, me_lag1
                (lagged market cap — use from crsp_clean.parquet)
    n_deciles : int — number of portfolios (default 10)

    Returns
    -------
    pd.DataFrame with columns: model, date, decile, port_ret, n_stocks, total_me
    """
    global N_DECILES
    N_DECILES = n_deciles

    # Merge in lagged market cap
    df = preds_df.merge(crsp_me[["permno", "date", "me_lag1"]], on=["permno", "date"], how="left")

    # Drop stocks with missing market cap (can't value-weight)
    n_before = len(df)
    df = df[df["me_lag1"].notna() & (df["me_lag1"] > 0)]
    if len(df) < n_before:
        log.info(f"  Dropped {n_before - len(df):,} rows with missing/zero me_lag1")

    # Assign deciles within each (model, date)
    log.info("Assigning decile ranks…")
    df["decile"] = (
        df.groupby(["model", "date"])
        .apply(lambda g: _assign_deciles(g), include_groups=False)
        .explode()
        .values
    )

    # Value-weighted return within each (model, date, decile)
    def vw_ret(grp: pd.DataFrame) -> float:
        w = grp["me_lag1"] / grp["me_lag1"].sum()
        return float((w * grp["ret_exc"]).sum())

    log.info("Computing value-weighted portfolio returns…")
    port = (
        df.groupby(["model", "date", "decile"])
        .apply(
            lambda g: pd.Series({
                "port_ret": vw_ret(g),
                "n_stocks": len(g),
                "total_me": float(g["me_lag1"].sum()),
            }),
            include_groups=False,
        )
        .reset_index()
    )

    return port.sort_values(["model", "date", "decile"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Long-short returns
# ---------------------------------------------------------------------------

def ls_returns(decile_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct long P10, short P1, and long-short (L/S) portfolio returns.

    Parameters
    ----------
    decile_df : output of build_portfolios()

    Returns
    -------
    pd.DataFrame with columns: model, date, long_ret, short_ret, ls_ret
    """
    p1 = decile_df[decile_df["decile"] == 1][["model", "date", "port_ret"]].rename(
        columns={"port_ret": "short_ret"}
    )
    p10 = decile_df[decile_df["decile"] == N_DECILES][["model", "date", "port_ret"]].rename(
        columns={"port_ret": "long_ret"}
    )
    ls = p1.merge(p10, on=["model", "date"], how="inner")
    ls["ls_ret"] = ls["long_ret"] - ls["short_ret"]
    return ls.sort_values(["model", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Factor model alpha
# ---------------------------------------------------------------------------

def ff_alpha(
    ls_series: pd.Series,
    ff_df: pd.DataFrame,
    model: str = "ff5",
) -> dict:
    """
    Regress monthly L/S returns on Fama-French factors to compute alpha.

    Parameters
    ----------
    ls_series : pd.Series indexed by date — monthly L/S excess returns (in %)
    ff_df     : pd.DataFrame — FF factors (columns: Mkt-RF, SMB, HML, RMW, CMA, RF)
                index or 'date' column of month-end Timestamps
    model     : 'ff3' or 'ff5'

    Returns
    -------
    dict with keys: alpha, t_alpha, p_alpha, sharpe, annual_ret, annual_vol,
                    r2_adj, factor_betas
    """
    if "date" not in ff_df.columns:
        ff_df = ff_df.reset_index().rename(columns={"index": "date"})

    ls_df = ls_series.rename("ls_ret").reset_index()
    ls_df.columns = ["date", "ls_ret"]

    merged = ls_df.merge(ff_df, on="date", how="inner")
    if len(merged) < 12:
        log.warning("Fewer than 12 observations for FF regression — results may be unreliable.")

    y = merged["ls_ret"].values
    if model == "ff3":
        factors = ["Mkt-RF", "SMB", "HML"]
    elif model == "ff5":
        factors = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
        # Fallback to FF3 if RMW/CMA not available (early dates)
        if merged["RMW"].isna().all():
            log.warning("RMW/CMA are all NaN — falling back to FF3.")
            factors = ["Mkt-RF", "SMB", "HML"]
    else:
        raise ValueError(f"Unknown factor model: {model!r}. Use 'ff3' or 'ff5'.")

    X = sm.add_constant(merged[factors].values)
    res = sm.OLS(y, X).fit(cov_type="HC3")

    alpha_monthly = res.params[0]
    t_alpha = res.tvalues[0]
    p_alpha = res.pvalues[0]
    r2_adj = res.rsquared_adj

    annual_ret = float(y.mean() * 12)
    annual_vol = float(y.std(ddof=1) * np.sqrt(12))
    sharpe = annual_ret / annual_vol if annual_vol > 0 else np.nan

    return {
        "alpha_monthly": round(alpha_monthly, 6),
        "alpha_annual": round(alpha_monthly * 12, 6),
        "t_alpha": round(t_alpha, 3),
        "p_alpha": round(p_alpha, 4),
        "sharpe": round(sharpe, 3),
        "annual_ret": round(annual_ret, 6),
        "annual_vol": round(annual_vol, 6),
        "r2_adj": round(r2_adj, 4),
        "n_months": len(merged),
        "factor_betas": dict(zip(factors, res.params[1:].round(4).tolist())),
    }


# ---------------------------------------------------------------------------
# Summary performance table
# ---------------------------------------------------------------------------

def performance_table(
    ls_df: pd.DataFrame,
    ff_df: pd.DataFrame,
    factor_model: str = "ff5",
) -> pd.DataFrame:
    """
    Produce a model-level performance summary for the L/S portfolio.

    Parameters
    ----------
    ls_df        : output of ls_returns()
    ff_df        : FF factors DataFrame (from ff_factors.parquet)
    factor_model : 'ff3' or 'ff5'

    Returns
    -------
    pd.DataFrame indexed by model name with columns:
        annual_ret, annual_vol, sharpe,
        alpha_annual, t_alpha, p_alpha, r2_adj, n_months
    """
    rows = []
    for model, grp in ls_df.groupby("model"):
        series = grp.set_index("date")["ls_ret"]
        stats = ff_alpha(series, ff_df, model=factor_model)
        rows.append({
            "model": model,
            "annual_ret": stats["annual_ret"],
            "annual_vol": stats["annual_vol"],
            "sharpe": stats["sharpe"],
            "alpha_annual": stats["alpha_annual"],
            "t_alpha": stats["t_alpha"],
            "p_alpha": stats["p_alpha"],
            "r2_adj": stats["r2_adj"],
            "n_months": stats["n_months"],
        })

    return (
        pd.DataFrame(rows)
        .set_index("model")
        .sort_index()
    )
