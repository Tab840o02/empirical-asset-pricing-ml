"""
src/features/trading_friction_features.py
===========================================
Market-microstructure and risk characteristics from GKX (2020) Table A.1.

All features are derived from the CRSP daily file.  A fully-vectorised
``pandas.groupby.rolling`` approach is used — no Python loops per stock.

Features implemented
--------------------
beta        – CAPM beta from rolling 252-day regression vs. vwretd
betasq      – Beta squared
idiovol     – Idiosyncratic return vol (std of CAPM residuals, 252 days)
retvol      – Std-dev of daily returns, past 21 days
maxret      – Maximum daily return, past 21 days
ill         – Amihud (2002) ILLIQ: mean(|ret|/dollar_vol), 252 days
zerotrade   – Fraction of zero-volume days, past 21 days
me          – Log market equity (from monthly panel)
age         – Years since first CRSP appearance (from monthly panel)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import RAW_DIR, BETA_WINDOW_DAYS, RETVOL_WINDOW_DAYS

log = logging.getLogger(__name__)

_SHORT_WIN = RETVOL_WINDOW_DAYS    # 21
_LONG_WIN = BETA_WINDOW_DAYS       # 252
_MIN_SHORT = 10
_MIN_LONG = 120


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_all_daily(years: list[int]) -> pd.DataFrame:
    parts = []
    for yr in years:
        path = RAW_DIR / f"crsp_daily_{yr}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["permno", "date", "ret", "vol", "prc"])
        df["permno"] = df["permno"].astype("int32")
        for col in ["ret", "vol", "prc"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
        parts.append(df)
    if not parts:
        raise FileNotFoundError("No crsp_daily_*.parquet files found in data/raw/")
    return pd.concat(parts, ignore_index=True)


def _load_market_index() -> pd.DataFrame:
    path = RAW_DIR / "crsp_daily_index.parquet"
    df = pd.read_parquet(path, columns=["date", "vwretd"])
    df["vwretd"] = df["vwretd"].astype("float32")
    return df.dropna()


# ---------------------------------------------------------------------------
# Rolling computations (vectorised)
# ---------------------------------------------------------------------------

def _compute_long_window_features(daily: pd.DataFrame) -> pd.DataFrame:
    """252-day rolling beta, betasq, idiovol, ill via groupby.rolling."""
    daily = daily.sort_values(["permno", "date"]).reset_index(drop=True)
    grp = daily.groupby("permno", sort=False)

    daily["_r_m"] = (daily["ret"] * daily["vwretd"]).astype("float32")
    daily["_m2"]  = (daily["vwretd"] ** 2).astype("float32")

    log.info("  252-day rolling means (beta) ...")
    roll = grp.rolling(_LONG_WIN, min_periods=_MIN_LONG)
    mean_r  = roll["ret"].mean().reset_index(level=0, drop=True).astype("float32")
    mean_m  = roll["vwretd"].mean().reset_index(level=0, drop=True).astype("float32")
    mean_rm = roll["_r_m"].mean().reset_index(level=0, drop=True).astype("float32")
    mean_m2 = roll["_m2"].mean().reset_index(level=0, drop=True).astype("float32")

    cov_rm = mean_rm - mean_r * mean_m
    var_m  = mean_m2 - mean_m ** 2

    daily["beta"]   = (cov_rm / var_m.replace(0.0, np.nan)).astype("float32")
    daily["betasq"] = (daily["beta"] ** 2).astype("float32")

    log.info("  idiovol ...")
    resid = (daily["ret"] - daily["beta"] * daily["vwretd"]).astype("float32")
    daily["_resid"]  = resid
    daily["_resid2"] = (resid ** 2).astype("float32")

    roll2    = daily.groupby("permno", sort=False).rolling(_LONG_WIN, min_periods=_MIN_LONG)
    mean_e   = roll2["_resid"].mean().reset_index(level=0, drop=True)
    mean_e2  = roll2["_resid2"].mean().reset_index(level=0, drop=True)
    daily["idiovol"] = np.sqrt((mean_e2 - mean_e ** 2).clip(lower=0.0)).astype("float32")

    log.info("  Amihud ILLIQ ...")
    dvol = (daily["prc"].abs() * daily["vol"]).astype("float32")
    daily["_ill"] = (daily["ret"].abs() / dvol.replace(0.0, np.nan)).astype("float32")
    daily["ill"] = (
        daily.groupby("permno", sort=False)["_ill"]
        .rolling(_LONG_WIN, min_periods=_MIN_LONG)
        .mean()
        .reset_index(level=0, drop=True)
        .astype("float32")
    )

    daily = daily.drop(columns=["_r_m", "_m2", "_resid", "_resid2", "_ill"],
                       errors="ignore")
    return daily


def _compute_short_window_features(daily: pd.DataFrame) -> pd.DataFrame:
    """21-day rolling retvol, maxret, zerotrade."""
    log.info("  21-day rolling features ...")
    grp = daily.groupby("permno", sort=False)
    roll_s = grp.rolling(_SHORT_WIN, min_periods=_MIN_SHORT)

    daily["retvol"]    = roll_s["ret"].std().reset_index(level=0, drop=True).astype("float32")
    daily["maxret"]    = roll_s["ret"].max().reset_index(level=0, drop=True).astype("float32")
    daily["_is_zero"]  = (daily["vol"] == 0).astype("float32")
    daily["zerotrade"] = (grp["_is_zero"]
                          .rolling(_SHORT_WIN, min_periods=_MIN_SHORT)
                          .mean()
                          .reset_index(level=0, drop=True)
                          .astype("float32"))
    daily = daily.drop(columns=["_is_zero"], errors="ignore")
    return daily


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily-based trading friction features for every (permno, month).

    Parameters
    ----------
    panel : pd.DataFrame
        Merged monthly panel.  Provides year range, me and age.

    Returns
    -------
    pd.DataFrame  Columns: permno, date (month-end Timestamp),
                  beta, betasq, idiovol, retvol, maxret, ill, zerotrade,
                  me, age.
    """
    panel_years = sorted(panel["date"].dt.year.unique())
    load_years  = list(range(min(panel_years) - 1, max(panel_years) + 1))

    log.info("Loading %d years of daily data ...", len(load_years))
    daily = _load_all_daily(load_years)
    mkt   = _load_market_index()
    daily = daily.merge(mkt, on="date", how="left")

    log.info("Daily data: %s rows, %d permnos",
             f"{len(daily):,}", daily["permno"].nunique())

    daily = _compute_long_window_features(daily)
    daily = _compute_short_window_features(daily)

    # Resample to monthly: last value per (permno, ym)
    log.info("Resampling daily → monthly ...")
    daily["ym"] = daily["date"].dt.to_period("M")

    feat_cols = ["beta", "betasq", "idiovol", "retvol", "maxret", "ill", "zerotrade"]
    monthly = (
        daily[["permno", "date", "ym"] + feat_cols]
        .sort_values(["permno", "date"])
        .groupby(["permno", "ym"])
        .last()
        .reset_index()
    )
    monthly["date"] = monthly["ym"].dt.to_timestamp("M")
    monthly = monthly.drop(columns=["ym"])

    # Winsorise ILLIQ at 99th percentile per month
    p99 = monthly.groupby("date")["ill"].transform(lambda x: x.quantile(0.99))
    monthly["ill"] = monthly["ill"].clip(upper=p99)

    # Add me (log) and age from the monthly panel
    panel_slim = panel[["permno", "date", "me"]].copy()
    panel_slim["date"] = panel_slim["date"].dt.to_period("M").dt.to_timestamp("M")
    panel_slim["me"]   = np.log(panel_slim["me"].clip(lower=1e-6))

    first_date = panel.groupby("permno")["date"].min().rename("first_date")
    panel_slim = panel_slim.merge(first_date, on="permno", how="left")
    panel_slim["age"] = (panel_slim["date"] - panel_slim["first_date"]).dt.days / 365.25
    panel_slim = panel_slim.drop(columns=["first_date"])

    monthly = monthly.merge(panel_slim, on=["permno", "date"], how="left")

    log.info("Trading friction features done: %s rows, %d permnos",
             f"{len(monthly):,}", monthly["permno"].nunique())
    return monthly
