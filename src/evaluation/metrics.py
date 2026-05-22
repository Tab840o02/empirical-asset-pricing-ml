"""
src/evaluation/metrics.py
==========================
Prediction-quality metrics for GKX (2020) replication.

All functions operate on a `preds_df` DataFrame with schema:
    permno    int64
    date      datetime64[ns]  (month-end timestamps)
    model     str
    pred_ret  float32         (predicted excess return)
    ret_exc   float32         (realised excess return) — merged from panel

Usage
-----
    from src.evaluation.metrics import summary_table
    preds = pd.read_parquet("data/processed/predictions.parquet")
    panel = pd.read_parquet("data/processed/features_panel.parquet")[["permno","date","ret_exc"]]
    df = preds.merge(panel, on=["permno","date"], how="left")
    print(summary_table(df, r_bench))
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core OOS R²
# ---------------------------------------------------------------------------

def oos_r2_pooled(preds_df: pd.DataFrame, r_bench: float) -> pd.Series:
    """
    Pooled OOS R² (Campbell & Thompson 2008 / GKX eq. 1) for each model.

    R²_OOS = 1 − Σ(r_it − ĝ_it)² / Σ(r_it − r̄)²

    Parameters
    ----------
    preds_df : DataFrame with columns permno, date, model, pred_ret, ret_exc
    r_bench  : scalar — historical training-window mean of ret_exc

    Returns
    -------
    pd.Series indexed by model name
    """
    results = {}
    for model, grp in preds_df.groupby("model"):
        r = grp["ret_exc"].values
        g = grp["pred_ret"].values
        ss_res = float(np.sum((r - g) ** 2))
        ss_tot = float(np.sum((r - r_bench) ** 2))
        results[model] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return pd.Series(results, name="oos_r2").sort_index()


def oos_r2_by_year(preds_df: pd.DataFrame, r_bench: float) -> pd.DataFrame:
    """
    OOS R² computed separately for each calendar year.

    Returns
    -------
    pd.DataFrame with columns: year, model, oos_r2
    """
    df = preds_df.copy()
    df["year"] = df["date"].dt.year
    rows = []
    for (model, year), grp in df.groupby(["model", "year"]):
        r = grp["ret_exc"].values
        g = grp["pred_ret"].values
        ss_res = float(np.sum((r - g) ** 2))
        ss_tot = float(np.sum((r - r_bench) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rows.append({"year": year, "model": model, "oos_r2": r2})
    return pd.DataFrame(rows).sort_values(["model", "year"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Monthly information coefficient (rank correlation)
# ---------------------------------------------------------------------------

def monthly_ic(preds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly Spearman rank correlation between pred_ret and ret_exc,
    computed separately per model and calendar month.

    Returns
    -------
    pd.DataFrame with columns: date, model, ic
    """
    rows = []
    for (model, date), grp in preds_df.groupby(["model", "date"]):
        if len(grp) < 5:
            continue
        corr, _ = spearmanr(grp["pred_ret"].values, grp["ret_exc"].values)
        rows.append({"date": date, "model": model, "ic": corr})
    return pd.DataFrame(rows).sort_values(["model", "date"]).reset_index(drop=True)


def ic_stats(ic_df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean IC, IC standard deviation, and ICIR (mean / std) per model.

    Parameters
    ----------
    ic_df : output of monthly_ic()

    Returns
    -------
    pd.DataFrame indexed by model with columns: mean_ic, std_ic, icir
    """
    out = (
        ic_df.groupby("model")["ic"]
        .agg(mean_ic="mean", std_ic="std")
        .assign(icir=lambda d: d["mean_ic"] / d["std_ic"])
    )
    return out


# ---------------------------------------------------------------------------
# Summary table (combines pooled R² and IC stats)
# ---------------------------------------------------------------------------

def summary_table(preds_df: pd.DataFrame, r_bench: float) -> pd.DataFrame:
    """
    Produce a model-level summary table combining OOS R² and IC statistics.

    Parameters
    ----------
    preds_df : DataFrame with columns permno, date, model, pred_ret, ret_exc
    r_bench  : scalar — historical training-window mean of ret_exc

    Returns
    -------
    pd.DataFrame indexed by model name with columns:
        oos_r2, mean_ic, std_ic, icir
    """
    r2 = oos_r2_pooled(preds_df, r_bench)

    ic_df = monthly_ic(preds_df)
    ic_s = ic_stats(ic_df)

    summary = r2.to_frame().join(ic_s, how="left")
    return summary.sort_index()
