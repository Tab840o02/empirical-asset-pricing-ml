"""
src/extensions/post2020_eval.py
===============================
Phase 5a — Post-2020 out-of-sample evaluation.

This module evaluates the extension predictions over the post-2016 test
window, splitting the sample into macro sub-periods and reporting:

* pooled OOS R^2
* monthly information coefficient statistics
* long-short decile portfolio performance
* FF5 alpha with Newey-West standard errors (12 lags, matching Phase 4)

Degenerate model handling
-------------------------
nn4 and nn5 produce a single constant prediction value for every stock in
every month of the 2017-2024 extension window (zero cross-sectional
variance throughout).  nn3 collapses to a constant in all of 2020, 2022,
and 2024.  Constant predictions cannot produce valid IC, OOS R², or
meaningful portfolio deciles (decile assignment becomes arbitrary row-order
tiebreaking).  Consequently:
  * nn4 and nn5 are excluded from all Phase 5a outputs (FULLY_EXCLUDED).
  * Any (model, month) pair with zero cross-sectional prediction variance
    is filtered out before metric computation; affected (model, period)
    combinations appear as NaN in the output CSVs.

Run with:
    python -m src.extensions.post2020_eval
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CRSP_CLEAN_PATH,
    EVAL_EXT_IC_PATH,
    EVAL_EXT_OOS_R2_PATH,
    EVAL_EXT_PORT_PATH,
    FEATURES_PANEL_PATH,
    EXT_END,
    PREDICTIONS_EXT_PATH,
    PROCESSED_DIR,
    RAW_DIR,
    RUN_MANIFEST_EXT_PATH,
)
from src.evaluation.metrics import ic_stats, monthly_ic, oos_r2_pooled
from src.evaluation.portfolio import build_decile_portfolios, ff_alpha, ls_returns

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Benchmark: historical mean of ret_exc for all months before 2017-01-01.
# Loaded from the extension manifest so the value stays in sync with the
# actual training data; falls back to the value computed on 2026-05-26.
# ---------------------------------------------------------------------------

def _load_bench_from_manifest() -> float:
    if RUN_MANIFEST_EXT_PATH.exists():
        try:
            with open(RUN_MANIFEST_EXT_PATH, encoding="utf-8") as _f:
                _m = json.load(_f)
            _bench = _m.get("pre_test_mean_ret")
            if _bench is not None:
                return float(_bench)
        except Exception:  # noqa: BLE001
            pass
    # Fallback: value computed from features_panel.parquet for dates < 2017-01-01
    return 0.008027891628444195


R_BENCH_EXT: float = _load_bench_from_manifest()
NW_LAGS_EXT: int = 12
MODEL_ORDER: tuple[str, ...] = (
    "ols3",
    "ols_all",
    "pcr",
    "pls",
    "enet",
    "glm",
    "rf",
    "gbrt",
    "nn1",
    "nn2",
    "nn3",
    "nn4",
    "nn5",
)

# ---------------------------------------------------------------------------
# Degenerate model exclusion (Audit C-1 / C-2)
# ---------------------------------------------------------------------------

# Models that produce constant predictions across the entire extension window.
# Their IC, OOS R², and portfolio statistics are all invalid and must be
# excluded from every Phase 5a output table.
FULLY_EXCLUDED: frozenset[str] = frozenset({"nn4", "nn5"})


def _zero_variance_months(preds: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame of (model, date) pairs where pred_ret has zero
    cross-sectional variance.  Constant predictions produce undefined IC
    and arbitrary decile rankings.
    """
    std = (
        preds.groupby(["model", "date"])["pred_ret"]
        .std()
        .rename("pred_std")
        .reset_index()
    )
    return std.loc[std["pred_std"] == 0.0, ["model", "date"]].copy()


def _filter_degenerate_predictions(preds: pd.DataFrame) -> pd.DataFrame:
    """
    Remove prediction rows that cannot support valid metric computation:
      1. All rows for models in FULLY_EXCLUDED.
      2. Any (model, month) pair with zero cross-sectional prediction variance.

    Logs a summary of every exclusion so the audit trail is visible.
    """
    # Step 1: fully excluded models
    mask_excl = preds["model"].isin(FULLY_EXCLUDED)
    if mask_excl.any():
        log.warning(
            "Phase 5a: removing %d rows for fully-excluded models %s (constant "
            "predictions throughout entire extension window — model collapse).",
            int(mask_excl.sum()),
            sorted(FULLY_EXCLUDED),
        )
        preds = preds.loc[~mask_excl].copy()

    # Step 2: zero-variance (model, month) pairs for remaining models
    zero_var = _zero_variance_months(preds)
    if not zero_var.empty:
        n_pairs = len(zero_var)
        affected = sorted(zero_var["model"].unique().tolist())
        log.warning(
            "Phase 5a: removing %d (model, month) prediction-pairs with zero "
            "cross-sectional variance for models %s.",
            n_pairs,
            affected,
        )
        preds = preds.merge(
            zero_var.assign(_drop=True), on=["model", "date"], how="left"
        )
        preds = preds.loc[preds["_drop"].isna()].drop(columns=["_drop"]).copy()

    return preds

SUB_PERIODS: dict[str, tuple[str, pd.Timestamp | None]] = {
    "Pre-COVID (2017-2019)": ("2017-01-01", pd.Timestamp("2019-12-31")),
    "COVID (2020)": ("2020-01-01", pd.Timestamp("2020-12-31")),
    "Reflation (2021)": ("2021-01-01", pd.Timestamp("2021-12-31")),
    "Rate hikes (2022)": ("2022-01-01", pd.Timestamp("2022-12-31")),
    "Post-norm (2023+)": ("2023-01-01", EXT_END),
    "Full ext (2017+)": ("2017-01-01", EXT_END),
}


def resolve_ext_end(features_path: Path = FEATURES_PANEL_PATH, crsp_path: Path = CRSP_CLEAN_PATH) -> pd.Timestamp:
    """Resolve the last complete month available for Phase 5a."""
    if not features_path.exists():
        raise FileNotFoundError(f"Missing features panel: {features_path}")
    if not crsp_path.exists():
        raise FileNotFoundError(f"Missing CRSP clean file: {crsp_path}")

    features_end = pd.Timestamp(pd.read_parquet(features_path, columns=["date"])["date"].max())
    crsp_end = pd.Timestamp(pd.read_parquet(crsp_path, columns=["date"])["date"].max())
    crsp_end = crsp_end.to_period("M").to_timestamp("M")
    features_end = features_end.to_period("M").to_timestamp("M")

    ext_end = min(features_end, crsp_end)
    log.info("Resolved Phase 5a end date: features=%s crsp=%s -> ext_end=%s", features_end.date(), crsp_end.date(), ext_end.date())
    return ext_end


def _materialize_subperiods(ext_end: pd.Timestamp) -> dict[str, tuple[str, pd.Timestamp]]:
    resolved: dict[str, tuple[str, pd.Timestamp]] = {}
    for period, (start, end) in SUB_PERIODS.items():
        resolved[period] = (start, ext_end if end is None else pd.Timestamp(end))
    return resolved


def _load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_EXT_PATH.exists():
        raise FileNotFoundError(f"Missing extension predictions: {PREDICTIONS_EXT_PATH}")
    preds = pd.read_parquet(PREDICTIONS_EXT_PATH)
    preds["date"] = pd.to_datetime(preds["date"]).dt.to_period("M").dt.to_timestamp("M")
    return preds


def _load_actual_returns() -> pd.DataFrame:
    if not FEATURES_PANEL_PATH.exists():
        raise FileNotFoundError(f"Missing features panel: {FEATURES_PANEL_PATH}")
    actual = pd.read_parquet(FEATURES_PANEL_PATH, columns=["permno", "date", "ret_exc"])
    actual["date"] = pd.to_datetime(actual["date"]).dt.to_period("M").dt.to_timestamp("M")
    return actual


def _load_crsp_me() -> pd.DataFrame:
    if not CRSP_CLEAN_PATH.exists():
        raise FileNotFoundError(f"Missing CRSP clean file: {CRSP_CLEAN_PATH}")
    crsp_me = pd.read_parquet(CRSP_CLEAN_PATH, columns=["permno", "date", "me_lag1"])
    crsp_me["date"] = pd.to_datetime(crsp_me["date"]).dt.to_period("M").dt.to_timestamp("M")
    return crsp_me


def _load_ff_factors() -> pd.DataFrame:
    ff_path = RAW_DIR / "ff_factors.parquet"
    if not ff_path.exists():
        raise FileNotFoundError(f"Missing FF factors file: {ff_path}")
    ff = pd.read_parquet(ff_path).reset_index().rename(columns={"index": "date"})
    ff["date"] = pd.to_datetime(ff["date"]).dt.to_period("M").dt.to_timestamp("M")
    return ff


def _scope_frame(df: pd.DataFrame, start: str, end: pd.Timestamp) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    return df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].copy()


def _check_group_coverage(preds: pd.DataFrame, ext_end: pd.Timestamp, start: str) -> None:
    expected_months = pd.period_range(pd.Timestamp(start), ext_end, freq="M").to_timestamp("M")
    scoped = preds[(preds["date"] >= pd.Timestamp(start)) & (preds["date"] <= ext_end)]
    missing_models = []
    for model, grp in scoped.groupby("model"):
        if set(grp["date"].drop_duplicates()) != set(expected_months):
            missing_models.append(model)
    if missing_models:
        raise ValueError(f"Incomplete predictions for models: {missing_models}")


def _compute_period_metrics(
    period: str,
    start: str,
    end: pd.Timestamp,
    preds: pd.DataFrame,
    crsp_me: pd.DataFrame,
    ff: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scoped_preds = _scope_frame(preds, start, end)
    scoped_crsp = _scope_frame(crsp_me, start, end)

    r2 = (
        oos_r2_pooled(scoped_preds, R_BENCH_EXT)
        .rename("oos_r2")
        .reset_index()
        .rename(columns={"index": "model"})
    )
    r2.insert(1, "period", period)

    ic_df = monthly_ic(scoped_preds)
    ic = ic_stats(ic_df).reset_index().rename(columns={"index": "model"})
    ic.insert(1, "period", period)

    deciles = build_decile_portfolios(scoped_preds, scoped_crsp)
    ls = ls_returns(deciles)

    rows = []
    for model, grp in ls.groupby("model"):
        stats = ff_alpha(grp.set_index("date")["ls_ret"], ff, model="ff5", nw_lags=NW_LAGS_EXT)
        rows.append(
            {
                "model": model,
                "period": period,
                "annual_ret": stats["annual_ret"],
                "sharpe": stats["sharpe"],
                "alpha_annual": stats["alpha_annual"],
                "t_alpha": stats["t_alpha"],
                "p_alpha": stats["p_alpha"],
            }
        )
    port = pd.DataFrame(rows)

    model_index = {model: idx for idx, model in enumerate(MODEL_ORDER)}
    for frame in (r2, ic, port):
        frame["model"] = pd.Categorical(frame["model"], categories=MODEL_ORDER, ordered=True)
        frame.sort_values(["period", "model"], inplace=True)
        frame["model"] = frame["model"].astype(str)
        frame.reset_index(drop=True, inplace=True)

    r2["model"] = r2["model"].astype(str)
    ic["model"] = ic["model"].astype(str)
    port["model"] = port["model"].astype(str)

    return r2, ic, port


def run_analysis(ext_end: pd.Timestamp | None = None) -> dict[str, pd.DataFrame]:
    """Run the Phase 5a analysis and persist the metric CSVs."""
    if ext_end is None:
        ext_end = resolve_ext_end()

    preds = _load_predictions()
    actual = _load_actual_returns()
    crsp_me = _load_crsp_me()
    ff = _load_ff_factors()

    preds = preds.merge(actual, on=["permno", "date"], how="left", validate="many_to_one")
    if preds["ret_exc"].isna().any():
        missing = int(preds["ret_exc"].isna().sum())
        raise ValueError(f"{missing} extension predictions are missing realised excess returns.")

    sub_periods = _materialize_subperiods(ext_end)

    # Remove degenerate predictions before any metric computation.
    preds = _filter_degenerate_predictions(preds)

    r2_frames: list[pd.DataFrame] = []
    ic_frames: list[pd.DataFrame] = []
    port_frames: list[pd.DataFrame] = []

    for period, (start, end) in sub_periods.items():
        log.info("Evaluating %s (%s → %s)", period, start, pd.Timestamp(end).date())
        r2, ic, port = _compute_period_metrics(period, start, pd.Timestamp(end), preds, crsp_me, ff)
        r2_frames.append(r2)
        ic_frames.append(ic)
        port_frames.append(port)

    r2_out = pd.concat(r2_frames, ignore_index=True)[["model", "period", "oos_r2"]]
    ic_out = pd.concat(ic_frames, ignore_index=True)[["model", "period", "mean_ic", "std_ic", "icir"]]
    port_out = pd.concat(port_frames, ignore_index=True)[
        ["model", "period", "annual_ret", "sharpe", "alpha_annual", "t_alpha", "p_alpha"]
    ]

    # Re-index to ensure every (model, period) combination appears in the
    # output.  Models excluded from computation are present as NaN rows so
    # downstream readers can see which models were excluded and why.
    all_periods = list(sub_periods.keys())
    full_index = pd.MultiIndex.from_product(
        [MODEL_ORDER, all_periods], names=["model", "period"]
    ).to_frame(index=False)

    r2_out = full_index.merge(r2_out, on=["model", "period"], how="left")
    ic_out = full_index.merge(ic_out, on=["model", "period"], how="left")
    port_out = full_index.merge(port_out, on=["model", "period"], how="left")

    # Restore canonical model ordering.
    for frame in (r2_out, ic_out, port_out):
        frame["model"] = pd.Categorical(frame["model"], categories=MODEL_ORDER, ordered=True)
        frame.sort_values(["period", "model"], inplace=True)
        frame["model"] = frame["model"].astype(str)
        frame.reset_index(drop=True, inplace=True)

    r2_out.to_csv(EVAL_EXT_OOS_R2_PATH, index=False)
    ic_out.to_csv(EVAL_EXT_IC_PATH, index=False)
    port_out.to_csv(EVAL_EXT_PORT_PATH, index=False)

    log.info("Wrote %s", EVAL_EXT_OOS_R2_PATH.name)
    log.info("Wrote %s", EVAL_EXT_IC_PATH.name)
    log.info("Wrote %s", EVAL_EXT_PORT_PATH.name)

    return {"oos_r2": r2_out, "ic_stats": ic_out, "portfolio_perf": port_out}


def main() -> dict[str, pd.DataFrame]:
    """CLI entry point for `python -m src.extensions.post2020_eval`."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    return run_analysis()


if __name__ == "__main__":
    main()