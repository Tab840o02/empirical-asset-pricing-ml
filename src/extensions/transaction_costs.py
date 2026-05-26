"""
Phase 5b - transaction-cost-aware model comparison.

This module estimates one-way transaction costs from two spread proxies and
re-ranks the long-short model performance after cost deductions.

Implemented spread methods:
1) fixed_schedule: 20 bps for the bottom NYSE size quintile (stocks at or
   below the NYSE 20th-percentile market-cap breakpoint), 5 bps for the top
   NYSE size quintile (at or above the 80th-percentile breakpoint), and 10 bps
   for the middle three quintiles.  Breakpoints are computed from NYSE-listed
   stocks only (exchcd == 1), with an all-stock fallback for months where
   NYSE breakpoints are unavailable.
2) amihud_calibrated: monthly Amihud ILLIQ proxy (|ret| / |prc x vol|, using
   monthly CRSP data as a proxy for the standard daily-average measure) mapped
   to bps via spread_bps = clip(20000 * sqrt(ILLIQ), 1, 300), with monthly
   99th-percentile winsorization on ILLIQ before the mapping.

Note on Amihud proxy: the standard Amihud (2002) measure averages daily
|ret_d| / (prc_d x vol_d) within each month.  This module uses a monthly
proxy (single-observation |ret_m| / (prc_m x vol_m)) because the daily CRSP
files are not merged here.  The monthly proxy introduces noise relative to the
daily average; results should be interpreted accordingly and compared against
a daily-CRSP robustness check if the paper advances to submission.

CLI usage:
    python -m src.extensions.transaction_costs            # Phase 4 window
    python -m src.extensions.transaction_costs --ext      # Phase 5a extension

Output files are written to data/processed:
    Phase 4 window : eval_tc_monthly.csv / eval_tc_summary.csv
    Extension window: eval_tc_ext_monthly.csv / eval_tc_ext_summary.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CRSP_CLEAN_PATH,
    EVAL_TC_EXT_MONTHLY_PATH,
    EVAL_TC_EXT_SUMMARY_PATH,
    EVAL_TC_MONTHLY_PATH,
    EVAL_TC_SUMMARY_PATH,
    FEATURES_PANEL_PATH,
    PREDICTIONS_EXT_PATH,
    PREDICTIONS_PATH,
    RAW_DIR,
)
from src.evaluation.portfolio import ff_alpha

log = logging.getLogger(__name__)

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


def _to_month_end(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.to_period("M").dt.to_timestamp("M")


def _load_inputs(
    predictions_path: Path,
    features_path: Path,
    crsp_path: Path,
    ff_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {predictions_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Missing features panel: {features_path}")
    if not crsp_path.exists():
        raise FileNotFoundError(f"Missing CRSP clean file: {crsp_path}")
    if not ff_path.exists():
        raise FileNotFoundError(f"Missing FF factors file: {ff_path}")

    preds = pd.read_parquet(predictions_path)
    panel = pd.read_parquet(features_path, columns=["permno", "date", "ret_exc"])
    crsp = pd.read_parquet(
        crsp_path,
        columns=["permno", "date", "me_lag1", "exchcd", "ret", "prc", "vol"],
    )
    ff = pd.read_parquet(ff_path).reset_index().rename(columns={"index": "date"})

    preds["date"] = _to_month_end(preds["date"])
    panel["date"] = _to_month_end(panel["date"])
    crsp["date"] = _to_month_end(crsp["date"])
    ff["date"] = _to_month_end(ff["date"])

    df = preds.merge(panel, on=["permno", "date"], how="left", validate="many_to_one")
    if df["ret_exc"].isna().any():
        missing = int(df["ret_exc"].isna().sum())
        raise ValueError(f"Missing realized returns for {missing} prediction rows.")

    df = df.merge(crsp, on=["permno", "date"], how="left", validate="many_to_one")
    required = ["me_lag1", "exchcd", "ret", "prc", "vol"]
    if df[required].isna().any().any():
        before = len(df)
        df = df.dropna(subset=required)
        dropped = before - len(df)
        log.info("Dropped %d rows with missing CRSP fields for TC estimation.", dropped)

    # Match portfolio implementation: only positive lagged cap rows can be weighted.
    df = df[df["me_lag1"] > 0].copy()
    return df, ff


def _assign_deciles(df: pd.DataFrame, n_deciles: int = 10) -> pd.DataFrame:
    ranked = df.copy()
    ranked["decile"] = (
        ranked.groupby(["model", "date"])["pred_ret"]
        .transform(
            lambda s: pd.qcut(
                s.rank(method="first"),
                q=n_deciles,
                labels=range(1, n_deciles + 1),
            ).astype(int)
        )
    )
    return ranked


def _build_signed_weights(ranked: pd.DataFrame, n_deciles: int = 10) -> pd.DataFrame:
    selected = ranked[ranked["decile"].isin([1, n_deciles])].copy()

    # Within each side, weights sum to 1 before side-scaling.
    selected["raw_w"] = selected.groupby(["model", "date", "decile"])["me_lag1"].transform(
        lambda x: x / x.sum()
    )

    # Dollar-neutral gross=1 convention: +0.5 long, -0.5 short.
    selected["signed_w"] = np.where(selected["decile"] == n_deciles, 0.5 * selected["raw_w"], -0.5 * selected["raw_w"])
    selected["abs_w"] = selected["signed_w"].abs()

    cols = [
        "model",
        "date",
        "permno",
        "decile",
        "signed_w",
        "abs_w",
        "ret_exc",
        "exchcd",
        "me_lag1",
        "ret",
        "prc",
        "vol",
    ]
    return selected[cols].sort_values(["model", "date", "permno"]).reset_index(drop=True)


def _add_fixed_schedule_spread(holdings: pd.DataFrame) -> pd.DataFrame:
    """
    Assign fixed spread schedules using NYSE 20th/80th-percentile size breakpoints
    (standard Fama-French NYSE-quintile convention).

    Spread schedule:
      - Bottom quintile (me_lag1 <= NYSE p20) : 20 bps
      - Middle three quintiles                : 10 bps
      - Top quintile    (me_lag1 >= NYSE p80) :  5 bps
    """
    out = holdings.copy()

    nyse = out[out["exchcd"] == 1]
    bp = (
        nyse.groupby("date")["me_lag1"]
        .quantile([0.2, 0.8])
        .unstack()
        .rename(columns={0.2: "q20", 0.8: "q80"})
        .reset_index()
    )

    # All-stock fallback for months with no NYSE observations.
    all_bp = (
        out.groupby("date")["me_lag1"]
        .quantile([0.2, 0.8])
        .unstack()
        .rename(columns={0.2: "q20_all", 0.8: "q80_all"})
        .reset_index()
    )

    out = out.merge(bp, on="date", how="left")
    out = out.merge(all_bp, on="date", how="left")

    out["q20"] = out["q20"].fillna(out["q20_all"])
    out["q80"] = out["q80"].fillna(out["q80_all"])

    out["spread_bps_fixed"] = np.where(
        out["me_lag1"] <= out["q20"],
        20.0,
        np.where(out["me_lag1"] >= out["q80"], 5.0, 10.0),
    )

    return out.drop(columns=["q20", "q80", "q20_all", "q80_all"])


def _add_amihud_spread(holdings: pd.DataFrame) -> pd.DataFrame:
    out = holdings.copy()

    dollar_volume = (out["prc"].abs() * out["vol"]).replace(0, np.nan)
    out["illiq"] = (out["ret"].abs() / dollar_volume).replace([np.inf, -np.inf], np.nan)

    # Monthly winsorization at 99th percentile.
    p99 = out.groupby("date")["illiq"].quantile(0.99).rename("illiq_p99")
    out = out.merge(p99, on="date", how="left")
    out["illiq_w"] = np.minimum(out["illiq"], out["illiq_p99"])

    # Fill remaining gaps conservatively with monthly median, then global median.
    med = out.groupby("date")["illiq_w"].transform("median")
    out["illiq_w"] = out["illiq_w"].fillna(med)
    out["illiq_w"] = out["illiq_w"].fillna(out["illiq_w"].median())

    # Hasbrouck-style square-root mapping; clipped for robustness.
    out["spread_bps_amihud"] = (20000.0 * np.sqrt(out["illiq_w"])).clip(lower=1.0, upper=300.0)

    return out.drop(columns=["illiq_p99"])


def _portfolio_monthly(holdings: pd.DataFrame, spread_col: str, method: str) -> pd.DataFrame:
    gross = (
        holdings.groupby(["model", "date"]) 
        .apply(lambda g: float((g["signed_w"] * g["ret_exc"]).sum()), include_groups=False)
        .rename("gross_ls_ret")
        .reset_index()
    )

    spread = (
        holdings.groupby(["model", "date"]) 
        .apply(lambda g: float((g["abs_w"] * g[spread_col]).sum() / g["abs_w"].sum()), include_groups=False)
        .rename("avg_one_way_spread_bps")
        .reset_index()
    )

    turnover_rows: list[dict[str, object]] = []
    for model, grp in holdings.groupby("model"):
        grp = grp.sort_values(["date", "permno"])
        prev = pd.Series(dtype=float)
        for dt, gdt in grp.groupby("date"):
            cur = gdt.set_index("permno")["signed_w"]
            if prev.empty:
                to = np.nan
            else:
                to = 0.5 * float(cur.sub(prev, fill_value=0.0).abs().sum())
            turnover_rows.append({"model": model, "date": dt, "turnover": to})
            prev = cur

    turnover = pd.DataFrame(turnover_rows)

    monthly = gross.merge(spread, on=["model", "date"], how="inner")
    monthly = monthly.merge(turnover, on=["model", "date"], how="left")

    # Do not charge costs in the first month because turnover is undefined there.
    monthly["turnover"] = monthly["turnover"].fillna(0.0)
    monthly["tc_cost"] = monthly["turnover"] * (monthly["avg_one_way_spread_bps"] / 10000.0)
    monthly["net_ls_ret"] = monthly["gross_ls_ret"] - monthly["tc_cost"]
    monthly["tc_method"] = method

    return monthly.sort_values(["model", "date"]).reset_index(drop=True)


def _annualize(series: pd.Series) -> tuple[float, float, float]:
    ann_ret = float(series.mean() * 12.0)
    ann_vol = float(series.std(ddof=1) * np.sqrt(12.0))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    return ann_ret, ann_vol, sharpe


def _summary_table(monthly: pd.DataFrame, ff: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (method, model), grp in monthly.groupby(["tc_method", "model"]):
        grp = grp.sort_values("date")
        gross_ann_ret, gross_ann_vol, gross_sharpe = _annualize(grp["gross_ls_ret"])
        net_ann_ret, net_ann_vol, net_sharpe = _annualize(grp["net_ls_ret"])

        gross_alpha = ff_alpha(grp.set_index("date")["gross_ls_ret"], ff, model="ff5", nw_lags=12)
        net_alpha = ff_alpha(grp.set_index("date")["net_ls_ret"], ff, model="ff5", nw_lags=12)

        rows.append(
            {
                "tc_method": method,
                "model": model,
                "n_months": int(grp["date"].nunique()),
                "annual_ret_gross": gross_ann_ret,
                "annual_vol_gross": gross_ann_vol,
                "sharpe_gross": gross_sharpe,
                "alpha_annual_gross": gross_alpha["alpha_annual"],
                "t_alpha_gross": gross_alpha["t_alpha"],
                "annual_ret_net": net_ann_ret,
                "annual_vol_net": net_ann_vol,
                "sharpe_net": net_sharpe,
                "alpha_annual_net": net_alpha["alpha_annual"],
                "t_alpha_net": net_alpha["t_alpha"],
                "mean_turnover": float(grp["turnover"].mean()),
                "mean_spread_bps": float(grp["avg_one_way_spread_bps"].mean()),
                "mean_tc_cost_monthly": float(grp["tc_cost"].mean()),
                "sharpe_delta": net_sharpe - gross_sharpe,
            }
        )

    out = pd.DataFrame(rows)
    out["model"] = pd.Categorical(out["model"], categories=MODEL_ORDER, ordered=True)
    out = out.sort_values(["tc_method", "model"]).reset_index(drop=True)
    out["model"] = out["model"].astype(str)

    # Add rank diagnostics by method.
    out["rank_gross"] = out.groupby("tc_method")["sharpe_gross"].rank(method="dense", ascending=False)
    out["rank_net"] = out.groupby("tc_method")["sharpe_net"].rank(method="dense", ascending=False)
    out["rank_change"] = out["rank_net"] - out["rank_gross"]

    return out


def run_transaction_cost_analysis(
    predictions_path: Path = PREDICTIONS_PATH,
    features_path: Path = FEATURES_PANEL_PATH,
    crsp_path: Path = CRSP_CLEAN_PATH,
    ff_path: Path = RAW_DIR / "ff_factors.parquet",
    monthly_out: Path = EVAL_TC_MONTHLY_PATH,
    summary_out: Path = EVAL_TC_SUMMARY_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Phase 5b and persist monthly and summary outputs."""
    df, ff = _load_inputs(predictions_path, features_path, crsp_path, ff_path)
    ranked = _assign_deciles(df, n_deciles=10)
    holdings = _build_signed_weights(ranked, n_deciles=10)

    fixed = _add_fixed_schedule_spread(holdings)
    fixed_monthly = _portfolio_monthly(fixed, spread_col="spread_bps_fixed", method="fixed_schedule")

    amihud = _add_amihud_spread(holdings)
    amihud_monthly = _portfolio_monthly(amihud, spread_col="spread_bps_amihud", method="amihud_calibrated")

    monthly = pd.concat([fixed_monthly, amihud_monthly], ignore_index=True)
    summary = _summary_table(monthly, ff)

    monthly_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(monthly_out, index=False)
    summary.to_csv(summary_out, index=False)

    log.info("Wrote %s", monthly_out)
    log.info("Wrote %s", summary_out)

    return monthly, summary


def main() -> tuple[pd.DataFrame, pd.DataFrame]:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Phase 5b transaction cost analysis.")
    parser.add_argument(
        "--ext",
        action="store_true",
        help="Run on the Phase 5a extension predictions (2017-2024) instead of the "
             "Phase 4 replication window (1987-2016).",
    )
    args = parser.parse_args()

    if args.ext:
        log.info("Phase 5b: running on extension predictions (2017-2024).")
        return run_transaction_cost_analysis(
            predictions_path=PREDICTIONS_EXT_PATH,
            monthly_out=EVAL_TC_EXT_MONTHLY_PATH,
            summary_out=EVAL_TC_EXT_SUMMARY_PATH,
        )
    return run_transaction_cost_analysis()


if __name__ == "__main__":
    main()
