"""
Phase 5a driver.

This script checks whether `predictions_ext.parquet` already covers the full
post-2016 window and, if not, runs the extension training loop before
executing the post-2020 evaluation.

Run with:
    python scripts/run_phase5a.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure the repository root is importable when the script is run directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PREDICTIONS_EXT_PATH, RUN_MANIFEST_EXT_PATH
from src.evaluation.portfolio import build_decile_portfolios, ff_alpha, ls_returns
from src.extensions.post2020_eval import MODEL_ORDER, resolve_ext_end, run_analysis
from src.models.train_eval import run as run_train_eval

log = logging.getLogger(__name__)


def _expected_dates(start: str, end: pd.Timestamp) -> set[pd.Timestamp]:
    return set(pd.period_range(pd.Timestamp(start), end, freq="M").to_timestamp("M"))


def _predictions_complete(path: Path, start: str, end: pd.Timestamp) -> bool:
    if not path.exists():
        return False

    preds = pd.read_parquet(path)
    preds["date"] = pd.to_datetime(preds["date"]).dt.to_period("M").dt.to_timestamp("M")

    if set(preds["model"].unique()) != set(MODEL_ORDER):
        return False
    if preds["date"].min() != pd.Timestamp(start):
        return False
    if preds["date"].max() != end:
        return False

    expected_dates = _expected_dates(start, end)
    for model, grp in preds.groupby("model"):
        if set(grp["date"].drop_duplicates()) != expected_dates:
            return False
    return True


def _print_summary(results: dict[str, pd.DataFrame]) -> None:
    r2 = results["oos_r2"]
    port = results["portfolio_perf"]

    best_r2 = (
        r2.sort_values(["period", "oos_r2"], ascending=[True, False])
        .groupby("period", as_index=False)
        .first()[["period", "model", "oos_r2"]]
        .rename(columns={"model": "best_r2_model", "oos_r2": "best_r2"})
    )
    best_sharpe = (
        port.sort_values(["period", "sharpe"], ascending=[True, False])
        .groupby("period", as_index=False)
        .first()[["period", "model", "sharpe", "alpha_annual", "t_alpha"]]
        .rename(
            columns={
                "model": "best_sharpe_model",
                "sharpe": "best_sharpe",
            }
        )
    )

    summary = best_r2.merge(best_sharpe, on="period", how="inner")
    print(summary.to_string(index=False))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    ext_end = resolve_ext_end()

    if _predictions_complete(PREDICTIONS_EXT_PATH, "2017-01-01", ext_end):
        log.info("predictions_ext.parquet is complete; skipping training.")
    else:
        log.info("predictions_ext.parquet is missing or incomplete; running training.")
        run_train_eval(
            test_start="2017-01-01",
            test_end=str(ext_end.date()),
            models_to_run=list(MODEL_ORDER),
            output_path=PREDICTIONS_EXT_PATH,
            manifest_path=RUN_MANIFEST_EXT_PATH,
            append=False,
            resume=False,
        )

    results = run_analysis(ext_end)
    _print_summary(results)


if __name__ == "__main__":
    main()