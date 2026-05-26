"""
Phase 5b driver.

Runs transaction-cost-aware performance analysis on the canonical predictions
file and writes:
- data/processed/eval_tc_monthly.csv
- data/processed/eval_tc_summary.csv

Usage:
    python scripts/run_phase5b.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import EVAL_TC_MONTHLY_PATH, EVAL_TC_SUMMARY_PATH
from src.extensions.transaction_costs import run_transaction_cost_analysis


def _print_top_rank_changes(summary):
    cols = [
        "tc_method",
        "model",
        "sharpe_gross",
        "sharpe_net",
        "rank_gross",
        "rank_net",
        "rank_change",
        "mean_turnover",
        "mean_spread_bps",
        "mean_tc_cost_monthly",
    ]
    print("\nTop models by net Sharpe (per TC method):")
    top = summary.sort_values(["tc_method", "sharpe_net"], ascending=[True, False]).groupby("tc_method").head(5)
    print(top[cols].to_string(index=False))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    monthly, summary = run_transaction_cost_analysis()

    print(f"Wrote monthly output: {EVAL_TC_MONTHLY_PATH}")
    print(f"Wrote summary output: {EVAL_TC_SUMMARY_PATH}")
    print(f"Monthly rows: {len(monthly):,}")
    print(f"Summary rows: {len(summary):,}")
    _print_top_rank_changes(summary)


if __name__ == "__main__":
    main()
