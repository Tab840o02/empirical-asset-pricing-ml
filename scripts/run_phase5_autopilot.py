"""
Unattended Phase 5 autopilot.

This script chains the Phase 5a extension and the Phase 5c parsimony run in
model batches so the work can proceed with minimal supervision.

Recommended usage:
    python -m scripts.run_phase5_autopilot --time-budget-hours 10

The script is conservative: it finishes the current batch, checkpoints via the
existing phase scripts, and exits before the budget is exceeded.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

# Ensure the repository root is importable when the script is run directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.extensions.feature_parsimony import build_parsimonious_panel  # noqa: E402
from src.extensions.post2020_eval import resolve_ext_end  # noqa: E402
from src.config import (  # noqa: E402
    FEATURES_PARSIMONIOUS_PATH,
    PREDICTIONS_EXT_PATH,
    PREDICTIONS_PARSIMONY_PATH,
)

log = logging.getLogger(__name__)

PHASE5C_BATCHES: list[list[str]] = [
    ["ols3", "ols_all", "pcr", "pls", "enet", "glm", "rf", "gbrt"],
    ["nn1", "nn2"],
    ["nn3", "nn4", "nn5"],
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GKX Phase 5 autopilot")
    parser.add_argument(
        "--time-budget-hours",
        type=float,
        default=10.0,
        help="Soft stop budget for the full run. Default: 10 hours.",
    )
    parser.add_argument(
        "--skip-phase5a",
        action="store_true",
        help="Skip Phase 5a and only run Phase 5c batches.",
    )
    parser.add_argument(
        "--skip-phase5c",
        action="store_true",
        help="Skip Phase 5c and only run Phase 5a.",
    )
    parser.add_argument(
        "--rebuild-parsimonious-panel",
        action="store_true",
        help="Force regeneration of the parsimonious feature panel before Phase 5c.",
    )
    return parser.parse_args()


def _run(cmd: list[str], label: str) -> None:
    log.info("Starting %s", label)
    log.info("Command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    log.info("Finished %s", label)


def _elapsed_hours(t0: float) -> float:
    return (time.time() - t0) / 3600.0


def _budget_exceeded(t0: float, budget_hours: float, reserve_hours: float = 0.25) -> bool:
    return _elapsed_hours(t0) >= max(budget_hours - reserve_hours, 0.0)


def _phase5a_complete() -> bool:
    if not PREDICTIONS_EXT_PATH.exists():
        return False
    try:
        ext_end = resolve_ext_end()
    except Exception:
        return False
    from scripts.run_phase5a import _predictions_complete

    return _predictions_complete(PREDICTIONS_EXT_PATH, "2017-01-01", ext_end)


def _phase5c_complete() -> bool:
    return PREDICTIONS_PARSIMONY_PATH.exists() and FEATURES_PARSIMONIOUS_PATH.exists()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args()
    t0 = time.time()

    log.info("Budget: %.1f hours", args.time_budget_hours)
    log.info("Phase 5a complete: %s", _phase5a_complete())
    log.info("Phase 5c complete: %s", _phase5c_complete())

    if not args.skip_phase5a and not _phase5a_complete():
        _run([sys.executable, "-m", "scripts.run_phase5a"], "Phase 5a")
    elif not args.skip_phase5a:
        log.info("Skipping Phase 5a — predictions_ext.parquet is already complete.")

    if args.skip_phase5c:
        log.info("Skipping Phase 5c by request.")
        return

    if args.rebuild_parsimonious_panel or not FEATURES_PARSIMONIOUS_PATH.exists():
        log.info("Building parsimonious panel directly …")
        build_parsimonious_panel()

    for batch in PHASE5C_BATCHES:
        if _budget_exceeded(t0, args.time_budget_hours):
            log.info(
                "Time budget nearly exhausted after %.2f hours; stopping before the next batch.",
                _elapsed_hours(t0),
            )
            return

        _run(
            [
                sys.executable,
                "-m",
                "scripts.run_phase5c",
                "--models",
                *batch,
                "--append",
                "--resume",
            ],
            f"Phase 5c batch: {' '.join(batch)}",
        )

    log.info("All requested Phase 5 runs finished in %.2f hours.", _elapsed_hours(t0))


if __name__ == "__main__":
    main()