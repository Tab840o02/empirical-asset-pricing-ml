"""
Phase 5c driver — feature parsimony retraining.

Workflow:
1. Build a parsimonious 15–20 feature panel if one does not already exist.
2. Retrain the Phase 4 model catalogue on the reduced panel.
3. Write a separate prediction file and manifest for auditability.

Run locally or in Kaggle:
    python -m scripts.run_phase5c --models all
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the repository root is importable when the script is run directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    FEATURES_PARSIMONIOUS_PATH,
    PREDICTIONS_PARSIMONY_PATH,
    RUN_MANIFEST_PARSIMONY_PATH,
)
from src.extensions.feature_parsimony import (  # noqa: E402
    build_parsimonious_panel,
    run_phase5c_training,
)

log = logging.getLogger(__name__)

ALL_MODELS = [
    "ols3", "ols_all", "pcr", "pls", "enet", "glm",
    "rf", "gbrt",
    "nn1", "nn2", "nn3", "nn4", "nn5",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GKX Phase 5c feature parsimony run")
    parser.add_argument(
        "--models",
        nargs="+",
        default=ALL_MODELS,
        help="Models to retrain. Use 'all' for the full catalogue.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append new predictions to an existing phase5c predictions file.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last completed year when used with --append.",
    )
    parser.add_argument(
        "--rebuild-panel",
        action="store_true",
        help="Force regeneration of the parsimonious feature panel.",
    )
    return parser.parse_args()


def _normalise_models(models: list[str]) -> list[str]:
    if len(models) == 1 and models[0] == "all":
        return ALL_MODELS
    invalid = [m for m in models if m not in ALL_MODELS]
    if invalid:
        raise ValueError(f"Unknown models: {invalid}. Valid choices: {ALL_MODELS}")
    return models


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args()
    models = _normalise_models(args.models)

    if args.rebuild_panel or not FEATURES_PARSIMONIOUS_PATH.exists():
        build_parsimonious_panel()

    results = run_phase5c_training(
        models_to_run=models,
        predictions_path=PREDICTIONS_PARSIMONY_PATH,
        manifest_path=RUN_MANIFEST_PARSIMONY_PATH,
        append=args.append,
        resume=args.resume,
    )

    print(f"Wrote predictions to {PREDICTIONS_PARSIMONY_PATH}")
    print(f"Wrote manifest to {RUN_MANIFEST_PARSIMONY_PATH}")
    print(f"Rows: {len(results):,}")


if __name__ == "__main__":
    main()