"""
Run Phase 5c models sequentially in Kaggle and checkpoint outputs after each model.

Usage inside Kaggle after bootstrap:
    python scripts/kaggle/kaggle_phase5c_runner.py --models ols3 ols_all pcr pls enet glm rf gbrt nn1 nn2 nn3 nn4 nn5

The script assumes the repository root is the current working directory.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequential Phase 5c runner with checkpoints")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "ols3", "ols_all", "pcr", "pls", "enet", "glm",
            "rf", "gbrt",
            "nn1", "nn2", "nn3", "nn4", "nn5",
        ],
        help="Models to run in order (e.g. ols3 ols_all rf gbrt nn1 nn2)",
    )
    parser.add_argument(
        "--no-append",
        action="store_true",
        help="Disable append mode (append is enabled by default).",
    )
    return parser.parse_args()


def checkpoint_outputs(root: Path, model: str) -> None:
    processed = root / "data" / "processed"
    checkpoints = processed / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    pred = processed / "predictions_parsimony.parquet"
    manifest = processed / "run_manifest_parsimony.json"

    if pred.exists():
        shutil.copy2(pred, checkpoints / f"predictions_parsimony_after_{model}_{ts}.parquet")
    if manifest.exists():
        shutil.copy2(manifest, checkpoints / f"run_manifest_parsimony_after_{model}_{ts}.json")


def run_one_model(args: argparse.Namespace, model: str) -> None:
    cmd = [
        sys.executable,
        "-m",
        "scripts.run_phase5c",
        "--models",
        model,
    ]

    if not args.no_append:
        cmd.append("--append")
        cmd.append("--resume")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    root = Path.cwd()

    valid = {
        "ols3", "ols_all", "pcr", "pls", "enet", "glm",
        "rf", "gbrt",
        "nn1", "nn2", "nn3", "nn4", "nn5",
    }
    bad = [m for m in args.models if m not in valid]
    if bad:
        raise ValueError(f"Invalid models: {bad}. Valid: {sorted(valid)}")

    for model in args.models:
        run_one_model(args, model)
        checkpoint_outputs(root, model)
        print(f"Checkpoint complete for {model}")

    print("All requested models finished.")


if __name__ == "__main__":
    main()