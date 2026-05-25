"""
Run NN models sequentially in Kaggle and checkpoint outputs after each model.

Usage (inside Kaggle notebook terminal/cell):
    python scripts/kaggle/kaggle_nn_runner.py --models nn3 nn4 nn5

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
    p = argparse.ArgumentParser(description="Sequential NN runner with checkpoints")
    p.add_argument(
        "--models",
        nargs="+",
        default=["nn3", "nn4", "nn5"],
        help="NN models to run in order (e.g., nn3 nn4 nn5)",
    )
    p.add_argument(
        "--test-start",
        default="1987-01-01",
        help="First test month (YYYY-MM-DD)",
    )
    p.add_argument(
        "--test-end",
        default="2016-12-31",
        help="Last test month (YYYY-MM-DD)",
    )
    p.add_argument(
        "--no-append",
        action="store_true",
        help="Disable append mode (append is enabled by default)",
    )
    return p.parse_args()


def checkpoint_outputs(root: Path, model: str) -> None:
    processed = root / "data" / "processed"
    checkpoints = processed / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    pred = processed / "predictions.parquet"
    manifest = processed / "run_manifest.json"

    if pred.exists():
        shutil.copy2(pred, checkpoints / f"predictions_after_{model}_{ts}.parquet")
    if manifest.exists():
        shutil.copy2(manifest, checkpoints / f"run_manifest_after_{model}_{ts}.json")


def run_one_model(args: argparse.Namespace, model: str) -> None:
    cmd = [
        sys.executable,
        "-m",
        "src.models.train_eval",
        "--models",
        model,
        "--test-start",
        args.test_start,
        "--test-end",
        args.test_end,
    ]

    if not args.no_append:
        cmd.append("--append")
        cmd.append("--resume")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    root = Path.cwd()

    valid = {"nn1", "nn2", "nn3", "nn4", "nn5"}
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
