"""
Bootstrap GKX project inside a Kaggle notebook session.

Usage in a Kaggle code cell:
    !python scripts/kaggle/kaggle_bootstrap.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def find_required_file(pattern: str) -> Path:
    candidate = next(Path("/kaggle/input").rglob(pattern), None)
    if candidate is None:
        raise FileNotFoundError(f"Could not find {pattern} under /kaggle/input")
    return candidate


def main() -> None:
    work = Path("/kaggle/working/gkx")
    work.mkdir(parents=True, exist_ok=True)

    src_init = find_required_file("src/__init__.py")
    code_root = src_init.parent.parent
    shutil.copytree(code_root, work, dirs_exist_ok=True)

    processed_dest = work / "data" / "processed"
    processed_dest.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.parquet", "*.json"):
        for source in Path("/kaggle/input").rglob(pattern):
            shutil.copy2(source, processed_dest / source.name)

    print("CWD:", work)
    print("features_panel:", processed_dest / "features_panel.parquet")

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(work / "requirements.txt")],
        check=True,
    )

    import tensorflow as tf

    print("TensorFlow:", tf.__version__)
    print("GPUs:", tf.config.list_physical_devices("GPU"))


if __name__ == "__main__":
    main()
