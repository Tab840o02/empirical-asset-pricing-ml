"""
Bootstrap GKX project inside a Kaggle notebook session.

Usage in a Kaggle code cell:
    !python scripts/kaggle/kaggle_bootstrap.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path


def find_required_zip(name: str) -> Path:
    candidates = list(Path("/kaggle/input").rglob(name))
    if not candidates:
        raise FileNotFoundError(f"Could not find {name} under /kaggle/input")
    return candidates[0]


def unzip_to_work(zip_path: Path, dest: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)


def main() -> None:
    work = Path("/kaggle/working/gkx")
    work.mkdir(parents=True, exist_ok=True)

    code_zip = find_required_zip("gkx_code_bundle.zip")
    data_zip = find_required_zip("gkx_processed_bundle.zip")

    unzip_to_work(code_zip, work)
    unzip_to_work(data_zip, work)

    os.chdir(work)
    print("CWD:", Path.cwd())

    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)

    import tensorflow as tf

    print("TensorFlow:", tf.__version__)
    print("GPUs:", tf.config.list_physical_devices("GPU"))


if __name__ == "__main__":
    main()
