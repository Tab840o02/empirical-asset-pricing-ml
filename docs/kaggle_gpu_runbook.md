# Kaggle GPU Runbook (Deadline Mode)

This runbook migrates NN3-NN5 training from local CPU to Kaggle GPU without changing the project methodology.

## 1) Local prep (already automated)

From repository root, run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/kaggle/make_kaggle_bundles.ps1
```

Expected output files:
- `artifacts/kaggle/gkx_code_bundle.zip`
- `artifacts/kaggle/gkx_processed_bundle.zip`

## 2) Upload to Kaggle (manual)

1. Create private Kaggle dataset `gkx-code-bundle` and upload `gkx_code_bundle.zip`.
2. Create private Kaggle dataset `gkx-processed-bundle` and upload `gkx_processed_bundle.zip`.
3. Create a new Kaggle Notebook with GPU accelerator.
4. Add both datasets as Notebook Inputs.

## 3) Kaggle notebook bootstrap cells

Cell A:

```python
import os
import zipfile
from pathlib import Path

WORK = Path('/kaggle/working/gkx')
WORK.mkdir(parents=True, exist_ok=True)

code_zip = next(Path('/kaggle/input').rglob('gkx_code_bundle.zip'))
data_zip = next(Path('/kaggle/input').rglob('gkx_processed_bundle.zip'))

with zipfile.ZipFile(code_zip, 'r') as z:
    z.extractall(WORK)
with zipfile.ZipFile(data_zip, 'r') as z:
    z.extractall(WORK)

os.chdir(WORK)
print('CWD:', Path.cwd())
```

Cell B:

```python
import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
```

Cell C:

```python
import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
```

## 4) Run NN queue in priority order

If NN2 is already trained and local NN3 is only half done, start on Kaggle with NN3 and continue deeper models.

Cell D:

```python
!python scripts/kaggle/kaggle_nn_runner.py --models nn3 nn4 nn5 --test-start 1987-01-01 --test-end 2016-12-31
```

This runs each model with `--append` and creates checkpoint files after every model in:
- `data/processed/checkpoints/`

## 5) Persist outputs before session ends

At the end of each completed model, use the right-side file browser in Kaggle and download:
- `data/processed/predictions.parquet`
- `data/processed/run_manifest.json`
- latest files under `data/processed/checkpoints/`

## 6) Merge back into local repo

Copy downloaded files into local:
- `data/processed/predictions.parquet`
- `data/processed/run_manifest.json`
- optional archive: `data/processed/checkpoints/*`

Then run your local evaluation scripts.

## 7) Deadline safety policy

- Prefer one model per session if Kaggle runtime limits are tight:
  - run `nn3`, download outputs
  - run `nn4`, download outputs
  - run `nn5`, download outputs
- Keep `--append` on to avoid losing already-completed model predictions.
- If quota is exhausted, finalize report with NN1-NN3 and state scope constraint explicitly.
