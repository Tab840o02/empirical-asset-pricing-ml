# Kaggle Runbook — Phase 5c Feature Parsimony

This runbook is the shortest safe path to execute Phase 5c on Kaggle within a 10-hour session.

## Current status context (2026-05-26)

- Phase 4 canonical replication outputs are complete (`eval_*_latest.csv`).
- Phase 5a run execution is complete at manifest level (`run_manifest_ext.json`, 2017-01 to 2024-11, all 13 models completed).
- Phase 5c is still in progress; this runbook is the operational handoff for finishing Phase 5c outputs.

## What this run does

1. Bootstraps the repository inside Kaggle.
2. Builds a parsimonious 15–20 feature panel from the 94-feature Phase 4 panel.
3. Retrains the Phase 4 model catalogue on the reduced panel.
4. Checkpoints the predictions and manifest after each model so the session can be resumed.

## What to upload to Kaggle

From your local machine, create the bundles with:

```powershell
.
scripts\kaggle\make_kaggle_bundles.ps1
```

Upload both generated archives to a Kaggle notebook or dataset:

* `artifacts/kaggle/gkx_code_bundle.zip`
* `artifacts/kaggle/gkx_processed_bundle.zip`

## Kaggle notebook setup

1. Create a Kaggle notebook with GPU turned on.
2. Add both zip files as input datasets.
3. Run the bootstrap cell:

```python
!python scripts/kaggle/kaggle_bootstrap.py
```

4. Change into the working copy:

```python
%cd /kaggle/working/gkx
```

## Fast execution plan under a 10-hour cap

Run the models in this order:

1. Non-NN models first: `ols3 ols_all pcr pls enet glm rf gbrt`
2. NN1 and NN2 next if time remains.
3. NN3–NN5 only if the session is still comfortably under the time limit.

This order front-loads the cheapest, most informative models. If Kaggle stops the session, the checkpoint files preserve completed work.

## Recommended Kaggle command

The most robust way to run the full Phase 5c job is the checkpoint runner:

```python
!python scripts/kaggle/kaggle_phase5c_runner.py --models ols3 ols_all pcr pls enet glm rf gbrt nn1 nn2 nn3 nn4 nn5
```

If you want a single unattended launcher that first checks Phase 5a, then runs Phase 5c in batches, use:

```python
!python -m scripts.run_phase5_autopilot --time-budget-hours 10
```

If you need to split the run into two Kaggle sessions, use these batches:

```python
!python scripts/kaggle/kaggle_phase5c_runner.py --models ols3 ols_all pcr pls enet glm rf gbrt
!python scripts/kaggle/kaggle_phase5c_runner.py --models nn1 nn2 nn3 nn4 nn5
```

## Output files you should expect

* `data/processed/features_panel_parsimonious.parquet`
* `data/processed/phase5c_selected_features.json`
* `data/processed/predictions_parsimony.parquet`
* `data/processed/run_manifest_parsimony.json`
* `data/processed/checkpoints/` with per-model snapshots

## Practical tips

* Keep the notebook output open and watch the first model finish before starting the next batch.
* If Kaggle begins timing out, split the models into smaller groups and rerun; completed models are checkpointed.
* Use the reduced panel only for Phase 5c. Do not overwrite the canonical `features_panel.parquet` or `predictions.parquet`.