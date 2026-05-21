"""
src/models/train_eval.py
=========================
Rolling-window model training and prediction loop for GKX (2020) replication.

Usage
-----
    python -m src.models.train_eval                     # all non-NN models
    python -m src.models.train_eval --models ols3 ols_all pcr pls enet glm
    python -m src.models.train_eval --models rf gbrt
    python -m src.models.train_eval --models nn1 nn2 nn3 nn4 nn5
    python -m src.models.train_eval --models all          # includes NNs (slow)
    python -m src.models.train_eval --test-end 2024-11-30  # Phase 5a extension

Rolling-window scheme (GKX §3)
--------------------------------
For each test year Y ∈ {test_start_year, ..., test_end_year}:
  - Training data : all months where date < 1-Jan-Y  (expanding window)
  - Predictions   : all months in year Y within [test_start, test_end]

Hyperparameter selection (once, before the test window):
  - Training: all months before 1975-01-01  (1957–1974)
  - Validation: 1975-01-01 – 1986-12-31

Output
------
data/processed/predictions.parquet  — columns: permno, date, model, pred_ret
data/processed/run_manifest.json    — seeds, hyperparams, timing

Environment
-----------
Activate the venv and set WRDS_USER (not needed here, but consistent):
    .venv/Scripts/Activate.ps1
    python -m src.models.train_eval
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    FEATURES_PANEL_PATH,
    PREDICTIONS_PATH,
    RUN_MANIFEST_PATH,
    VAL_START,
    VAL_END,
    REPLICATE_TEST_END,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VAL_PRE_END: str = "1974-12-31"       # Training data end for hyperparameter selection
TEST_START: str = "1987-01-01"        # First test month (replication window)

ALL_MODELS: list[str] = [
    "ols3", "ols_all", "pcr", "pls", "enet", "glm",
    "rf", "gbrt",
    "nn1", "nn2", "nn3", "nn4", "nn5",
]
DEFAULT_MODELS: list[str] = [
    "ols3", "ols_all", "pcr", "pls", "enet", "glm",
    "rf", "gbrt",
]


# ---------------------------------------------------------------------------
# Feature columns
# ---------------------------------------------------------------------------

def get_feature_cols(panel: pd.DataFrame) -> list[str]:
    return [c for c in panel.columns if c not in ("permno", "date", "ret_exc")]


# ---------------------------------------------------------------------------
# Model catalogue — built after hyperparameter selection
# ---------------------------------------------------------------------------

def build_model_catalogue(
    hp_linear: dict,
    hp_tree: dict,
    feature_cols: list[str],
    models_to_run: list[str],
) -> dict:
    """
    Return a dict {model_name: callable_that_returns_fresh_model}.
    The catalogue only includes requested models.
    """
    from src.models import linear_models as lm
    from src.models import tree_models as tm

    catalogue = {}

    if "ols3" in models_to_run:
        catalogue["ols3"] = lm.make_ols3
    if "ols_all" in models_to_run:
        catalogue["ols_all"] = lm.make_ols_all
    if "pcr" in models_to_run:
        catalogue["pcr"] = lambda: lm.make_pcr(hp_linear["pcr_n"])
    if "pls" in models_to_run:
        catalogue["pls"] = lambda: lm.make_pls(hp_linear["pls_n"])
    if "enet" in models_to_run:
        catalogue["enet"] = lambda: lm.make_enet(hp_linear["enet_alpha"], hp_linear["enet_l1"])
    if "glm" in models_to_run:
        catalogue["glm"] = lambda: lm.make_glm(hp_linear["glm_alpha"])
    if "rf" in models_to_run:
        catalogue["rf"] = lambda: tm.make_rf(hp_tree["rf_depth"])
    if "gbrt" in models_to_run:
        catalogue["gbrt"] = lambda: tm.make_gbrt(hp_tree["gbrt_lr"], hp_tree["gbrt_depth"])

    if any(m.startswith("nn") for m in models_to_run):
        from src.models import neural_nets as nn
        for k in ["nn1", "nn2", "nn3", "nn4", "nn5"]:
            if k in models_to_run:
                n_layers = int(k[2])
                catalogue[k] = lambda nl=n_layers: nn.make_nn(nl)

    return catalogue


# ---------------------------------------------------------------------------
# Hyperparameter selection
# ---------------------------------------------------------------------------

def select_all_hyperparams(
    panel: pd.DataFrame,
    feature_cols: list[str],
    models_to_run: list[str],
) -> tuple[dict, dict]:
    """
    Select hyperparameters for linear and tree models using the validation
    window (train ≤ 1974, validate 1975–1986).
    """
    from src.models import linear_models as lm
    from src.models import tree_models as tm

    log.info("Selecting hyperparameters on validation window (1957–1974 train, 1975–1986 val)…")

    pre_val = panel[panel["date"] <= pd.Timestamp(VAL_PRE_END)]
    val = panel[
        (panel["date"] >= pd.Timestamp(VAL_START)) &
        (panel["date"] <= pd.Timestamp(VAL_END))
    ]

    X_pre = pre_val[feature_cols].values.astype(np.float32)
    y_pre = pre_val["ret_exc"].values.astype(np.float32)
    X_val = val[feature_cols].values.astype(np.float32)
    y_val = val["ret_exc"].values.astype(np.float32)

    hp_linear: dict = {}
    hp_tree: dict = {}

    needs_linear_hp = any(m in models_to_run for m in ["pcr", "pls", "enet", "glm"])
    needs_tree_hp = any(m in models_to_run for m in ["rf", "gbrt"])

    if needs_linear_hp:
        log.info("  Linear model hyperparameter search…")
        hp_linear = lm.select_hyperparams(X_pre, y_pre, X_val, y_val, len(feature_cols))

    if needs_tree_hp:
        log.info("  Tree model hyperparameter search…")
        hp_tree = tm.select_hyperparams(X_pre, y_pre, X_val, y_val)

    log.info(f"  Hyperparams selected: linear={hp_linear}  tree={hp_tree}")
    return hp_linear, hp_tree


# ---------------------------------------------------------------------------
# Main training and prediction loop
# ---------------------------------------------------------------------------

def run(
    test_start: str = TEST_START,
    test_end: str = REPLICATE_TEST_END,
    models_to_run: list[str] = DEFAULT_MODELS,
    features_path: Path = FEATURES_PANEL_PATH,
    output_path: Path = PREDICTIONS_PATH,
    manifest_path: Path = RUN_MANIFEST_PATH,
) -> pd.DataFrame:
    """
    Run the full rolling-window training and prediction loop.

    Returns
    -------
    pd.DataFrame — predictions (permno, date, model, pred_ret)
    """
    t0 = time.time()
    log.info(f"Loading features panel from {features_path} …")
    panel = pd.read_parquet(features_path)
    panel = panel.sort_values(["date", "permno"]).reset_index(drop=True)

    feature_cols = get_feature_cols(panel)
    log.info(f"  {len(panel):,} rows, {len(feature_cols)} features, "
             f"{panel['date'].nunique()} months")

    ts_start = pd.Timestamp(test_start)
    ts_end = pd.Timestamp(test_end)
    test_years = sorted(
        panel.loc[
            (panel["date"] >= ts_start) & (panel["date"] <= ts_end),
            "date",
        ].dt.year.unique()
    )
    log.info(f"  Test window: {ts_start.date()} → {ts_end.date()}  "
             f"({len(test_years)} years)")

    # Hyperparameter selection
    hp_linear, hp_tree = select_all_hyperparams(panel, feature_cols, models_to_run)

    # Build model catalogue
    catalogue = build_model_catalogue(hp_linear, hp_tree, feature_cols, models_to_run)
    log.info(f"Models to train: {list(catalogue.keys())}")

    # ols3 uses only 3 features
    from src.models.linear_models import OLS3_FEATURES
    ols3_idx = [feature_cols.index(c) for c in OLS3_FEATURES if c in feature_cols]

    # Historical mean (training benchmark for OOS R²)
    pre_test_mean = float(
        panel.loc[panel["date"] < ts_start, "ret_exc"].mean()
    )
    log.info(f"  Pre-test mean excess return: {pre_test_mean:.4%}/mo")

    all_preds: list[pd.DataFrame] = []
    manifest: dict = {
        "run_timestamp": datetime.utcnow().isoformat(),
        "test_start": test_start,
        "test_end": test_end,
        "models": list(catalogue.keys()),
        "hp_linear": hp_linear,
        "hp_tree": hp_tree,
        "pre_test_mean_ret": pre_test_mean,
        "years": {},
    }

    for year in test_years:
        year_t0 = time.time()
        train_mask = panel["date"] < pd.Timestamp(f"{year}-01-01")
        test_mask = (
            (panel["date"].dt.year == year) &
            (panel["date"] >= ts_start) &
            (panel["date"] <= ts_end)
        )

        train = panel[train_mask]
        test = panel[test_mask]

        if len(test) == 0:
            continue

        X_train_full = train[feature_cols].values.astype(np.float32)
        y_train = train["ret_exc"].values.astype(np.float32)
        X_test_full = test[feature_cols].values.astype(np.float32)

        meta = test[["permno", "date"]].copy()

        for model_name, model_factory in catalogue.items():
            # OLS-3 uses only 3 features
            if model_name == "ols3":
                X_tr = X_train_full[:, ols3_idx]
                X_te = X_test_full[:, ols3_idx]
            else:
                X_tr = X_train_full
                X_te = X_test_full

            model = model_factory()
            model.fit(X_tr, y_train)
            preds = model.predict(X_te)
            if hasattr(preds, "ravel"):
                preds = preds.ravel()

            df_pred = meta.copy()
            df_pred["model"] = model_name
            df_pred["pred_ret"] = preds.astype(np.float32)
            all_preds.append(df_pred)

        elapsed = time.time() - year_t0
        manifest["years"][str(year)] = {"n_train": int(len(train)),
                                         "n_test": int(len(test)),
                                         "elapsed_s": round(elapsed, 1)}
        log.info(f"  {year}: {len(train):,} train obs, {len(test):,} test obs  "
                 f"({elapsed:.0f}s)")

    predictions = pd.concat(all_preds, ignore_index=True)
    predictions = predictions.sort_values(["model", "date", "permno"]).reset_index(drop=True)

    # Save predictions
    log.info(f"Writing {len(predictions):,} predictions to {output_path} …")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_path, index=False)

    # Save run manifest
    manifest["total_elapsed_s"] = round(time.time() - t0, 1)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"Run manifest written to {manifest_path}")
    log.info(f"Total time: {manifest['total_elapsed_s']:.0f}s")

    return predictions


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GKX (2020) rolling-window model training and prediction."
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=(
            f"Models to train. Use 'all' for all 13 models including NNs. "
            f"Choices: {ALL_MODELS + ['all']}"
        ),
    )
    p.add_argument(
        "--test-start",
        default=TEST_START,
        help="First test month (YYYY-MM-DD). Default: 1987-01-01",
    )
    p.add_argument(
        "--test-end",
        default=REPLICATE_TEST_END,
        help=f"Last test month (YYYY-MM-DD). Default: {REPLICATE_TEST_END}",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args()

    models_to_run = args.models
    if len(models_to_run) == 1 and models_to_run[0] == "all":
        models_to_run = ALL_MODELS

    invalid = [m for m in models_to_run if m not in ALL_MODELS]
    if invalid:
        raise ValueError(f"Unknown models: {invalid}. Valid choices: {ALL_MODELS}")

    run(
        test_start=args.test_start,
        test_end=args.test_end,
        models_to_run=models_to_run,
    )


if __name__ == "__main__":
    main()
