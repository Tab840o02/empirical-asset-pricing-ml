"""
src/extensions/feature_parsimony.py
===================================
Phase 5c — Feature parsimony.

Build a reduced feature panel using a compact, interpretable shortlist of
characteristics selected from validation-period RF importance and an economic
allow-list.  The reduced panel is then used to retrain the Phase 4 model
catalogue on the same rolling window.

This implementation is optimized for Kaggle execution: it uses a sampled
validation set for permutation importance and writes a standalone reduced
panel so the expensive selection step runs once per environment.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.config import (
    FEATURES_PANEL_PATH,
    FEATURES_PARSIMONIOUS_PATH,
    PREDICTIONS_PARSIMONY_PATH,
    PROCESSED_DIR,
    RUN_MANIFEST_PARSIMONY_PATH,
)
from src.models.tree_models import make_rf

log = logging.getLogger(__name__)

ALLOW_LIST: tuple[str, ...] = (
    "mom12m", "mom1m", "chmom", "indmom",
    "bm", "ep", "cfp",
    "roaq", "gp", "roe",
    "agr", "invest", "noa",
    "me", "ill", "idiovol", "beta", "retvol",
)

TARGET_FEATURE_COUNT_MIN = 15
TARGET_FEATURE_COUNT_MAX = 20
SAMPLE_ROWS = 100_000


def _feature_cols(panel: pd.DataFrame) -> list[str]:
    return [c for c in panel.columns if c not in {"permno", "date", "ret_exc"}]


def _load_panel(path: Path = FEATURES_PANEL_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing features panel: {path}")
    panel = pd.read_parquet(path)
    panel["date"] = pd.to_datetime(panel["date"]).dt.to_period("M").dt.to_timestamp("M")
    return panel.sort_values(["date", "permno"]).reset_index(drop=True)


def _validation_sample(val: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    if len(val) <= SAMPLE_ROWS:
        return val
    return val.sample(n=SAMPLE_ROWS, random_state=random_state)


def select_parsimonious_features(panel: pd.DataFrame | None = None) -> list[str]:
    """Return a compact shortlist of 15–20 economically interpretable features."""
    if panel is None:
        panel = _load_panel()

    feature_cols = _feature_cols(panel)
    pre_val = panel[panel["date"] <= pd.Timestamp("1974-12-31")]
    val = panel[
        (panel["date"] >= pd.Timestamp("1975-01-31"))
        & (panel["date"] <= pd.Timestamp("1986-12-31"))
    ]

    if pre_val.empty or val.empty:
        raise ValueError("Validation window is empty; cannot select parsimonious features.")

    X_pre = pre_val[feature_cols].astype(np.float32)
    y_pre = pre_val["ret_exc"].astype(np.float32)

    rf = make_rf(max_depth=2)
    log.info("Fitting RF on pre-validation window for parsimony selection …")
    rf.fit(X_pre, y_pre)

    val_sample = _validation_sample(val)
    X_val = val_sample[feature_cols].astype(np.float32)
    y_val = val_sample["ret_exc"].astype(np.float32)

    log.info(
        "Computing permutation importance on %d validation rows / %d features …",
        len(val_sample),
        len(feature_cols),
    )
    importance = permutation_importance(
        rf,
        X_val,
        y_val,
        n_repeats=1,
        random_state=42,
        scoring="r2",
        n_jobs=-1,
    )
    ranked = pd.Series(importance.importances_mean, index=feature_cols).sort_values(ascending=False)

    allow_ranked = [feat for feat in ranked.index if feat in ALLOW_LIST]
    if not allow_ranked:
        raise ValueError("No allow-list features survived the importance ranking.")

    chosen: list[str] = []
    upper = TARGET_FEATURE_COUNT_MIN
    while len(chosen) < TARGET_FEATURE_COUNT_MIN and upper <= len(feature_cols):
        chosen = [feat for feat in ranked.index[:upper] if feat in ALLOW_LIST]
        upper += 5

    if len(chosen) < TARGET_FEATURE_COUNT_MIN:
        chosen = allow_ranked[:TARGET_FEATURE_COUNT_MIN]

    chosen = chosen[:TARGET_FEATURE_COUNT_MAX]

    log.info("Selected %d parsimonious features: %s", len(chosen), ", ".join(chosen))
    return chosen


def build_parsimonious_panel(
    panel: pd.DataFrame | None = None,
    output_path: Path = FEATURES_PARSIMONIOUS_PATH,
    selected_features_path: Path | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build and persist a reduced feature panel for Phase 5c."""
    if panel is None:
        panel = _load_panel()

    chosen = select_parsimonious_features(panel)
    cols = ["permno", "date", "ret_exc", *chosen]
    reduced = panel[cols].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    reduced.to_parquet(output_path, index=False)

    if selected_features_path is None:
        selected_features_path = output_path.with_name("phase5c_selected_features.json")
    with open(selected_features_path, "w", encoding="utf-8") as f:
        json.dump({"selected_features": chosen}, f, indent=2)

    log.info("Wrote parsimonious panel to %s", output_path)
    log.info("Wrote selected features to %s", selected_features_path)
    return reduced, chosen


def run_phase5c_training(
    models_to_run: list[str],
    panel_path: Path = FEATURES_PARSIMONIOUS_PATH,
    predictions_path: Path = PREDICTIONS_PARSIMONY_PATH,
    manifest_path: Path = RUN_MANIFEST_PARSIMONY_PATH,
    append: bool = False,
    resume: bool = False,
) -> pd.DataFrame:
    """Retrain the Phase 4 model catalogue on the parsimonious feature set."""
    from src.models.train_eval import run as train_eval_run

    if not panel_path.exists():
        build_parsimonious_panel(output_path=panel_path)

    return train_eval_run(
        test_start="1987-01-01",
        test_end="2016-12-31",
        models_to_run=models_to_run,
        features_path=panel_path,
        output_path=predictions_path,
        manifest_path=manifest_path,
        append=append,
        resume=resume,
    )