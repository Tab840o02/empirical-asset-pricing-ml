"""
src/models/tree_models.py
==========================
Tree-based predictors from GKX (2020) Table 3.

Models
------
rf    – Random Forest (scikit-learn)
gbrt  – Gradient Boosted Regression Trees (LightGBM — much faster than sklearn)

Hyperparameters are selected by `select_hyperparams()` using the
validation window before the test period begins.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def make_rf(max_depth: int = 2) -> Any:
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=max_depth,
        min_samples_leaf=1000,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )


def make_gbrt(learning_rate: float = 0.01, max_depth: int = 2) -> Any:
    """
    LightGBM regressor.  Interface is sklearn-compatible.
    subsample=0.5 and min_child_samples=1000 match GKX §3.
    """
    from lightgbm import LGBMRegressor
    return LGBMRegressor(
        n_estimators=300,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.5,
        subsample_freq=1,
        min_child_samples=1000,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )


# ---------------------------------------------------------------------------
# Hyperparameter selection (validation window)
# ---------------------------------------------------------------------------

def _oos_r2(y_true: np.ndarray, y_pred: np.ndarray, y_bench: float) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_bench) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def select_hyperparams(
    X_pre_val: np.ndarray,
    y_pre_val: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict[str, Any]:
    """
    Select max_depth for RF and (learning_rate, max_depth) for GBRT using
    a hold-out validation window.

    Returns
    -------
    dict with keys: rf_depth, gbrt_lr, gbrt_depth
    """
    y_bench = float(y_pre_val.mean())
    result: dict[str, Any] = {}

    # --- Random Forest ---
    best_r2, best_depth = -np.inf, 2
    for depth in [1, 2, 4]:
        m = make_rf(max_depth=depth)
        m.fit(X_pre_val, y_pre_val)
        r2 = _oos_r2(y_val, m.predict(X_val), y_bench)
        if r2 > best_r2:
            best_r2, best_depth = r2, depth
    result["rf_depth"] = best_depth
    log.info(f"  RF best max_depth={best_depth} (val R²={best_r2:.4%})")

    # --- GBRT (LightGBM) ---
    best_r2 = -np.inf
    best_lr, best_d = 0.01, 2
    for lr in [0.01, 0.1]:
        for d in [1, 2]:
            m = make_gbrt(learning_rate=lr, max_depth=d)
            m.fit(X_pre_val, y_pre_val)
            r2 = _oos_r2(y_val, m.predict(X_val), y_bench)
            if r2 > best_r2:
                best_r2, best_lr, best_d = r2, lr, d
    result["gbrt_lr"] = best_lr
    result["gbrt_depth"] = best_d
    log.info(f"  GBRT best lr={best_lr}, max_depth={best_d} (val R²={best_r2:.4%})")

    return result
