import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — derived from this file's location; never hard-code elsewhere
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SRC_DIR.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Date constants (YYYY-MM strings)
# ---------------------------------------------------------------------------
TRAIN_END = "1974-12"       # end of expanding training window warm-up
VAL_END = "1986-12"         # end of validation / hyperparameter selection window
TEST_END = "2016-12"        # end of replication test window (matches GKX)
EXT_END = None              # resolved at runtime from latest available CRSP month

# ---------------------------------------------------------------------------
# WRDS credentials — read from environment; never hard-code
# ---------------------------------------------------------------------------
WRDS_USER = os.environ["WRDS_USER"]

# ---------------------------------------------------------------------------
# Model hyperparameter defaults (can be overridden per run)
# ---------------------------------------------------------------------------
HYPERPARAMS = {
    "pcr": {"n_components": [3, 5, 10, 20, 50]},
    "pls": {"n_components": [3, 5, 10, 20, 50]},
    "enet": {"alpha": [0.001, 0.01, 0.1], "l1_ratio": [0.1, 0.5, 0.9]},
    "rf": {"max_depth": [1, 2, 4], "n_estimators": 300, "min_samples_leaf": 1000},
    "gbrt": {"learning_rate": [0.01, 0.1], "max_depth": [1, 2], "subsample": 0.5},
    "nn": {
        "hidden_units": 32,
        "dropout": 0.30,
        "lr": 0.001,
        "n_seeds": 10,
    },
}


# ---------------------------------------------------------------------------
# Smoke-test: verify WRDS connectivity
# ---------------------------------------------------------------------------
def connect_wrds():
    """Open and return a wrds.Connection(). First call caches credentials interactively."""
    import wrds
    return wrds.Connection(wrds_username=WRDS_USER)
