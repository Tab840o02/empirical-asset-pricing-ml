"""
scripts/build_unified_manifest.py
==================================
Sweep all run artefacts and produce a single, immutable audit manifest at
data/processed/unified_manifest.json.

For each model it records:
  - seeds used (hard-coded from known run history)
  - GKX-compliance flag (10-seed ensemble requirement)
  - environment where the model was trained (Local / Kaggle / Unknown)
  - per-year training counts and elapsed seconds (from whichever manifest
    captured that model; flagged "no_manifest" when absent)
  - total training wall-clock time
  - OOS R², mean IC, ICIR, and portfolio Sharpe (from the precomputed CSVs)
  - SHA-256 checksum of the model's prediction slice (sorted by permno, date)
    — provides an immutable fingerprint of the exact predictions on disk
  - row count and date range of predictions

Run:
    python scripts/build_unified_manifest.py

Output:
    data/processed/unified_manifest.json
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

PREDICTIONS_PATH = PROCESSED / "predictions.parquet"
OOS_R2_CSV       = PROCESSED / "eval_oos_r2_latest.csv"
IC_CSV           = PROCESSED / "eval_ic_stats_latest.csv"
PORTFOLIO_CSV    = PROCESSED / "eval_portfolio_perf_latest.csv"
OUTPUT_PATH      = PROCESSED / "unified_manifest.json"

# Individual run-manifest files in discovery order (most-recent first so the
# first hit for a model wins).
MANIFEST_FILES = [
    PROCESSED / "run_manifest.json",
    PROCESSED / "run_manifest_kaggle_nn3.json",
    PROCESSED / "run_manifest_before_kaggle_nn3_merge.json",
    ROOT / "artifacts" / "kaggle" / "data_stage" / "data" / "processed" / "run_manifest.json",
]

# ---------------------------------------------------------------------------
# Hard-coded provenance metadata
# -----------------------------------------------------------------------
# This table captures facts that were never written to a manifest file but
# are known from the project history.  Update here if you re-run anything.
#
# "seeds" meanings:
#   list of ints  → explicit random seeds passed to NNEnsemble
#   {"random_state": 42} → sklearn/LightGBM deterministic; no seed ensemble
#   None          → fully deterministic (OLS, PCR, PLS, ElasticNet w/ fixed alpha)
#
# "gkx_compliant":
#   True  → matches the paper's 10-seed NN ensemble (or deterministic linear/tree)
#   False → protocol break (< 10 seeds)
#   None  → not applicable (model not run)
# ---------------------------------------------------------------------------
KNOWN_METADATA: dict[str, dict] = {
    "ols3": {
        "seeds": None,
        "gkx_compliant": True,
        "environment": "Local",
        "notes": "Deterministic OLS on 3 characteristics. No random state.",
    },
    "ols_all": {
        "seeds": None,
        "gkx_compliant": True,
        "environment": "Local",
        "notes": "Deterministic OLS on all 94 characteristics. No random state.",
    },
    "pcr": {
        "seeds": None,
        "gkx_compliant": True,
        "environment": "Local",
        "notes": "Deterministic truncated SVD. No random state.",
    },
    "pls": {
        "seeds": None,
        "gkx_compliant": True,
        "environment": "Local",
        "notes": "Deterministic PLS. No random state.",
    },
    "enet": {
        "seeds": None,
        "gkx_compliant": True,
        "environment": "Local",
        "notes": "ElasticNet with fixed alpha/l1_ratio selected on validation set. No random state.",
    },
    "glm": {
        "seeds": None,
        "gkx_compliant": True,
        "environment": "Local",
        "notes": "Lasso (GLM) with fixed alpha selected on validation set. No random state.",
    },
    "rf": {
        "seeds": {"random_state": 42},
        "gkx_compliant": True,
        "environment": "Local",
        "notes": "RandomForestRegressor random_state=42, n_estimators=300, max_depth=2.",
    },
    "gbrt": {
        "seeds": {"random_state": 42},
        "gkx_compliant": True,
        "environment": "Local",
        "notes": "LGBMRegressor random_state=42, lr=0.01, max_depth=2. OOS R² negative — open divergence.",
    },
    "nn1": {
        "seeds": list(range(10)),       # seeds 0–9, 10-seed GKX-compliant ensemble
        "gkx_compliant": True,
        "environment": "Local",         # trained locally before Kaggle handoff
        "notes": "10-seed ensemble. GKX-compliant. No manifest captured; provenance from project log.",
    },
    "nn2": {
        "seeds": list(range(10)),       # seeds 0–9, 10-seed GKX-compliant ensemble
        "gkx_compliant": True,
        "environment": "Kaggle",        # run_manifest in artifacts/kaggle/data_stage
        "notes": "10-seed ensemble. GKX-compliant. Manifest captured in artifacts/kaggle/data_stage.",
    },
    "nn3": {
        "seeds": [0, 1, 2],            # 3-seed deadline mode
        "gkx_compliant": False,
        "environment": "Kaggle",
        "notes": (
            "3-seed deadline-mode ensemble. NOT GKX-compliant (paper requires 10 seeds). "
            "Cross-model comparisons involving NN3 are unreliable due to seed-variance."
        ),
    },
    "nn4": {
        "seeds": [0, 1, 2],            # 3-seed deadline mode
        "gkx_compliant": False,
        "environment": "Local",
        "notes": (
            "3-seed deadline-mode ensemble. NOT GKX-compliant (paper requires 10 seeds). "
            "Cross-model comparisons involving NN4 are unreliable due to seed-variance."
        ),
    },
    "nn5": {
        "seeds": None,
        "gkx_compliant": None,
        "environment": None,
        "notes": "Not run. Excluded from all analysis.",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_manifests() -> dict[str, dict]:
    """
    Read all manifest files and index per-model data by model name.
    Returns dict: model_name -> manifest_entry dict (keys: run_timestamp,
    environment_from_manifest, years, total_elapsed_s, source_file).
    The first manifest that mentions a model wins (most-recent-first ordering).
    """
    per_model: dict[str, dict] = {}
    for mpath in MANIFEST_FILES:
        if not mpath.exists():
            print(f"  [skip] manifest not found: {mpath.relative_to(ROOT)}")
            continue
        with open(mpath) as f:
            data = json.load(f)
        models = data.get("models", [])
        years  = data.get("years", {})
        ts     = data.get("run_timestamp", "unknown")
        elapsed_by_year = {y: v.get("elapsed_s", 0) for y, v in years.items()}
        total_elapsed   = sum(elapsed_by_year.values())
        for model in models:
            if model in per_model:
                continue  # already captured from a higher-priority manifest
            per_model[model] = {
                "run_timestamp":  ts,
                "source_manifest": str(mpath.relative_to(ROOT)),
                "total_elapsed_s": round(total_elapsed, 1),
                "years":          elapsed_by_year,
                "n_train_by_year": {
                    y: v.get("n_train") for y, v in years.items()
                },
            }
        print(f"  [ok]   {mpath.relative_to(ROOT)} → models: {models}")
    return per_model


def load_eval_metrics() -> dict[str, dict]:
    """Load the three precomputed evaluation CSVs, return dict keyed by model."""
    metrics: dict[str, dict] = {}

    if OOS_R2_CSV.exists():
        r2 = pd.read_csv(OOS_R2_CSV, index_col=0)
        for model, row in r2.iterrows():
            metrics.setdefault(model, {})["oos_r2"] = round(float(row["oos_r2"]), 8)

    if IC_CSV.exists():
        ic = pd.read_csv(IC_CSV)
        for _, row in ic.iterrows():
            m = row["model"]
            metrics.setdefault(m, {})["mean_ic"]  = round(float(row["mean_ic"]),  6)
            metrics.setdefault(m, {})["std_ic"]   = round(float(row["std_ic"]),   6)
            metrics.setdefault(m, {})["icir"]     = round(float(row["icir"]),     6)

    if PORTFOLIO_CSV.exists():
        port = pd.read_csv(PORTFOLIO_CSV)
        for _, row in port.iterrows():
            m = row["model"]
            metrics.setdefault(m, {})["sharpe"]       = round(float(row["sharpe"]),      4)
            metrics.setdefault(m, {})["alpha_annual"]  = round(float(row["alpha_annual"]), 6)
            metrics.setdefault(m, {})["t_alpha"]       = round(float(row["t_alpha"]),      4)
            metrics.setdefault(m, {})["p_alpha"]       = round(float(row["p_alpha"]),      6)
            metrics.setdefault(m, {})["n_portfolio_months"] = int(row["n_months"])

    return metrics


def compute_prediction_checksum(df_model: pd.DataFrame) -> str:
    """
    Compute a SHA-256 fingerprint of a model's predictions.

    The slice is sorted by (permno, date) before hashing so the fingerprint
    is stable regardless of the order rows were written to disk.  We hash
    the raw bytes of the numpy arrays for permno, date (as int64 epoch ns),
    and pred_ret (float64).
    """
    df_sorted = df_model.sort_values(["permno", "date"]).reset_index(drop=True)
    h = hashlib.sha256()
    h.update(df_sorted["permno"].to_numpy().astype("int64").tobytes())
    h.update(df_sorted["date"].to_numpy().astype("int64").tobytes())
    h.update(df_sorted["pred_ret"].to_numpy().astype("float64").tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("build_unified_manifest.py")
    print("=" * 60)

    # 1. Load manifests
    print("\n[1] Loading run manifests ...")
    manifest_data = load_manifests()

    # 2. Load eval metrics
    print("\n[2] Loading evaluation metrics ...")
    eval_metrics = load_eval_metrics()
    print(f"  Metrics found for {len(eval_metrics)} models.")

    # 3. Load predictions and compute per-model checksums
    print("\n[3] Loading predictions.parquet and computing checksums ...")
    if not PREDICTIONS_PATH.exists():
        print(f"  ERROR: {PREDICTIONS_PATH} not found. Aborting.", file=sys.stderr)
        sys.exit(1)

    predictions = pd.read_parquet(PREDICTIONS_PATH)
    print(f"  Total rows: {len(predictions):,}")

    checksum_data: dict[str, dict] = {}
    for model, grp in predictions.groupby("model"):
        n_rows  = len(grp)
        n_months = grp["date"].nunique()
        date_min = grp["date"].min().strftime("%Y-%m")
        date_max = grp["date"].max().strftime("%Y-%m")
        sha256   = compute_prediction_checksum(grp)
        checksum_data[model] = {
            "n_rows":   n_rows,
            "n_months": n_months,
            "date_min": date_min,
            "date_max": date_max,
            "sha256":   sha256,
        }
        print(f"  {model:10s}  rows={n_rows:>9,}  months={n_months}  sha256={sha256[:16]}...")

    # 4. Assemble unified manifest
    print("\n[4] Assembling unified manifest ...")
    models_entry: dict[str, dict] = {}

    all_model_names = sorted(
        set(KNOWN_METADATA) | set(manifest_data) | set(eval_metrics) | set(checksum_data)
    )

    for model in all_model_names:
        km = KNOWN_METADATA.get(model, {})
        md = manifest_data.get(model)
        ev = eval_metrics.get(model, {})
        cs = checksum_data.get(model)

        seeds_val = km.get("seeds")
        if isinstance(seeds_val, list):
            n_seeds_val = len(seeds_val)
            seeds_label = seeds_val
        elif isinstance(seeds_val, dict):
            n_seeds_val = "deterministic"
            seeds_label = seeds_val
        else:
            n_seeds_val = "deterministic"
            seeds_label = None

        entry: dict = {
            "seeds":           seeds_label,
            "n_seeds":         n_seeds_val,
            "gkx_seed_compliant": km.get("gkx_compliant"),
            "environment":     km.get("environment"),
            "notes":           km.get("notes"),
        }

        # Manifest provenance
        if md:
            entry["run_timestamp"]    = md["run_timestamp"]
            entry["source_manifest"]  = md["source_manifest"]
            entry["total_elapsed_s"]  = md["total_elapsed_s"]
            entry["years"]            = md["years"]
            entry["n_train_by_year"]  = md["n_train_by_year"]
        else:
            entry["run_timestamp"]   = "no_manifest"
            entry["source_manifest"] = "no_manifest"
            entry["total_elapsed_s"] = None
            entry["years"]           = {}
            entry["n_train_by_year"] = {}

        # Eval metrics
        entry["metrics"] = ev if ev else {}

        # Prediction fingerprint
        if cs:
            entry["predictions"] = cs
        else:
            entry["predictions"] = None

        models_entry[model] = entry

    unified = {
        "generated_at":        datetime.now(timezone.utc).isoformat(),
        "predictions_path":    str(PREDICTIONS_PATH.relative_to(ROOT)),
        "total_prediction_rows": int(len(predictions)),
        "protocol_notes": (
            "NN1/NN2 use 10-seed GKX-compliant ensembles. "
            "NN3/NN4 use 3-seed deadline-mode ensembles — NOT GKX-compliant; "
            "cross-model comparisons involving these models are unreliable. "
            "NN5 was not run. "
            "Tree model OOS R² is negative; root cause unresolved (open divergence). "
            "Linear/tree models are deterministic (fixed random_state=42 for RF/GBRT)."
        ),
        "models": models_entry,
    }

    # 5. Write output
    with open(OUTPUT_PATH, "w") as f:
        json.dump(unified, f, indent=2)

    print(f"\n[5] Written: {OUTPUT_PATH.relative_to(ROOT)}")
    print("=" * 60)

    # 6. Print a human-readable summary table
    print("\nSummary\n" + "-" * 60)
    header = f"{'Model':<10} {'Seeds':>6} {'GKX?':>5} {'Env':>6}  {'OOS R²':>9}  {'Sharpe':>7}  {'SHA256 prefix'}"
    print(header)
    print("-" * len(header))
    for model, entry in sorted(unified["models"].items()):
        n_seeds  = entry["n_seeds"]
        n_seeds_str = str(n_seeds) if isinstance(n_seeds, int) else n_seeds[:4]
        compliant = {True: "yes", False: "NO", None: "n/a"}[entry["gkx_seed_compliant"]]
        env      = (entry["environment"] or "n/a")[:6]
        r2       = entry["metrics"].get("oos_r2")
        sharpe   = entry["metrics"].get("sharpe")
        r2_str   = f"{r2*100:+.4f}%" if r2 is not None else "    n/a"
        sh_str   = f"{sharpe:+.3f}" if sharpe is not None else "    n/a"
        pred     = entry.get("predictions")
        sha_str  = pred["sha256"][:12] + "..." if pred else "no predictions"
        print(f"{model:<10} {n_seeds_str:>6} {compliant:>5} {env:>6}  {r2_str:>9}  {sh_str:>7}  {sha_str}")


if __name__ == "__main__":
    main()
