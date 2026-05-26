"""
scripts/run_final_eval.py
=========================
Recompute all Phase-4 evaluation outputs from the canonical predictions file.

Outputs (written to data/processed/):
  eval_oos_r2_latest.csv       — pooled OOS R² per model
  eval_ic_stats_latest.csv     — mean IC, IC std, ICIR per model
  eval_pred_std_latest.csv     — prediction dispersion per model
  eval_portfolio_perf_latest.csv — L/S portfolio performance per model

Run from project root:
  python scripts/run_final_eval.py
"""

import warnings
warnings.filterwarnings("ignore")

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from src.evaluation.metrics import oos_r2_pooled, monthly_ic, ic_stats, summary_table
from src.evaluation.portfolio import build_portfolios, ls_returns, performance_table
from src.config import PROCESSED_DIR, RAW_DIR

PROCESSED = Path(PROCESSED_DIR)
RAW = Path(RAW_DIR)

# ── Benchmark: pre-test mean excess return ─────────────────────────────────
# Value stored in the NN5 run_manifest (consistent across all model runs).
R_BENCH = 0.006853426806628704

print("Loading data…")
preds = pd.read_parquet(PROCESSED / "predictions.parquet")
panel = pd.read_parquet(PROCESSED / "features_panel.parquet")[["permno", "date", "ret_exc"]]
crsp  = pd.read_parquet(PROCESSED / "crsp_clean.parquet")[["permno", "date", "me_lag1"]]
ff    = pd.read_parquet(RAW / "ff_factors.parquet").reset_index()

print(f"  Predictions: {len(preds):,} rows, {sorted(preds['model'].unique())}")

# Merge realized returns
df = preds.merge(panel, on=["permno", "date"], how="left")
print(f"  Merge coverage: {df['ret_exc'].notna().mean():.2%}\n")

# ── OOS R² ─────────────────────────────────────────────────────────────────
print("Computing pooled OOS R²…")
r2 = oos_r2_pooled(df, R_BENCH)
print(r2.apply(lambda x: f"{x:+.6f}").to_string())
r2.to_frame("oos_r2").to_csv(PROCESSED / "eval_oos_r2_latest.csv")
print()

# ── IC stats ───────────────────────────────────────────────────────────────
print("Computing monthly rank IC…")
ic_df = monthly_ic(df)
ic_s  = ic_stats(ic_df)
print(ic_s.round(4).to_string())
ic_s.to_csv(PROCESSED / "eval_ic_stats_latest.csv")
print()

# ── Prediction dispersion ──────────────────────────────────────────────────
print("Computing prediction dispersion…")
pred_std = df.groupby("model")["pred_ret"].std().rename("pred_std").round(6)
print(pred_std.to_string())
pred_std.to_frame().to_csv(PROCESSED / "eval_pred_std_latest.csv")
print()

# ── Portfolio performance ───────────────────────────────────────────────────
print("Building decile portfolios (this may take a few minutes)…")
deciles = build_portfolios(df, crsp)
ls = ls_returns(deciles)

# Month coverage sanity check
months_by_model = ls.groupby("model")["date"].nunique()
print(f"  L/S months per model:\n{months_by_model.to_string()}")

print("\nComputing FF5 alpha and performance table…")
perf = performance_table(ls, ff, factor_model="ff5")
print(perf.round(4).to_string())
perf.to_csv(PROCESSED / "eval_portfolio_perf_latest.csv")
print()

print("=== ALL DONE ===")
print(f"Outputs written to {PROCESSED}/eval_*_latest.csv")
