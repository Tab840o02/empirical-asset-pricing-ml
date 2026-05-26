"""
scripts/run_phase5c_eval.py
===========================
Evaluate Phase 5c (feature parsimony) predictions and compare with Phase 4.

Outputs (written to data/processed/):
  eval_parsimony_oos_r2.csv        — Phase 4 vs Phase 5c OOS R² with delta
  eval_parsimony_ic_stats.csv      — mean IC, IC std, ICIR per model (Phase 5c)
  eval_parsimony_portfolio_perf.csv — L/S portfolio performance (Phase 5c)

Run from project root:
  python scripts/run_phase5c_eval.py
"""

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.evaluation.metrics import oos_r2_pooled, monthly_ic, ic_stats
from src.evaluation.portfolio import build_portfolios, ls_returns, performance_table
from src.config import (
    PROCESSED_DIR, RAW_DIR,
    PREDICTIONS_PARSIMONY_PATH, FEATURES_PARSIMONIOUS_PATH,
    PREDICTIONS_PATH, FEATURES_PANEL_PATH,
    EVAL_PARSIMONY_OOS_R2_PATH, EVAL_PARSIMONY_IC_PATH, EVAL_PARSIMONY_PORT_PATH,
)

PROCESSED = Path(PROCESSED_DIR)
RAW = Path(RAW_DIR)

R_BENCH_5C = 0.006853425409644842
R_BENCH_4 = 0.006853426806628704

print("Loading parsimony predictions…")
pred5c = pd.read_parquet(PREDICTIONS_PARSIMONY_PATH)
panel5c = pd.read_parquet(FEATURES_PARSIMONIOUS_PATH)[["permno", "date", "ret_exc"]]
df5c = pred5c.merge(panel5c, on=["permno", "date"], how="left")
cov = df5c["ret_exc"].notna().mean()
print(f"  Rows: {len(pred5c):,}  Models: {sorted(pred5c['model'].unique())}")
print(f"  Merge coverage: {cov:.2%}\n")

print("Loading Phase-4 predictions…")
pred4 = pd.read_parquet(PREDICTIONS_PATH)
panel4 = pd.read_parquet(FEATURES_PANEL_PATH)[["permno", "date", "ret_exc"]]
df4 = pred4.merge(panel4, on=["permno", "date"], how="left")

# ── OOS R² comparison ──────────────────────────────────────────────────────
print("Computing pooled OOS R²…")
r2_5c = oos_r2_pooled(df5c, R_BENCH_5C)
r2_4 = oos_r2_pooled(df4, R_BENCH_4)

r2_cmp = pd.DataFrame({"phase4_r2": r2_4, "phase5c_r2": r2_5c})
r2_cmp["delta_r2"] = r2_cmp["phase5c_r2"] - r2_cmp["phase4_r2"]
r2_cmp.to_csv(EVAL_PARSIMONY_OOS_R2_PATH)
print(r2_cmp.round(6).to_string())

# ── IC stats ───────────────────────────────────────────────────────────────
print("\nComputing monthly rank IC (Phase 5c)…")
ic5c = ic_stats(monthly_ic(df5c))
ic5c.to_csv(EVAL_PARSIMONY_IC_PATH)
print(ic5c.round(4).to_string())

# ── Portfolio performance ──────────────────────────────────────────────────
print("\nBuilding decile portfolios (Phase 5c) — this may take a few minutes…")
crsp = pd.read_parquet(PROCESSED / "crsp_clean.parquet")[["permno", "date", "me_lag1"]]
ff_path = RAW / "ff_factors.parquet"
if not ff_path.exists():
    ff_path = PROCESSED / "ff_factors.parquet"
ff = pd.read_parquet(ff_path).reset_index()
deciles = build_portfolios(df5c, crsp)
ls = ls_returns(deciles)
port_perf = performance_table(ls, ff)
port_perf.to_csv(EVAL_PARSIMONY_PORT_PATH)
print(port_perf.round(4).to_string())

print("\nDone. CSVs written to data/processed/.")
