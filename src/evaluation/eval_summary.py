"""Quick evaluation summary — run once after train_eval.py completes."""
import warnings
warnings.filterwarnings("ignore")

import json
import pandas as pd
import numpy as np

from src.evaluation.metrics import oos_r2_pooled, monthly_ic, ic_stats

# ── Load data ──────────────────────────────────────────────────────────────
preds = pd.read_parquet("data/processed/predictions.parquet")
panel = pd.read_parquet("data/processed/features_panel.parquet")[["permno", "date", "ret_exc"]]

print("=== predictions.parquet ===")
print(f"Rows       : {len(preds):,}")
print(f"Models     : {sorted(preds['model'].unique())}")
print(f"Date range : {preds['date'].min().date()} -> {preds['date'].max().date()}")
print(f"Months     : {preds['date'].nunique()}")

with open("data/processed/run_manifest.json") as f:
    manifest = json.load(f)
r_bench = manifest["pre_test_mean_ret"]
print(f"Benchmark  : {r_bench:.4%}/mo (pre-test mean excess return)\n")

# ── Merge realised returns ──────────────────────────────────────────────────
df = preds.merge(panel, on=["permno", "date"], how="left")
print(f"Merge coverage: {df['ret_exc'].notna().mean():.1%}\n")

# ── Pooled OOS R² ──────────────────────────────────────────────────────────
r2 = oos_r2_pooled(df, r_bench)
print("=== Pooled OOS R² (1987-2016) ===")
for model, val in r2.items():
    print(f"  {model:<10} {val:+.4%}")

# ── Monthly IC ─────────────────────────────────────────────────────────────
ic_df = monthly_ic(df)
ic_s  = ic_stats(ic_df)
print("\n=== Monthly Rank IC ===")
print(ic_s.round(4).to_string())

# ── pred_ret distribution check ────────────────────────────────────────────
print("\n=== pred_ret std by model (signal dispersion) ===")
print(df.groupby("model")["pred_ret"].std().round(5).to_string())
