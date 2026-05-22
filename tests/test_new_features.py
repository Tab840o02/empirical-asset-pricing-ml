"""Quick functional test of the 10 new features on a small panel subset."""
import pandas as pd
import numpy as np

panel = pd.read_parquet("data/processed/merged_panel.parquet")
panel["date"] = pd.to_datetime(panel["date"])
panel["date"] = panel["date"].dt.to_period("M").dt.to_timestamp("M")
permnos = panel["permno"].unique()[:500]
panel = panel[panel["permno"].isin(permnos)].copy()
panel["me"] = panel["me"].where(panel["me"] > 0, other=np.nan)
print(f"Rows: {len(panel):,}  Permnos: {panel['permno'].nunique()}")

from src.features import value_features, profitability_features, momentum_features

print("\n--- value_features ---")
val = value_features.compute(panel)
for f in ["pchsale_pchrect", "pchcurrat", "sin"]:
    s = val[f]
    print(f"  {f}: {s.notna().mean():.1%} non-null  mean={s.mean():.4f}  range=[{s.min():.3f}, {s.max():.3f}]")

print("\n--- profitability_features ---")
prof = profitability_features.compute(panel)
for f in ["depr", "pchdepr", "rna", "rsup", "nincr", "ps", "orgcap"]:
    s = prof[f]
    print(f"  {f}: {s.notna().mean():.1%} non-null  mean={s.mean():.4f}  range=[{s.min():.3f}, {s.max():.3f}]")

print("\n--- momentum_features (herf) ---")
mom = momentum_features.compute(panel)
s = mom["herf"]
print(f"  herf: {s.notna().mean():.1%} non-null  mean={s.mean():.4f}  range=[{s.min():.3f}, {s.max():.3f}]")

# Sanity checks
assert val["sin"].isin([0.0, 1.0, np.nan]).all(), "sin should be 0/1/NaN"
assert prof["nincr"].between(0, 8).dropna().all(), "nincr should be in [0,8]"
assert prof["ps"].between(0, 9).dropna().all(), "ps should be in [0,9]"
assert (s.dropna() <= 1.0).all(), "herf should be <= 1 (HHI <= 1)"
assert (s.dropna() >= 0.0).all(), "herf should be >= 0"
print("\nAll sanity checks passed.")
