# GKX (2020) Replication — Results vs Paper Comparison

> **Run date:** 2026-05-26 (Phase 4 complete — all 13 models trained)  
> **Features:** **94/94** GKX characteristics (all implemented)  
> **Test window:** 1987-01 → 2016-12 (360 months, matching GKX exactly)  
> **Training scheme:** Expanding window, hyperparams selected on 1957–1974 train / 1975–1986 val  
> **Models trained:** OLS-3, OLS-all, PCR, PLS, ElasticNet, GLM, RF, GBRT, NN1, NN2, NN3, NN4, NN5  
> **Total predictions:** 18,165,186 stock-month observations  
> **Runtime:** ~84 min full non-NN run + NN1/NN2 full expanding-window runs + NN3 (Kaggle, 23,331.6s) + NN4 (local, 20,003s) + NN5 (local, 38,665.7s)
>
> ⚠️ **Replication status: PARTIAL.** This document reports a partial replication with several unresolved comparability gaps. See Section 0 for the full methodological deviation register before interpreting any results.

---

## 0. Methodological Deviations and Limitations

> **This section must be read before interpreting any quantitative comparison in this document.** The replication is partial. Several deviations from GKX (2020) are known, quantified where possible, and disclosed below. Claims previously stated as "✅ Replicated" have been restated as approximate matches subject to the caveats enumerated here.

### 0.1 Neural Network Ensemble Protocol Break (P0 — Cross-Model Comparisons Unreliable)

GKX (2020) trains each neural network model (NN1–NN5) as an ensemble of **10 independently seeded runs**, and reports the average prediction across all 10 seeds as the model output. This is explicitly stated in the paper as a variance-reduction measure; single-seed NN runs are not considered valid model evaluations under the GKX protocol.

**This replication used a mixed protocol:**

| Model | Seeds Used | GKX Protocol | Status |
|---|---|---|---|
| NN1 | 10 (seeds 0–9) | 10 | ✅ Compliant |
| NN2 | 10 (seeds 0–9) | 10 | ✅ Compliant |
| NN3 | 3 (seeds 0–2) | 10 | ❌ Deadline mode — 7 seeds missing |
| NN4 | 3 (seeds 0–2) | 10 | ❌ Deadline mode — 7 seeds missing |
| NN5 | 3 (seeds 0–2) | 10 | ❌ Deadline mode — 7 seeds missing |

**Inferential consequence:** A 3-seed ensemble has substantially higher variance than a 10-seed ensemble. The reported OOS R² for NN3 (+0.023%), NN4 (+0.003%), and NN5 (+0.007%) cannot be compared on equal footing against GKX's 10-seed results or against our own NN1/NN2 results. Any conclusion that "deeper networks underperform shallower networks in this dataset" is not supported at the current seed count — the observed deterioration could partially reflect seed-variance rather than a genuine architectural effect. **No cross-model ranking involving NN3, NN4, or NN5 should be treated as conclusive.**

### 0.2 Tree Model Failure — Open Divergence (P1 — Result Unreliable)

Both tree-based models produce **negative pooled OOS R²**, in sharp contrast to GKX Table 3:

| Model | Our OOS R² | GKX OOS R² | Gap |
|---|---|---|---|
| RF | −0.509% | +0.39% | −0.899 pp |
| GBRT | −0.989% | +0.34% | −1.329 pp |

Root cause is **unresolved**. Investigated and ruled out: stump-only RF depth, GBRT learning-rate pathology (fixed from −3.80% to −0.989%), missing features (all 94 implemented). Neither tree model was retrained with alternative specifications before this deadline. This is treated as an **open divergence requiring future research**, not a concluded finding. Portfolio results for RF and GBRT (Sharpe 0.10 and 0.04 respectively) are reported for completeness but should not be used for cross-model inference.

### 0.3 CRSP–Compustat Linkage: Custom Implementation vs. Canonical CCM (P1 — Universe Composition Unverified)

GKX use the WRDS CCM (Center for Research in Security Prices–Compustat Merged) link table (`comp.ccmxpf_lnkhist`) as the canonical PERMNO–GVKEY bridge. This replication uses a **custom CUSIP-primary / Jaccard name-similarity fallback** linker (`src/data/crsp_compustat_linker.py`).

The delta between the custom linker and canonical CCM has **not been quantified**. Universe composition (number of firms per month, coverage by size decile) may differ from GKX. This affects all models equally but introduces a systematic comparability gap that is not eliminated by matching OOS R² values closely.

### 0.4 Annual Compustat Attachment Window: Missing Upper Cap (P2 — Stale Filing Risk)

`project_plan.md` specifies that annual Compustat data should be attached to the CRSP panel only when `public_date ≤ crsp_month ≤ public_date + 12 months`, enforcing a 12-month upper cap to prevent very stale filings from propagating forward. The implementation in `src/data/ccm_merger.py` uses `merge_asof` with `direction="backward"` but **does not enforce the 12-month upper cap**. A fiscal-year filing from, say, June 1992 could in principle be attached to a CRSP observation in December 1998 if no newer filing is available. The practical impact is expected to be small (most firms file annually) but has not been measured.

### 0.5 Quarterly Fallback Logic: Planned vs. Implemented (P2 — Feature Imputation Divergence)

**Planned behavior** (`project_plan.md §Phase 2`): When a quarterly Compustat value is missing, fall back to the most recent annual filing value for that variable.

**Implemented behavior**: Quarterly columns are left as `NaN` in `merged_panel.parquet`. `src/features/feature_assembler.py` zero-imputes all `NaN` values after cross-sectional rank normalization. Post-normalization zero-imputation is equivalent to assigning the cross-sectional median rank — not the annual filing value. For firms with systematically missing quarterly data, this may introduce a systematic attenuation bias in quarterly-based features (e.g., `roaq`, `stdacc`, `rsup`).

### 0.6 Portfolio Date Normalization Bug — Fixed (P0 — Previously Affected All Results)

A date-format mismatch between CRSP trading dates (raw business-day end, e.g., `1987-05-29`) and prediction dates (calendar month-end, `1987-05-31`) caused the portfolio merge to silently drop **108 of 360 months** (30% of the test window). This affected all previously reported L/S returns and Sharpe ratios. **Fixed** in `src/evaluation/portfolio.py` via `pd.offsets.MonthEnd(0)` normalization. All portfolio results in this document reflect the corrected 360-month window. Prior versions of this document should be disregarded for portfolio metrics.

### 0.7 Linear Model L/S Returns: Persistent Upward Divergence (P2 — Magnitude Unexplained)

After the portfolio date-fix, linear model L/S monthly returns remain **2.7–4.4× the GKX Table 3 benchmark** (e.g., PCR: 1.10% vs ~0.25%/month). This is not consistent with a small normalization difference. Probable causes include differences in universe construction (see §0.3), value-weighting methodology, or decile boundary definitions, none of which have been isolated. These results are reported without the claim that they match GKX.

### 0.8 NN1/NN2 L/S Returns: Large Upward Divergence (P1 — Structural Concern)

NN1 and NN2 produce L/S monthly returns of 2.50% and 2.44% respectively, versus GKX's ~0.39–0.40%/month — approximately 6× the paper's values. OOS R² for these models is close to GKX (+0.344% and +0.178% vs +0.39% and +0.40%), which makes the portfolio magnitude divergence difficult to explain through prediction quality alone. This divergence is flagged as an unresolved structural concern.

### 0.9 Claim Restatement Summary

| Prior claim | Restated claim |
|---|---|
| "✅ Replicated — within ±0.09 pp of GKX" (linear OOS R²) | Approximate match on OOS R² metric only; portfolio magnitudes diverge substantially (see §0.7) |
| "✅ Exact match" (GLM OOS R²) | OOS R² point estimate matches; portfolio performance does not |
| "✅ Close to paper" (NN1 OOS R²) | OOS R² close (+0.344% vs +0.39%); portfolio return 6× GKX; 10-seed protocol compliant |
| "⚠️ Positive but below expectation" (NN2) | 10-seed compliant; OOS R² below GKX and below NN1; portfolio magnitude unexplained |
| "⚠️ Completed with reduced-rigor setup" (NN3/NN4) | 3-seed non-compliant ensembles; results not comparable to GKX or to NN1/NN2; reported for transparency only |
| "⏳ Pending" (NN5) | Now complete; 3-seed non-compliant; OOS R² +0.007%, Sharpe 0.10, alpha not significant; consistent with NN4 pattern |
| "❌ Still not replicated" (tree models) | Open divergence; root cause unresolved; excluded from cross-model ranking conclusions |

---

## 1. Methodology Differences vs Paper

| Aspect | GKX (2020) | This Replication |
|---|---|---|
| Features | 94 characteristics | **94 characteristics** (all implemented) |
| GBRT implementation | Not specified (likely sklearn GBRT) | LightGBM `LGBMRegressor` |
| Portfolio weighting | Value-weighted (lagged market cap) | Value-weighted (lagged `me_lag1`) |
| Portfolio months covered | 360 (1987–2016) | **360** — date-normalization fix applied to `portfolio.py` |
| NN models | NN1–NN5 (ensemble of 10 seeds each) | NN1 and NN2 complete with 10 seeds (GKX-compliant); NN3, NN4, and NN5 complete with a 3-seed deadline-mode ensemble (not GKX-compliant) |

---

## 2. Pooled OOS R² (1987–2016)

GKX reports this as the primary in-sample predictability metric (Table 3, column "R²OOS"). Values are in percent.

$$R^2_{\text{OOS}} = 1 - \frac{\sum_t (r_{i,t} - \hat{r}_{i,t})^2}{\sum_t (r_{i,t} - \bar{r}_{\text{train}})^2}$$

| Model | **Our R²OOS** | **GKX Table 3** | Δ vs GKX | Status |
|---|---|---|---|---|
| OLS-3 | +0.025% | +0.06% | −0.035 pp | ✅ Close |
| OLS-all | +0.152% | +0.09% | +0.062 pp | ✅ Close (more features) |
| PCR | +0.169% | +0.19% | −0.021 pp | ✅ Very close |
| PLS | +0.163% | +0.25% | −0.087 pp | ✅ Close |
| ElasticNet | +0.180% | +0.22% | −0.040 pp | ✅ Close |
| GLM (Lasso) | +0.063% | +0.06% | +0.003 pp | ✅ **Exact match** |
| RF | −0.509% | +0.39% | −0.899 pp | ❌ Negative |
| GBRT | −0.989% | +0.34% | −1.329 pp | ⚠️ Improved, still negative |
| NN1 | +0.344% | +0.39% | −0.046 pp | ✅ Close |
| NN2 | +0.178% | +0.40% | −0.222 pp | ⚠️ Positive, below GKX and below NN1 |
| NN3 | +0.023% | +0.41% | −0.387 pp | ⚠️ Positive, far below GKX |
| NN4 | +0.003% | +0.45% | −0.447 pp | ❌ Near-zero OOS R² |
| NN5 | +0.007% | +0.55% | −0.543 pp | ❌ Near-zero OOS R² |

**Key observations:**
- All **linear models** match GKX within ±0.09 pp. **GLM** reproduces the paper's +0.06% exactly.
- **RF** remains negative at `max_depth=2`; the stump-only explanation was not the main bottleneck.
- **GBRT** improved from −3.80% to −0.99% after restricting to `lr=0.01`, reducing prediction dispersion from 0.0363 to 0.0158. Still negative vs GKX's +0.34%.
- **NN1** remains the strongest neural model so far at +0.344% vs GKX +0.39%.
- **NN2** is positive (+0.178%) but underperforms both GKX NN2 (+0.40%) and our NN1, so added depth did not improve OOS R² in this run.
- **NN3** completed but is materially weaker than GKX and weaker than NN1/NN2.
- **NN4** completed with near-zero OOS R², indicating additional depth did not help this pipeline.
- **NN5** completed under the same 3-seed deadline-mode setup. OOS R² = +0.007%, near-zero — consistent with NN4 pattern. Portfolio alpha not statistically significant (t = 1.05).

---

## 3. Monthly Rank IC (Information Coefficient)

Rank IC = Spearman rank correlation between `pred_ret` and realized `ret_exc` each month, averaged over 1987–2016.  
ICIR = IC Mean / IC Std × √12 (annualized).

GKX reports rank IC in Figure 1 of the paper (not Table 3); approximate paper values range from 0.02 to 0.06.

| Model | **Our Mean IC** | **Our IC Std** | **Our ICIR** | GKX approx. range |
|---|---|---|---|---|
| OLS-3 | 0.0269 | 0.0772 | 0.3480 | 0.02–0.03 |
| OLS-all | 0.0519 | 0.0639 | 0.8114 | 0.04–0.05 |
| PCR | 0.0541 | 0.0653 | 0.8292 | 0.04–0.05 |
| PLS | 0.0509 | 0.0589 | 0.8638 | 0.04–0.05 |
| ElasticNet | 0.0537 | 0.0611 | 0.8783 | 0.04–0.05 |
| GLM (Lasso) | 0.0601 | 0.0765 | 0.7857 | 0.03–0.05 |
| NN1 | 0.0492 | 0.0713 | 0.6896 | 0.04–0.06 |
| NN2 | 0.0488 | 0.0806 | 0.6052 | 0.04–0.06 |
| NN3 | 0.0169 | 0.0715 | 0.2364 | 0.04–0.06 |
| NN4 | 0.0291 | 0.0491 | 0.5931 | 0.04–0.06 |
| NN5 | 0.0364 | 0.0522 | 0.6973 | 0.04–0.06 |
| RF | 0.0213 | 0.0728 | 0.2922 | 0.03–0.05 |
| GBRT | 0.0300 | 0.0662 | 0.4535 | 0.04–0.06 |

**Key observation:** All models have **positive rank IC**. NN3 rank signal is weakest (IC 0.0169). NN4 and NN5 recover partially (IC 0.0291 and 0.0364) but do not translate into useful OOS R² (+0.003% and +0.007%).

---

## 4. Value-Weighted L/S Decile Portfolio Performance (1987–2016)

Long P10 − Short P1, value-weighted using lagged market cap. FF5 alpha from Newey-West regression (12 lags).  
**Full 360-month window (1987–2016).** Fix: CRSP dates normalised to calendar month-end in `portfolio.py` before the merge with predictions.

### 4a. Our Results

| Model | Monthly L/S | Annual Return | Sharpe | FF5 α (annual) | t(α) | p(α) |
|---|---|---|---|---|---|---|
| PLS | **1.07%** | 12.9% | **0.85** | 13.21% | **4.76** | <0.001 |
| PCR | 1.10% | 13.2% | 0.81 | 14.12% | 4.42 | <0.001 |
| OLS-all | 1.00% | 12.0% | 0.74 | 12.54% | 3.81 | <0.001 |
| ElasticNet | 0.99% | 11.8% | 0.72 | 12.53% | 3.84 | <0.001 |
| OLS-3 | 0.86% | 10.3% | 0.70 | 9.27% | 3.35 | <0.001 |
| GLM (Lasso) | 0.75% | 9.0% | 0.56 | 9.04% | 3.57 | <0.001 |
| NN1 | **2.50%** | **30.0%** | **1.60** | **28.30%** | **6.21** | <0.001 |
| NN2 | 2.44% | 29.3% | 1.51 | 27.99% | 5.92 | <0.001 |
| NN3 | 0.58% | 7.0% | 0.41 | 8.39% | 2.55 | 0.011 |
| NN4 | 0.11% | 1.4% | 0.09 | 1.63% | 0.55 | 0.582 |
| NN5 | 0.11% | 1.3% | 0.10 | 2.71% | 1.05 | 0.293 |
| RF | 0.13% | 1.6% | 0.10 | 2.27% | 0.61 | 0.541 |
| GBRT | 0.06% | 0.8% | 0.04 | 1.10% | 0.30 | 0.764 |

### 4b. GKX Table 3 Benchmarks

Values from GKX (2020) Table 3. Portfolio is long-short, value-weighted, 1987–2016 (360 months).

| Model | GKX Monthly L/S | GKX Annual Sharpe | GKX FF3 α significant? |
|---|---|---|---|
| OLS-3 | ~0.31% | ~0.41 | Yes (t ≈ 2.4) |
| OLS-all | ~0.24% | ~0.33 | Marginal |
| PCR | ~0.25% | ~0.40 | Yes |
| PLS | ~0.39% | ~0.54 | Yes |
| ElasticNet | ~0.26% | ~0.37 | Yes |
| GLM (Lasso) | ~0.17% | ~0.23 | Marginal |
| RF | ~0.39% | ~0.49 | Yes |
| GBRT | ~0.34% | ~0.42 | Yes |
| NN1 | ~0.39% | ~0.55 | Yes |
| NN2 | ~0.40% | ~0.60 | Yes |
| NN3 | ~0.41% | ~0.61 | Yes |
| NN4 | ~0.45% | ~0.70 | Yes |
| NN5 | ~0.55% | ~0.77 | Yes |

> ⚠️ GKX L/S monthly returns are **approximate** values reconstructed from Table 3 and related slides. The exact figures should be verified against the published paper (RFS 33(5), Table 3).

### 4c. Comparison and Discussion

| Model | Our Monthly | GKX Monthly | Ratio | Our Sharpe | GKX Sharpe | Match? |
|---|---|---|---|---|---|---|
| OLS-3 | 0.86% | ~0.31% | 2.8× | 0.70 | ~0.41 | ⚠️ Higher |
| OLS-all | 1.00% | ~0.24% | 4.2× | 0.74 | ~0.33 | ❌ Higher |
| PCR | 1.10% | ~0.25% | 4.4× | 0.81 | ~0.40 | ❌ Higher |
| PLS | 1.07% | ~0.39% | 2.7× | 0.85 | ~0.54 | ⚠️ Higher |
| ElasticNet | 0.99% | ~0.26% | 3.8× | 0.72 | ~0.37 | ❌ Higher |
| GLM | 0.75% | ~0.17% | 4.4× | 0.56 | ~0.23 | ⚠️ Higher |
| NN1 | 2.50% | ~0.39% | 6.4× | 1.60 | ~0.55 | ❌ Much higher |
| NN2 | 2.44% | ~0.40% | 6.1× | 1.51 | ~0.60 | ❌ Much higher |
| NN3 | 0.58% | ~0.41% | 1.4× | 0.41 | ~0.61 | ⚠️ Lower Sharpe |
| NN4 | 0.11% | ~0.45% | 0.2× | 0.09 | ~0.70 | ❌ Much lower |
| NN5 | 0.11% | ~0.55% | 0.2× | 0.10 | ~0.77 | ❌ Much lower |
| RF | 0.13% | ~0.39% | 0.3× | 0.10 | ~0.49 | ❌ Lower |
| GBRT | 0.06% | ~0.34% | 0.2× | 0.04 | ~0.42 | ❌ Much lower |

**Why our linear model L/S returns are still higher than GKX:**

1. **Sample recovered (252 → 360 months):** The 108 missing months were caused by a date-format mismatch — CRSP stores raw trading dates (e.g., `1987-05-29`) while predictions use calendar month-end (`1987-05-31`). The merge silently dropped any month where the CRSP date fell before the calendar month-end. Fixed in `portfolio.py` with `pd.offsets.MonthEnd(0)` normalization. Linear L/S returns fell from ~0.91–1.29%/mo to ~0.75–1.10%/mo, partially closing the gap with GKX.

2. **Persistent upward bias in linear models:** Even on the full 360-month sample, linear L/S returns remain 2.7–4.4× GKX. Most likely causes: (a) rank normalization amplifies signal in the 1990s–2000s bull market, (b) GKX uses a different universe definition or FF3 (not FF5) for alpha benchmarking, (c) our value-weighting and decile boundaries may differ slightly from the paper.

**Tree-model diagnosis:**

1. **GBRT variance problem was real:** restricting to `lr=0.01` reduced prediction std from `0.0363` to `0.0158` and improved pooled OOS R² by ~2.81 pp.

2. **RF is not rescued by deeper trees alone:** `max_depth=2` selected on validation but pooled OOS R² worsened to −0.509%. Both tree models now show very low portfolio returns (RF 0.13%, GBRT 0.06%) vs GKX's ~0.34–0.39%.

**Neural-model diagnosis:**

1. **Depth beyond NN2 is not helping in this replication:** NN3 is weak (+0.023% OOS R²) and NN4 is near-zero (+0.003%), both well below GKX's monotonic NN gains.

2. **Risk-adjusted performance deteriorates with depth past NN2:** NN3 Sharpe = 0.41, NN4 = 0.09, NN5 = 0.10, versus NN1 1.60 and NN2 1.51. The pattern is consistent across OOS R² and portfolio Sharpe: deeper networks do not improve performance in this 3-seed deadline-mode replication.

---

## 5. OOS R² by Model Class

| Model Class | GKX Range | Our Range | Gap |
|---|---|---|---|
| Linear (OLS, PCR, PLS, ENet, GLM) | +0.06% to +0.25% | +0.025% to +0.180% | Small (≤0.09 pp) |
| Tree (RF, GBRT) | +0.34% to +0.39% | −0.99% to −0.51% | **Still large (>0.8 pp)** |
| Neural (NN1–NN5) | +0.37% to +0.55% | NN1: +0.344%, NN2: +0.178%, NN3: +0.023%, NN4: +0.003%, NN5: +0.007% | Mixed; NN1/NN2 partial match; NN3–NN5 near-zero (3-seed, non-compliant) |

---

## 6. Signal Dispersion (pred_ret std)

Extreme dispersion in tree predictions inflates MSE without improving rank correlation. GKX do not report this, but it is diagnostic.

| Model | pred_ret Std |
|---|---|
| GLM (Lasso) | 0.0035 |
| OLS-3 | 0.0056 |
| OLS-all | 0.0103 |
| ElasticNet | 0.0096 |
| NN1 | 0.0092 |
| NN2 | 0.0050 |
| NN3 | 0.0017 |
| NN4 | 0.0019 |
| NN5 | 0.0022 |
| PCR | 0.0100 |
| PLS | 0.0113 |
| RF | 0.0112 |
| GBRT | **0.0158** |

GBRT no longer shows the earlier extreme high-dispersion behavior, which validates the learning-rate fix. RF dispersion rises modestly with deeper trees but still does not translate into positive pooled OOS R².

---

## 7. Hyperparameters Selected

| Model | Parameter | Value Selected | GKX Paper |
|---|---|---|---|
| PCR | n_components | 50 | Not disclosed |
| PLS | n_components | 10 | Not disclosed |
| ElasticNet | alpha | 0.001 | λ ∈ {10⁻³, 10⁻², 10⁻¹} |
| ElasticNet | l1_ratio | 0.1 | ρ ∈ {0.1, 0.5, 0.9} |
| GLM (Lasso) | alpha | 0.001 | λ grid search |
| RF | max_depth | **2** | 1, 2, or 4 |
| GBRT | learning_rate | **0.01** | 0.01 or 0.1 |
| GBRT | max_depth | 2 | 1, 2, or other |

> The current tree rerun validates that `lr=0.01` is the correct GBRT region for this pipeline. RF still fails to reach positive pooled OOS R² even at `max_depth=2`, so further gains will likely require broader tree-specification changes rather than a one-line depth tweak.

---

## 8. Summary Assessment

> All claims have been restated per §0.9. See Section 0 for full deviation register.

| Category | Assessment |
|---|---|
| Linear models OOS R² | ⚠️ **Approximate match** — within ±0.09 pp of GKX; portfolio magnitudes diverge 2.7–4.4× (see §0.7); not a full replication |
| GLM OOS R² | ⚠️ **OOS R² point estimate matches** (+0.06% vs +0.06%); portfolio performance does not; partial match only |
| Tree models OOS R² | ❌ **Open divergence** — both RF and GBRT negative; root cause unresolved; excluded from cross-model ranking |
| Tree model L/S Sharpe | ❌ **Not interpretable** — Sharpe 0.04–0.10 vs GKX ~0.42–0.49; excluded from conclusions pending resolution |
| NN1 OOS R² | ⚠️ **Close partial match** — +0.344% vs +0.39%; 10-seed compliant; portfolio 6× GKX (§0.8) |
| NN2 OOS R² | ⚠️ **Positive partial match** — +0.178% vs GKX +0.40%; 10-seed compliant; magnitude below GKX |
| NN3/NN4/NN5 | ❌ **Non-compliant ensembles** — 3 seeds vs GKX 10; results reported for transparency only; not comparable to GKX or to NN1/NN2 |
| Portfolio coverage | ✅ **Fixed** — 360/360 months after date-normalization fix in `portfolio.py` (§0.6) |
| Linear model L/S | ❌ **Unexplained upward divergence** — 2.7–4.4× GKX after fix; cause not isolated |
| NN1/NN2 L/S | ❌ **Large unexplained divergence** — ~6× GKX monthly return despite close OOS R² |

### Root causes of remaining gaps

1. ~~**Missing `hire` and `ear`**~~ **Resolved**: All 94/94 GKX features are implemented. Adding these features made no measurable difference to the tree models.

2. **`me_lag1` data gaps in CRSP**: backfill applied in `crsp_cleaner.py`. Not the root cause of missing months — see item 3.

3. **Portfolio date mismatch (fixed):** CRSP trading dates (e.g., `1987-05-29`) did not match calendar month-end prediction dates (`1987-05-31`), silently dropping 108 months. Fixed in `portfolio.py` with `pd.offsets.MonthEnd(0)`. Coverage is now **360/360**.

3. **Tree-spec mismatch remains**: The latest tree rerun removed the stump-only RF explanation and fixed the GBRT learning-rate pathology, yet both tree models remain negative. Any further tree work should be treated as a broader specification exercise, not a quick unblocker before NN training.

4. **Neural queue complete**: NN1 and NN2 complete under the original 10-seed setup; NN3 (Kaggle), NN4 (local), and NN5 (local) all complete under 3-seed deadline-mode. Phase 4 training is done.
---

## 9. Next Steps

- [x] ~~Implement all 94/94 GKX characteristics~~ Done
- [x] ~~Fix `me_lag1` gaps in `crsp_cleaner.py`~~ Done (backfill applied; not the root cause)
- [x] ~~Fix missing 108 portfolio months~~ Done — date-normalization fix in `portfolio.py`; coverage now **360/360**
- [x] Train NN1 (10 seeds) and evaluate
- [x] Train NN2 (10 seeds) and evaluate
- [x] Train NN3 (3 seeds, Kaggle GPU deadline mode) and merge outputs safely
- [x] Train NN4 (3 seeds, local CPU deadline mode) and merge outputs safely
- [x] Train NN5 (3 seeds, deadline mode) → Phase 4 complete
- [ ] Revisit tree-model specification (optional — not a blocker for Phase 5)
- [ ] Phase 5a: Post-2020 OOS extension (`src/extensions/post2020_eval.py`)
- [ ] Phase 5b: Net of transaction costs (`src/extensions/transaction_costs.py`)
- [ ] Phase 5c: Feature parsimony (`src/extensions/feature_parsimony.py`)
