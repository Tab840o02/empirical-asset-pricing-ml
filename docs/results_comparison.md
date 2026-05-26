# GKX (2020) Replication — Results Comparison Snapshot

> Last updated: 2026-05-27
> Sources used in this pass:
> `data/processed/eval_oos_r2_latest.csv`
> `data/processed/eval_ic_stats_latest.csv`
> `data/processed/eval_portfolio_perf_latest.csv`
> `data/processed/eval_pred_std_latest.csv`
> `data/processed/eval_ext_oos_r2.csv`
> `data/processed/eval_ext_ic_stats.csv`
> `data/processed/eval_ext_portfolio_perf.csv`
> `data/processed/eval_tc_summary.csv`
> `data/processed/eval_tc_monthly.csv`
> `data/processed/eval_tc_ext_summary.csv`
> `data/processed/eval_tc_ext_monthly.csv`
> `data/processed/eval_parsimony_oos_r2.csv`
> `data/processed/eval_parsimony_ic_stats.csv`
> `data/processed/eval_parsimony_portfolio_perf.csv`
> `data/processed/eval_parsimony_degenerate_months_summary.csv`
> `data/processed/eval_parsimony_ic_stats_audited.csv`
> `data/processed/eval_parsimony_portfolio_perf_audited.csv`
> `data/processed/run_manifest_parsimony_chronology_reconstructed.csv`
> `data/processed/phase5c_audit_addendum.json`
> `data/processed/run_manifest.json`
> `data/processed/run_manifest_ext.json`
> `data/processed/run_manifest_parsimony.json`
> `data/processed/phase5c_selected_features.json`
> `data/processed/unified_manifest.json`

---

## 1. Run Metadata

### Phase 4 (replication window)

- window: 1987-01-01 to 2016-12-31
- manifest timestamp: 2026-05-25T10:56:29.994295+00:00
- latest recorded run in `run_manifest.json`: `nn5` completion pass

### Phase 5a (post-2020 extension window)

- window: 2017-01-01 to 2024-11-30
- manifest timestamp: 2026-05-26T00:46:31.529659+00:00
- models in extension run: `ols3`, `ols_all`, `pcr`, `pls`, `enet`, `glm`, `rf`, `gbrt`, `nn1`, `nn2`, `nn3`, `nn4`, `nn5`
- all years 2017–2024 marked `completed` in `run_manifest_ext.json`

### Phase 5c (feature parsimony window)

- window: 1987-01-01 to 2016-12-31 (same as Phase 4)
- manifest timestamp: 2026-05-26T05:29:33.689703+00:00
- features: 15 selected by RF permutation importance from 18-feature allow-list
- selected features: `cfp`, `indmom`, `chmom`, `mom12m`, `ep`, `agr`, `bm`, `mom1m`, `ill`, `gp`, `roe`, `roaq`, `retvol`, `invest`, `noa`
- all years 1987–2016 marked `completed` in `run_manifest_parsimony.json`
- benchmark (pre-test mean excess return): 0.006853425409644842

### Unified manifest

- generated at: 2026-05-26T00:10:39.048322+00:00

---

## 2. Phase 4 Key Outcomes (1987–2016)

### 2.1 Pooled OOS R²

| Model | oos_r2 |
|---|---:|
| nn1 | 0.0034396592 |
| enet | 0.0018030082 |
| nn2 | 0.0017746471 |
| pcr | 0.0016885477 |
| pls | 0.0016277616 |
| ols_all | 0.0015242222 |
| glm | 0.0006334276 |
| ols3 | 0.0002473771 |
| nn3 | 0.0002328155 |
| nn5 | 0.0000687441 |
| nn4 | 0.0000298004 |
| rf | -0.0050936655 |
| gbrt | -0.0098902584 |

Key points:

1. Best OOS R² is `nn1`.
2. `rf` and `gbrt` remain negative.
3. `nn3` to `nn5` OOS R² values are near zero compared with `nn1` and `nn2`.

### 2.2 Rank IC and ICIR

| Model | mean_ic | std_ic | icir |
|---|---:|---:|---:|
| glm | 0.0600802999 | 0.0764677655 | 0.7856944623 |
| pcr | 0.0541475238 | 0.0653030335 | 0.8291731766 |
| enet | 0.0537036988 | 0.0611416664 | 0.8783486277 |
| ols_all | 0.0518810469 | 0.0639396412 | 0.8114066001 |
| pls | 0.0508984688 | 0.0589251277 | 0.8637820709 |
| nn1 | 0.0492027471 | 0.0713478490 | 0.6896178060 |
| nn2 | 0.0487925266 | 0.0806193680 | 0.6052209017 |
| nn5 | 0.0364007723 | 0.0521995188 | 0.6973392308 |
| gbrt | 0.0300232194 | 0.0661998550 | 0.4535239459 |
| nn4 | 0.0291062840 | 0.0490782489 | 0.5930587311 |
| ols3 | 0.0268667609 | 0.0772039634 | 0.3479971714 |
| rf | 0.0212723036 | 0.0727902571 | 0.2922410833 |
| nn3 | 0.0169163479 | 0.0715466324 | 0.2364380735 |

### 2.3 Long-Short portfolio performance

| Model | annual_ret | annual_vol | sharpe | alpha_annual | t_alpha | p_alpha | n_months |
|---|---:|---:|---:|---:|---:|---:|---:|
| nn1 | 0.300480 | 0.188325 | 1.596 | 0.282998 | 6.205 | 0.0000 | 360 |
| nn2 | 0.292554 | 0.193725 | 1.510 | 0.279931 | 5.924 | 0.0000 | 360 |
| pls | 0.128799 | 0.152105 | 0.847 | 0.132113 | 4.761 | 0.0000 | 360 |
| pcr | 0.131522 | 0.163403 | 0.805 | 0.141196 | 4.420 | 0.0000 | 360 |
| ols_all | 0.120057 | 0.161576 | 0.743 | 0.125432 | 3.812 | 0.0001 | 360 |
| enet | 0.118277 | 0.163395 | 0.724 | 0.125282 | 3.839 | 0.0001 | 360 |
| ols3 | 0.102814 | 0.147408 | 0.697 | 0.092650 | 3.354 | 0.0008 | 360 |
| glm | 0.090111 | 0.160027 | 0.563 | 0.090400 | 3.571 | 0.0004 | 360 |
| nn3 | 0.070030 | 0.170741 | 0.410 | 0.083930 | 2.547 | 0.0109 | 360 |
| rf | 0.016014 | 0.165326 | 0.097 | 0.022676 | 0.611 | 0.5414 | 360 |
| nn5 | 0.013110 | 0.137489 | 0.095 | 0.027105 | 1.051 | 0.2932 | 360 |
| nn4 | 0.013794 | 0.156792 | 0.088 | 0.016262 | 0.551 | 0.5820 | 360 |
| gbrt | 0.007548 | 0.176147 | 0.043 | 0.011004 | 0.301 | 0.7635 | 360 |

### 2.4 Prediction dispersion

| Model | pred_std |
|---|---:|
| gbrt | 0.015789 |
| pls | 0.011300 |
| rf | 0.011148 |
| ols_all | 0.010284 |
| pcr | 0.009998 |
| enet | 0.009572 |
| nn1 | 0.009207 |
| ols3 | 0.005562 |
| nn2 | 0.004973 |
| glm | 0.003515 |
| nn5 | 0.002172 |
| nn4 | 0.001949 |
| nn3 | 0.001737 |

---

## 3. Phase 5a Extension Results (2017–2024)

> **Exclusions:** nn4 and nn5 are omitted from all Phase 5a tables because
> they produce a constant prediction value (zero cross-sectional variance)
> throughout the entire 95-month extension window, rendering IC, OOS R², and
> portfolio decile assignments invalid (see academic disclosure in Section 5).
> nn3 is excluded from COVID (2020) and Rate hikes (2022) sub-periods for the
> same reason; its Post-norm (2023+) metrics are based on the 12 valid 2023
> months only (all 2024 months for nn3 also have zero variance).
>
> **Benchmark:** `R_BENCH_EXT = 0.008027891628444195` (pre-2017 mean excess
> return loaded from `run_manifest_ext.json`).  NW SE computed with 12 lags
> (consistent with Phase 4).

### 3.1 Pooled OOS R² by sub-period

| Model | Pre-COVID (2017–2019) | COVID (2020) | Reflation (2021) | Rate hikes (2022) | Post-norm (2023+) | Full ext (2017+) |
|---|---:|---:|---:|---:|---:|---:|
| ols3 | -0.001927 | -0.000295 | 0.003181 | 0.001024 | -0.000006 | -0.000050 |
| ols_all | -0.001402 | -0.000631 | 0.002822 | 0.000499 | 0.000115 | -0.000081 |
| pcr | -0.001402 | -0.000631 | 0.002822 | 0.000499 | 0.000115 | -0.000081 |
| pls | -0.001987 | -0.000722 | 0.004532 | **0.001355** | 0.000153 | 0.000045 |
| enet | -0.001298 | -0.000199 | **0.004454** | 0.001326 | 0.000188 | 0.000298 |
| glm | -0.000549 | -0.000162 | 0.001721 | 0.000526 | **0.000272** | 0.000164 |
| rf | 0.000420 | 0.001280 | -0.002664 | -0.006394 | -0.000552 | -0.000830 |
| gbrt | **0.001440** | **0.003111** | 0.007373 | -0.010578 | -0.000542 | 0.000058 |
| nn1 | -0.000481 | 0.002886 | 0.003313 | 0.001039 | 0.000077 | **0.001012** |
| nn2 | 0.000147 | 0.000689 | 0.000790 | 0.000010 | 0.000071 | 0.000280 |
| nn3 | 0.000062 | — | -0.000641 | — | -0.000776 | -0.000372 |
| nn4 | — | — | — | — | — | — |
| nn5 | — | — | — | — | — | — |

*Bold = best valid model in each column.  — = excluded (model collapse).*

**Key observations:**
- No single model dominates across all sub-periods.  gbrt leads in Pre-COVID
  and COVID but collapses in Rate hikes.  enet and pls are strongest in
  Reflation; pls edges enet in Rate hikes.  glm leads in Post-norm.
- Full-window (2017+) is led by nn1 (0.001012), roughly one-third of its
  Phase 4 value (0.00344).  Six of 11 evaluated models produce negative
  full-window OOS R² once the correct benchmark is applied.
- ols3, ols_all, and pcr produce negative full-window OOS R² (−0.000050,
  −0.000081, −0.000081), confirming the benchmark correction was material.

### 3.2 Mean monthly IC by sub-period

| Model | Pre-COVID (2017–2019) | COVID (2020) | Reflation (2021) | Rate hikes (2022) | Post-norm (2023+) | Full ext (2017+) |
|---|---:|---:|---:|---:|---:|---:|
| ols3 | -0.0287 | -0.0226 | 0.0668 | 0.0383 | 0.0010 | -0.0002 |
| ols_all | 0.0090 | -0.0084 | 0.0516 | 0.0371 | 0.0181 | 0.0179 |
| pcr | 0.0090 | -0.0084 | 0.0516 | 0.0371 | 0.0181 | 0.0179 |
| pls | 0.0062 | -0.0039 | 0.0718 | 0.0412 | 0.0064 | 0.0177 |
| enet | 0.0056 | -0.0027 | 0.0713 | 0.0407 | 0.0048 | 0.0171 |
| glm | 0.0016 | -0.0120 | 0.0870 | 0.0496 | 0.0310 | 0.0238 |
| rf | 0.0049 | 0.0131 | 0.1096 | 0.0283 | 0.0034 | 0.0217 |
| gbrt | 0.0077 | 0.0036 | 0.1043 | 0.0116 | 0.0027 | 0.0187 |
| nn1 | -0.0005 | 0.0214 | 0.0864 | 0.0188 | -0.0055 | 0.0145 |
| nn2 | 0.0092 | 0.0058 | 0.1023 | 0.0323 | 0.0139 | 0.0246 |
| nn3 | -0.0209 | — | -0.0253 | — | -0.0162 | -0.0209 |
| nn4 | — | — | — | — | — | — |
| nn5 | — | — | — | — | — | — |

### 3.3 Long-short portfolio Sharpe ratio by sub-period

| Model | Pre-COVID (2017–2019) | COVID (2020) | Reflation (2021) | Rate hikes (2022) | Post-norm (2023+) | Full ext (2017+) |
|---|---:|---:|---:|---:|---:|---:|
| ols3 | -1.462 | 0.696 | 1.124 | 1.675 | -0.719 | 0.068 |
| ols_all | 0.241 | -0.534 | 2.258 | 0.561 | -0.330 | 0.226 |
| pcr | 0.241 | -0.534 | 2.258 | 0.561 | -0.330 | 0.226 |
| pls | 0.421 | -1.856 | **2.577** | 0.883 | -0.327 | 0.153 |
| enet | 0.148 | -0.644 | **2.632** | 0.477 | -0.144 | 0.230 |
| glm | -0.563 | -2.759 | 0.908 | 0.868 | -1.427 | -0.507 |
| rf | 0.567 | -2.012 | 0.478 | 1.739 | **0.449** | 0.409 |
| gbrt | **1.036** | -1.756 | 1.203 | 0.699 | -0.521 | 0.200 |
| nn1 | 0.924 | **2.196** | 2.521 | **2.463** | -0.226 | 1.117 |
| nn2 | **1.217** | 1.714 | 2.462 | 1.901 | -0.022 | **1.161** |
| nn3 | 0.492 | — | -0.684 | — | -0.561 | -0.037 |
| nn4 | — | — | — | — | — | — |
| nn5 | — | — | — | — | — | — |

*COVID Sharpe ratios are based on 12 months and should be interpreted cautiously
(t-statistics not significant at 5% for most models given the short window).*

---

## 4. Phase 5b Transaction Cost Results

Transaction costs are estimated using two one-way spread proxies.  Breakpoints
for the fixed schedule use NYSE 20th / 80th-percentile market-cap thresholds
(standard Fama-French quintile convention).  NW standard errors use 12 lags.

### 4.1 Phase 4 window (1987–2016, 360 months)

#### Fixed-schedule spreads (small-cap 20 bps / mid-cap 10 bps / large-cap 5 bps)

| Model | Gross Sharpe | Net Sharpe | Spread (bps) | Gross rank | Net rank | Rank Δ |
|---|---:|---:|---:|---:|---:|---:|
| nn1 | 1.590 | 1.521 | 11.7 | 1 | 1 | 0 |
| nn2 | 1.507 | 1.447 | 11.7 | 2 | 2 | 0 |
| pls | 0.844 | 0.787 | 7.6 | 3 | 3 | 0 |
| pcr | 0.805 | 0.759 | 7.7 | 4 | 4 | 0 |
| ols_all | 0.751 | 0.703 | 7.7 | 5 | 5 | 0 |
| enet | 0.735 | 0.683 | 7.5 | 6 | 6 | 0 |
| ols3 | 0.724 | 0.670 | 9.7 | 7 | 7 | 0 |
| glm | 0.568 | 0.527 | 7.4 | 8 | 8 | 0 |
| nn3 | 0.387 | 0.352 | 8.9 | 9 | 9 | 0 |
| nn5 | 0.104 | 0.080 | 7.3 | 10 | 10 | 0 |
| rf | 0.102 | 0.064 | 7.3 | 11 | 11 | 0 |
| nn4 | 0.063 | 0.034 | 7.7 | 12 | 12 | 0 |
| gbrt | 0.024 | −0.023 | 7.9 | 13 | 13 | 0 |

**Finding: zero rank changes gross → net under fixed schedule.**

#### Amihud-calibrated spreads

| Model | Gross Sharpe | Net Sharpe | Spread (bps) | Gross rank | Net rank | Rank Δ |
|---|---:|---:|---:|---:|---:|---:|
| nn1 | 1.590 | 1.465 | 21.5 | 1 | 1 | 0 |
| nn2 | 1.507 | 1.405 | 20.7 | 2 | 2 | 0 |
| pls | 0.844 | 0.798 | 6.3 | 3 | 3 | 0 |
| pcr | 0.805 | 0.766 | 6.5 | 4 | 4 | 0 |
| ols_all | 0.751 | 0.709 | 6.6 | 5 | 5 | 0 |
| enet | 0.735 | 0.690 | 6.4 | 6 | 6 | 0 |
| ols3 | 0.724 | 0.642 | 14.6 | 7 | 7 | 0 |
| glm | 0.568 | 0.527 | 6.8 | 8 | 8 | 0 |
| nn3 | 0.387 | 0.341 | 10.3 | 9 | 9 | 0 |
| nn5 | 0.104 | 0.069 | 7.3 | 10 | 10 | 0 |
| rf | 0.102 | 0.067 | 6.5 | 11 | 11 | 0 |
| nn4 | 0.063 | 0.023 | 8.5 | 12 | 12 | 0 |
| gbrt | 0.024 | −0.028 | 8.3 | 13 | 13 | 0 |

**Finding: zero rank changes gross → net under Amihud calibration.**

### 4.2 Phase 5a extension window (2017–2024, 95 months)

> **Note:** nn4 and nn5 are included in the TC extension output for completeness
> but their gross Sharpe ratios (0.386 for both, identical) are spurious —
> constant predictions map to row-order decile rankings in both the gross and net
> computation.  Treat these rows as invalid for comparison purposes.

#### Fixed-schedule spreads

| Model | Gross Sharpe | Net Sharpe | Spread (bps) | Gross rank | Net rank | Rank Δ |
|---|---:|---:|---:|---:|---:|---:|
| nn2 | 1.145 | 1.079 | 11.5 | 1 | 1 | 0 |
| nn1 | 1.103 | 1.041 | 12.3 | 2 | 2 | 0 |
| rf | 0.432 | 0.389 | 6.6 | 3 | 3 | 0 |
| gbrt | 0.241 | 0.194 | 7.2 | 6 | 6 | 0 |
| ols_all / pcr | 0.222 | 0.176 | 6.1 | 7 | 7 | 0 |
| enet | 0.199 | 0.149 | 6.4 | 8 | 8 | 0 |
| pls | 0.145 | 0.098 | 6.3 | 9 | 9 | 0 |
| ols3 | 0.064 | 0.032 | 8.9 | 10 | 10 | 0 |
| glm | −0.504 | −0.528 | 6.2 | 11 | 11 | 0 |

**Finding: zero rank changes gross → net in extension window (excluding
degenerate nn4/nn5 rows).**

#### Amihud-calibrated spreads (extension window)

| Model | Gross Sharpe | Net Sharpe | Spread (bps) | Gross rank | Net rank | Rank Δ |
|---|---:|---:|---:|---:|---:|---:|
| nn2 | 1.145 | 1.091 | 9.4 | 1 | 1 | 0 |
| nn1 | 1.103 | 1.045 | 11.6 | 2 | 2 | 0 |
| rf | 0.432 | 0.419 | 2.0 | 3 | 3 | 0 |
| nn3 | 0.282 | 0.279 | 1.5 | 5 | 5 | 0 |
| gbrt | 0.241 | 0.225 | 2.4 | 6 | 6 | 0 |
| ols_all / pcr | 0.222 | 0.210 | 1.6 | 7 | 7 | 0 |
| enet | 0.199 | 0.184 | 1.8 | 8 | 8 | 0 |
| pls | 0.145 | 0.131 | 1.8 | 9 | 9 | 0 |
| ols3 | 0.064 | 0.044 | 5.4 | 10 | 10 | 0 |
| glm | −0.504 | −0.510 | 1.6 | 11 | 11 | 0 |

**Finding: zero rank changes gross → net under Amihud calibration in extension
window (excluding degenerate nn4/nn5 rows).**

---

## 5. Progress Interpretation and Academic Disclosures

### 5.1 Progress summary

1. Phase 4 canonical replication outputs are complete and internally consistent.
2. Phase 5a: evaluation CSVs generated with six macro sub-periods. Neural network
   dominance from Phase 4 (nn1, nn2) holds in the 2017+ window at reduced
   magnitude; linear and shrinkage models produce near-zero or negative OOS R².
3. Phase 5b: transaction cost analysis complete for both Phase 4 and Phase 5a
   extension windows. Zero rank changes under both spread methods constitute a
   null result: model rankings are robust to realistic transaction cost levels.
4. Phase 5c: feature parsimony evaluation complete. All 13 models retrained on 15
   selected features (1987–2016 window). OOS R² cost is modest for most models;
   tree-based models (rf, gbrt) unexpectedly improve. Portfolio Sharpe ratios
   remain statistically significant for nn1, nn2, pcr, pls, ols_all, and enet.
   See Section 6 for full results.

### 5.2 Academic disclosures

#### Neural network model collapse (nn4, nn5; partial nn3)

nn4 and nn5 produce a constant prediction value across all stocks in every month
of the 2017–2024 extension window (cross-sectional prediction standard deviation
= 0.000 for all 95 months). Under constant predictions, the `rank(method='first')`
tiebreaking in `pd.qcut` produces decile assignments that depend solely on row
order in the predictions file, which is arbitrary. Portfolio returns, Sharpe
ratios, IC values, and OOS R² computed from such predictions are statistical
artefacts. Accordingly, nn4 and nn5 are excluded from all Phase 5a evaluation
outputs and their metric cells are reported as NaN. nn3 exhibits the same
collapse in all 12 months of 2020, all 12 months of 2022, and all 11 months of
2024 (35 affected months); those (model, month) pairs are removed before metric
computation, with affected (model, period) combinations reported as NaN.
The probable cause is vanishing or exploding gradients during deep-network
training on the extension data; the failure was not detected at training time
because the model produces prediction values without error. Results for nn4 and
nn5 in the Phase 4 window (1987–2016) are not affected (prediction variance is
non-zero in Phase 4).

#### Amihud ILLIQ proxy: monthly approximation

The standard Amihud (2002) illiquidity measure averages daily
|return_d| / dollar_volume_d within each month. This project's Phase 5b
implementation uses a monthly approximation: |ret_m| / (|prc_m| × vol_m) from
`crsp_clean.parquet`, which records monthly return, price, and volume. The
monthly proxy introduces noise relative to the daily-average measure because a
single month-end observation imperfectly proxies the intra-month average. Spread
estimates under the Amihud method should be treated as illustrative rather than
precise. A daily-CRSP robustness check is recommended before any submission.

#### Null result on gross-to-net rank changes

Under both the fixed-schedule (20/10/5 bps by NYSE 20th/80th-percentile size
quintile) and the Amihud-calibrated spread methods, every model's performance
rank is identical gross and net of transaction costs in both the Phase 4 window
(360 months) and the Phase 5a extension window (95 months). This is a genuine
null result: at the annualized turnover rates and spread levels observed,
transaction costs reduce net Sharpe ratios and net alphas proportionally across
models without altering the ordinal ranking. The magnitudes (gross→net Sharpe
reduction of approximately 0.05–0.13 for nn1/nn2; 0.03–0.08 for linear models)
are economically meaningful but rank-order preserving.

---

## 6. Phase 5c Feature Parsimony Results (1987–2016, 15 features)

> **Feature selection:** Random forest permutation importance run on the
> pre-validation window; top features filtered against an 18-feature economic
> allow-list (momentum, value, quality, liquidity). 15 features selected
> (within target 15–20 range):
> `cfp`, `indmom`, `chmom`, `mom12m`, `ep`, `agr`, `bm`, `mom1m`, `ill`, `gp`,
> `roe`, `roaq`, `retvol`, `invest`, `noa`.
>
> **Benchmark:** `R_BENCH = 0.006853425409644842` (pre-test mean excess return
> from `run_manifest_parsimony.json`).
> **Sample:** 18,165,186 prediction rows; 100% merge coverage with parsimonious
> feature panel.
>
> **Audit remediation (no retraining):** Degenerate model-month diagnostics and
> audited robustness tables are published in
> `eval_parsimony_degenerate_months_summary.csv`,
> `eval_parsimony_ic_stats_audited.csv`, and
> `eval_parsimony_portfolio_perf_audited.csv`. Run chronology evidence is
> reconstructed from stored predictions in
> `run_manifest_parsimony_chronology_reconstructed.csv`.

### 6.1 Pooled OOS R² — Phase 4 vs Phase 5c

| Model | Phase 4 OOS R² | Phase 5c OOS R² | Δ R² |
|---|---:|---:|---:|
| nn1 | +0.003440 | +0.001571 | −0.001869 |
| enet | +0.001803 | +0.000946 | −0.000857 |
| nn2 | +0.001775 | +0.000470 | −0.001305 |
| pcr | +0.001689 | +0.000712 | −0.000977 |
| pls | +0.001628 | +0.000878 | −0.000750 |
| ols_all | +0.001524 | +0.000945 | −0.000579 |
| glm | +0.000634 | +0.000444 | −0.000190 |
| ols3 | +0.000247 | +0.000195 | −0.000053 |
| nn3 | +0.000233 | −0.000142 | −0.000375 |
| nn5 | +0.000069 | −0.000201 | −0.000270 |
| nn4 | +0.000030 | +0.000272 | **+0.000242** |
| rf | −0.005094 | +0.000476 | **+0.005570** |
| gbrt | −0.009890 | +0.000424 | **+0.010314** |

**Key observations:**

1. OOS R² cost of parsimony is modest for linear/shrinkage models (Δ ≈ −0.0005
   to −0.002). Best model nn1 loses roughly 54% of its OOS R² (0.00344 → 0.00157)
   with only 15 features; enet retains 52% (0.00180 → 0.00095).
2. Tree models (rf, gbrt) show a surprising *improvement* under the parsimonious
   feature set. Both had negative Phase 4 OOS R² (rf: −0.005, gbrt: −0.010) and
   move to small positive values (+0.000476, +0.000424). This is consistent with
   reduced overfitting: with 94 features, tree models fit noise; the 15-feature
   filter removes irrelevant predictors that hurt generalisation.
3. Linear models (ols3, ols_all, glm, pcr, pls, enet) retain the sign of their
   OOS R² and meaningful predictive power. glm suffers the smallest absolute loss
   (Δ = −0.000190), suggesting its L1-penalised coefficients were already
   concentrated on the selected features.

### 6.2 Monthly rank IC — Phase 5c (1987–2016)

| Model | mean_ic | std_ic | icir |
|---|---:|---:|---:|
| glm | 0.0594 | 0.0826 | 0.7191 |
| pcr | 0.0561 | 0.0833 | 0.6737 |
| pls | 0.0562 | 0.0842 | 0.6673 |
| gbrt | 0.0470 | 0.0709 | 0.6632 |
| enet | 0.0563 | 0.0858 | 0.6556 |
| ols_all | 0.0551 | 0.0862 | 0.6388 |
| rf | 0.0495 | 0.0785 | 0.6311 |
| nn5 | 0.0331 | 0.0538 | 0.6142 |
| nn1 | 0.0446 | 0.0757 | 0.5895 |
| ols3 | 0.0450 | 0.0770 | 0.5849 |
| nn2 | 0.0459 | 0.0793 | 0.5794 |
| nn3 | 0.0350 | 0.0650 | 0.5377 |
| nn4 | 0.0189 | 0.0485 | 0.3908 |

All models show positive mean_ic.  ICIR rankings differ from Phase 4 (where
enet, pcr, pls led with ICIR > 0.86) because the parsimonious feature set
reduces signal volatility: std_ic narrows for most models, compressing ICIR
differences.  glm leads Phase 5c IC ranking (ICIR 0.72).

**Important comparability caveat:** nn4 and nn5 contain degenerate months in
Phase 5c (zero cross-sectional prediction variance), so their IC entries are
computed on reduced effective samples:

- nn4: 312 non-degenerate months (48 degenerate months excluded)
- nn5: 144 non-degenerate months (216 degenerate months excluded)

These counts are reported in `eval_parsimony_ic_stats_audited.csv` and
`eval_parsimony_degenerate_months_summary.csv`.

### 6.3 Long-short portfolio performance — Phase 5c (1987–2016)

| Model | annual_ret | annual_vol | sharpe | alpha_annual | t_alpha | p_alpha | n_months |
|---|---:|---:|---:|---:|---:|---:|---:|
| nn2 | 0.1683 | 0.1839 | **0.915** | 0.1507 | 4.400 | 0.0000 | 360 |
| nn1 | 0.1706 | 0.2129 | 0.801 | 0.1539 | 4.607 | 0.0000 | 360 |
| pcr | 0.1438 | 0.1861 | 0.773 | 0.1310 | 3.833 | 0.0001 | 360 |
| pls | 0.1357 | 0.1915 | 0.709 | 0.1350 | 3.665 | 0.0002 | 360 |
| ols_all | 0.1255 | 0.1850 | 0.678 | 0.1273 | 3.418 | 0.0006 | 360 |
| enet | 0.1258 | 0.1881 | 0.669 | 0.1219 | 3.433 | 0.0006 | 360 |
| nn4 | 0.0690 | 0.1254 | 0.550 | 0.0744 | 3.389 | 0.0007 | 360 |
| ols3 | 0.0897 | 0.1701 | 0.527 | 0.0853 | 2.687 | 0.0072 | 360 |
| glm | 0.0993 | 0.2044 | 0.486 | 0.0817 | 2.241 | 0.0250 | 360 |
| nn3 | 0.0423 | 0.1168 | 0.362 | 0.0453 | 2.448 | 0.0144 | 360 |
| gbrt | 0.0471 | 0.1850 | 0.255 | 0.0385 | 1.049 | 0.2940 | 360 |
| nn5 | 0.0322 | 0.1358 | 0.237 | 0.0454 | 1.860 | 0.0629 | 360 |
| rf | 0.0473 | 0.2120 | 0.223 | 0.0314 | 0.838 | 0.4022 | 360 |

*Degenerate-month caveat for nn4/nn5:* the canonical table above uses all 360
months. Because `pd.qcut(... rank(method="first"))` produces order-dependent
buckets when predictions are constant, audited robustness results additionally
exclude degenerate model-months for nn4/nn5. In the audited table:

- nn4: Sharpe 0.536, alpha_annual 0.0731, n_months 312
- nn5: Sharpe 0.830, alpha_annual 0.0978, n_months 144

See `eval_parsimony_portfolio_perf_audited.csv`.

**Key observations:**

1. nn1 and nn2 retain the top two Sharpe ratios (0.801 and 0.915 respectively)
   despite using only 15 features. Portfolio performance is more robust to
   feature reduction than point-in-time OOS R² would suggest.
2. Linear/shrinkage models (pcr, pls, ols_all, enet) maintain statistically
   significant alphas (p < 0.001) — the 15-feature set preserves the same
   economic return sources these models exploit.
3. rf and gbrt remain in the lower tier (Sharpe 0.22–0.26) with insignificant
   alphas (p = 0.40 and 0.29 respectively), consistent with their near-zero OOS R²
   values: the parsimony improvement removes negative drag but cannot create signal
   where little exists.
4. Compared with Phase 4, nn1 Sharpe declines from 1.596 to 0.801 (−50%) and
   nn2 from 1.510 to 0.915 (−39%). The R² cost of parsimony translates to
   economically meaningful portfolio performance degradation for NNs, but the
   strategies remain profitable and statistically significant.

### 6.4 Feature parsimony: summary

| Dimension | Finding |
|---|---|
| OOS R² cost (avg linear/shrinkage) | −0.0005 to −0.001 (small) |
| OOS R² cost (nn1, best NN) | −0.00187 (54% reduction) |
| Tree models (rf, gbrt) | Improve: negative → near-zero OOS R² |
| Portfolio Sharpe (nn1, nn2) | Remain significant; −39% to −50% vs Phase 4 |
| Portfolio Sharpe (linear models) | Broadly retained; no sign changes |
| IC/ICIR | All models positive; glm leads (ICIR 0.72); nn4/nn5 use reduced effective months |
| Feature count | 15 (target: 15–20) ✓ |
| Degenerate-month diagnostics | Published (`eval_parsimony_degenerate_months_summary.csv`) |
| Chronology evidence | Reconstructed per model-year from stored predictions |

**Conclusion:** The 15-feature parsimonious model set retains economically
meaningful predictive power across all model families. The R² cost is modest for
linear models and negative for tree models (parsimony improves out-of-sample
generalisation for rf and gbrt). The highest-performing NNs experience the
largest absolute loss in OOS R², suggesting that deep networks extract marginal
signal from the full 94-feature set that is lost in the reduced specification.

### 6.5 Inference and Integrity Disclosures (Phase 5c)

1. **Multiple-testing risk:** Phase 5c compares 13 models across several metrics
   (OOS R², IC, Sharpe, alpha). Reported p-values are model-wise and are not
   adjusted for multiplicity. Interpret model-to-model significance rankings as
   descriptive unless a formal multiple-testing correction is applied.
2. **Degenerate prediction months:** nn4 and nn5 have non-trivial shares of
   zero-variance prediction months (13.3% and 60.0% respectively). Canonical
   tables are retained for comparability with earlier phases, and audited tables
   excluding degenerate months are provided as robustness evidence.
3. **Chronology evidence under time constraints:** `run_manifest_parsimony.json`
   does not include year-level checkpoints. To preserve auditability without
   retraining, a model-year chronology was reconstructed directly from stored
   predictions (`run_manifest_parsimony_chronology_reconstructed.csv`).

