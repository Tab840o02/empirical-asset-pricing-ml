# Project Status

> Last updated: 2026-05-27 (Phases 5a, 5b, and 5c complete; Phase 5c audit remediation artifacts added)

---

## Current Phase Snapshot

| Phase | Name | Status | Evidence |
|-------|------|--------|----------|
| 0 | Environment setup | ✅ Done | Project environment and dependencies in place |
| 1 | WRDS data extraction | ✅ Done | Raw tables available in `data/raw/` |
| 2 | Data cleaning and CCM merge | ✅ Done | `tests/test_no_lookahead.py` suite passing historically (11/11) |
| 3 | Feature engineering (94 characteristics) | ✅ Done | `features_panel.parquet` built with 94 characteristics |
| 4 | Model training and evaluation | ✅ Done | `eval_*_latest.csv` generated for 13 models, 1987-01 to 2016-12 |
| 5a | Extension: post-2020 OOS | ✅ Done | `eval_ext_oos_r2.csv`, `eval_ext_ic_stats.csv`, `eval_ext_portfolio_perf.csv` generated; 6 sub-periods; nn4/nn5 excluded (model collapse); nn3 partial exclusion |
| 5b | Extension: net transaction costs | ✅ Done | Phase 4 window: `eval_tc_summary.csv` (360 months, 20/80 NYSE breakpoints, 0 rank changes). Extension window: `eval_tc_ext_summary.csv` (95 months, same null result) |
| 5c | Extension: feature parsimony | ✅ Done | Canonical outputs generated plus audited robustness artifacts (`eval_parsimony_degenerate_months_summary.csv`, `eval_parsimony_ic_stats_audited.csv`, `eval_parsimony_portfolio_perf_audited.csv`) using stored predictions only |
| 6 | Notebooks and visualization | 🔲 Not started | Pending extension completion |
| 7 | Report | 🔲 Not started | Pending final extension outputs |

---

## Latest Canonical Outputs Read

### Phase 5a extension evaluation outputs

- `data/processed/eval_ext_oos_r2.csv` (13 models × 6 sub-periods; nn4/nn5 all NaN; nn3 NaN in COVID/Rate hikes)
- `data/processed/eval_ext_ic_stats.csv` (same exclusion pattern)
- `data/processed/eval_ext_portfolio_perf.csv` (same exclusion pattern)

### Phase 5b transaction cost outputs

- `data/processed/eval_tc_monthly.csv` (Phase 4 window, 360 months)
- `data/processed/eval_tc_summary.csv` (Phase 4 window summary, fixed + Amihud methods)
- `data/processed/eval_tc_ext_monthly.csv` (extension window, 95 months)
- `data/processed/eval_tc_ext_summary.csv` (extension window summary)

### Phase 5c feature parsimony outputs

- `data/processed/eval_parsimony_oos_r2.csv` (13 models: Phase 4 vs Phase 5c OOS R² with delta)
- `data/processed/eval_parsimony_ic_stats.csv` (mean IC, std IC, ICIR per model on 15-feature panel)
- `data/processed/eval_parsimony_portfolio_perf.csv` (L/S portfolio performance, 1987–2016)
- `data/processed/eval_parsimony_degenerate_months_summary.csv` (zero-variance prediction month diagnostics; nn4=48, nn5=216)
- `data/processed/eval_parsimony_ic_stats_audited.csv` (IC stats with degenerate model-months excluded; includes effective month counts)
- `data/processed/eval_parsimony_portfolio_perf_audited.csv` (portfolio stats with degenerate model-months excluded for affected models)
- `data/processed/run_manifest_parsimony_chronology_reconstructed.csv` (model-year chronology reconstructed from stored predictions)
- `data/processed/phase5c_audit_addendum.json` (compact audit metadata)
- `data/processed/phase5c_selected_features.json` (15 selected features list)
- `data/processed/predictions_parsimony.parquet` (18.2M rows, all 13 models, 1987–2016)
- `data/processed/features_panel_parsimonious.parquet` (15-feature + ret_exc panel)

### Run manifests

- `data/processed/run_manifest.json`
    - `run_timestamp`: 2026-05-25T10:56:29.994295+00:00
    - window: 1987-01-01 to 2016-12-31
    - models in this run: `nn5`
    - status: all years 1987–2016 marked `completed`
- `data/processed/run_manifest_ext.json`
    - `run_timestamp`: 2026-05-26T00:46:31.529659+00:00
    - window: 2017-01-01 to 2024-11-30
    - models: all 13 (`ols3`, `ols_all`, `pcr`, `pls`, `enet`, `glm`, `rf`, `gbrt`, `nn1`–`nn5`)
    - status: all years 2017–2024 marked `completed`
- `data/processed/run_manifest_parsimony.json`
    - `run_timestamp`: 2026-05-26T05:29:33.689703+00:00
    - window: 1987-01-01 to 2016-12-31
    - models: all 13 (append mode, final batch: nn3, nn4, nn5)
  - note: this manifest does not store year-level checkpoints in current artifact version
    - `pre_test_mean_ret`: 0.006853425409644842
- `data/processed/unified_manifest.json`
    - `generated_at`: 2026-05-26T00:10:39.048322+00:00

---

## Key Model Outcomes (from current `eval_*_latest.csv`)

### Pooled OOS R² (1987–2016)

Top models by OOS R²:

1. `nn1`: 0.0034396592
2. `enet`: 0.0018030082
3. `nn2`: 0.0017746471
4. `pcr`: 0.0016885477
5. `pls`: 0.0016277616

Tree models remain negative:

- `rf`: -0.0050936655
- `gbrt`: -0.0098902584

### Rank IC (mean)

Highest mean IC:

1. `glm`: 0.0600802999
2. `pcr`: 0.0541475238
3. `enet`: 0.0537036988

### L/S portfolio (annualized, value-weighted)

Highest annual return / Sharpe:

1. `nn1`: annual_ret 0.300480, sharpe 1.596
2. `nn2`: annual_ret 0.292554, sharpe 1.510
3. `pls`: annual_ret 0.128799, sharpe 0.847

Weakest Sharpe:

- `gbrt`: 0.043
- `nn4`: 0.088
- `nn5`: 0.095

### Prediction dispersion (`pred_std`)

- highest: `gbrt` 0.015789
- lowest: `nn3` 0.001737

---

## Notes on Evidence Boundaries

- Phase 5a completion is evidenced by `run_manifest_ext.json` completion metadata and the three
  `eval_ext_*.csv` output files.  nn4 and nn5 are excluded from all Phase 5a outputs (constant
  predictions = model collapse).  nn3 is excluded from the COVID (2020) and Rate hikes (2022)
  sub-periods; its Post-norm (2023+) metric is based on 2023-only (12 months).
- `R_BENCH_EXT` is loaded dynamically from `run_manifest_ext.json` (`pre_test_mean_ret =
  0.008027891628444195`) rather than hard-coded.  NW SE uses 12 lags (consistent with Phase 4).
- Phase 5b outputs use NYSE 20th/80th-percentile market-cap breakpoints for the fixed schedule
  (corrected from a prior 40th/60th-percentile implementation).  The Amihud spread proxy is a
  monthly approximation (|ret_m| / |prc_m × vol_m|) rather than the standard daily-average; see
  `results_comparison.md` Section 5.2 for the academic disclosure on this limitation.
- Both TC methods produce zero rank changes gross→net in both the Phase 4 and the extension
  windows (null finding on transaction cost robustness).
- Phase 5c nn4/nn5 exhibit degenerate prediction months (zero cross-sectional variance). Audited
  robustness files exclude degenerate model-months for affected models and report effective month
  counts for IC and portfolio inference.
- Phase 5c chronology was reconstructed from stored predictions at model-year level to preserve
  auditability without retraining (`run_manifest_parsimony_chronology_reconstructed.csv`).

---

## Immediate Next Steps

1. Proceed to Phase 6: notebooks and visualization (EDA plots, extension comparison charts, parsimony feature importance chart).
2. Proceed to Phase 7: LaTeX report write-up.
