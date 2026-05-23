# Submission Package README

This package contains the promoted final C2O close-to-open strategy submission.

## Final Strategy

- Experiment id: `phase2_g5_05_expanding`
- Target: `volatility_scaled_overnight_return`
- Target definition: `overnight_next / (vol20 / sqrt(252))`
- Training window: expanding
- Alpha learning: 50% prior / 50% learned (transparent correlation weights)
- Basket: top/bottom 3%, equal weight, minimum 15 per side
- Ranking: raw alpha score
- Short treatment: tiered borrow cost only (no hard exclusions)
- Cutoff: 2024-12-31
- ML comparison: LightGBM tested; results in `final_outputs/model_comparison.csv`

## Headline 250M Result

- Net annual return: `7.31%`
- Net annual volatility: `4.97%`
- Net Sharpe: `1.445`
- Lo-corrected Sharpe: `1.578`
- Max drawdown: `-10.19%`
- IC mean: `0.029`, t-stat: `9.93`

## Key Files

### Report
- `report/final_report.html` — comprehensive 15+ page HTML report with figures
- `report/final_report.md` — markdown source
- `report/innovation_declaration.md`

### Strategy Outputs
- `final_outputs/performance_summary.csv`
- `final_outputs/daily_returns_{50m,250m,1b}.csv`
- `final_outputs/positions_{50m,250m,1b}.parquet`
- `final_outputs/sharpe_lo_corrected.csv` — Lo (2002) autocorrelation correction
- `final_outputs/model_comparison.csv` — Linear vs LightGBM IC comparison
- `final_outputs/lgbm_diagnostics.csv` — LightGBM feature importances by year

### Figures (10 figures)
- `final_outputs/figures/equal_weight_return_decomposition.png`
- `final_outputs/figures/strategy_cumulative.png`
- `final_outputs/figures/rolling_ic.png`
- `final_outputs/figures/yearly_sharpe_250m.png`
- `final_outputs/figures/gross_to_net_waterfall.png`
- `final_outputs/figures/universe_eligible_count.png`
- `final_outputs/figures/borrow_tier_distribution.png`
- `final_outputs/figures/feature_ablation.png`
- `final_outputs/figures/drawdown_250m.png`
- `final_outputs/figures/ic_comparison_linear_vs_lgbm.png`

### QuantStats
- `quantstats/quantstats_250m.html`

### Audit Evidence
- `audit_evidence/phase2_promotion_audit.md`
- `audit_evidence/final_reproduction_log.md`

## Reproduction

From the repository root:

```bash
make reproduce
make test
```

Latest recorded result: `13 passed`.
