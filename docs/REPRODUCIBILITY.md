# Reproducibility Notes

This document records the experimental settings used throughout the manuscript so the
results in `figures/`, `tables/`, and `results/` can be reproduced.

## Environment

- Python ≥ 3.9
- See `requirements.txt` (numpy, pandas, torch, ruptures, scipy)
- All experiments run on CPU; no GPU is required.
- Global random seed: `7` (M5 script uses `--seed`, default in code).

## Common hyperparameters (manuscript Table 3)

| Hyperparameter | Value |
|----------------|-------|
| Lookback length L | 12 quarters |
| Forecast horizon H | 4 quarters |
| LSTM layers | 1 |
| Hidden size | 32 |
| Optimizer | Adam |
| Learning rate | 1e-2 |
| Epochs | 40 |
| Training scheme | global (one model across all series) |
| Standardization | per-series z-score (training statistics only) |
| PELT cost / penalty | l2 / β = 2.0 |
| Critical half-width w | 1 quarter |
| λ values explored | {1, 5, 10, 20, 50} |

## Changepoint detection

PELT (Pruned Exact Linear Time) with an l2 cost on each z-scored series, via the
`ruptures` package. The series endpoint returned by PELT is dropped. A time point `t` is
"critical" if it lies within `w` periods of any detected changepoint.

## Train/test split

The split follows the calendar. With L = 12 and H = 4, a 24-quarter series yields nine valid
(input, target) windows. The last calendar year (four quarters) is held out, so each series
contributes four evaluation windows. Over the 5,000-series main subset this gives 20,000 test
windows; the 34,177 figure quoted in the Diebold–Mariano tests counts individual critical-region
time points, not windows.

## Metrics

```
e_t        = |y_hat_t - y_t|
MAE_cp     = mean(e_t for t in critical neighbourhoods K)
MAE_normal = mean(e_t for t in stable region N)
rho        = MAE_cp / MAE_normal
```

Significance is assessed with the Diebold–Mariano test on squared errors at critical points
and, for the M5 transfer, additionally with a Wilcoxon signed-rank test.

## Datasets

- **Medicaid SDUD** (main): download from CMS, define a series as
  `utilization type × state × NDC`, signal = Units Reimbursed, period 2014Q1–2019Q4
  (24 quarters), keep series with mean ≥ 2 and ≥ 3 detected changepoints.
- **M5** (validation): download from Kaggle; aggregate to weekly and monthly resolution;
  keep changepoint-heavy series as in `src/m5_changepoint_validation.py`.

## Notes on the conditioning experiment (Section 5.16)

The conditioning **rule** in the paper is established from the contrast between the two real
datasets (persistent Medicaid level shifts → augmentation helps; transient M5 spikes → it does
not). The synthetic persistence sweep in
`src/persistence_conditioning_experiment.py` is provided for transparency; at the scale tested
it does not produce a clean monotone trend, and we report this openly rather than overclaiming.
A stronger, data-driven version would bin actual Medicaid changepoints by retained
post-break level difference and measure the augmentation's benefit per bin.
