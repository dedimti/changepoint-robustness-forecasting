# Changepoint Robustness Forecasting

Companion repository for the paper:

> **A Changepoint Robustness Metric and Causal Regime-Shift Knowledge Injection for Reliable Demand Forecasting under Structural Change**
> Dedi Irawan, Sudarmaji, Imam Samsudin
> Faculty of Computer Science, Muhammadiyah University of Metro, Lampung, Indonesia
> *Submitted to Knowledge-Based Systems (Elsevier).*

This repository provides reproducibility code, results, figures, and tables for the
manuscript. The headline contribution is **diagnostic**: a metric (ρ) that exposes
forecasting fragility at structural breaks, and an empirically grounded principle for
when that fragility can be repaired.

---

## What the paper introduces

1. **ρ (rho) — a changepoint robustness ratio.** A simple, model-agnostic diagnostic that
   compares forecasting error in the neighbourhood of detected changepoints (`MAE_cp`)
   against error in stable regions (`MAE_normal`):

   ```
   ρ = MAE_cp / MAE_normal
   ```

   `ρ = 1` means equal accuracy at changepoints and in stable regions (full robustness);
   `ρ > 1` quantifies fragility.

2. **An honest negative result.** A changepoint-aware weighted loss lowers ρ only by
   *inflating stable-region error*, not by improving changepoint accuracy — showing that
   error re-weighting alone cannot confer robustness; the model must instead be *given*
   regime information.

3. **A causal local-shift augmentation.** A lightweight, causal input feature
   `s_t = mean(y[t-2:t]) − mean(y[t-5:t-3])` that signals recent regime change to the
   forecaster, with no added recurrent parameters. It significantly reduces changepoint
   error on the persistent-shift Medicaid corpus and transfers to an iTransformer backbone.

4. **A conditioning rule (Section 5.16).** The augmentation's benefit grows with the
   *persistence* of a series' changepoints. Because ρ flags fragility without assuming the
   nature of the breaks, it doubles as a screening criterion for when the remedy applies.

---

## Datasets

| Dataset | Role | Availability |
|---------|------|--------------|
| US Medicaid State Drug Utilization Data (SDUD) | **Main experiments** (Sections 5.1–5.14, 5.16) | Public — [CMS](https://www.medicaid.gov/medicaid/prescription-drugs/state-drug-utilization-data) |
| M5 Forecasting (Walmart retail) | **Cross-dataset validation** (Section 5.15) | Public — [Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) |

> Note: map lines and institutional affiliations follow Elsevier's neutral-jurisdiction policy.
> The raw Medicaid series are not redistributed here; download them from CMS at the link above.

---

## Repository layout

```
.
├── src/
│   ├── m5_changepoint_validation.py          # Section 5.15 — M5 cross-dataset validation (PELT + global LSTM)
│   └── persistence_conditioning_experiment.py# Section 5.16 — controlled persistence sweep (synthetic)
├── results/
│   ├── m5_results.json                        # Aggregated M5 metrics (rho, MAE_cp, DM, Wilcoxon)
│   └── persistence_sweep.json                 # Output of the persistence experiment
├── figures/
│   └── Figure_1.png … Figure_10.png           # Manuscript figures, 300 dpi, ≥2244 px wide
├── tables/
│   ├── Table_1.csv … Table_15.csv             # Manuscript tables, machine-readable
│   └── table_captions.txt
├── manuscript/
│   ├── KBS_Manuscript_Final.docx              # Clean manuscript (title page embedded)
│   ├── Title_Page.docx
│   ├── Highlights.docx
│   ├── Cover_Letter.docx
│   ├── Declaration_of_Interest.docx
│   ├── Tables.docx                            # All tables, Elsevier booktabs style
│   └── Figure_Captions.docx
├── docs/
│   ├── FIGURES.md                             # Figure index + captions
│   └── REPRODUCIBILITY.md                     # Step-by-step reproduction notes
├── requirements.txt
├── CITATION.cff
├── LICENSE                                    # MIT
└── README.md
```

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate     # optional
pip install -r requirements.txt
```

Requires Python ≥ 3.9. Core dependencies: `numpy`, `pandas`, `torch`, `ruptures`, `scipy`.

---

## Reproducing the results

### 1. M5 cross-dataset validation (Section 5.15)

Download the M5 competition data from Kaggle, then:

```bash
# Weekly resolution
python src/m5_changepoint_validation.py --data ./m5-forecasting-accuracy \
    --resolution weekly --out results/m5_weekly.json

# Monthly resolution
python src/m5_changepoint_validation.py --data ./m5-forecasting-accuracy \
    --resolution monthly --out results/m5_monthly.json
```

Expected (matches `results/m5_results.json`): ρ ≈ 1.504 (weekly) / 1.545 (monthly),
confirming the diagnostic reproduces the fragility pattern; the augmentation does **not**
transfer to retail (monthly DM = −3.64, p < 10⁻⁵; weekly DM = −3.47 but Wilcoxon p = 0.76,
i.e. statistically indistinguishable from baseline).

### 2. Persistence conditioning experiment (Section 5.16)

```bash
python src/persistence_conditioning_experiment.py
```

Sweeps a persistence parameter from transient (0.0) to fully persistent (1.0) on synthetic
series, reusing the same `causal_local_shift`, PELT detection, and global-LSTM training as
the M5 script. Writes `results/persistence_sweep.json`.

> **Caveat (reported honestly).** On synthetic series the per-level signal is noisy and does
> not yield a clean monotone trend at the scale tested; the conditioning rule in the paper is
> established from the contrast between the two **real** datasets (persistent Medicaid shifts
> vs transient M5 spikes), not from this synthetic sweep. The script is provided for
> transparency and as a starting point for a binned, data-driven persistence analysis on the
> raw Medicaid series.

### 3. Main Medicaid experiments

The Medicaid pipeline (Sections 5.1–5.14) operates on the raw CMS SDUD download. The figures
and tables in `figures/` and `tables/` are the outputs of that pipeline. Reproduction notes
and the metric definitions used throughout are in `docs/REPRODUCIBILITY.md`.

---

## Metric definitions (as used in code)

```python
# Critical neighbourhood of a changepoint c with half-width w:
K = { t : |t - c| <= w for some detected changepoint c }
N = { t : t not in K }                       # stable region

e_t       = |y_hat_t - y_t|
MAE_cp    = mean(e_t for t in K)
MAE_normal= mean(e_t for t in N)
rho       = MAE_cp / MAE_normal

# Causal local-shift feature:
s_t = mean(y[t-2 : t+1]) - mean(y[t-5 : t-2])
```

---

## Citation

If you use this code, metric, or findings, please cite the paper (see `CITATION.cff`).

```bibtex
@article{irawan_changepoint_robustness,
  title   = {A Changepoint Robustness Metric and Causal Regime-Shift Knowledge
             Injection for Reliable Demand Forecasting under Structural Change},
  author  = {Irawan, Dedi and Sudarmaji and Samsudin, Imam},
  journal = {Knowledge-Based Systems},
  note    = {Submitted},
  year    = {2025}
}
```

---

## License

Code is released under the **MIT License** (see `LICENSE`). The manuscript files in
`manuscript/` are © the authors and provided for transparency; reuse of the text is subject
to the publisher's copyright once published.
