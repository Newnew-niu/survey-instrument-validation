# Survey instrument validation: a 40-item questionnaire on graduate students' psychosocial development

**Author:** Faye Niu · **Portfolio:** [`portfolio/Faye_Niu_Survey_Research_Portfolio.pdf`](portfolio/Faye_Niu_Survey_Research_Portfolio.pdf)

This repository contains the code for my BSc thesis survey (Xi'an Technological University, 2024–25) and the 2026 re-analysis behind my survey research portfolio. It covers the full process: questionnaire design, fieldwork, data-quality screening, reliability and validity testing, a weighted scoring model, segment comparisons and a short-form analysis.

## Key results

| | |
|---|---|
| Sample | 154 usable responses out of 175 (88%), master's students, collected on Wenjuanxing |
| Reliability | Cronbach's α .91–.92 per block; composite reliability .91–.92 |
| CFA (4 factors) | χ²/df = 1.41, RMSEA = .052, CFI = .93, TLI = .93 |
| Discriminant validity | **Not supported.** Latent correlations ≈ 1.0. Parallel analysis supports one factor, and the one-factor model has the lowest AIC and BIC, so the total score is used as the main measure |
| Scoring model | Fuzzy comprehensive evaluation: Good 64, Normal 21, Mild 67, Severe 2. 87–97% of classifications stay the same under alternative weights |
| Segments | No meaningful differences by gender (d = 0.04) or year of study (η² = .01) |
| Short form | 8 items: r = .96 with the 40-item total and 87% level agreement on held-out halves (baseline 40%) |

## Repository structure

```
analysis/
  01_psychometrics.py        data quality, item analysis, reliability, EFA, CFA
  02_scoring_segments.py     fuzzy scoring, weight sensitivity, model comparison, segment tests
  03_figures.py              figures 1–4
  04_short_form.py           8-item short form, split-half validation x 200
  05_robustness_checks.py    outlier exclusion; GA-BP input/target check
outputs/                     printed results of each script
figures/                     charts used in the portfolio (SVG)
matlab/                      original thesis GA-BP scripts
questionnaire/               all 40 items in English and Chinese
data/                        data dictionary (respondent-level data available on request)
portfolio/                   portfolio PDF
```

## Reproducing the results

```bash
pip install -r requirements.txt
# put survey_responses.csv and scoring_memberships.csv in data/ (see data/README.md)
python analysis/01_psychometrics.py
python analysis/02_scoring_segments.py      # also writes outputs/scored.csv
python analysis/03_figures.py
python analysis/04_short_form.py
python analysis/05_robustness_checks.py
```

Tested with Python 3.10. Random processes use fixed seeds, so the numbers match the portfolio exactly.

## Data availability

Respondent-level data is not published, because participants were told their answers would be used for research only. The anonymised dataset is available on request. The data dictionary is in [`data/README.md`](data/README.md).

## Tools

Wenjuanxing (fieldwork) · SPSS (descriptives, EFA, reliability) · AMOS (CFA) · Python: pandas, SciPy, factor_analyzer, semopy, scikit-learn, matplotlib (scoring model and re-analysis) · MATLAB (GA-BP model) · Excel
