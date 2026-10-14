# Data

The respondent-level data is **not included** in this repository. Participants were told their answers would be used for research only, so I have not published the raw file. The anonymised dataset is available on request: please contact me through GitHub.

The scripts expect two files in this folder:

## `survey_responses.csv`: 154 rows, one per respondent

| Column | Type | Values |
|---|---|---|
| `respondent_id` | integer | 1–154 |
| `A1`–`A10` | integer | 1–5 · Mental health block |
| `B1`–`B10` | integer | 1–5 · Social adaptation block |
| `C1`–`C10` | integer | 1–5 · Academic pressure block |
| `D1`–`D10` | integer | 1–5 · Self-actualisation block |
| `gender` | integer | 1 = male, 2 = female |
| `year` | integer | 1, 2, 3 = year of master's study |

Scale: 1 = strongly disagree … 5 = strongly agree. Higher means more difficulty on every item. Item wording is in [`../questionnaire/questionnaire.md`](../questionnaire/questionnaire.md).

The file has no missing values and no free-text fields. Of the 175 responses collected on Wenjuanxing, 154 were kept after removing incomplete and careless responses. The rules used for that step were not recorded at the time; [`05_robustness_checks.py`](../analysis/05_robustness_checks.py) and section 3 of the portfolio document the screen I applied afterwards.

## `scoring_memberships.csv`: input to the MATLAB model (Appendix B)

| Column | Description |
|---|---|
| `mu_good`, `mu_normal`, `mu_mild`, `mu_severe` | Membership degrees from the fuzzy scoring model (0–1) |
| `level` | Assigned level: 0 = Good, 1 = Normal, 2 = Mild, 3 = Severe |
