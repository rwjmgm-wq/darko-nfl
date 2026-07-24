# H3 Landing-Spot Report

Pre-registered in SPEC_V3.md (H3). One frozen pass.

Sample: 125 drafted QBs, classes 2007-2021, 60 starters. Landing coverage 100.0%.

## LODCO OOF AUC

| Model | AUC | 95% CI |
|---|---|---|
| M0_draft | 0.887 | [0.809, 0.950] |
| M3_landing | 0.875 | [0.800, 0.941] |
| M2_both | 0.894 | [0.829, 0.951] |
| M4_all | 0.881 | [0.817, 0.941] |

- **PRIMARY** DeLong M3_landing vs M0_draft: AUC 0.875 vs 0.887, p = 0.185
- **secondary** DeLong M4_all vs M2_both: AUC 0.881 vs 0.894, p = 0.013

## Quasi-holdout (classes 2022-2024, directional only)

Sample: 29 QBs, 14 starters so far.

- M0_draft: AUC 0.819
- M3_landing: AUC 0.829
- M2_both: AUC 0.795
- M4_all: AUC 0.786

## Landing coefficients (M3, full dev fit, standardized)

- l1_team_pass_epa_prev: -0.176
- l2_hc_new: +0.065
- l3_qb_churn3: -0.292

**Verdict (pre-registered criteria): NULL** — OOF dAUC -0.012 (DeLong p = 0.185), holdout directional dAUC 0.01.
