# V3 Evaluation Report

Implements SPEC_V3.md (pre-registered). All numbers out-of-fold, leave-one-draft-class-out.

Sample: 125 drafted QBs, classes 2007-2021, 60 became primary starters within 5 seasons (48%).

## T1: primary starter within 5 seasons

| Model | CV AUC | 95% CI (class bootstrap) |
|---|---|---|
| M0_draft | 0.887 | [0.809, 0.950] |
| M1_college | 0.733 | [0.666, 0.803] |
| M2_both | 0.894 | [0.829, 0.951] |

- DeLong M2_both vs M0_draft: AUC 0.894 vs 0.887, **p = 0.718**
- DeLong M1_college vs M0_draft: AUC 0.733 vs 0.887, **p = 0.002**

## Calibration (reliability intercept / slope; ideal 0 / 1)

- M0_draft: intercept -0.27, slope 0.83
- M1_college: intercept -0.06, slope 0.61
- M2_both: intercept -0.38, slope 0.70

## Quasi-holdout (classes 2022-2024, observed horizons)

Sample: 29 QBs, 14 starters so far.

- M0_draft: AUC 0.819
- M1_college: AUC 0.652
- M2_both: AUC 0.795

## T2: veteran contract (APY % of cap) within 6 years, classes <= 2020

| Threshold | M0 AUC | M1 AUC | M2 AUC | base rate |
|---|---|---|---|---|
| 3% | 0.826 | 0.661 | 0.783 | 34% |
| 4% | 0.799 | 0.650 | 0.775 | 31% |
| 5% | 0.772 | 0.605 | 0.733 | 29% |
