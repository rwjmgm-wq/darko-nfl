# V2 Evaluation Report (out-of-sample)

Fixed feature set, no search. Scaling + regularization fit inside training folds. Folds = draft classes.

College features: `['epa_per_play_adj', 'success_rate', 'big_play_rate', 'attempts']`

## Target 1: NFL rookie LEAF

Every metric below is out-of-fold (leave-one-draft-class-out). CV R^2 can be negative: that means worse than predicting the mean.

| Model | n | CV R^2 | CV r | MAE |
|---|---|---|---|---|
| Draft capital only: log(pick) | 108 | +0.002 | +0.081 | 0.260 |
| College only (raw EPA) | 108 | -0.029 | -0.198 | 0.273 |
| College only (adjusted EPA) | 108 | -0.028 | -0.120 | 0.273 |
| College (adjusted) + log(pick) | 108 | -0.020 | -0.021 | 0.269 |

## Target 2: became a starter (85+ starts or 30+ consecutive), classes <= 2019

Newer classes are right-censored (not enough seasons to reach the threshold) and are excluded from training and scoring.

Sample: 106 QBs, 34 starters (32%).

| Model | n | CV AUC | CV log-loss | Base-rate log-loss |
|---|---|---|---|---|
| Draft capital only: log(pick) | 106 | 0.828 | 0.467 | 0.627 |
| College only (adjusted EPA) | 106 | 0.680 | 0.577 | 0.627 |
| College (adjusted) + log(pick) | 106 | 0.813 | 0.471 | 0.627 |

## Sensitivity: aggregation method x EPA variant (starter target, college-only)

| Aggregation | EPA variant | CV AUC | CV AUC (+log pick) |
|---|---|---|---|
| career_average | raw | 0.663 | 0.812 |
| career_average | adjusted | 0.680 | 0.813 |
| recency_weighted | raw | 0.643 | 0.815 |
| recency_weighted | adjusted | 0.662 | 0.821 |
| final_season | raw | 0.560 | 0.823 |
| final_season | adjusted | 0.561 | 0.815 |

## 4-class outcome model (Elite / Solid Starter / Journeyman / Bust), classes <= 2019

Sample: 92 QBs. Class counts: {'Bust': np.int64(37), 'Solid Starter': np.int64(28), 'Journeyman': np.int64(21), 'Elite': np.int64(6)}.

- CV log-loss: **1.280** vs base-rate 1.244 (does NOT beat always-predict-frequencies)
- CV accuracy: **0.413** vs majority-class 0.402
