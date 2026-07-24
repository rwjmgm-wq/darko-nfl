# LEAF v3 Results

See docs/SPEC_LEAF_V3.md for the pre-registered design. Engine:
scripts/v3_honest/build_leaf_v3.py -> data/production/leaf_v3_ratings.csv
(walk-forward per-game ratings, safe for any downstream use without leakage).

## Executive summary

- **The honest predictive ceiling for QB EPA with public data is r ~ 0.45-0.48**,
  for both next-season and next-16-games targets. The previously advertised
  r=0.906 came from a leaky validation (see validate_weights.py header) and is
  retracted everywhere it was quoted.
- **On ranking (r), LEAF v3 ties the best naive baselines** - the pre-registered
  bar for claiming an edge (bootstrap CI on the difference excluding zero) was
  NOT met on either target (T1: dr = -0.018 [-0.088, +0.049]; T2: dr = +0.027
  [-0.034, +0.093]).
- **On calibration (RMSE), LEAF v3 is clearly the best predictor on both
  targets**: T1 RMSE 0.115 vs 0.126-0.157 for the baselines; T2 0.106 vs
  0.117-0.281. Ranking QBs is easy; being right about the LEVEL of future
  EPA is where the state-space machinery earns its keep.
- Layer-by-layer: opponent adjustment adds a hair (+0.005 r), informed
  priors/age another hair (+0.004), fusion converts r into RMSE. Nothing is
  dramatic; everything is honest. NFL opponent adjustment matters far less than
  in college (defense ratings predict a game's EPA allowed at only r ~ 0.06).
- The fusion regression independently reproduces the college-side finding:
  success rate (consistency) carries weight comparable to EPA; CPOE adds little.
- The steep-recency weighting scheme, re-tested with corrected orientation,
  is genuinely the worst choice but by ~13% relative (not the retracted "47%,
  #57/60"); uniform vs gentle exponential is a coin flip.


## Frozen test-era results (2019�2025)

All predictors walk-forward; all hyperparameters tuned on 2006�2018 only.

### T1: next-season EPA/play (n = 199, 61 QBs)

| Predictor | r | 95% CI | RMSE |
|---|---|---|---|
| B1 expanding mean | +0.457 | [+0.319, +0.579] | 0.126 |
| B2 prev-season EPA | +0.468 | [+0.346, +0.569] | 0.133 |
| B3 EWMA (hl=3) | +0.395 | [+0.259, +0.536] | 0.157 |
| B4 last-12 mean | +0.483 | [+0.373, +0.582] | 0.133 |
| V1 Kalman raw EPA | +0.471 | [+0.343, +0.581] | 0.120 |
| V2 Kalman opp-adj | +0.476 | [+0.352, +0.596] | 0.120 |
| V3 + priors/age | +0.480 | [+0.343, +0.595] | 0.119 |
| LEAF v3 (fused) | +0.468 | [+0.342, +0.574] | 0.115 |

**LEAF v3 vs best baseline (B4 last-12 mean):** dr = -0.018 [-0.088, +0.049] � CI includes zero.

80% predictive-interval coverage (state variance only): 35%. State variance alone under-covers because the target season has its own sampling noise; a full predictive interval adds target noise variance.

### T2: next-16-games EPA/play (n = 287, 54 QBs)

| Predictor | r | 95% CI | RMSE |
|---|---|---|---|
| B1 expanding mean | +0.421 | [+0.218, +0.582] | 0.117 |
| B2 prev-season EPA | +0.145 | [+0.022, +0.399] | 0.281 |
| B3 EWMA (hl=3) | +0.285 | [+0.130, +0.435] | 0.178 |
| B4 last-12 mean | +0.413 | [+0.260, +0.548] | 0.129 |
| V1 Kalman raw EPA | +0.447 | [+0.271, +0.584] | 0.112 |
| V2 Kalman opp-adj | +0.452 | [+0.281, +0.594] | 0.112 |
| V3 + priors/age | +0.456 | [+0.267, +0.606] | 0.112 |
| LEAF v3 (fused) | +0.446 | [+0.282, +0.588] | 0.106 |

**LEAF v3 vs best baseline (B1 expanding mean):** dr = +0.027 [-0.034, +0.093] � CI includes zero.

## Fair weight-scheme re-test (corrected orientation)

The original optimizer reversed the linear/proposed schemes (35% landed on the
oldest game; the five newest games got zero weight). Corrected results,
next-16-games EPA target, frozen test era:

| Scheme | train r | test r |
|---|---|---|
| uniform | +0.4637 | +0.4078 |
| expo_10 | +0.4705 | +0.4046 |
| step_4 | +0.4460 | +0.4032 |
| expo_15 | +0.4647 | +0.3998 |
| step_6 | +0.4637 | +0.3981 |
| expo_20 | +0.4548 | +0.3937 |
| linear_25 | +0.4648 | +0.3873 |
| expo_25 | +0.4428 | +0.3870 |
| linear_30 | +0.4644 | +0.3868 |
| linear_35 | +0.4641 | +0.3864 |
| linear_40 | +0.4639 | +0.3861 |
| linear_45 | +0.4637 | +0.3859 |
| expo_30 | +0.4298 | +0.3800 |
| step_8 | +0.4733 | +0.3778 |
| proposed_steep | +0.3868 | +0.3531 |

Spread across all correctly-oriented schemes: 0.055 in test-era r — weighting choice within a 12-game window is a second-order decision.

## Amendment A1 results (frozen test era, single evaluation)

| Model | test r | test RMSE | dr vs LEAF v3 [95% CI] |
|---|---|---|---|
| LEAF v3 (baseline) | +0.4680 | 0.1153 | - |
| + L5 (change/missed) | +0.4640 | 0.1157 | -0.0033 [-0.0323, +0.0271] |
| + L6 (schedule) | +0.4676 | 0.1153 | -0.0004 [-0.0010, +0.0001] |
| + L7 (components) | +0.4622 | 0.1160 | -0.0049 [-0.0277, +0.0171] |
| v3.1 (all layers) | +0.4583 | 0.1166 | -0.0091 [-0.0460, +0.0299] |

Skill-only ceiling: r = 0.52. Scramble EPA unavailable (rusher-attributed); logged limitation.

### A1 verdict

The pre-registered campaign to break the r=0.52 skill-only ceiling FAILED on
every front, and the failures are informative:

- **L5 (team change, games missed): dr = -0.003 [CI on zero].** The earlier
  +0.008 from the team-change flag alone was noise; under a joint train-era
  fit it evaporates.
- **L6 (next-season schedule): dr = -0.000.** NFL schedule spread is ~0.008
  EPA/play (SD) - parity makes the schedule door negligible.
- **L7 (play-level components - CPOE overall/deep, sack rate, QB hits,
  air/YAC split): dr = -0.005.** The components carry no next-season
  information beyond the fused EPA state at n=338 training pairs.
- **v3.1 (everything): dr = -0.009 [-0.046, +0.030].** Rejected per the
  pre-registered criterion; LEAF v3 remains the production model.

**What did improve: predictive intervals.** Full predictive variance =
skill-change variance (sd 0.030 EPA/season) + state variance + target
sampling noise (per-play var 4.24). Calibrated on train era, validated on the
frozen era: nominal 80% -> actual 77%, nominal 50% -> 53% (previously 35% at
nominal 80%). Parameters stored in leaf_v3_params.json.

**Standing conclusion:** with public play-by-play data, next-season QB EPA
prediction is solved to within ~0.05 of its theoretical ceiling. The remaining
gap is not reachable through more modeling of the same data; it would require
information that is not in the box score (tracking data, charting, injury
reports, scheme knowledge).

## Amendment A2 results (frozen test era, single evaluation)

| Model | test r | test RMSE | dr vs LEAF v3 [95% CI] |
|---|---|---|---|
| LEAF v3 (baseline) | +0.4680 | 0.1153 | - |
| W2_robust | +0.4758 | 0.1147 | +0.0078 [-0.0150, +0.0295] |
| W3_expq | +0.4705 | 0.1151 | +0.0018 [-0.0265, +0.0293] |
| W1_gt | +0.4650 | 0.1155 | -0.0031 [-0.0318, +0.0247] |
| W123 | +0.4545 | 0.1162 | -0.0139 [-0.0468, +0.0148] |
| W4_nonlinear | +0.4658 | 0.1157 | -0.0030 [-0.0346, +0.0272] |
| W5_stack | +0.4560 | 0.1161 | -0.0119 [-0.0204, -0.0041] |
| v3.2 combined | +0.4724 | 0.1151 | +0.0040 [-0.0208, +0.0263] |

### A2 verdict

The estimation campaign also failed the pre-registered bar. Ten distinct ideas
across two campaigns (A1 features, A2 reweighting) have now been tested against
the frozen era; none produced a dr CI excluding zero in the right direction:

- **W2 robust filtering (+0.008 [-0.015, +0.030])** - the best idea of either
  campaign, directionally consistent with fat-tailed game EPA, but not
  significant. Not adopted per the pre-registered criterion; flagged as the
  first thing to re-examine when more test-era seasons accumulate.
- W3 experience-q: the tuning itself is informative (rookie-era process noise
  came out 4x veteran-era, confirming young QBs' skill moves faster) but it
  does not improve next-season prediction (+0.002).
- W1 garbage-time filtering: hurts. Filtered EPA is noisier per game (fewer
  plays) and one-step wMSE rose 20%; blowout production evidently carries
  real signal at the QB level.
- W5 stacking: significantly WORSE (-0.012 [-0.020, -0.004]) - stacking highly
  correlated predictors overfit even 338 training pairs.

**Program conclusion.** LEAF v3 (r = 0.468) stands at the effective frontier of
public play-by-play data for next-season QB EPA. The distance to the
theoretical skill-only ceiling (0.52) is within the noise of our ability to
measure improvements (CI half-widths ~0.03). Further gains require either new
information (tracking/charting data, injury reports) or a different question
(multi-season targets, game-level markets).

## Amendment A3 results (frozen test era, single evaluation)

Pairs restricted to Y >= 2016 (charting era): train 59, test 199.

| Model | test r | test RMSE | dr vs LEAF v3 [95% CI] |
|---|---|---|---|
| LEAF v3 (restricted) | +0.4680 | 0.1160 | - |
| P1 pressure states | +0.4084 | 0.1207 | -0.0594 [-0.1074, -0.0158] |
| P2 RAPM | +0.4394 | 0.1187 | -0.0281 [-0.0590, -0.0008] |
| v3.3 (P1+P2) | +0.3983 | 0.1226 | -0.0687 [-0.1198, -0.0226] |
| P1 clean-only | +0.4408 | 0.1174 | -0.0280 [-0.0563, -0.0026] |

Standardized train coefficients (v3.3): leaf_v3=+0.051, k_epa_clean=-0.009, k_epa_pressure=+0.017, k_pressure_rate=-0.005, k_tt_mean=-0.009, rapm=-0.005


### A3/A3b verdict

The new-information campaign is decisively negative - the strongest negative
of the three campaigns. Under both the pre-registered fixed-era protocol (A3)
and the disclosed adaptive expanding-window protocol (A3b), every model using
the new information is SIGNIFICANTLY worse than LEAF v3 on the frozen era:

| Model (A3b rolling protocol) | dr vs baseline |
|---|---|
| P1 pressure states | -0.057 [-0.102, -0.016] |
| P1 clean-pocket EPA only | -0.016 [-0.035, -0.000] |
| P2 QB RAPM (teammate-adjusted) | -0.033 [-0.059, -0.009] |
| All combined | -0.084 [-0.124, -0.044] |

Why genuinely new information HURT at the season horizon:
1. Splitting ~550 plays/season into clean/pressured makes each component
   noisier than the whole; the Kalman EPA state already integrates their
   season-relevant content at full sample size.
2. Per-play RAPM with lineup controls is a far noisier QB estimate than the
   filtered EPA state at NFL sample sizes; the bias it removes (supporting
   cast) is smaller than the variance it adds.
3. Season-horizon QB prediction is sample-efficiency-bound, not bias-bound.
   Decompositions that add nuance subtract precision.

These are informative negatives: knowing that pressure splits and teammate
adjustment do NOT improve season-ahead QB projection - measured properly -
is itself non-public knowledge. The pressure data may still matter at other
horizons (single-game, matchup-specific) where its mechanisms operate.

**Final program status after A1 + A2 + A3/A3b:** LEAF v3 (r = 0.468,
RMSE-best, calibrated intervals) is the confirmed frontier. Three campaigns,
seventeen ideas, zero improvements, three informative rejections.

## Amendment A4 results — environment layer (frozen pass, 2019-2025)

Provenance: leaf_v3_ratings.csv was regenerated 2026-07-14 (weekly data refresh), so the T1 universe here is 200 pairs / 60 QBs vs the published 199 / 61; LEAF v3 raw re-scores slightly differently than the frozen table for that reason alone. Team-fallback pairs excluded per deviation 2 (5 train, 1 test).

T1 pairs scored: n = 199 (60 QBs). All three predictors scored on this identical set.

| Predictor | r | 95% CI | RMSE |
|---|---|---|---|
| LEAF v3 (raw) | +0.466 | [+0.327, +0.582] | 0.113 |
| LEAF v3-recal (control) | +0.463 | [+0.335, +0.584] | 0.113 |
| LEAF v4 env | +0.480 | [+0.347, +0.586] | 0.112 |

**Primary: dr(v4 − v3-recal) = +0.016 [-0.017, +0.049]. Secondary: dr(v4 − v3 raw) = +0.014 [-0.019, +0.049]. Verdict: NULL.**

Per-target-season breakdown:

| Season | n | r v3-recal | r v4 | dr |
|---|---|---|---|---|
| 2019 | 27 | +0.344 | +0.381 | +0.037 |
| 2020 | 29 | +0.528 | +0.516 | -0.012 |
| 2021 | 27 | +0.635 | +0.630 | -0.005 |
| 2022 | 30 | +0.441 | +0.484 | +0.043 |
| 2023 | 26 | +0.507 | +0.522 | +0.015 |
| 2024 | 29 | +0.491 | +0.536 | +0.044 |
| 2025 | 31 | +0.404 | +0.420 | +0.016 |

Season-demeaned (within-season) pooled r: v3-recal +0.475, v4 +0.494, dr +0.019 — isolates QB-specific environment signal from league-trend tracking.

Walk-forward coefficients (env standardized on train):

| target_season | n_train | intercept | leaf_v3 | e1_new_team | e2_coach_change | e3_ret_rec_epa | e4_team_pass_env | e5_sched_def | recal_int | recal_slope |
|---|---|---|---|---|---|---|---|---|---|---|
| 2019.0 | 306.0 | 0.0054 | 0.8845 | -0.0344 | -0.0064 | -0.0014 | 0.0101 | -0.003 | -0.0088 | 1.0258 |
| 2020.0 | 333.0 | 0.0094 | 0.8649 | -0.0323 | -0.0138 | -0.0018 | 0.0095 | -0.0019 | -0.0063 | 1.0072 |
| 2021.0 | 362.0 | 0.0115 | 0.8645 | -0.0258 | -0.0145 | -0.0014 | 0.0109 | -0.0012 | -0.0045 | 1.0198 |
| 2022.0 | 389.0 | 0.0081 | 0.9022 | -0.0355 | -0.0118 | -0.0003 | 0.0075 | -0.0002 | -0.0077 | 1.0372 |
| 2023.0 | 419.0 | 0.0088 | 0.8693 | -0.0408 | -0.0105 | 0.002 | 0.0067 | 0.0011 | -0.0087 | 1.0188 |
| 2024.0 | 445.0 | 0.008 | 0.827 | -0.0382 | -0.0086 | 0.0023 | 0.0109 | 0.0051 | -0.0114 | 1.02 |
| 2025.0 | 474.0 | 0.0114 | 0.7756 | -0.034 | -0.0093 | 0.0005 | 0.0168 | 0.0022 | -0.0109 | 1.0197 |

### Program status after A4

Four campaigns, twenty-two ideas. A4 is the first to produce a positive
point estimate of any size: dr = +0.016 [-0.017, +0.049] vs the recalibrated
control, positive in 5 of 7 test seasons, with +0.019 of it surviving
season-demeaning (QB-specific environment, not league-trend tracking) and
sign-stable coefficients across all seven walk-forward fits (new team
~ -0.034 EPA/play, new coach ~ -0.011, team environment ~ +0.010/SD).
Under the pre-registered criteria it is NULL: the sample cannot certify an
effect this small. The honest summary is that the environment layer likely
carries a real but small signal (~ +0.01-0.02 r) that seven seasons of
test data cannot distinguish from zero. It does not ship in LEAF; the
result stands as registered.
