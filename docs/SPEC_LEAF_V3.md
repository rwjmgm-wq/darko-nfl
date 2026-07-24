# LEAF v3 (QB DARKO) — Pre-Registered Specification

**Written July 2, 2026, BEFORE any v3 evaluation was run.** Deviations logged at
the bottom with reasons. Results in docs/LEAF_V3_RESULTS.md must trace to this.

## Why v3

The v2 audit found: (1) the advertised r=0.906 came from a validation where the
Kalman predictor had already seen the test games (honest next-season number
measured on production data: r ≈ 0.46); (2) defense ratings were fit on the full
period and applied retroactively; (3) the weight-scheme optimizer implemented
linear/proposed schemes with reversed orientation (the "proposed" scheme
zero-weighted the five most recent games), invalidating its published ranking;
(4) in a simple honest check, the adjustment/filter machinery did not beat raw
season EPA for next-season prediction.

v3 rebuilds the rating as a strictly walk-forward system, DARKO-style, where
every layer must earn its keep out-of-sample against naive baselines.

## Data

`data/production/leaf_v2_game_by_game_20251111.csv` — 12,899 QB-games,
648 QBs, seasons 2006–2025, per-game EPA / CPOE / success rate / attempts /
opponent. Only raw per-game components are used (the file's precomputed
adjusted/Kalman columns are tainted by full-period fitting).
Supplemented by: QB birthdates (nflverse players), draft picks (nflverse).

## Discipline

- **Walk-forward everywhere:** a rating at game t uses only information from
  games strictly before t (plus the QB's own game t for the *filtered* state
  used to predict t+1 onward — standard filtering convention).
- **Era split:** ALL hyperparameters (decay rates, shrinkage, Kalman noises,
  fusion weights, prior coefficients, age curve) are fit/tuned on seasons
  **2006–2018** only. Seasons **2019–2025** are the frozen test era, evaluated
  once, all variants reported.

## Targets

- **T1 (primary): next-season EPA/play.** Predictor = rating at end of season
  Y; target = raw EPA/play in season Y+1. Both seasons ≥ 150 plays. Test-era
  pairs: Y+1 in 2019–2025.
- **T2: next-16-games EPA/play.** Evaluated at non-overlapping checkpoints
  (every 8th game of each QB's career with ≥ 16 subsequent games), target
  period starting in the test era.

## Baselines (walk-forward, no tuning or tuned on train era)

- B0: league mean (constant)
- B1: expanding career mean of game EPA
- B2: previous-season mean EPA
- B3: EWMA of game EPA, halflife tuned on train era
- B4: uniform mean of last 12 games

## System layers (each a variant in the final table)

- **L1 — Opponent adjustment:** defense rating at time t = shrunk,
  time-decayed mean of EPA allowed in games strictly before t, centered on the
  rolling league mean. Shrinkage and decay tuned on train era.
  `adj_epa_g = raw_epa_g − def_rating(t)`.
- **L2 — Kalman filter** on adjusted game EPA: random-walk state, observation
  noise scaled by 1/attempts (r_t = r_play / attempts_g), q and r_play tuned by
  one-step-ahead MSE on train era. Filtered state after game t is the rating.
- **L3 — Component fusion:** parallel Kalman states for EPA, CPOE, success
  rate; linear fusion weights fit on train era to predict next-16-games EPA.
- **L4 — Informed priors + age:** rookie initial state mean = linear function
  of log(draft pick) fit on train-era rookies; age drift added to the state
  transition, estimated from train-era within-QB year-over-year changes.
- **LEAF v3 = L1+L2+L3+L4.**

## Metrics & inference

Pearson r and RMSE per target, test era only. Cluster bootstrap by QB (2000
resamples) for CIs on r and on differences vs the best baseline. Kalman
predictive-interval coverage (nominal 80%) reported for T1.
Confirmatory bar, stated in advance: LEAF v3 must beat the best naive baseline
on T1 with a bootstrap CI on the difference excluding zero to claim
"predictive edge"; otherwise report as a tie.

## Fair weight-scheme re-test

Correct-orientation implementations of: uniform, linear-decay family,
exponential family, step family, and the originally proposed steep scheme
(35/25/17/12/7/3/1, newest→oldest). Tuning comparisons on train era; single
frozen comparison on the test era. This replaces the invalid finding in
WEIGHT_OPTIMIZATION_FINDINGS.md.

## Outputs

- `data/production/leaf_v3_ratings.csv` — per-QB-game walk-forward ratings +
  uncertainty (safe to use in any downstream analysis without leakage).
- `docs/LEAF_V3_RESULTS.md`, corrections to README + affected docs.

## Amendment A1 — "break the wall" campaign (registered July 2, 2026, before evaluation)

Ceiling analysis on the full data: season-EPA reliability 0.657, true-skill
persistence 0.641, so a perfect knower of CURRENT skill caps at r = 0.52 on T1.
LEAF v3 sits at 0.468. A1 attacks the two components the ceiling math leaves
open, plus the remaining skill-estimation gap:

**New layers (v3.1), all walk-forward, all tuned/fit on train era <= 2018:**
- L5 skill-change: team-change flag (validated: train coef -0.034, frozen test
  r 0.468->0.476); games missed in season Y (injury proxy).
- L6 target-component: expected schedule effect for season Y+1 = mean
  end-of-Y defense rating of Y+1 opponents (schedules are public before the
  season; walk-forward legitimate).
- L7 play-level components from nflfastR pbp: CPOE overall + deep (15+ air
  yards), deep-attempt rate, sack rate + sack EPA, scramble rate + EPA,
  air-EPA vs YAC-EPA split (receiver after-catch contribution separated),
  QB-hit rate. Each gets its own tuned Kalman state; expanded fusion refit
  on train era.

**Pre-registered success criteria (single frozen test-era evaluation):**
- Primary: v3.1 vs LEAF v3 on T1, cluster-bootstrap CI on delta-r excluding zero.
- Reference points reported: the 0.52 skill-only ceiling, and a schedule-aware
  ceiling recomputed with L6's contribution.
- All layers reported individually (ablation) — no silent dropping.

## Amendment A2 — signal-extraction campaign (registered July 2, 2026, before evaluation)

A1 (new features) failed cleanly. A2 attacks the same 0.468 -> 0.52 gap through
ESTIMATION rather than new information — reweighting the plays and games we
already have:

- **W1 garbage-time weighting:** game EPA recomputed with plays at win
  probability < 0.05 or > 0.95 excluded (re-extracted from pbp with wp).
- **W2 robust filtering:** Kalman innovations winsorized at c * sqrt(innovation
  variance), c tuned on train era (fat-tailed game EPA violates the Gaussian
  observation model).
- **W3 experience-dependent process noise:** q_early for a QB's first G games,
  q_late after; (q_early, q_late, G) tuned on train era.
- **W4 variance-aware nonlinear fusion:** train-era regression adding a
  quadratic state term and a state x state-variance interaction (uncertain and
  extreme estimates regress harder).
- **W5 stack:** train-era ridge over {LEAF v3, B2 prev-season, B4 last-12}.

Same rules as A1: all tuning on seasons <= 2018; ONE frozen evaluation on
2019-2025 T1; ablations all reported; success = cluster-bootstrap dr CI vs
LEAF v3 excluding zero. Anticipated honestly: combined gain of +0.01 to +0.03
at best; the 0.52 skill-only ceiling still binds.

## Amendment A3 — new-information campaign (registered July 2, 2026, before evaluation)

A1 (features from the same data) and A2 (estimation) both failed. A3 introduces
genuinely NEW information: nflverse participation data (2016-2025), which
carries play-level lineups and NGS-derived charting (was_pressure,
time_to_throw, coverage type).

- **P1 pressure-split states:** per-game clean-pocket EPA, EPA under pressure,
  pressure rate faced, mean time to throw. Each gets a Kalman state (v3 noise
  settings reused where possible — the 2016-2018 tuning window is thin and
  will not be over-fit).
- **P2 teammate adjustment (QB RAPM):** walk-forward ridge on play-level EPA
  with QB + offensive-teammate indicator variables, fit on seasons strictly
  before the prediction point; the QB's coefficient is the
  supporting-cast-independent value estimate.

Constraints declared in advance: features exist only 2016+; fusion training
pairs are limited to Y+1 <= 2018 with Y >= 2016 (~60-70 pairs), so the fusion
is a RIDGE with alpha fixed at 1.0 (not tuned) on standardized features.
Evaluation: ONE frozen pass, T1 test-era pairs restricted to Y >= 2016;
LEAF v3 re-scored on the same restricted pairs for a fair delta; cluster
bootstrap CI as before. Success bar unchanged (dr CI excluding zero).

## Amendment A3b — expanding-window fusion (registered before running, July 2, 2026)

A3's fixed-train-era protocol left 59 fusion-training pairs (charting begins
2016). The result (all A3 models significantly WORSE) reflects unlearnable
fusion weights, not necessarily uninformative features. A3b re-scores the SAME
features and hyperparameters under rolling-origin weight refits: the
prediction for a pair with predictor season Y uses ridge weights (alpha=1.0,
unchanged) fit on all pairs with target season <= Y and predictor season
>= 2016. Strictly walk-forward: every weight is estimable at prediction time.

Disclosed risk: this is the second protocol applied to the same frozen era for
A3, after seeing A3's failure. It is scientifically motivated (structural
training shortage) but adaptive; the verdict will say so, and A3b is FINAL for
this campaign regardless of outcome. Baseline (LEAF v3 alone) is re-scored
under the identical rolling protocol for a fair delta.

## Amendment A4 — environment layer (registered July 23, 2026, before any construction or evaluation)

**Motivation.** The r ≈ 0.52 disattenuated-persistence ceiling binds predictors
built from the QB's own performance history. The T1 target is next-season RAW
EPA/play — skill x environment x sampling noise. Part of the Y+1 environment
is public knowledge at Y+1 kickoff, before any Y+1 snap: the QB's team, the
head coach, the Week-1 roster, and the full schedule. These are information
channels outside the QB's history, so the relevant cap for a skill+environment
model is the measurement cap (r ≈ 0.81), not 0.52. A1-A3 never touched this
lane.

**Forecast origin (declared).** Predictions for season Y+1 are made at Y+1
kickoff. All skill states use games through end of Y (unchanged). Environment
covariates may use only facts public before Y+1 Week 1. Data proxies: the
QB's Y+1 team is taken from the Y+1 Week-1 roster (fallback: posteam of his
first Y+1 game); the Y+1 coach is taken from the team's first Y+1 scheduled
game. Both proxy real preseason knowledge; fallback counts will be disclosed.
Franchise relocations (STL→LA 2016, SD→LAC 2017, OAK→LV 2020) are mapped to
stable franchise codes before any cross-season comparison.

**Covariates (fixed in advance — no selection, no additions after this point):**
- **E1 new_team** — 1 if the QB's Y+1 team differs from the team of his last Y game.
- **E2 coach_change** — 1 if the Y+1 team's Week-1 head coach differs from that
  franchise's final-game Y coach.
- **E3 ret_rec_epa** — sum of season-Y total receiving EPA (nflverse weekly
  player stats) over WR/TE/RB on the Y+1 team's Week-1 roster, scaled /17.
  Players without Y stats contribute 0 (that IS the preseason information state).
- **E4 team_pass_env** — the Y+1 team's season-Y team pass EPA/dropback
  (qb_games_base aggregated by posteam). Collinear with the QB's own state for
  stay-put QBs; most informative for movers. Disclosed, kept.
- **E5 sched_def** — mean over the Y+1 team's scheduled opponents of the
  opponent's season-Y defensive EPA/dropback allowed (qb_games_base by defteam),
  one term per scheduled game.

If a covariate cannot be built from available data it is DROPPED and logged as
a deviation — never replaced or redefined.

**Model.** LEAF v4 = walk-forward OLS: `target ~ leaf_v3 + E1..E5`, continuous
covariates standardized on training data only. For each test target season
s in 2019-2025: train on all pairs with target season ≤ s-1 (target ≥ 2008),
predict pairs with target season = s. Plain OLS, 7 parameters, no
regularization, therefore NO hyperparameters and no tuning-contamination risk.
Rolling-origin refitting is estimation, not tuning (A3b precedent).

**Honesty control (registered).** OLS on top of any predictor improves RMSE by
recalibration alone. The fair comparator is therefore **LEAF v3-recal**: the
identical walk-forward OLS with leaf_v3 as the sole feature. The environment
claim is measured as v4 vs v3-recal; v4 vs raw v3 is reported but secondary.

**Evaluation.** ONE frozen pass on the identical T1 pair set (both seasons
≥ 150 plays), restricted to pairs with non-null env features. LEAF v3 and
v3-recal re-scored on the same restricted set. Cluster bootstrap by QB (2000
resamples) on r and on Δr. T2 is skipped: env covariates are season-indexed
and T2 windows cross season boundaries.

**Success criteria:** decisive — Δr(v4 − v3-recal) 95% CI excludes zero;
suggestive — point Δr ≥ +0.02 with CI including zero; otherwise null. RMSE
and per-fold coefficient stability reported alongside. A4 is final for this
campaign regardless of outcome; a null is published as a null.

## Deviations

All logged July 23, 2026, BEFORE the A4 frozen evaluation ran. Sources:
feature-build diagnostics and a pre-run adversarial audit (4 confirmed
findings out of 22 raised).

1. **(A4, pre-eval) Team-code harmonization fix.** Older nflverse roster
   files use PFR-style codes (CLV/BLT/SL/ARZ/HST); the first feature build
   mis-flagged moves and dropped coach/env lookups for those franchises.
   Franchise map extended, features rebuilt. No evaluation had run.
2. **(A4, pre-eval) Fallback pairs excluded.** Audit showed the registered
   first-game team fallback can inject post-origin information for QBs
   unsigned at Y+1 kickoff (one frozen-era pair: Flacco 2022→2023, signed
   by CLE in Nov 2023 — all five covariates would have encoded that signing,
   in v4's favor). A `team_fallback` flag was added to the feature file and
   flagged pairs are EXCLUDED from training and test, counts disclosed in
   the results. Stricter than the registered include-and-disclose.
3. **(A4, pre-eval) Verdict statistic clarified.** The registered "point Δr"
   is the plug-in correlation difference; the first evaluator draft
   mistakenly fed the bootstrap-mean Δr to the verdict thresholds. Fixed:
   plug-in point estimate, bootstrap for the CI only.
4. **(A4, pre-eval) Disclosure-only reporting added to the same pass:**
   per-target-season n/r/Δr, season-demeaned pooled r for v4 and the
   control (audit showed e5 is ~65% between-season variance, so pooled Δr
   could reflect league-trend tracking rather than QB-specific environment),
   v3-recal fold coefficients, and a provenance note — leaf_v3_ratings.csv
   was regenerated 2026-07-14 (weekly data refresh), so the T1 universe is
   200 pairs / 60 QBs vs the published 199 / 61.
