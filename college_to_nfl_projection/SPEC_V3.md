# V3 Pre-Registered Analysis Specification

**Written:** July 2, 2026, BEFORE the v3 evaluation was run.
Any deviation from this spec is logged in the "Deviations" section at the bottom,
with the reason. Results in RESULTS_V3.md must be traceable to this document.

## Motivation

V2 established (leave-one-draft-class-out): college stats predict career
outcomes at CV AUC ~0.68, draft position alone ~0.83, and college stats do not
add to draft position. V3 upgrades the methodology — survival modeling,
measurement-error shrinkage, game-level opponent adjustment, rushing/age/accuracy
covariates, stronger baselines and inference — and tests whether the v2
conclusions survive.

## Targets

**T1 (primary): became a primary NFL starter within 5 seasons of draft.**
A "starter season" is a season with >= 10 games of >= 10 pass attempts
(playoffs included), from nflverse play-by-play 2007-2025. The event is the
first starter season; QBs are censored after their last observed season.
Classes 2007-2021 are fully observed for the 5-season window; later classes
enter the hazard model as censored person-periods.

**T2 (secondary): veteran contract within 6 years of draft.**
Signed a non-rookie QB contract with APY >= 4% of the salary cap (OTC data via
nflverse). 4% is chosen a priori as clearly-above-backup, starter-bridge money.
Sensitivity at 3% and 5% will be reported (all three, not the best).

Rookie-season performance is NOT a target: v2 showed it is unpredictable
out-of-sample (CV R^2 <= 0 for all models including draft position).

## Predictors

**College block** (one fixed set, no search):
1. `epa_adj` — opponent-adjusted EPA/play. Game-level PPA adjustment
   (garbage time excluded, per-game opponent defense SP+, ridge/regression
   adjustment) for careers with 2013+ seasons; v2 season-level schedule
   adjustment otherwise. A `game_level_adj` indicator marks which.
2. `success_rate` — from play-level career files (raw).
3. `big_play_rate` — plays >= 15 yards (raw).
4. `log_attempts` — log career pass plays.
5. `rush_share` — rushing yards / (passing + rushing yards), career, from
   CFBD season stats.
6. `rush_ypg` — rushing yards per game (career average).
7. `comp_pct` — career completion % from CFBD season stats.
8. `age_at_draft` — from nflverse draft data; median-imputed with a
   missingness indicator.
9. `seasons_played` — count of college seasons with plays.

**Measurement model:** `epa_adj` and `success_rate` are shrunk toward the
sample mean with attempts-based empirical-Bayes weights
(w = n / (n + k), k estimated once from the between/within variance
decomposition of season-level EPA; same k applied everywhere).

**Draft block:** natural-log pick plus a 3-knot natural cubic spline of
log(pick), so the baseline can bend like a real draft value curve.
Undrafted QBs in scope are assigned pick 263.

## Models

Discrete-time logistic hazard on person-period rows (QB-season-since-draft,
t = 1..5, dummy per t), L2 penalty C = 1.0 (v2's setting, fixed a priori),
features standardized within training folds.

- **M0**: draft block only.
- **M1**: college block only.
- **M2**: college + draft blocks.

P(starter within 5 seasons) = 1 - prod_t (1 - hazard_t).

## Evaluation

- **Primary:** leave-one-draft-class-out OOF P(starter within 5) for classes
  2007-2021, scored by AUC.
- **Uncertainty:** cluster bootstrap over draft classes (2000 resamples) for
  AUC CIs; DeLong test for M2 vs M0 and M1 vs M0.
- **Calibration:** OOF reliability intercept/slope for M1 and M2.
- **Quasi-holdout:** classes 2022-2024, never used in any v2/v3 development:
  score P(starter within observed horizon) against outcomes to date.
- **T2:** same CV design, logistic (non-survival, since contract timing is
  not modeled), AUC + bootstrap CI.

Success criteria stated in advance:
- M1 vs v2 college-only (0.68): does the upgraded college block do better?
- M2 vs M0: DeLong p < 0.05 would be evidence college stats add to draft
  capital. Expectation stated in advance: they will NOT.

## Registered follow-up hypothesis H2 (late-round edge)

**Registered July 2, 2026, BEFORE scoring the quasi-holdout classes on it.**
Motivated by a post-hoc finding on the 2007-2021 OOF scores (college model
0.703 vs pick 0.756 outside round 1, dead heat; pick order uninformative
within Day 3; top-tercile college-model hit rate 48% vs 27% base at equal
average pick).

**H2:** Among QBs drafted OUTSIDE round 1 (pick >= 33), the college-only
model (M1) discriminates future starters at least as well as draft position
(M0). Test set: quasi-holdout classes 2022-2024 (never used in any v2/v3
development or in the post-hoc subgroup analysis), scored at observed
horizons with models trained on classes <= 2021.

Metrics, fixed in advance: (a) AUC of M1 vs M0 within pick >= 33, with
DeLong p; (b) starter rate in the college-model top HALF vs bottom half of
that subset (halves, not terciles - the expected n is ~15-20). Confirmatory
direction: M1 AUC >= M0 AUC and top-half lift > 1.5x. Expected event count
is small (~5-8); this is a directional first replication, not a definitive
test, and will be re-scored as each future class matures.

## Deviations

Logged after implementation, before RESULTS_V3.md was written:

1. **`has_box_stats` indicator + median imputation** added for `rush_share`,
   `rush_ypg`, `comp_pct` (7% of QBs lack CFBD box stats). The spec did not
   state a missingness policy for these; median + indicator mirrors the
   pre-specified `age_missing` treatment. Decided before evaluation ran.
2. **Bug fix before evaluation:** the first implementation of the game-level
   coverage check counted seasons instead of games (4% coverage instead of
   67%). Fixed prior to any evaluation run.
3. **Deployment-only recalibration:** displayed probabilities in the explorer
   are Platt-recalibrated using the OOF reliability fit and clipped to
   [0.01, 0.99]. This does not affect any evaluation metric (AUC is
   rank-invariant); done because OOF calibration slopes were < 1
   (overconfidence at the extremes).
4. **Post-hoc diagnostic (reported, not selected on):** within-class mean AUC
   computed alongside pooled AUC to rule out era-indicator inflation
   (`game_level_adj` is confounded with draft era). Within-class AUCs were
   slightly HIGHER than pooled for all three models.
5. Person-periods end at the 2025 NFL season (data limit), so the 2021 class
   is the newest with a complete 5-season window — as anticipated in the spec.
6. **Age backfill after the first evaluation run** (v3_backfill_ages.py):
   nflverse birthdates (exact) + CFBD roster class years (flagged estimates).
   Training-set age coverage was already ~98% via the draft file, so the
   primary metrics moved by <= 0.01; the backfill mainly improves pre-draft
   prospect predictions. Both before/after numbers preserved in git-less form:
   college-only 0.742 -> 0.733, draft 0.884 -> 0.887, combined 0.890 -> 0.894
   (the shift also includes deviation 7's data fixes).
7. **Data corrections after inspecting deployed prospect output** (not model
   changes): (a) rushing yards clipped at 0 before computing rush_share
   (sack yardage made shares negative); (b) 12 stale "prospect" rows relabeled
   to their true draft class (they had already been drafted); (c) one duplicate
   player row removed; (d) undrafted players whose class-year age implies they
   are not draft-eligible (year <= 2) are excluded from deployed projections
   rather than extrapolated.
