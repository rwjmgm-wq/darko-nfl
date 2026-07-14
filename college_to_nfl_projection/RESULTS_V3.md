# V3 Results: Pre-Registered Survival Analysis

**Date:** July 2, 2026.
**Spec:** SPEC_V3.md (written before evaluation; deviations logged there).
**Full numbers:** results/v3_evaluation_report.md. OOF predictions: results/v3_oof_scores.csv.

## What v3 changed over v2

| Component | v2 | v3 |
|---|---|---|
| Outcome model | binary logistic, classes ≤2019 only (n=106) | discrete-time hazard, censoring-aware, classes 2007–2021 fully scored (n=125), later classes contribute censored person-periods |
| Target | 85+ career starts or 30+ streak (~32%) | primary-starter season (10+ starts) within 5 years (~48%) |
| NFL outcome data | career totals, games with ≥1 attempt | per-season counts, games with ≥10 attempts (relief appearances excluded) |
| Opponent adjustment | season-level schedule average | game-level PPA vs. specific opponent, garbage time excluded, for 2013+ careers (67% of QBs); β = +0.0125 (SE 0.0003, n = 34,310 QB-games) |
| Measurement error | none | EB shrinkage of EPA/success toward the mean, k = 112 plays |
| Covariates | EPA, success rate, big-play rate, attempts | + rushing share, rushing yds/game, completion %, age at draft, seasons played |
| Draft baseline | log(pick) | natural cubic spline of log(pick) |
| Inference | point AUCs | class-bootstrap CIs, DeLong tests, calibration, within-class robustness, quasi-holdout |
| Selection control | post-hoc honesty caveat | spec pre-registered before evaluation |

## Headline results (all out-of-sample, leave-one-draft-class-out)

**T1: primary starter within 5 seasons (125 drafted QBs, classes 2007–2021)**

(final numbers, after the age backfill and data corrections logged in
SPEC_V3.md deviations 6–7)

| Model | CV AUC | 95% CI |
|---|---|---|
| Draft position only | 0.887 | [0.809, 0.950] |
| College stats only | 0.733 | [0.666, 0.803] |
| College + draft | 0.894 | [0.829, 0.951] |

- **College + draft vs. draft alone: statistical tie** (0.894 vs 0.887, DeLong
  p = 0.72). Exactly the pre-registered expectation: college stats do not add
  measurably to draft position.
- **Draft position is significantly better than college stats alone**
  (DeLong p = 0.002).
- **The college-only model improved from 0.68 (v2) to ~0.73–0.74** — the
  methodology upgrades bought real signal.
- **Quasi-holdout** (2022–2024 classes, untouched during all development):
  same ordering — draft 0.819, combined 0.795, college 0.648 (n = 29).
- **Within-class check:** mean within-class AUCs are slightly *higher* than
  pooled (college 0.761, draft 0.899, combined 0.908) — discrimination is
  genuine within draft classes, not an era artifact.
- **T2 (veteran contract ≥4% of cap within 6 years):** same ordering at all
  three pre-specified thresholds (e.g., at 4%: draft 0.799, college 0.650,
  combined 0.775).

## The interesting substantive finding

Standardized hazard coefficients in the college-only model:

| Feature | Coef | Reading |
|---|---|---|
| age at draft | −0.68 | younger = much better. Strongest college signal. |
| rushing share | +0.54 | dual-threat production translates |
| completion % | +0.47 | accuracy travels |
| success rate (shrunk) | +0.11 | almost fully absorbed by the above |
| adjusted EPA (shrunk) | +0.10 | ditto |
| big-play rate | −0.02 | explosiveness adds nothing |

**The v2 box-score signal was largely a proxy for age, rushing, and
accuracy.** Once those are in the model, opponent-adjusted EPA and success
rate contribute almost nothing. This matches the public research consensus
(breakout age, rushing production, and accuracy are the durable college
predictors; EPA-style efficiency is mostly priced into them). The hazard
shape is also sensible: starter events concentrate in the first two seasons.

Calibration: OOF reliability slopes 0.58–0.85 (mild overconfidence at the
extremes). Deployed probabilities are Platt-recalibrated and capped at 99%.

## Bottom line

The v2 conclusion **survives an academic-grade replication**: college
performance data contains real, improvable signal about NFL careers
(0.68 → 0.74 with better methods), but the draft market at 0.88 remains
unbeaten, and adding college stats to draft position yields a statistical
tie, not an edge. The model's legitimate uses are pre-draft projection
(no pick exists yet) and flagging QBs where the college model and the
market sharply disagree — e.g., the model gave Brock Purdy (pick 262) a
62% college-only starter probability against a draft-implied ~14%.

## H2: the late-round edge (registered follow-up, first replication)

Post-hoc analysis of the 2007–2021 OOF scores showed the market's advantage
is a **round-1 phenomenon**: outside round 1, college model 0.703 vs pick
0.756 (dead heat, p = 0.57); within Day 3, pick order is uninformative
(AUC 0.365) while the college model holds 0.678; sorting non-first-rounders
by the college model gives starter hit rates of 48% / 18% / 14% by tercile
at essentially equal average pick.

H2 was then registered in SPEC_V3.md and scored once on the untouched
quasi-holdout classes (2022–2024, picks ≥ 33, n = 20, 5 starters so far):

- **AUC: college 0.627 vs draft position 0.493** (DeLong p = 0.61) —
  direction confirmed, draft order again ~coin-flip outside round 1.
- **Top half by college model: 4/10 starters (40%) vs bottom half 1/10
  (10%) — 4.0x lift** at nearly equal average pick (151 vs 163). Clears the
  pre-registered 1.5x bar. Hits: Howell, Purdy; visible misses: Willis
  (ranked 2nd, bust so far), O'Connell (ranked last, became a starter).

Both pre-registered confirmatory directions were met. With 5 events nothing
is statistically significant — this is a successful first replication of a
directional hypothesis, to be re-scored as each class matures. The honest
claim: **within the draft's late rounds, where pick number carries almost no
information, opponent-adjusted college production still does.**

## Limitations

- **Age at draft is now exact for 100% of drafted QBs** (nflverse draft data +
  birthdates). Pre-draft prospects use CFBD class-year estimates (~±1 year
  noise from redshirts) for half the pool and median imputation for the rest,
  so prospect rankings lean partly on an estimated age.
- The starter definition is a proxy (10+ games with 10+ attempts, playoffs
  included), not official starts.
- Careers file staleness: some 2025-class QBs are labeled 2026 prospects.
- Drafted-QB-only sample: range restriction attenuates college-stat effects;
  unavoidable without outcomes for undrafted players.
- ~10 modeling decisions were locked in SPEC_V3.md before evaluation, but the
  v2→v3 direction itself was informed by v2 results on overlapping data; the
  quasi-holdout classes (2022–2024) are the only fully untouched test set, and
  they agree with the headline ordering.
