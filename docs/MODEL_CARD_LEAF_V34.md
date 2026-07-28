# Model Card — LEAF v3.4 (QB rating / next-season EPA forecast)

Status: **re-certified baseline, not yet deployed.** v3 remains in production
(`data/production/leaf_v3_ratings.csv`, the Railway explorer) until Phase 2
aligns the deployed forecast path. Nothing in this card describes the live site.

## What it is

A walk-forward state-space rating for NFL quarterbacks. Each QB has a latent
skill state updated game by game with a scalar Kalman filter on
opponent-adjusted EPA per dropback, plus draft-pick-informed rookie priors and
an age drift term. A fusion layer combines EPA, CPOE and success-rate states.

Version artifacts: `scripts/v3_4/`, `data/production/leaf_v34_*.csv|json`,
`docs/LEAF_V34_RECERTIFICATION.md`.

## Forecast origin and targets

- **Rating at game _t_** uses only information from games strictly before _t_;
  the filtered state additionally incorporates game _t_ itself and is used to
  predict _t+1_ onward (standard filtering convention).
- **T1 (primary target):** raw EPA per dropback in season _Y+1_, forecast from
  the QB's state at the end of season _Y_. Both seasons must have ≥ 150 plays.
- **Fusion training target:** the QB's next 16 **appearances**, not next 16
  calendar games. Because inactive stretches can stretch that window for years,
  the deployed fit is restricted to spans ≤ 730 days; the unrestricted
  "next-16-appearances" fit is reported separately and is **not** deployed.

## Population filters

- Regular season only (`season_type == 'REG'`) by default. A postseason-inclusive
  file can be built explicitly and is written under a different name so it cannot
  be confused with the default.
- QB passers only, by position from nflverse player metadata. Non-QB passers
  (wide receivers, punters, running backs on trick plays) are excluded.
- Excluded rows are counted in `qb_games_base_v34_dropreport.csv`, never
  silently discarded. For 2006–2025: 1,497 non-QB passer plays and 17,357
  postseason dropbacks removed.
- Grouping is on player identity, not name, so nflverse spelling variants within
  one game cannot split a QB-game.

## Training / evaluation split

All hyperparameters, priors, age curves, fusion weights and predictive-interval
variance components are fit on **2006–2018 only**. 2019–2025 is the frozen
reporting era, used here solely to re-certify the leak corrections. It must not
be used to select candidate models (see the Phase 3 gate in the audit brief).

## Uncertainty interpretation — two explicitly separate modes

Only Mode A is deployable. Reporting Mode B numbers as if they were forecast
accuracy would overstate what the model can actually do at forecast time.

### Mode A — `forecast_origin` (deployable)

```
var = scale * (state_var + skill_change_var + play_noise_var * E[1/plays])
```

Uses only information available at the origin. Because a forecaster does not
know how many plays a QB will take next season, the sampling-noise term uses
the **training-era expectation** `E[1/plays] = 0.002454` (implied ≈ 407 plays),
never the realized count.

`scale` is fit to **rolling out-of-fold training residuals** (n = 203, target
seasons 2012–2018): for each calibration season, fusion weights are refit on
pairs ending at or before the prior season and used to predict it. This matters
for the fused rating specifically — `leaf_v34` is a linear combination of three
correlated Kalman states plus a fusion residual, so its variance is **not**
`k_informed_var`. Rather than assert an independence structure we do not model,
the calibration absorbs fusion-coefficient uncertainty, CPOE/success state error
and their covariance jointly. Fused and informed means carry separate scales
(0.745 and 0.792).

Measured on the frozen era: nominal 80% → **80.0%** actual (PIT KS 0.084) for
the fused mean; **80.5%** (KS 0.054) for the informed state.

### Mode B — `conditional_realized_volume` (retrospective diagnostic only)

```
var = state_var + skill_change_var + play_noise_var / target_plays
```

Uses the realized next-season play count, which no forecast can know. Reported
only to show how much interval width is attributable to volume uncertainty.
Frozen-era coverage 85.5% (fused). **Never deploy this.**

The v3 behaviour of using state variance alone gave **31.5%** coverage at a
nominal 80% and is retracted.

## Known limitations

- **The fusion layer may not improve ranking — SUGGESTIVE, not confirmatory.**
  After removing label leakage the fused rating ranks below the un-fused
  informed state (Δr = −0.0259), but the primary QB-clustered 95% CI
  [−0.059, +0.006] **includes zero**. The season-clustered CI [−0.046, −0.003]
  does exclude zero, but rests on only 7 clusters and is a sensitivity check,
  not a decisive test. Leave-one-target-season-out Δr stays negative across all
  seven refits, range [−0.0337, −0.0202], so no single season drives it; the
  per-season Δr is nonetheless positive in 2019 (+0.048) and negative in the
  other six. The honest reading is a consistent but statistically unconfirmed
  deficit. Fused remains best on RMSE and MAE — ranking and level are not
  optimized by the same quantity.
- **Two fusion target definitions are carried, neither chosen on frozen data.**
  The deployed fit uses a dense target (span ≤ 730 days); the unrestricted
  next-16-appearances fit is reported alongside it. The 730-day rule exists
  because "next 16 games" really means "next 16 appearances", which for QBs
  with long inactive stretches reached 4,375 days — not a short-horizon label
  in any useful sense. Threshold sensitivity must be explored on training-era
  data only.
- **Not comparable to v3's published table.** The population changed, so the
  targets differ. No v3-vs-v3.4 horse race on identical data is claimed.
- **Survivor-conditioned.** T1 requires ≥ 150 plays in both seasons, so all
  reported accuracy is conditional on a QB continuing to play. Availability is
  not modeled (Phase 3 item).
- **Multi-season forecasts are not evaluated.** Only the one-season-ahead
  horizon has been scored. Longer horizons must not be presented as model
  output until recursively evaluated.
- **Opponent adjustment is weak.** NFL defense ratings predict a game's EPA
  allowed at roughly r ≈ 0.06; the adjustment is close to a small constant.
- **Deployment mismatch is open.** The v3 export labels the fused column as the
  rating while the dashboard forecasts from the informed state. Unresolved until
  Phase 2.

## Reproduction

```
python scripts/v3_4/preflight.py           # deps check; installs nothing
python scripts/v3_4/build_base_v34.py      # ~20 min, downloads nflverse pbp
python scripts/v3_4/engine_v34.py
python scripts/v3_4/smoke_test.py          # tests + evaluator + manifest
```

Non-default populations write to distinct filenames and can never overwrite the
default artifact: `--include-postseason` and/or `--keep-non-qb` produce
`qb_games_base_v34_withpost.csv`, `_allpassers.csv`, `_withpost_allpassers.csv`.

Tracked artifacts and their SHA-256 hashes are listed in
`docs/LEAF_V34_MANIFEST.md`, so the baseline can be verified from a clean
checkout without re-downloading nflverse data.
