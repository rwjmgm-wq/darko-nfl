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

## Uncertainty interpretation

The predictive variance for a one-season-ahead forecast is

```
var = state_var + skill_change_var + play_noise_var / target_plays
```

covering three distinct sources: what we don't know about current skill, how
much true skill moves between seasons, and how noisily a finite season measures
it. `skill_change_var` and `play_noise_var` are estimated on the training era.

Measured on the frozen era: nominal 80% interval → **85.5%** actual coverage
(PIT KS = 0.097). Intervals are mildly conservative. The v3 behaviour of using
state variance alone gave **31.5%** coverage and should be considered retracted.

## Known limitations

- **The fusion layer does not improve ranking.** After removing label leakage,
  the fused rating ranks worse than the un-fused informed state (Δr = −0.026,
  season-clustered 95% CI excludes zero). It remains the best predictor on RMSE,
  MAE and CRPS. Ranking and level are not optimized by the same quantity.
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
python scripts/v3_4/build_base_v34.py      # ~20 min, downloads nflverse pbp
python scripts/v3_4/engine_v34.py
python scripts/v3_4/evaluate_v34.py
python scripts/v3_4/test_v34.py            # invariant tests, must exit 0
```
