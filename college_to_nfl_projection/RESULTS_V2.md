# V2: Corrected Model & Honest Results

**Date:** July 2, 2026
**Supersedes:** the conclusions in `IMPLEMENTATION_RESULTS.md` and `FEASIBILITY_REPORT.md`.

## Why a v2 exists

A methodology review found the published results were artifacts of bugs and
overfitting, not predictive power:

1. **The SP+ adjustment was a no-op** (`apply_sp_plus_to_careers.py`): it
   compared the national-average defense rating to itself, so `sos_percentile`
   was always exactly 50 and "adjusted" EPA always equaled raw EPA. Every
   downstream model trained on "SP+ adjusted" data was actually trained on raw data.
2. **The older 50-QB pipeline had the adjustment sign flipped**
   (`calculate_college_leaf_full.py` shipped with "REVERSED LOGIC TEST"
   constants that rewarded playing weak defenses), used a *multiplicative*
   adjustment on signed EPA (which punishes bad plays against good opponents —
   backwards), and fabricated offense/defense SP+ as fixed 60/40 shares of the
   overall rating.
3. **All headline metrics were in-sample.** The "r = 0.64, R² = 41%, beats
   draft capital by 318%" numbers came from exhaustively searching ~10,000
   feature combinations on ~50 QBs and reporting the best training fit.
   `validate_predictions.py` re-fit the same model on the same data and called
   it validation.
4. **Right-censoring ignored:** recent draft classes that hadn't had time to
   reach 85 starts were labeled busts.
5. **The explorer's 5-class outcome probabilities** came from a ~40-parameter
   multinomial fit on ~90 QBs with no out-of-sample check.

## What v2 does

| Component | v1 | v2 |
|---|---|---|
| Opponent strength | national average (no-op) / sign-flipped multiplier | each team-season's actual schedule from the CFBD games table |
| Adjustment form | `EPA × multiplier` (arbitrary constants) | `EPA − β·(avg_opp_def − season mean)`, β estimated from within-QB schedule variation on 19.6k plays: **β = +0.0094 EPA/play per defense-rating point (SE 0.0018)** |
| SP+ data | fabricated 60/40 split; 6 seasons at 0% coverage | real offense/defense components, 2003–2025, 98% QB coverage; FCS schools capped at worst-FBS-defense schedule strength |
| Evaluation | in-sample fit after feature search | leave-one-draft-class-out CV, fixed features, scaling/regularization inside folds |
| Baseline | raw pick number | log(draft pick) + incremental (college **on top of** pick) test |
| Censoring | ignored | outcome models train only on classes ≤ 2019 |
| Explorer output | 5-class probabilities (worse than base rates OOS) | binary starter probability, college-only and college+pick |

Pipeline: `v2_fetch_data.py → v2_build_adjusted_stats.py → v2_evaluate.py → v2_build_projections.py`

## Honest results (leave-one-draft-class-out CV)

### Rookie-season performance: **not predictable**

| Model | CV R² | CV r |
|---|---|---|
| log(draft pick) | +0.002 | +0.08 |
| College stats (raw EPA) | −0.029 | −0.20 |
| College stats (adjusted EPA) | −0.028 | −0.12 |
| College + log(pick) | −0.020 | −0.02 |

Nothing — including the NFL's own draft market — predicts rookie LEAF
out-of-sample (n = 108). The v1 claim of 41% variance explained was overfitting.
V2 does not ship a rookie projection.

### Career outcome (starter = 85+ starts or 30+ consecutive; n = 106, classes 2007–2019)

| Model | CV AUC | CV log-loss (base 0.627) |
|---|---|---|
| log(draft pick) only | **0.828** | 0.467 |
| College stats only (adjusted) | 0.680 | 0.577 |
| College stats + log(pick) | 0.813 | 0.471 |

Three honest conclusions:

1. **College stats carry real signal** (AUC 0.68 beats chance and base-rate
   log-loss). Success rate — down-to-down consistency — is the workhorse;
   EPA/play alone scores only 0.556.
2. **The opponent adjustment helps**: adjusted beats raw in 5 of 6
   aggregation-method variants (e.g., 0.680 vs 0.663 for career-average).
3. **College stats do not beat, or add to, draft capital.** The draft market
   already prices everything in the box score and more. The model's value is
   for *pre-draft* contexts (prospects without a pick yet) or as a
   market-disagreement flag, not as a replacement for draft position.

Caveats: n = 106 with 34 positives; ~10 model variants were compared on these
folds during development (all reported above or in
`results/v2_evaluation_report.md`), so treat 0.68 as the optimistic end of a
0.62–0.68 range. The 4-class outcome model was evaluated and **dropped**: its
CV log-loss (1.28) was worse than quoting base rates (1.24).

## Known remaining limitations

- Schedule strength is season-level (games the team played), not weighted by
  the QB's actual snaps per game; the career play files don't record opponents.
- The EPA models behind the two play-by-play sources differ; a few QBs have
  implausible values (e.g., Wentz's raw career EPA is slightly negative).
- Drafted-QB-only sample: range restriction attenuates every college-stat
  effect. This can't be fixed without outcomes for undrafted players.
- `nfl_outcomes_comprehensive.csv` counts games with a pass attempt, not true
  starts, and matches names as F.Last; a spot-check found no collisions among
  QBs, but it's a fragile join.
