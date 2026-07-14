# LEAF v2 Data Quality Bug Investigation

**Date:** November 2024
**Issue:** `leaf_rating` column shows constant values per player instead of game-by-game progression

## Executive Summary

The LEAF v2 QB rating system has a data quality bug where the `leaf_rating` column is incorrectly set to each player's **final career rating** across all their games, rather than tracking game-by-game performance. This affects 98.2% of QBs (378 out of 385).

## Impact

- **Analysis scripts** using `leaf_rating` produce incorrect results
- **Sam Darnold case study** initially showed all games at -0.007, which is his current career average
- **Elite QB thresholds** were incorrectly calculated as too high

## Root Cause

### Bug Location

**File:** `src/features/integrated_leaf_pipeline.py`
**Lines:** 204-212 (Stage 3: Kalman filtering)

### The Problem

Stage 3 calls `kalman_filter.process_all_players()` which returns:
- **ONE row per player** with only their final Kalman estimate
- Uses `estimates[-1]` instead of full time series

When this gets merged back into game-level data (line 220-224), it broadcasts the final value to ALL games for each player.

### Original Buggy Code

```python
# Stage 3: Kalman filtering
kalman_stats = self.kalman_filter.process_all_players(
    game_stats=result,
    metrics=kalman_metrics,
    prior_means=kalman_prior_means
)

result = result.merge(
    kalman_stats[['passer_player_id', 'opp_adj_base_epa_kalman']],
    on='passer_player_id',  # ← BUG: merges on player_id only, not game_id
    how='left'
)

result['leaf_rating'] = result['opp_adj_base_epa_kalman']  # ← Constant per player
```

## Evidence

### Investigation Results

Running `investigate_leaf_rating.py` showed:

```
QBs with constant leaf_rating (like Darnold): 378/385
QBs with varying leaf_rating: 7/385

Diagnosis: 98.2% of QBs have constant leaf_rating.
```

### Examples

| QB | leaf_rating (constant) | Final opp_adj_base_epa_kalman | Match? |
|----|----------------------|------------------------------|--------|
| Tom Brady | -0.002341 | -0.002341 | ✓ |
| Patrick Mahomes | 0.085405 | 0.085405 | ✓ |
| Sam Darnold | -0.007058 | -0.007058 | ✓ |

The `leaf_rating` equals each QB's **most recent** `opp_adj_base_epa_kalman` value.

## The Fix

### Solution Applied

Modified `src/features/integrated_leaf_pipeline.py`:

**Stage 3 (lines 204-212):** Removed buggy player-level merge
```python
# Stage 3: Kalman filtering (game-by-game in Stage 5)
if self.use_kalman:
    logger.info("\n[Stage 3/4] Kalman filtering enabled (applied in Stage 5)")
    # NOTE: Game-by-game Kalman filtering is done in Stage 5
    # Stage 3 is skipped to avoid buggy player-level merge
    pass
```

**Stage 5 (after line 250):** Set `leaf_rating` correctly
```python
# Set leaf_rating to the correct game-by-game Kalman filtered values
if self.use_kalman:
    game_by_game['leaf_rating'] = game_by_game['opp_adj_base_epa_kalman']
else:
    game_by_game['leaf_rating'] = game_by_game['opp_adj_base_epa']
```

### Why This Works

`process_game_by_game()` in Stage 5 correctly creates game-by-game `opp_adj_base_epa_kalman` values by:
1. Processing each player individually
2. Applying Kalman filter to get full time series (`estimates` array)
3. Assigning all estimates to player_games: `player_games[f'{metric}_kalman'] = estimates`
4. Concatenating all players together

## Workaround (Until Regeneration)

### For Analysis Scripts

Use `opp_adj_base_epa_kalman` instead of `leaf_rating`:

```python
# OLD (BUGGY)
qb_data = qb_data.rename(columns={
    'leaf_rating': 'raw_composite'
})

# NEW (CORRECT)
qb_data = qb_data.rename(columns={
    'opp_adj_base_epa_kalman': 'raw_composite'
})
```

### Files Updated

- `scripts/analysis/analyze_qb_early_predictors.py`
- `scripts/analysis/analyze_qb_early_predictors_extended.py`

## Impact on Analysis Results

### Before Fix (Using Buggy `leaf_rating`)

- Elite threshold at game 8: **+0.085**
- NO Elite QBs below 0.00 at game 8
- Sam Darnold: constant -0.007 across all games

### After Fix (Using Correct `opp_adj_base_epa_kalman`)

- Elite threshold at game 8: **-0.018**
- **3 Elite QBs** started below 0.00 at game 8:
  - Aaron Rodgers: -0.008 → +0.124 (game 16)
  - Andrew Luck: -0.018 → +0.048
  - Jameis Winston: -0.009 → +0.052
- Sam Darnold: game 8 rating was -0.054 (below Elite threshold)

## Sam Darnold Case Study

### Question
"What about Sam Darnold? He has been playing really well but started off quite poorly."

### Answer with Correct Data

**Early Career:**
- Game 8: -0.054 (below Elite threshold -0.018)
- Game 16: -0.102
- Game 32: -0.155
- Never achieved Elite trajectory

**2024 Season Paradox:**
- Raw EPA/play: **+0.063** (positive - explains "playing well" perception)
- Kalman rating: **-0.190** (still below average)

**Why the discrepancy?**
- Kalman filter weights entire career history
- Poor early career (-0.063 avg first 32 games) drags down current rating
- Raw EPA shows improvement, but overall trajectory remains below-average

**Conclusion:** Darnold is playing better in 2024 (positive raw EPA), but was never on an Elite trajectory and the Kalman filter correctly reflects his career-long struggles.

## Required Actions

### To Fully Fix

1. ✅ **Fixed:** Updated `integrated_leaf_pipeline.py`
2. ✅ **Fixed:** Updated analysis scripts to use correct column
3. ⏳ **Required:** Regenerate `leaf_v2_game_by_game_*.csv` with fixed pipeline
4. ⏳ **Required:** Re-run all analyses with regenerated data

### To Regenerate Data

```bash
cd DARKO_NFL
python scripts/legacy/deploy_leaf_v2.py --seasons 2016 2017 2018 2019 2020 2021 2022 2023 2024
```

This will create a new `leaf_v2_game_by_game_YYYYMMDD.csv` with correct game-by-game `leaf_rating` values.

## Testing the Fix

After regenerating data, verify:

```python
import pandas as pd
df = pd.read_csv('data/production/leaf_v2_game_by_game_YYYYMMDD.csv')

# Check that leaf_rating varies game-by-game
qb_variability = df.groupby('passer_player_name').agg({
    'leaf_rating': 'nunique',
    'game_number': 'max'
}).reset_index()

# Should NOT be 1 for most QBs
constant_rating = qb_variability[qb_variability['leaf_rating'] == 1]
print(f"QBs with constant leaf_rating: {len(constant_rating)}/{len(qb_variability)}")

# Should be close to 0% (only QBs with 1 game should be constant)
```

Expected result: Only QBs with 1 game should have `nunique == 1`.

## Lessons Learned

1. **Always check data distributions** - 98% constant values is a red flag
2. **Merge operations are risky** - Always merge on full composite keys (player_id + game_id)
3. **Test edge cases** - Sam Darnold's "improvement" paradox revealed the bug
4. **Validate pipeline outputs** - Each stage should have data quality checks
5. **Document data schema** - Clear column definitions prevent misuse

## References

- Bug discovered: November 2024
- Investigation script: `investigate_leaf_rating.py`
- Test case: `check_darnold.py`
- Fixed files:
  - `src/features/integrated_leaf_pipeline.py`
  - `scripts/analysis/analyze_qb_early_predictors.py`
  - `scripts/analysis/analyze_qb_early_predictors_extended.py`
