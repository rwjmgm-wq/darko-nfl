# WR Composite Rating - Implementation Guide

## Overview

The WR Composite Rating has been successfully integrated into the WR LEAF pipeline. This rating system uses 5 metrics validated across 2020-2024 to capture WR-specific performance independent of QB play.

---

## Quick Start

```python
from data_pipeline.nfl_data_fetcher import NFLDataFetcher
from data_pipeline.receiver_stats_aggregator import ReceiverStatsAggregator
from features.wr_leaf_pipeline import WRLEAFPipeline

# Fetch data
fetcher = NFLDataFetcher()
pbp = fetcher.fetch_pbp_data([2024])
rosters = fetcher.fetch_rosters([2024])

# Aggregate WR stats (WR-only filtering)
aggregator = ReceiverStatsAggregator(min_targets=1, filter_wr_only=True)
game_stats = aggregator.aggregate_game_stats(pbp, roster_data=rosters)
season_stats = aggregator.aggregate_season_stats(game_stats, min_targets=30)

# Calculate composite rating
pipeline = WRLEAFPipeline()
season_with_ratings = pipeline.calculate_composite_rating(
    season_stats,
    use_optimized_weights=False  # Use equal weighting (recommended)
)

# Access ratings
top_wrs = season_with_ratings.sort_values('wr_composite_rating', ascending=False)
print(top_wrs[['receiver_player_name', 'wr_composite_rating',
              'wr_composite_percentile']].head(20))
```

---

## What the Composite Measures

The composite rating captures **WR-specific performance** through 5 metrics:

1. **targets_per_game** (20%): Opportunity/usage
2. **receptions** (20%): Production
3. **total_yac_epa** (20%): After-catch value creation (WR skill)
4. **first_downs** (20%): Impact plays
5. **targets** (20%): Overall volume

### Why These Metrics?

- **YAC-focused**: total_yac_epa minimizes QB dependency (vs air_epa)
- **Volume-based**: Captures usage patterns and opportunity
- **Impact-oriented**: first_downs measures game-changing plays
- **Stable**: Multi-year correlation r = 0.66 (highest stability)

---

## Equal vs Optimized Weights

The method supports two weighting schemes:

### Equal Weighting (Recommended)
```python
pipeline.calculate_composite_rating(season_stats, use_optimized_weights=False)
```

**Weights**: 20% each metric
**Pros**: Simplest, most stable, interpretable
**Correlation with optimized**: r = 0.989

### Optimized Weighting
```python
pipeline.calculate_composite_rating(season_stats, use_optimized_weights=True)
```

**Weights** (Ridge alpha=10.0):
- 27.8% × total_yac_epa (dominant)
- 20.2% × targets
- 19.1% × first_downs
- 18.1% × receptions
- 14.9% × targets_per_game

**Pros**: Slightly higher predictive power
**Cons**: More complex, marginal improvement

**Recommendation**: Use equal weighting unless you need maximum predictive accuracy for a specific year pair.

---

## Output Columns

The method adds two columns to your season_stats DataFrame:

1. **wr_composite_rating**: Z-score standardized composite (mean=0, std~0.7)
2. **wr_composite_percentile**: Percentile rank (0-100)

---

## Integration with Existing Pipeline

### Option 1: Season-Level Composite (Recommended)

Use the composite for season-level WR evaluations:

```python
# Aggregate to season level
season_stats = aggregator.aggregate_season_stats(game_stats, min_targets=30)

# Add composite rating
season_with_composite = pipeline.calculate_composite_rating(season_stats)

# Use for rankings, comparisons, etc.
top_wrs = season_with_composite.sort_values('wr_composite_rating', ascending=False)
```

### Option 2: Combine with LEAF Ratings

You can use both the traditional LEAF rating (air_epa focus) and the composite:

```python
# Traditional LEAF (game-by-game, air EPA focus)
processed = pipeline.process_full_pipeline(game_stats, pbp, secondary_ratings)
current_leaf = pipeline.get_current_ratings(processed, min_targets=30)

# Composite (season-level, YAC + volume focus)
season_with_composite = pipeline.calculate_composite_rating(season_stats)

# Merge for comparison
combined = current_leaf.merge(
    season_with_composite[['receiver_player_id', 'wr_composite_rating']],
    on='receiver_player_id'
)
```

**Use cases**:
- **LEAF rating**: Game-by-game tracking, coverage ability (air EPA)
- **Composite rating**: Season summary, overall WR quality (YAC + volume)

---

## Validation Results

### Multi-Year Performance (2020-2024)

| Year Pair | Correlation | Sample Size |
|-----------|-------------|-------------|
| 2020→2021 | 0.5612 | 83 WRs |
| 2021→2022 | 0.6232 | 83 WRs |
| 2022→2023 | 0.7515 | 85 WRs |
| 2023→2024 | 0.7096 | 86 WRs |
| **Average** | **0.6614** | **84 WRs** |

**Consistency**: 88.8% (std = 0.074)

### Comparison to Single Metrics

- **Best single metric** (racr): r = 0.597
- **Composite improvement**: +20.4% predictive power
- **Better than air_epa**: Air EPA r = 0.559 (more QB-dependent)

---

## Use Cases

### 1. WR Rankings & Evaluations

```python
# Season-end WR rankings
ranked = season_with_composite.sort_values('wr_composite_rating', ascending=False)
print(ranked[['receiver_player_name', 'wr_composite_rating',
              'wr_composite_percentile', 'targets', 'total_yac_epa']])
```

### 2. QB Supporting Cast Quality

```python
# Calculate team WR quality for QB context
team_wr_quality = season_with_composite.groupby('team').agg({
    'wr_composite_rating': 'mean',
    'targets': 'sum',
    'total_yac_epa': 'sum'
}).rename(columns={'wr_composite_rating': 'team_wr_composite'})

# Merge with QB stats for context adjustments
qb_with_wr_quality = qb_stats.merge(team_wr_quality, on='team')
```

### 3. Year-over-Year Projection

```python
# Use 2024 composite to project 2025 performance
# Multi-year correlation r = 0.66 suggests ~44% variance explained
predictions_2025 = season_with_composite[['receiver_player_id',
                                           'receiver_player_name',
                                           'wr_composite_rating']]
predictions_2025['projected_2025_composite'] = (
    0.66 * predictions_2025['wr_composite_rating']  # Regression to mean
)
```

### 4. Trade Value / Contract Analysis

```python
# High composite = sustainable production (less QB-dependent)
undervalued_wrs = season_with_composite[
    (season_with_composite['wr_composite_rating'] > 0.5) &  # Above average
    (season_with_composite['age'] <= 25)  # Young
]
```

---

## Comparison: Traditional LEAF vs Composite

| Aspect | LEAF Rating | Composite Rating |
|--------|-------------|------------------|
| **Focus** | Air EPA (coverage beating) | YAC EPA + Volume |
| **Level** | Game-by-game | Season aggregate |
| **QB Dependency** | Moderate (air yards) | Low (YAC focus) |
| **Stability** | High (Kalman filtered) | High (multi-metric) |
| **Use Case** | Tracking weekly changes | Season summary/rankings |
| **Adjustments** | Context + Opponent | None (raw production) |

**Recommendation**: Use both! LEAF for temporal tracking, Composite for overall quality assessment.

---

## Technical Details

### Standardization Method

Each metric is z-score standardized:
```
z = (value - mean) / std
```

This ensures:
- All metrics on same scale
- Mean = 0, Std ≈ 1
- Fair weighting regardless of metric units

### Weighting Validation

Weights were optimized using:
- **Ridge regression** (L2 regularization)
- **Alpha = 10.0** (high regularization for stability)
- **Target**: 2024 equal-weighted composite
- **Validation**: 4 year pairs (2020-2024)

### Missing Data Handling

- Requires all 5 metrics present in season_stats
- If any metric missing, raises `ValueError`
- Minimum 30 targets recommended for stable estimates

---

## API Reference

### `WRLEAFPipeline.calculate_composite_rating()`

```python
def calculate_composite_rating(
    self,
    season_stats: pd.DataFrame,
    use_optimized_weights: bool = False
) -> pd.DataFrame
```

**Parameters**:
- `season_stats` (pd.DataFrame): Season-aggregated statistics with required metrics
- `use_optimized_weights` (bool): If True, use Ridge-optimized weights. Default: False (equal weighting)

**Returns**:
- pd.DataFrame: Input DataFrame with added columns:
  - `wr_composite_rating`: Z-score composite rating
  - `wr_composite_percentile`: Percentile rank (0-100)

**Raises**:
- `ValueError`: If required metrics are missing from season_stats

**Required Metrics**:
- `targets_per_game`
- `receptions`
- `total_yac_epa`
- `first_downs`
- `targets`

---

## Performance Benchmarks

### 2024 Results (Test Run)

- **WRs analyzed**: 119 (30+ targets)
- **Computation time**: <1 second
- **Top WR**: Amon-Ra St. Brown (rating: 2.23)
- **Rating range**: [-1.19, 2.23]
- **Mean**: 0.00 (by design)
- **Std**: 0.75

### Equal vs Optimized Correlation

- **r = 0.989**: Nearly identical rankings
- **Biggest differences**: ±0.34 rating points (rare)
- **Top 20 overlap**: 18/20 WRs identical

---

## Future Enhancements

### Potential Additions

1. **Age adjustment**: Weight younger WRs higher for projection purposes
2. **Route-running metrics**: If available (e.g., separation data)
3. **Contested catch rate**: Target quality dimension
4. **Depth of target distribution**: Versatility metric

### Not Recommended

- **Air-based metrics**: Increases QB dependency
- **TD rate**: High variance, luck-dependent
- **Single-season optimization**: Overfitting risk

---

## FAQs

### Q: Why not use air_epa like the original analysis?

**A**: Air EPA is highly QB-dependent. The WR-only analysis revealed that YAC metrics (racr, total_yac_epa) are more predictive for WRs specifically because they capture WR skill after the catch.

### Q: Should I use equal or optimized weights?

**A**: Equal weights (recommended). They're simpler, more stable across years (r = 0.66 avg), and nearly identical to optimized (r = 0.989 correlation).

### Q: Can I add more metrics?

**A**: Not recommended. The 5-metric combination was exhaustively tested (3,003 combinations). Adding metrics risks overfitting and reduced stability.

### Q: What about TEs and RBs?

**A**: This composite is WR-specific. TEs/RBs have different roles (blocking, pass protection) and should use different metrics. Always use `filter_wr_only=True`.

### Q: How does this compare to PFF grades?

**A**:
- **PFF**: Subjective film grades
- **Composite**: Objective statistical performance
- **Use together**: PFF for technique, Composite for production

### Q: Why is correlation only r = 0.66?

**A**: WRs are inherently volatile year-over-year due to:
- QB changes (most impactful)
- Scheme changes
- Injuries
- Target share changes
- r = 0.66 (~44% variance explained) is excellent for WRs

---

## Troubleshooting

### Error: "Missing required metrics"

**Solution**: Ensure you're using season-level aggregated stats with all 5 metrics:
```python
# Correct
season_stats = aggregator.aggregate_season_stats(game_stats, min_targets=30)
season_with_composite = pipeline.calculate_composite_rating(season_stats)

# Incorrect - game-level stats won't have all metrics
game_stats = aggregator.aggregate_game_stats(pbp)
# Don't pass game_stats to calculate_composite_rating!
```

### Issue: Ratings seem off / unexpected WRs at top

**Check**:
1. **Position filtering**: Are you using `filter_wr_only=True`?
2. **Minimum targets**: Use at least 30 targets for stable estimates
3. **Data quality**: Check for duplicates, missing data

### Performance: Slow calculation

- Composite calculation is O(n) and should be instant
- If slow, check DataFrame size (should be ~100-200 WRs per season)
- Consider filtering to min_targets >= 30 earlier in pipeline

---

## Citation

If using this composite in research or production:

```
WR Composite Rating (2025)
- Metrics: targets_per_game, receptions, total_yac_epa, first_downs, targets
- Validation: 2020-2024 NFL seasons (r = 0.66 avg)
- Method: Exhaustive 5-metric search (3,003 combinations tested)
- Data: nflfastR play-by-play, WR-only filtered
```

---

## Summary

The WR Composite Rating provides a **stable, WR-specific performance metric** that:

✓ Focuses on WR skills (YAC, volume, impact)
✓ Minimizes QB dependency
✓ Validated across 5 seasons
✓ Simple equal weighting (interpretable)
✓ Seamlessly integrated into WR LEAF pipeline

**Use it for**: WR rankings, QB supporting cast evaluation, projection baselines, and trade analysis.

---

**Last Updated**: 2025-11-07
**Pipeline Version**: WR LEAF v1.0
**Test Coverage**: ✓ Passing
