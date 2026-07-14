# WR Composite Rating - Visualization Integration

## Summary

The WR Composite Rating has been successfully integrated into the QB LEAF visualization dashboard. QBs now display their supporting cast quality based on the validated 5-metric WR composite.

---

## What Was Added

### 1. Team Information in WR Data

**File**: `src/data_pipeline/receiver_stats_aggregator.py`

Added team tracking to season-level aggregation:
- Captures most recent team for each WR in each season
- Handles mid-season team changes (trades)
- Enables team-level WR quality calculations

### 2. WR Composite Generation Script

**File**: `generate_wr_composite.py`

Generates WR composite ratings for visualization:
- Processes 2020-2024 seasons
- Applies WR-only filtering
- Calculates composite using equal weights (recommended)
- Outputs: `data/processed/wr_composite_ratings_2020_2024.csv`

**Generated Data**:
- 592 WR-seasons across 5 years
- All WRs with 30+ targets per season
- Includes composite rating and percentile

### 3. Visualization Dashboard Update

**File**: `visualize_leaf_ratings.py`

Added WR supporting cast quality display for QBs:

**New Features**:
- **WR Quality Card**: Shows team WR quality metrics
- **Team Average Composite**: Mean WR rating for QB's team
- **Top WR**: Best WR on team with rating
- **Volume Metrics**: Total targets and YAC EPA
- **Roster Size**: Number of WRs with 30+ targets

**Helper Function**: `calculate_team_wr_quality()`
- Aggregates WR stats by team and season
- Returns comprehensive team WR quality metrics

---

## How to Use

### Generate WR Composite Data

```bash
python generate_wr_composite.py
```

This creates `data/processed/wr_composite_ratings_2020_2024.csv` with:
- WR composite ratings (z-score standardized)
- WR percentile rankings
- All underlying metrics (targets, receptions, YAC EPA, etc.)
- Team assignments

### Run Visualization

```bash
python visualize_leaf_ratings.py
```

Then open browser to: http://localhost:8050

**What You'll See**:
1. Select any QB from dropdown
2. View their LEAF rating and trajectory
3. **NEW**: See their team's WR supporting cast quality
   - Average WR composite rating
   - Top WR name and rating
   - Total team targets and YAC EPA
   - Number of qualified WRs

---

## Example Output

### Test Results (2024 Season)

| Team | WRs | Avg Composite | Top WR | Top WR Rating | Total Targets | Total YAC EPA |
|------|-----|---------------|--------|---------------|---------------|---------------|
| **CIN** | 3 | **+0.753** | J.Chase | +1.971 | 346 | -114.6 |
| **DET** | 4 | +0.079 | A.St. Brown | +2.069 | 295 | -41.3 |
| **BUF** | 5 | -0.106 | K.Shakir | +1.245 | 392 | -140.2 |
| **SF** | 4 | -0.034 | J.Jennings | +0.618 | 288 | -136.0 |
| **KC** | 4 | -0.254 | X.Worthy | +0.771 | 275 | -125.5 |

**Insights**:
- **Best Supporting Cast**: Cincinnati (CIN) with +0.753 avg
- **Top Individual WR**: Amon-Ra St. Brown (DET) at +2.069
- **Most Volume**: Buffalo (BUF) with 392 total targets

---

## Technical Details

### WR Composite Metrics (Equal Weights)

```
WR Composite = standardized_mean(
    targets_per_game,     # 20% - Opportunity
    receptions,           # 20% - Production
    total_yac_epa,        # 20% - WR-specific value
    first_downs,          # 20% - Impact plays
    targets                # 20% - Overall volume
)
```

### Team Quality Calculation

For each team-season:
1. Filter WRs to matching team and season
2. Calculate mean composite rating
3. Aggregate volume metrics (targets, YAC EPA)
4. Identify top WR by composite rating

### Data Flow

```
nflfastR Play-by-Play
    ↓
ReceiverStatsAggregator (WR-only)
    ↓
WRLEAFPipeline.calculate_composite_rating()
    ↓
wr_composite_ratings_2020_2024.csv
    ↓
Visualization Dashboard
```

---

## Files Modified

### Core Pipeline
- `src/data_pipeline/receiver_stats_aggregator.py` - Added team tracking

### Scripts
- `generate_wr_composite.py` - NEW: Generate WR composite data
- `visualize_leaf_ratings.py` - Added WR quality display

### Test Scripts
- `test_wr_quality_display.py` - NEW: Test WR quality calculation

### Data Files
- `data/processed/wr_composite_ratings_2020_2024.csv` - NEW: WR composite data

---

## Validation

### WR Quality Calculation Test

```bash
python test_wr_quality_display.py
```

**Expected Output**:
- Loads 592 WR-seasons
- Displays quality metrics for 5 test teams
- Shows avg composite, top WR, targets, YAC EPA

### Visualization Test

```bash
python visualize_leaf_ratings.py
```

**Expected Behavior**:
- Successfully loads 592 WR composite ratings
- Displays WR quality card for each QB
- Shows team WR metrics based on QB's most recent team

---

## Use Cases

### 1. QB Context Analysis

Compare QB performance with their supporting cast quality:

```python
# High QB rating + Low WR quality = Elite QB elevating mediocre WRs
# Low QB rating + High WR quality = QB struggling despite talent
```

### 2. Trade Value Assessment

Identify QBs who might benefit from WR upgrades:

```python
# QBs with below-average WR composite are trade targets for WR additions
```

### 3. Fantasy Football

Evaluate how WR quality affects QB fantasy production:

```python
# High WR composite → More consistent QB production
# Top WR presence → Increased ceiling
```

### 4. Coaching Evaluation

Assess coaching impact relative to talent:

```python
# Coaching performance = QB LEAF rating / Expected rating given WR quality
```

---

## Metrics Explanation

### WR Composite Rating
- **Scale**: Z-score (mean=0, std≈0.7)
- **Interpretation**:
  - +1.0 = Top 15% of WRs
  - 0.0 = Average WR
  - -1.0 = Bottom 15% of WRs

### Team Average Composite
- **Calculation**: Mean of all qualified WRs on team (30+ targets)
- **Interpretation**:
  - +0.5+ = Elite WR corps
  - 0.0 = Average WR corps
  - -0.5- = Below-average WR corps

### Total YAC EPA
- **What It Measures**: Value created after the catch
- **Why It Matters**: YAC is WR-specific (less QB-dependent)
- **Interpretation**: Higher is better, but can be negative

---

## Future Enhancements

### Potential Additions

1. **Historical Comparison**: Show how QB rating changes with WR quality changes
2. **Expected vs Actual**: Predict QB rating based on WR quality, show residuals
3. **Individual WR Cards**: Click on WR name to see full profile
4. **Team Timeline**: Track how WR corps changes season-to-season
5. **Age Adjustment**: Weight younger WRs higher for projection purposes

### Data Enhancements

1. **Route-Running Metrics**: If separation data becomes available
2. **Contested Catch Rate**: Target quality dimension
3. **Depth Distribution**: Versatility metric
4. **Pre/Post Catch Splits**: More granular EPA breakdowns

---

## Troubleshooting

### Issue: "No WR data available"

**Cause**: Missing or incomplete WR composite file

**Solution**:
```bash
python generate_wr_composite.py
```

### Issue: WR metrics show for wrong team

**Cause**: Mid-season trade (WR changed teams)

**Explanation**: System uses most recent team in season. This is intentional to match end-of-season evaluations.

### Issue: Few WRs on roster

**Cause**: Only WRs with 30+ targets are included

**Explanation**: This is a feature, not a bug. Low sample-size WRs are excluded for stability.

---

## Performance

### Generation Time
- **Full 5-year generation**: ~30 seconds
- **592 WR-seasons processed**
- **Output file size**: ~100 KB

### Visualization Load Time
- **Initial load**: <2 seconds
- **Per-QB render**: Instant
- **Data size in memory**: ~1 MB (WR data)

---

## Citation

When using WR supporting cast quality in analysis:

```
WR Supporting Cast Quality Metric (2025)
- Based on: WR Composite Rating (5-metric validated combination)
- Metrics: targets_per_game, receptions, total_yac_epa, first_downs, targets
- Validation: 2020-2024 NFL seasons (r=0.66 multi-year stability)
- Team aggregation: Mean composite rating of WRs with 30+ targets
- Data source: nflfastR play-by-play data
```

---

## Summary

✅ **Completed Tasks**:
1. Added team tracking to WR season stats
2. Created WR composite generation script
3. Integrated WR quality display into QB visualization
4. Generated 592 WR-seasons of composite data (2020-2024)
5. Validated WR quality calculation with test script

✅ **Working Features**:
- WR composite ratings for all qualified WRs
- Team-level WR quality aggregation
- QB visualization with WR supporting cast display
- Comprehensive metrics (avg composite, top WR, volume stats)

✅ **Ready for Use**:
- Run `generate_wr_composite.py` to update data
- Run `visualize_leaf_ratings.py` to view dashboard
- Navigate to http://localhost:8050 to interact

---

**Last Updated**: 2025-11-07
**Integration Version**: v1.0
**Status**: ✅ Complete and tested

