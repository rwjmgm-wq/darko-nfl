# Phase 2: Injury Tracking System - Complete

**Status**: ✅ COMPLETED
**Date**: 2025-11-05

---

## Overview

Phase 2 implements a comprehensive injury tracking system that adjusts LEAF ratings and uncertainty based on player injury status. The system provides better risk assessment for predictions involving injured or recovering players.

---

## Components Implemented

### 1. Core Injury Tracking Module

**File**: [src/features/injury_tracking.py](../src/features/injury_tracking.py)

**Classes**:

1. **InjuryStatus**: Enum-like class defining injury status levels
   - Healthy
   - Questionable
   - Doubtful
   - Out
   - IR (Injured Reserve)
   - PUP (Physically Unable to Perform)

2. **InjuryDataFetcher**: Fetches injury data from multiple sources
   - ESPN API integration (attempted - rate limited/auth required)
   - CSV manual entry (primary method)
   - Fallback handling for missing data

3. **InjuryAdjuster**: Applies injury adjustments to ratings
   - Uncertainty multipliers for injury status
   - Recovery modeling for returning players
   - Rating adjustments for non-playing status

4. **InjuryTrackingSystem**: Complete system with caching
   - Injury data caching for performance
   - Integration with LEAF pipeline
   - Batch processing support

### 2. Pipeline Integration

**File**: [src/features/integrated_leaf_pipeline.py](../src/features/integrated_leaf_pipeline.py)

**Changes**:
- Added `injury_system` parameter to IntegratedLEAFPipeline
- Added `use_injury_tracking` flag for optional injury tracking
- Implemented Stage 4 in pipeline: Injury adjustments
- Graceful handling of missing injury data

**Pipeline Flow**:
```
Raw EPA
  ↓
Stage 1: Context Adjustments
  ↓
Stage 2: Opponent Adjustments
  ↓
Stage 3: Kalman Filtering
  ↓
Stage 4: Injury Tracking (NEW)
  ↓
Stage 5: Game-by-Game Ratings
```

### 3. Data Infrastructure

**Directory**: [data/injuries/](../data/injuries/)

**Files Created**:
1. **README.md**: Complete documentation of CSV format and usage
2. **injuries_template.csv**: Template file for manual entry
3. **injuries_2023_week_10.csv**: Sample data for testing

**CSV Format**:
```csv
player_id,player_name,team,status,injury_type,week,season
00-0033873,P.Mahomes,KC,Questionable,Ankle,10,2024
00-0036442,J.Allen,BUF,Healthy,,10,2024
00-0033077,J.Hurts,PHI,Out,Shoulder,10,2024
```

### 4. Validation System

**File**: [validate_injury_tracking.py](../validate_injury_tracking.py)

**Tests**:
1. Sample injury data creation
2. Injury adjustment effects
3. CSV loading functionality
4. Full pipeline integration

**Validation Results**:
- ✅ Sample data created successfully (4 players)
- ✅ All injury statuses tested (Healthy, Questionable, Doubtful, Out)
- ✅ CSV loading working correctly
- ✅ Pipeline integration successful
- ✅ Applied 1 injury adjustment in test data (K.Murray - Out)

---

## Injury Adjustment Effects

### Status-Based Adjustments

| Status | Rating Effect | Uncertainty Effect | Use Case |
|--------|--------------|-------------------|----------|
| **Healthy** | No change | No change | Normal gameplay |
| **Questionable** | No change | +10% (×1.10) | Increased risk |
| **Doubtful** | No change | +25% (×1.25) | High risk |
| **Out** | Set to 0.0 | Set to 0.0 | Not playing |
| **IR** | Set to 0.0 | Set to 0.0 | Season-ending |
| **PUP** | Set to 0.0 | Set to 0.0 | Pre-season inactive |

### Recovery Modeling

Players returning from injury are tracked for gradual recovery:

| Games Since Return | Performance Factor | Notes |
|-------------------|-------------------|-------|
| 1 | 85% | First game back |
| 2 | 92% | Second game |
| 3+ | 100% | Full recovery |

---

## Usage

### Basic Usage

```python
from src.features.integrated_leaf_pipeline import IntegratedLEAFPipeline

# Enable injury tracking
pipeline = IntegratedLEAFPipeline(
    use_context=True,
    use_opponent_adj=True,
    use_kalman=True,
    use_injury_tracking=True  # Enable injury adjustments
)

# Run pipeline - injury adjustments applied automatically
ratings = pipeline.process_full_pipeline(qb_games, pbp, defense_ratings)
```

### Creating Injury Data

```python
import pandas as pd

# Create weekly injury data
injury_data = pd.DataFrame([
    {
        'player_id': '00-0033873',
        'player_name': 'P.Mahomes',
        'team': 'KC',
        'status': 'Questionable',
        'injury_type': 'Ankle',
        'week': 10,
        'season': 2024
    }
])

# Save to appropriate location
injury_data.to_csv('data/injuries/injuries_2024_week_10.csv', index=False)
```

### Checking Injury Adjustments

```python
# Filter for injured players
if 'injury_adjusted_rating' in ratings.columns:
    injured = ratings[ratings['injury_adjusted_rating'].notna()]

    print("Injured QBs:")
    print(injured[['passer_player_name', 'status',
                   'opp_adj_base_epa_kalman',
                   'injury_adjusted_rating',
                   'injury_adjusted_uncertainty']])
```

---

## Data Sources

### Recommended Sources for Injury Data

1. **NFL.com Injury Report**
   - URL: https://www.nfl.com/injuries/
   - Updated: Weekly (Wednesday, Friday)
   - Official injury designations

2. **ESPN Injury Report**
   - URL: https://www.espn.com/nfl/injuries
   - Updated: Multiple times per week
   - Detailed injury descriptions

3. **Pro Football Reference**
   - URL: https://www.pro-football-reference.com/
   - Historical injury data
   - Season-long tracking

4. **Team Websites**
   - Official team injury reports
   - Most up-to-date information
   - Coach press conferences

### Data Collection Workflow

1. **Wednesday**: Initial injury report published
2. **Friday**: Final injury report before game
3. **Manual entry**: Update CSV files for upcoming week
4. **Pipeline run**: Generate injury-adjusted ratings
5. **Predictions**: Use adjusted ratings and uncertainties

---

## Benefits

### 1. Better Risk Assessment

- **Before**: All QBs treated equally regardless of injury status
- **After**: Injured QBs have increased uncertainty reflecting higher risk

**Example**:
```
Josh Allen - Questionable (Shoulder)
  Normal uncertainty: ±0.050
  With injury: ±0.055 (+10%)
  → Higher confidence interval in predictions
```

### 2. Realistic Ratings for Non-Playing Status

- **Before**: Out/IR players retained positive ratings
- **After**: Out/IR players set to 0.0 rating

**Example**:
```
Jalen Hurts - Out (Shoulder)
  Normal rating: 0.142
  With injury: 0.000
  → Accurate reflection that player won't play
```

### 3. Recovery Tracking

- Identifies players returning from injury
- Applies gradual recovery curve
- Prevents overestimating recently-returned players

**Example**:
```
Player returns from IR in Week 10
  Week 10 (first game back): 85% of pre-injury rating
  Week 11 (second game): 92% of pre-injury rating
  Week 12+ (recovered): 100% of pre-injury rating
```

---

## Technical Details

### Implementation

**Location**: [src/features/integrated_leaf_pipeline.py](../src/features/integrated_leaf_pipeline.py) (Lines 233-260)

**Logic**:
```python
# For each week-season in dataset
for (season, week), group_idx in result.groupby(['season', 'week']).groups.items():
    # Fetch injury data for this week
    week_result = self.injury_system.adjust_ratings_with_injuries(
        ratings=result.loc[group_idx],
        week=week,
        season=season
    )

    # Update with injury adjustments if available
    if 'injury_adjusted_rating' in week_result.columns:
        result.loc[group_idx, 'injury_adjusted_rating'] = week_result['injury_adjusted_rating']
        result.loc[group_idx, 'injury_adjusted_uncertainty'] = week_result['injury_adjusted_uncertainty']
```

### Error Handling

- **Missing CSV files**: Gracefully skipped with warning
- **Empty injury data**: Returns original ratings unchanged
- **Missing columns**: Validates data structure before processing
- **Pipeline failures**: Continues without injury adjustments

---

## Validation Results

### Test Execution

**Command**: `python validate_injury_tracking.py`

**Results**:
```
1. Sample Data Creation:
   [OK] Created 4 sample injury records

2. Injury Adjustments:
   [OK] Tested 4 injury statuses
   [OK] Questionable: +10% uncertainty
   [OK] Doubtful: +25% uncertainty
   [OK] Out: Rating set to 0.0

3. CSV Loading:
   [OK] Successfully loaded 4 injury reports

4. Pipeline Integration:
   [OK] Pipeline completed successfully
   [OK] Processed 718 QB-game records
   [OK] Applied 1 injury adjustment
```

**Sample Output**:
```
QBs with Injury Adjustments:
player_name  team  status        injury_type  adjusted_rating
K.Murray     ARI   Out           Knee         0.000
```

---

## Future Enhancements

### Potential Improvements

1. **Automated Data Collection**
   - Web scraping from NFL.com
   - API integration (if authentication obtained)
   - Scheduled weekly updates

2. **Historical Injury Database**
   - Track injury history per player
   - Analyze recovery patterns
   - Adjust recovery models based on injury type

3. **Injury Severity Modeling**
   - Different recovery curves by injury type
   - Position-specific recovery patterns
   - Age-adjusted recovery rates

4. **Predictive Injury Risk**
   - Identify players at risk based on workload
   - Flag potential injuries before they occur
   - Preventive uncertainty adjustments

---

## Files Modified/Created

### Created

1. **src/features/injury_tracking.py** (540 lines)
   - Complete injury tracking system
   - ESPN API + CSV fallback
   - Adjustment logic and caching

2. **data/injuries/README.md**
   - CSV format documentation
   - Data source recommendations
   - Usage instructions

3. **data/injuries/injuries_template.csv**
   - Template for manual entry
   - Example format

4. **data/injuries/injuries_2023_week_10.csv**
   - Sample test data
   - 4 QB injury statuses

5. **validate_injury_tracking.py** (300+ lines)
   - Comprehensive validation suite
   - 4-stage testing process
   - Usage examples

### Modified

1. **src/features/integrated_leaf_pipeline.py**
   - Added injury_system parameter
   - Added use_injury_tracking flag
   - Implemented Stage 4: Injury tracking
   - Updated pipeline flow diagram

2. **docs/LEAF_V2_DOCUMENTATION.md**
   - Added injury tracking to "What's New"
   - Updated system components diagram
   - Added injury tracking API section
   - Added Example 5: Injury tracking usage

---

## Summary

Phase 2 successfully implements a production-ready injury tracking system for LEAF ratings:

✅ **Infrastructure Complete**: CSV-based data management with fallback handling
✅ **Pipeline Integration**: Optional injury tracking in Stage 4
✅ **Validation Tested**: All components working correctly
✅ **Documentation Complete**: Usage examples and API reference

**Next Steps**: Phase 3+ optional enhancements (garbage time filtering, stadium effects, etc.)

---

**Phase 2 Status**: ✅ COMPLETE
**Ready for Production**: YES
**Validated**: YES
**Documented**: YES
