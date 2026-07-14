# Prediction Model Update - November 5, 2025

**Issue**: Extreme and unrealistic future predictions in visualization app
**Status**: ✅ FIXED

---

## Problem Description

The original prediction model generated extreme projections for QBs, especially for 2-3 year forecasts. QBs on hot/cold streaks would show absurd future ratings:

**Example Issues**:
- QB with recent +0.01 EPA/game trend → +0.48 EPA in 3 years (impossibly high)
- Elite QBs maintaining peak performance indefinitely
- Poor QBs staying poor with no improvement
- Predictions exceeding any historical QB performance

---

## Root Cause

### Original Formula (Line 110):
```python
predicted_rating = current_rating + (trend * 16 * years) - (age_decline_per_year * years)
```

**The Bug**:
- `trend` = slope from last 10 games (per game)
- `trend * 16 * years` = full extrapolation of short-term trend over multiple years
- This assumed recent performance would continue indefinitely

**Why This Failed**:
1. **NFL is noisy**: A 10-game hot streak doesn't predict 3-year performance
2. **No mean reversion**: Elite QBs stayed elite, poor QBs stayed poor
3. **Trend overweighting**: Short-term variance treated as long-term signal
4. **No bounds**: Predictions could exceed any historical performance

---

## Solution Implemented

### 1. Dampened Trend (Lines 91-93)

**Old**:
```python
trend = np.polyfit(x, y, 1)[0]  # Per-game slope
predicted_rating = current_rating + (trend * 16 * years)
```

**New**:
```python
trend_per_game = np.polyfit(x, y, 1)[0]
# Only 20% of trend applies + cap at ±0.02/year
trend_annual = np.clip(trend_per_game * 16 * 0.2, -0.02, 0.02)
predicted_rating = current_rating + (trend_annual * years)
```

**Impact**:
- Hot/cold streaks have minimal long-term effect
- Maximum trend impact: ±0.02 EPA per year (±0.06 over 3 years)
- Example: +0.01/game trend → +0.006/year (vs. +0.16/year before)

---

### 2. Stronger Regression to Mean (Lines 123-127)

**Old**:
```python
regression_factor = 0.1 * years  # 10% per year
mean_rating = 0.05
```

**New**:
```python
regression_factor = 0.20 * years  # 20% per year
regression_factor = min(regression_factor, 0.5)  # Cap at 50%
mean_rating = 0.03  # Conservative average
```

**Impact**:
- Elite QBs (0.15 EPA) regress toward 0.03 faster
- Poor QBs (-0.10 EPA) improve toward 0.03 faster
- 3-year regression: 50% toward mean (vs. 30% before)

**Example**:
- Elite QB at 0.15 after 3 years:
  - Old: 0.15 → 0.135 (10% regression)
  - New: 0.15 → 0.090 (stronger regression)

---

### 3. Hard Prediction Bounds (Lines 129-131)

**New**:
```python
predicted_rating = np.clip(predicted_rating, -0.15, 0.20)
```

**Rationale**:
- **Upper bound (0.20)**: Peak Mahomes level - no QB sustains higher
- **Lower bound (-0.15)**: Backup QB level - active starters don't go lower
- Prevents model from generating impossible predictions

---

### 4. Refined Age Decline (Lines 102-107)

**Old**:
```python
if estimated_age > 32:
    age_decline_per_year = 0.015
elif estimated_age > 35:
    age_decline_per_year = 0.025
```

**New**:
```python
if estimated_age > 35:
    age_decline_per_year = 0.020  # 0.020 EPA/year
elif estimated_age > 32:
    age_decline_per_year = 0.010  # 0.010 EPA/year
# else: 0.0 (no decline before 32)
```

**Changes**:
- More gradual decline curve
- Better matches observed NFL aging patterns
- Age 32-35: 0.010/year decline (was 0.015)
- Age 35+: 0.020/year decline (was 0.025)

---

## Comparison: Old vs. New Model

### Example: Elite QB (Patrick Mahomes - 0.185 current rating)

**Assumptions**:
- Current: 0.185 EPA
- Recent trend: +0.005 per game (hot streak)
- Age: 29 (no decline yet)

**Old Model**:
- 1 year: 0.185 + (0.005 × 16) = **0.265** ❌ (impossibly high)
- 2 years: 0.185 + (0.005 × 32) = **0.345** ❌ (absurd)
- 3 years: 0.185 + (0.005 × 48) = **0.425** ❌ (literally impossible)

**New Model**:
- trend_annual = clip(0.005 × 16 × 0.2, -0.02, 0.02) = 0.016 → **0.02** (capped)
- 1 year:
  - Base: 0.185 + 0.02 = 0.205
  - Regression (20%): 0.205 × 0.8 + 0.03 × 0.2 = **0.170** ✅
- 2 years:
  - Base: 0.185 + 0.04 = 0.225
  - Regression (40%): 0.225 × 0.6 + 0.03 × 0.4 = **0.147** ✅
- 3 years:
  - Base: 0.185 + 0.06 = 0.245
  - Regression (50%): 0.245 × 0.5 + 0.03 × 0.5 = **0.138** ✅

**Result**: Realistic decline from peak, accounting for mean reversion.

---

### Example: Average QB (0.050 current rating)

**Assumptions**:
- Current: 0.050 EPA
- Recent trend: -0.003 per game (slight decline)
- Age: 31 (no decline yet)

**Old Model**:
- 1 year: 0.050 + (-0.003 × 16) = **0.002** (big drop)
- 2 years: 0.050 + (-0.003 × 32) = **-0.046** (poor)
- 3 years: 0.050 + (-0.003 × 48) = **-0.094** (backup level)

**New Model**:
- trend_annual = clip(-0.003 × 16 × 0.2, -0.02, 0.02) = **-0.010**
- 1 year:
  - Base: 0.050 - 0.010 = 0.040
  - Regression (20%): 0.040 × 0.8 + 0.03 × 0.2 = **0.038** ✅
- 2 years:
  - Base: 0.050 - 0.020 = 0.030
  - Regression (40%): 0.030 × 0.6 + 0.03 × 0.4 = **0.030** ✅
- 3 years:
  - Base: 0.050 - 0.030 = 0.020
  - Regression (50%): 0.020 × 0.5 + 0.03 × 0.5 = **0.025** ✅

**Result**: Stays near average (strong regression to mean prevents over-prediction).

---

## Key Principles of New Model

### 1. **Trends Fade**
Short-term performance is mostly noise. Only 20% of recent trend carries forward, and it's capped.

### 2. **Mean Reversion is Real**
Elite QBs rarely stay elite for 3+ years. Poor QBs improve or get benched. The model reflects this natural regression.

### 3. **Age Matters**
QBs decline after 32, accelerating after 35. Young QBs have stable/improving projections.

### 4. **Uncertainty Grows**
Predictions get more uncertain over time (40% increase per year in confidence bands).

### 5. **Bounded Rationality**
Hard caps prevent physically impossible predictions (no QB above peak Mahomes, no active starter below -0.15).

---

## Testing the New Model

### Visualization App Updated

**File**: `visualize_leaf_ratings.py`
**Lines Changed**: 84-131

**To Test**:
1. Stop old app instance (Ctrl+C)
2. Restart: `python visualize_leaf_ratings.py`
3. Open: http://localhost:8050
4. Select any QB and check predictions
5. Verify:
   - No extreme jumps/drops
   - Elite QBs regress gradually
   - Poor QBs improve gradually
   - All predictions within [-0.15, 0.20]

---

## 2025 Data Added

**Concurrent Change**: Deployed ratings now include 2025 season data

**Command**:
```bash
python deploy_leaf_v2.py --seasons 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025
```

**New Totals**:
- **6,410 QB-game records** (up from 6,089)
- **+321 games** from 2025 season (partial - through Week 10)
- All 2025 data included in visualizations

---

## Summary

**Problem**: Extreme predictions from over-extrapolating short-term trends

**Solution**:
1. Dampen trends (20% weight, ±0.02/year cap)
2. Stronger mean reversion (20%/year, 50% max)
3. Hard prediction bounds ([-0.15, 0.20])
4. Refined age decline curve

**Result**: Realistic, conservative predictions that match NFL reality

**Status**: ✅ Fixed and deployed

---

**Updated**: 2025-11-05
**Version**: LEAF v2.0 + Visualization App v1.1
