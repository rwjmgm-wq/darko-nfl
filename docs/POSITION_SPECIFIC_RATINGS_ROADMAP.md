# Position-Specific Rating Systems - Roadmap

**Date**: 2025-11-05
**Status**: Planning

---

## Overview

After validating that **teammate adjustments don't work** (all 4 methods failed due to circular dependencies), we're building **separate rating systems** for each position using **position-specific metrics** that avoid circularity.

### Key Principle

**Use direct, position-specific metrics that don't depend on other positions**

- WR: Separation, YAC, target quality
- RB: Yards before contact, evasion
- OL: Pressure rate, run blocking (requires advanced data)
- Defense: Position-specific pressure/coverage metrics

---

## Position Rating Systems

### ✅ QB LEAF v2.0 (COMPLETE)

**Status**: Production-ready, deployed

**Metrics**:
- EPA per play (context-adjusted)
- CPOE (Completion % Over Expected)
- Success rate
- Opponent-adjusted
- Kalman-filtered

**Performance**: 0.8951 correlation (170% improvement over raw EPA)

**Output**: [data/production/leaf_v2_current_ratings_20251105.csv](../data/production/leaf_v2_current_ratings_20251105.csv)

---

## Phase 1: WR/TE Rating System

### Objective

Create a LEAF-style rating for receivers using **receiving-specific metrics** that don't depend on QB quality.

### Available Metrics (nflfastR)

From [docs/POSITION_METRICS.md](POSITION_METRICS.md):

**Target-based**:
- `epa` - EPA on targets (36,298 targets - EXCELLENT sample)
- `air_epa` - EPA from throw (independent of YAC)
- `yac_epa` - EPA from yards after catch
- `xyac_epa` - Expected YAC EPA
- `air_yards` - Depth of target
- `yards_after_catch` - YAC gained

**Quality metrics**:
- `complete_pass` - Catch rate
- `touchdown` - TD rate
- `interception` - INT rate (QB-dependent, exclude)
- `fumble_lost` - Fumble rate

### WR LEAF Design

**Core Metric**: Target EPA (context-adjusted)

**Pipeline** (mirrors QB LEAF):
1. **Context adjustments**: Weather, home/away, down/distance
2. **Target quality adjustments**: Depth of target (air_yards), expected difficulty
3. **Opponent adjustments**: Defense quality (secondary coverage)
4. **Kalman filtering**: Adaptive learning over time
5. **Game-by-game tracking**: Real-time ratings

**Key Difference from QB LEAF**:
- Uses `air_epa` (EPA at catch point) instead of total EPA to minimize QB influence
- Adjusts for target difficulty (depth, separation - if available)
- YAC EPA captures receiver-specific ability

### Implementation Plan

**Step 1.1**: Create WR/TE stats aggregator
- File: `src/data_pipeline/receiver_stats_aggregator.py` (already exists!)
- Metrics: air_epa, yac_epa, catch_rate, air_yards

**Step 1.2**: Build WR context adjustment model
- File: `src/features/wr_context_adjustments.py`
- Adjust for: weather, target depth, field position

**Step 1.3**: Calculate secondary defense ratings
- File: `src/features/secondary_ratings.py`
- Rate defenses by pass coverage (EPA allowed to WRs)

**Step 1.4**: WR Kalman filter
- Reuse `QBKalmanFilter` with WR-specific metrics
- Metrics: `air_epa_per_target`, `yac_epa_per_target`

**Step 1.5**: Integrated WR LEAF pipeline
- File: `src/features/wr_leaf_pipeline.py`
- Mirror structure of QB pipeline

**Step 1.6**: Validation
- File: `validate_wr_leaf.py`
- Test: Does 2022 WR rating predict 2023 receiving production?
- Target: >0.5 correlation (lower than QB due to target volume variance)

---

## Phase 2: RB Rating System

### Objective

Rate running backs using **rushing-specific metrics**.

### Available Metrics (nflfastR)

From [docs/POSITION_METRICS.md](POSITION_METRICS.md):

**Rushing** (30,813 rushes - GOOD sample):
- `epa` - EPA on rushes
- `yards_gained` - Rushing yards
- `touchdown` - TD rate
- `fumble_lost` - Fumble rate
- `first_down` - Conversion rate

**Advanced** (if available):
- Yards before contact (requires NextGen Stats)
- Evasion rate (requires tracking data)
- Gap scheme success (requires charting)

### RB LEAF Design

**Core Metric**: Rush EPA (context-adjusted)

**Pipeline**:
1. **Context adjustments**: Weather, field position, down/distance, score
2. **Run blocking adjustments**: Opponent front-7 quality
3. **Kalman filtering**: Adaptive learning
4. **Game-by-game tracking**: Real-time ratings

**Key Challenges**:
- OL quality affects RB EPA (similar circular dependency as QB ↔ WR)
- Solution: Use **opponent front-7 quality** instead of adjusting for OL
- Accept that RB ratings include OL effects (like QB includes WR effects)

### Implementation Plan

**Step 2.1**: Create RB stats aggregator
- File: `src/data_pipeline/rb_stats_aggregator.py`
- Metrics: rush_epa, yards_per_carry, success_rate

**Step 2.2**: RB context adjustments
- File: `src/features/rb_context_adjustments.py`
- Adjust for: weather, field position, game script

**Step 2.3**: Calculate front-7 defense ratings
- File: `src/features/front_seven_ratings.py`
- Rate defenses by rush defense (EPA allowed to RBs)

**Step 2.4**: RB Kalman filter
- Reuse existing filter with RB metrics

**Step 2.5**: Integrated RB LEAF pipeline
- File: `src/features/rb_leaf_pipeline.py`

**Step 2.6**: Validation
- Does 2022 RB rating predict 2023 rushing production?

---

## Phase 3: Defensive Position Ratings

### Objective

Rate defensive players using **defense-specific metrics**.

### Challenges

**nflfastR limitations**:
- Play-by-play tracks OFFENSE, not individual defenders
- No player-level defensive stats (tackles, pressures, coverage)

**Requires additional data**:
- PFF grades (proprietary)
- NextGen Stats (tracking data)
- ESPN Analytics

### Defensive LEAF Options

#### Option A: Team Defense Ratings (Already Have!)

We already calculate team defense EPA in opponent adjustments:
- File: `src/features/opponent_adjustments.py`
- Output: Defense ratings by team-season

**Use case**: Adjust opponent difficulty for offensive players

#### Option B: Position Group Ratings

Calculate ratings for defensive **units** (not individuals):
- **Secondary** (pass defense): EPA allowed on pass plays
- **Front-7** (run defense): EPA allowed on run plays
- **Pass rush**: Pressure rate, sack rate (if available)

**Implementation**:
- File: `src/features/defensive_unit_ratings.py`
- Similar to defense ratings, but split by unit

#### Option C: Individual Defensive Players (Future - Requires PFF/NGS)

If you have access to:
- PFF grades
- NextGen Stats tracking data
- Play-by-play with defender assignments

**Would enable**:
- Individual CB ratings (coverage EPA)
- Individual pass rusher ratings (pressure rate)
- Individual LB ratings (run stop rate)

**Not feasible with nflfastR alone**

### Recommendation

**Phase 3A** (Immediate): Implement defensive unit ratings
- Secondary ratings (for WR opponent adjustments)
- Front-7 ratings (for RB opponent adjustments)

**Phase 3B** (Future): Individual defensive ratings if data becomes available

---

## Implementation Priority

### High Priority (Start Now)

1. **WR/TE LEAF** (Phase 1)
   - Most data available in nflfastR
   - Clear metrics (air_epa, yac_epa)
   - High impact (top fantasy position)

2. **RB LEAF** (Phase 2)
   - Good sample size (30k+ rushes)
   - Clear metrics (rush_epa)
   - Important for team analytics

### Medium Priority

3. **Defensive Unit Ratings** (Phase 3A)
   - Needed for WR/RB opponent adjustments
   - Feasible with nflfastR
   - Team-level insights

### Low Priority (Future)

4. **OL Ratings** (Requires external data)
   - nflfastR has no OL metrics
   - Would need PFF grades or NextGen Stats

5. **Individual Defensive Player Ratings** (Requires external data)
   - Would need PFF grades, NGS tracking, or play charting

---

## Comparison with QB LEAF

### Rating Scales

**Important**: Position ratings will NOT be on the same scale!

| Position | Scale | Interpretation |
|----------|-------|----------------|
| **QB** | EPA per play | 0.15 = elite, -0.05 = below average |
| **WR** | Air EPA per target | ~0.10 = elite, -0.05 = below average |
| **RB** | Rush EPA per carry | ~0.05 = elite, -0.10 = below average |

**Why different scales?**
- QB touches ball every play → higher EPA responsibility
- WR only on targets → lower EPA totals
- RB only on rushes → EPA affected by game script

### Validation Targets

| Position | Correlation Target | Notes |
|----------|-------------------|-------|
| **QB** | 0.895 | Achieved with LEAF v2.0 |
| **WR** | 0.5-0.7 | Lower due to target volume variance |
| **RB** | 0.4-0.6 | Lower due to OL/game script effects |

**Lower targets acceptable** because:
- Less data per player (QBs have most plays)
- More situational variance
- More dependent on team factors

---

## File Structure

```
src/
├── data_pipeline/
│   ├── qb_stats_aggregator.py        (✅ Complete)
│   ├── receiver_stats_aggregator.py  (✅ Exists, needs enhancement)
│   └── rb_stats_aggregator.py        (⏳ To create)
│
├── features/
│   ├── qb_leaf_pipeline.py           (✅ integrated_leaf_pipeline.py)
│   ├── wr_leaf_pipeline.py           (⏳ To create)
│   ├── rb_leaf_pipeline.py           (⏳ To create)
│   ├── wr_context_adjustments.py     (⏳ To create)
│   ├── rb_context_adjustments.py     (⏳ To create)
│   ├── secondary_ratings.py          (⏳ To create)
│   ├── front_seven_ratings.py        (⏳ To create)
│   └── defensive_unit_ratings.py     (⏳ To create)
│
└── validation/
    ├── validate_qb_leaf.py            (✅ Multiple validation scripts)
    ├── validate_wr_leaf.py            (⏳ To create)
    └── validate_rb_leaf.py            (⏳ To create)
```

---

## Next Steps (Recommended Order)

### Step 1: WR/TE LEAF (1-2 weeks)

1. ✅ Already have `receiver_stats_aggregator.py`
2. Create `wr_context_adjustments.py`
3. Create `secondary_ratings.py` (defensive pass coverage)
4. Build `wr_leaf_pipeline.py`
5. Validate on 2023 data
6. Deploy if validation succeeds

### Step 2: RB LEAF (1 week)

1. Create `rb_stats_aggregator.py`
2. Create `rb_context_adjustments.py`
3. Create `front_seven_ratings.py` (defensive run stopping)
4. Build `rb_leaf_pipeline.py`
5. Validate and deploy

### Step 3: Defensive Units (1 week)

1. Create `defensive_unit_ratings.py`
2. Split defense ratings into:
   - Pass defense (secondary)
   - Run defense (front-7)
3. Use for WR/RB opponent adjustments

---

## Success Criteria

### MVP (Minimum Viable Product)

- ✅ QB LEAF v2.0 deployed (0.895 correlation)
- ⏳ WR LEAF deployed (>0.5 correlation target)
- ⏳ RB LEAF deployed (>0.4 correlation target)

### Complete System

- All offensive skill positions rated
- Defensive unit ratings for opponent adjustments
- Unified deployment pipeline
- Production-ready outputs

---

## Key Differences from Original Multi-Position Plan

### What Changed

**Original Plan** (Failed):
- Teammate adjustments for QB ↔ WR
- Unified EPA scale across positions
- Circular dependency resolution

**New Plan** (This document):
- **Separate** rating systems per position
- **Position-specific** metrics (no cross-position dependencies)
- **Different** scales (accept incomparability)

### Why It Will Work

1. **No circular dependencies**: Each position uses own metrics
2. **Proven architecture**: Reuse QB LEAF pipeline structure
3. **Empirical validation**: Test each position separately
4. **Practical**: Works with available data (nflfastR)

---

## Summary

**Goal**: Build position-specific LEAF ratings avoiding circular dependencies

**Approach**: Use position-specific metrics (air_epa for WR, rush_epa for RB)

**Priority**:
1. WR/TE LEAF (start now)
2. RB LEAF (next)
3. Defensive units (support)

**Timeline**: 2-4 weeks for full offensive skill position coverage

**Status**: QB LEAF deployed ✅, WR LEAF next ⏳
