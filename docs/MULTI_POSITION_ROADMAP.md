# Multi-Position LEAF Implementation Roadmap

## Vision

Extend LEAF from QB-only to all positions, creating a unified rating system on a common EPA scale. Similar to NBA's DARKO, all players will be evaluated on their contribution to team success, adjusted for teammates, opponents, and context.

## Key Challenges

1. **Circular Dependencies**: QB ↔ WR, OL ↔ RB, etc.
2. **Position-Specific Metrics**: Each position contributes differently to EPA
3. **Sample Size Variation**: Some positions have more plays than others
4. **Data Availability**: Not all positions have direct EPA attribution in nflfastR

## Design Principles

1. **Empirical Validation**: Test every adjustment's predictive power before including
2. **Iterative Convergence**: Use simultaneous estimation to handle circularity (like defense ratings)
3. **Parsimony**: Prune insignificant interactions to keep model interpretable
4. **Position-Agnostic Infrastructure**: Build systems that work for any position

---

## Phase 1: Data Foundation & Exploration (Weeks 1-2)

**Goal**: Understand what metrics we can derive for each position from nflfastR

### Deliverables

1. **explore_position_metrics.py**
   - Comprehensive exploration of nflfastR play-by-play data
   - Extract EPA attribution for all positions
   - Calculate sample sizes and data quality metrics
   - Test what metrics are derivable vs directly available

2. **docs/POSITION_METRICS.md**
   - Documentation of available metrics by position
   - Sample size analysis (plays per game, games needed for confidence)
   - Data quality assessment
   - Recommendations for minimum thresholds

3. **Metric Inventory**:
   - **QB**: EPA, CPOE, success rate (already done)
   - **WR/TE**: EPA per target, catch rate over expected, YAC over expected, separation
   - **RB**: EPA per rush, EPA per target (receiving), pass blocking (inferred)
   - **OL**: Pass blocking (pressure rate, sack rate), run blocking (yards before contact)
   - **EDGE/DL**: Pressure rate, run stop rate, QB hits
   - **LB**: Coverage EPA, run defense, pressure
   - **DB**: Coverage EPA, completion % allowed, PBUs, interceptions

### Success Criteria
- Clear inventory of what's possible for each position
- Identified which positions have sufficient data for LEAF ratings
- Sample size requirements documented

---

## Phase 2: Infrastructure - Unified Adjustment Framework (Weeks 2-3)

**Goal**: Build position-agnostic infrastructure that handles circularity cleanly

### Deliverables

1. **config/position_interactions.yaml**
   ```yaml
   interactions:
     QB:
       affected_by: [WR, TE, OL, RB]
       affects: [WR, TE, RB]

     WR:
       affected_by: [QB, OL]
       affects: [QB]

     RB:
       affected_by: [OL, QB]
       affects: [OL, QB]
   ```

2. **src/features/position_interactions.py**
   - `InteractionMatrix` class - defines which positions affect each other
   - Methods to query interactions
   - Support for testing and pruning interactions

3. **src/features/teammate_adjustments.py**
   - `TeammateQualityCalculator` - calculate quality scores for position groups
   - `IterativeAdjustmentSolver` - converge to stable estimates (mirrors defense ratings)
   - Position-agnostic design
   - Time-weighted averaging (players change teams/lineups)

### Architecture

```python
# Pseudo-code structure
class TeammateQualityCalculator:
    def calculate_position_group_quality(self, position, players, games):
        """Calculate aggregate quality for position group (e.g., WR corps)"""

class IterativeAdjustmentSolver:
    def solve(self, player_stats, interaction_matrix, iterations=5):
        """
        Iteration 0: Raw EPA for all positions
        Iteration 1: Adjust each position for teammates
        Iteration 2-5: Recalculate until convergence
        """
```

### Success Criteria
- Infrastructure works for any position
- Converges to stable estimates (delta < threshold)
- Can add/remove interactions via config

---

## Phase 3: Proof of Concept - QB + Pass Catchers (Weeks 3-4)

**Goal**: Prove the concept works before extending to all positions

### Deliverables

1. **src/data_pipeline/receiver_stats_aggregator.py**
   - Aggregate WR/TE performance from play-by-play
   - Calculate receiver EPA per target, catch rate over expected
   - Game-level and season-level aggregation

2. **Teammate Adjustment Implementation**
   - QB adjusted for WR/TE quality
   - WR/TE adjusted for QB quality
   - Iterative convergence (5 iterations)

3. **validate_teammate_adjustments.py**
   - Test predictive power of adjusted metrics
   - Compare: Raw → Opp-Adj → Teammate-Adj
   - Holdout validation (2020-2022 → 2023)

### Validation Tests

**Question**: Does adjusting QB for WR quality improve future prediction?
```python
# Test on 2020-2022 predicting 2023
qb_raw = predict(qb_epa_2020_2022) → correlation with 2023
qb_wr_adj = predict(qb_epa_adjusted_for_wr) → correlation with 2023

# Only proceed if: qb_wr_adj > qb_raw + threshold (e.g., 0.01)
```

**Question**: Does adjusting WR for QB quality improve prediction?
```python
wr_raw = predict(wr_epa_2020_2022)
wr_qb_adj = predict(wr_epa_adjusted_for_qb)
```

### Decision Point
**STOP if teammate adjustments don't improve prediction**. No point extending to all positions if the concept doesn't work for QB + WR.

### Success Criteria
- QB adjustment for WR improves future prediction by >1%
- WR adjustment for QB improves future prediction by >1%
- Iterative solver converges (< 10 iterations)

---

## Phase 4: Extend to Offensive Positions (Weeks 5-6)

**Goal**: Add RB and OL to the teammate adjustment system

### Deliverables

1. **src/data_pipeline/rb_stats_aggregator.py**
   - Rushing EPA, receiving EPA
   - Pass blocking contributions (inferred)
   - Situational performance

2. **src/features/ol_quality_estimator.py**
   - Reverse engineer OL quality from QB/RB performance
   - Pass blocking: Pressure rate, sack rate (QB-adjusted)
   - Run blocking: Yards before contact, stuff rate

3. **Full Offensive Network**
   - QB ↔ WR/TE (already done)
   - QB ↔ RB
   - QB ↔ OL
   - RB ↔ OL
   - WR/TE ↔ OL

4. **Validation**
   - Test each new interaction's predictive power
   - Prune insignificant interactions
   - Compare full model vs simpler versions

### Success Criteria
- All offensive positions have adjusted ratings
- Full model predicts better than pairwise adjustments
- OL quality estimation is stable and reasonable

---

## Phase 5: Add Defense & Universal LEAF (Weeks 7-8)

**Goal**: Extend to defensive positions and create unified all-position LEAF

### Deliverables

1. **src/data_pipeline/defensive_stats_aggregator.py**
   - EDGE/DL: Pressure rate, run defense
   - LB: Coverage, run defense, pressure
   - DB: Coverage EPA, completion % allowed

2. **src/projections/universal_leaf.py**
   - Unified LEAF calculator for all positions
   - All players on common EPA scale
   - Position-specific plays-per-game conversions
   - Universal WAR calculation

3. **Cross-Unit Interactions** (optional)
   - Offense ↔ Defense via game script
   - Field position effects
   - Time of possession

4. **Comprehensive Validation**
   - Test on 2021→2022, 2022→2023
   - Cross-validate weight optimizations
   - Validate all-position rankings

### Success Criteria
- All 22 positions have LEAF ratings
- Ratings are on comparable EPA scale
- System improves prediction vs position-specific models

---

## Phase 6: Production & Refinement (Weeks 9+)

### Deliverables

1. **Automation**
   - Daily/weekly updates
   - Automated validation checks
   - Historical tracking

2. **Documentation**
   - Methodology paper
   - User guide
   - API documentation

3. **Visualization**
   - Web dashboard (Streamlit)
   - Interactive position rankings
   - Player comparison tool
   - Teammate quality visualizations

4. **Continuous Improvement**
   - A/B test new interactions
   - Optimize convergence speed
   - Add new data sources (Next Gen Stats, PFF)

---

## Key Validation Framework

For every new adjustment or interaction:

1. **Predictive Power Test**: Does it improve future prediction?
2. **Stability Test**: Do estimates converge and remain stable?
3. **Reasonableness Test**: Do results pass the eye test?
4. **Sample Size Test**: Do we have enough data?

### Validation Template

```python
def validate_adjustment(train_years, test_year, adjustment_name):
    """
    Template for testing any adjustment
    """
    # Baseline
    baseline = predict_future_performance(raw_metric, train_years, test_year)

    # With adjustment
    adjusted = predict_future_performance(adjusted_metric, train_years, test_year)

    # Test improvement
    improvement = adjusted.correlation - baseline.correlation

    # Decision
    if improvement > THRESHOLD:
        print(f"{adjustment_name}: KEEP (+{improvement:.3f})")
        return True
    else:
        print(f"{adjustment_name}: REJECT (+{improvement:.3f})")
        return False
```

---

## Current Status

- **Phase 1**: Starting now
- **QB-Only LEAF**: Complete (correlation 0.906, validated)
- **Infrastructure**: Defense ratings system can be adapted for teammates
- **Next Steps**: explore_position_metrics.py

---

## Risk Mitigation

1. **Risk**: Teammate adjustments don't improve prediction
   - **Mitigation**: Validate QB+WR first (Phase 3 decision point)

2. **Risk**: Insufficient data for some positions
   - **Mitigation**: Phase 1 exploration identifies issues early

3. **Risk**: Circular dependencies don't converge
   - **Mitigation**: Use proven iterative methods (already works for defense)

4. **Risk**: Model becomes too complex
   - **Mitigation**: Prune insignificant interactions via validation

---

## Success Metrics

1. **Phase 3**: QB+WR adjustments improve prediction by >1%
2. **Phase 4**: Full offensive model improves by >2% vs raw metrics
3. **Phase 5**: All-position LEAF outperforms position-specific models
4. **Overall**: Unified system correlates r > 0.90 with future performance across positions

---

## Open Questions

1. How to handle OL (no direct EPA attribution)?
2. Should we use game-level or play-level adjustments?
3. What's the right convergence threshold?
4. How to handle players who change positions?
5. Should rookie priors differ by position?

These will be answered as we progress through phases.
