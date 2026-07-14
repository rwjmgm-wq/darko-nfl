# Unified NFL Player Rating System - Proposal

## Overview

Create a unified "DARKO NFL" style rating system that allows cross-position player comparisons and team-level aggregation.

---

## Current State

### What We Have

1. **QB LEAF Rating**: Kalman-filtered air_epa with context adjustments
   - NOT based on exhaustive metric search
   - Uses single metric (air_epa) as foundation

2. **WR Composite Rating**: 5-metric combination from exhaustive search
   - Validated across 2020-2024
   - YAC-focused, minimizes QB dependency
   - r = 0.66 multi-year stability

3. **Other Positions**: Not yet developed

---

## Proposed Plan

### Phase 1: Run Exhaustive QB Analysis

Apply the same 4-phase process we used for WRs:

**Step 1: Metric Candidate Selection**
- Test all available QB metrics for year-over-year correlation
- Identify top 15 most predictive metrics
- Consider: completion %, yards, TDs, INTs, EPA metrics, success rate, QBR components

**Step 2: Exhaustive Combination Search**
- Test all C(15,5) = 3,003 possible 5-metric combinations
- Find optimal equal-weighted combination
- Compare to current LEAF rating

**Step 3: Weight Optimization**
- Use Ridge regression to optimize weights
- Test alpha values: 0.1, 1.0, 10.0
- Compare to equal weighting

**Step 4: Multi-Year Validation**
- Test on 2020-2024 year pairs
- Measure stability and consistency
- Compare to single metrics

**Expected Output**:
- QB Composite Rating (similar structure to WR Composite)
- 5 optimal metrics with validated weights
- Multi-year stability metrics

---

### Phase 2: Develop RB Composite Rating

Same process for running backs:

**Key Considerations**:
- RBs have dual role: rushing + receiving
- Need metrics for both dimensions
- May need separate composites for:
  - Pure rushers (Derrick Henry)
  - Pass-catching backs (Christian McCaffrey)
  - 3-down backs (Alvin Kamara)

**Candidate Metrics**:
- Rushing: carries, yards, YPC, rush_epa, success_rate
- Receiving: targets, receptions, receiving_epa, YAC
- Impact: first_downs, TDs, broken tackles (if available)

---

### Phase 3: Develop TE Composite Rating

Same process for tight ends:

**Key Considerations**:
- TEs have triple role: receiving + blocking + run support
- Blocking data limited in nflfastR
- May focus on receiving metrics similar to WRs

**Candidate Metrics**:
- Similar to WRs but adjusted for:
  - Lower target volume
  - Different route trees
  - Red zone usage

---

### Phase 4: Create Unified "DARKO NFL" Score

Once we have position-specific composites, create unified system:

#### Option A: Percentile-Based Unification

```python
unified_score = percentile_within_position(composite_rating)
```

**Pros**:
- Simple, interpretable
- Automatically accounts for position differences
- Easy to aggregate to team level

**Cons**:
- Assumes all positions equally valuable
- Doesn't account for positional scarcity
- 50th percentile QB ≠ 50th percentile WR in terms of value

#### Option B: Value-Based Unification (Recommended)

```python
unified_score = position_weight × standardized_composite + baseline
```

**Position Weights** (based on positional value):
- QB: 1.5× (most important position)
- WR: 1.0× (baseline)
- RB: 0.8× (less valuable in modern NFL)
- TE: 0.9× (hybrid value)
- OL: 1.2× (if we expand)

**Baseline Adjustments**:
- Account for replacement level
- Use minimum starter threshold
- Adjust for position depth

**Example Calculation**:
```
QB with +1.0 composite → 1.5 × 1.0 + 0.0 = 1.5 unified score
WR with +1.0 composite → 1.0 × 1.0 + 0.0 = 1.0 unified score
RB with +1.0 composite → 0.8 × 1.0 + 0.0 = 0.8 unified score
```

#### Option C: WAR-Style Approach

Similar to baseball's WAR (Wins Above Replacement):

```python
unified_score = (composite - replacement_level) × games × position_leverage
```

**Components**:
- **composite**: Position-specific composite rating
- **replacement_level**: Average of bottom 10% at position
- **games**: Games played (durability matters)
- **position_leverage**: Impact on team wins

**Pros**:
- Most sophisticated
- Directly interpretable as "wins added"
- Accounts for durability

**Cons**:
- More complex to explain
- Requires win probability modeling
- Need replacement level baselines

---

## Implementation Roadmap

### Week 1: QB Exhaustive Analysis
1. Adapt WR analysis scripts for QB data
2. Run 4-phase analysis (2020-2024)
3. Generate QB composite rating
4. Compare to existing LEAF rating

### Week 2: RB & TE Analysis
1. Run exhaustive analysis for RBs
2. Run exhaustive analysis for TEs
3. Document optimal metrics for each position

### Week 3: Unified System Development
1. Choose unification approach (recommend Option B)
2. Calibrate position weights
3. Calculate unified scores for all positions
4. Validate against team wins (correlation check)

### Week 4: Visualization & Documentation
1. Create unified player dashboard
2. Add team-level aggregation view
3. Show position breakdowns
4. Documentation and testing

---

## Use Cases

### 1. Cross-Position Player Rankings

```python
# Top 50 players regardless of position
all_players.sort_values('unified_score', ascending=False).head(50)
```

### 2. Team Talent Evaluation

```python
# Aggregate team score by position
team_talent = players.groupby(['team', 'position']).agg({
    'unified_score': 'sum'
}).reset_index()
```

### 3. Positional Spending Efficiency

```python
# Compare unified score to salary cap hit
efficiency = unified_score / salary_cap_hit
```

### 4. Draft Value Analysis

```python
# Evaluate draft picks by unified score gained
draft_value = drafted_players.groupby('draft_round')['unified_score'].mean()
```

### 5. Trade Value Assessment

```python
# Compare players across positions for fair trades
trade_value_diff = player_a['unified_score'] - player_b['unified_score']
```

---

## Technical Considerations

### Data Requirements

1. **QB Data**: Already available in nflfastR
2. **RB Data**: Rushing + receiving stats available
3. **WR Data**: ✅ Already processed
4. **TE Data**: Similar to WR, available
5. **OL Data**: Limited in nflfastR (future enhancement)
6. **DEF Data**: Available but separate analysis needed

### Computational Complexity

- **QB Analysis**: ~30 minutes (similar to WR)
- **RB Analysis**: ~30 minutes
- **TE Analysis**: ~30 minutes
- **Unification**: ~5 minutes
- **Total**: ~2 hours for complete system

### Storage Requirements

- QB composite: ~500 QB-seasons (~100 KB)
- RB composite: ~600 RB-seasons (~100 KB)
- TE composite: ~400 TE-seasons (~100 KB)
- Unified scores: ~2,000 player-seasons (~300 KB)
- **Total**: <1 MB additional storage

---

## Validation Strategy

### Internal Validation

1. **Multi-year stability**: Test r on year pairs
2. **Cross-validation**: Hold out seasons
3. **Consistency**: Std dev across years

### External Validation

1. **Team wins correlation**: Do high-score teams win more?
2. **Playoff success**: Do high-score teams make playoffs?
3. **Awards correlation**: Do high-score players win MVP/OPOY?
4. **Expert rankings**: Compare to PFF, ESPN, etc.

### Robustness Checks

1. **Position balance**: Ensure no position dominates top 100
2. **Team balance**: Check for team bias
3. **Era adjustments**: Account for rule changes over time

---

## Expected Outcomes

### QB Composite Rating
- **Correlation with existing LEAF**: r = 0.85-0.95
- **Multi-year stability**: r = 0.60-0.70
- **Top metrics**: Likely completion %, EPA/play, success_rate, yards, TDs

### RB Composite Rating
- **Multi-year stability**: r = 0.50-0.60 (RBs more volatile)
- **Top metrics**: Likely carries, receiving_epa, YPC, first_downs, TDs

### TE Composite Rating
- **Multi-year stability**: r = 0.55-0.65
- **Top metrics**: Similar to WR but lower volume

### Unified Score
- **Team wins correlation**: r = 0.70-0.80
- **Playoff prediction**: AUC = 0.75-0.85
- **MVP correlation**: Top 5 players include 3-4 actual MVP candidates

---

## Alternative Approaches

### Approach 1: Single Metric for All Positions

Use EPA as universal currency:
```python
unified_score = total_epa / expected_epa_for_position
```

**Pros**: Simplest, directly comparable
**Cons**: EPA is QB-dependent for skill positions

### Approach 2: Neural Network Unification

Train ML model to predict unified score:
```python
model.fit(X=position_specific_stats, y=team_wins_contributed)
```

**Pros**: Captures complex interactions
**Cons**: Black box, hard to interpret

### Approach 3: Market Value Proxy

Use contract values as unified currency:
```python
unified_score = predicted_market_value
```

**Pros**: Real-world validation
**Cons**: Market inefficiencies, positional biases

**Recommendation**: Use Option B (Value-Based Unification) for balance of simplicity and accuracy.

---

## Next Steps

**Immediate Priority**:
1. Run exhaustive QB analysis (highest impact)
2. Compare QB composite to existing LEAF rating
3. Decide if we replace or complement LEAF

**Once QB Complete**:
4. Run RB and TE analyses
5. Develop unified scoring system
6. Create cross-position visualizations

**Should we proceed with QB exhaustive analysis first?**

This will take ~30-60 minutes to run but will give us:
- Optimal QB metric combination
- Comparison to existing LEAF rating
- Foundation for unified system

---

**Created**: 2025-11-07
**Status**: Proposal - Awaiting approval to proceed
