# College LEAF Model Design

## Overview
Adapt NFL LEAF framework to college football for QB evaluation with competition adjustment.

## Core Components

### 1. Base Metric: EPA (Expected Points Added)
- **Source**: CFBD API provides PPA (Predicted Points Added) per play
- **Alternative**: Calculate EPA from play outcomes using down/distance/field position models
- **Aggregation**: EPA per play, EPA per game, season total EPA

### 2. Context Adjustments (from NFL LEAF)
```
Context factors to adjust for:
- Down & Distance (3rd & long vs 1st & 10)
- Score Differential (garbage time vs competitive)
- Time Remaining (end of half vs middle of game)
- Field Position (redzone vs midfield)
```

### 3. Competition Adjustments (KEY INNOVATION)
```
Two-sided adjustment using SP+ ratings:

A. Opponent Strength (Defensive SP+)
   - Adjust EPA based on opponent defense quality
   - Example: +0.20 EPA vs SP+ 5 defense is more impressive than +0.30 vs SP+ -10 defense

B. Team Strength (Offensive SP+)
   - Adjust for supporting cast quality (OL, WRs, scheme)
   - Penalize QBs on elite teams, reward QBs on weak teams
   - Example: Alabama QB gets downward adjustment, FCS QB gets upward adjustment

Formula:
  Adjusted EPA = Raw EPA × (1 + α × Opponent_SP+ - β × Team_SP+)

  Where:
    α = opponent adjustment weight (positive - harder opponent = bonus)
    β = team adjustment weight (positive - better team = penalty)
```

### 4. Aggregation to Season Rating
```
College LEAF (season) =
  Σ(Context-Adjusted EPA × Competition Multiplier) / Plays

Normalize to scale similar to NFL LEAF (-2 to +2 range)
```

## Key Differences from NFL LEAF

| Aspect | NFL LEAF | College LEAF |
|--------|----------|--------------|
| **Sample Size** | 500-600 plays/season | 300-400 plays/season |
| **Competition** | Relatively uniform | Huge disparities (FCS vs Alabama) |
| **Team Quality Impact** | Moderate | Extreme (supporting cast matters more) |
| **Game-by-Game Tracking** | Yes (Kalman filter) | No (limited games) |
| **Key Innovation** | Opponent adjustment | **Competition adjustment (2-sided)** |

## Implementation Steps

### Phase 1: Data Collection
1. Fetch play-by-play data from CFBD API for each QB
2. Get opponent info for each game
3. Fetch SP+ ratings for all teams and opponents
4. Calculate plays, EPA, context variables

### Phase 2: EPA Calculation
- Use CFBD PPA (if available) OR
- Build simple EPA model from play outcomes

### Phase 3: Context Adjustment
- Apply down/distance/score/time adjustments
- Similar weights to NFL LEAF

### Phase 4: Competition Adjustment
- Calculate opponent adjustment multiplier
- Calculate team quality adjustment multiplier
- Apply two-sided adjustment to EPA

### Phase 5: Validation
- Test correlation: College LEAF → NFL Rookie LEAF
- Compare to baseline (draft capital)
- Tune adjustment weights (α, β) for best prediction

## Expected Outcomes

**If College LEAF works:**
- Correlation with NFL rookie LEAF: r > 0.30 (vs baseline r = -0.186)
- Competition-adjusted rating beats raw stats
- Can identify QBs who overperformed due to team strength
- Can identify QBs who are undervalued due to weak competition

**Success Criteria:**
- r > 0.30: Strong signal, proceed with full model
- r = 0.20-0.30: Moderate signal, useful as supplementary tool
- r < 0.20: Weak signal, project not viable

## Data Requirements

### From CFBD API:
```
For each QB in sample (62 QBs, 2015-2023 drafts):
  - Play-by-play data for final college season
  - Opponent for each game
  - PPA (EPA) per play
  - Context: down, distance, yard line, score, time
  - Game results
```

### From SP+ Ratings (already collected):
```
For each season (2014-2022):
  - Team SP+ overall, offense, defense
  - Opponent SP+ for each game
```

## Next Steps
1. Build play-by-play data collector
2. Calculate raw EPA metrics
3. Apply adjustments
4. Test predictions
5. Iterate on adjustment weights
