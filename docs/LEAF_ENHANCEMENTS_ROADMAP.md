# LEAF Enhancement Roadmap

**Date**: 2025-11-05
**Status**: Post Teammate Adjustment Validation

---

## Executive Summary

After comprehensive validation, **teammate adjustments do NOT improve prediction** (tested 4 methods, all failed or hurt performance). Instead, we're focusing on practical enhancements that avoid circular dependency issues:

1. **In-Game Updates**: Update LEAF ratings after every game (not just season aggregates)
2. **Injury Tracking**: Adjust for injury status and uncertainty
3. **Context Adjustments**: Weather, home/away, rest days, game script

These improvements work WITH the existing 0.8855 baseline rather than trying to decompose it.

---

## Teammate Adjustment Validation - FINAL RESULTS

### What We Tested

| Method | Approach | Result | Status |
|--------|----------|--------|--------|
| **Baseline** | Kalman + Opponent-Adj | 0.8855 | - |
| **V2** | Residual (team aggregate) | 0.8855 (no change) | ❌ |
| **C1** | Iterative simultaneous | 0.8252 (-6.8%) | ❌ |
| **C2** | Individual WR quality | 0.8151 (-8.0%) | ❌ |
| **C3** | Direct covariate | 0.8789 (-1.0%) | ❌ |

### Why They Failed

1. **Receiver quality already captured**: Kalman+Opponent adjustments already account for supporting cast through game-level variance
2. **Circular dependency**: Good QBs get good receivers, good receivers elevate QBs - fundamentally hard to untangle
3. **Adjustment removes signal**: Every explicit adjustment introduced more noise than signal
4. **Baseline is excellent**: 0.8855 represents ~78% of theoretical maximum

### Decision

**STOP teammate adjustment research. Focus on enhancements below.**

---

## Enhancement #1: In-Game Updates (Game-by-Game LEAF)

### Current System
- Aggregate to season level
- Single LEAF rating per QB per season
- Only updated between seasons

### Proposed: Rolling Game-by-Game Updates
- Update LEAF after every game
- Kalman filter processes each game sequentially
- Real-time rating trajectory throughout season

### Benefits
1. **Capture hot/cold streaks**: QB improving or declining mid-season
2. **Injury recovery tracking**: Gradual return to form after injury
3. **Real-time prediction**: Current rating at any point in season
4. **Uncertainty quantification**: Confidence intervals narrow with more games

### Implementation

**Phase 1A: Infrastructure**
- Modify QBKalmanFilter to output game-by-game estimates (not just final)
- Store trajectory: `{game_id: {rating, uncertainty, game_number}}`
- Add "games played" weight to confidence

**Phase 1B: API**
```python
class LEAFRatingSystem:
    def get_current_rating(self, player_id: str, date: str) -> dict:
        """Get player rating as of specific date."""
        return {
            'leaf_rating': 95.3,
            'uncertainty': 2.1,
            'games_played': 8,
            'trend': 'improving',  # +0.5 per game over last 4 games
            'last_updated': '2024-11-04'
        }

    def get_rating_trajectory(self, player_id: str, season: int) -> pd.DataFrame:
        """Get full season trajectory."""
        return pd.DataFrame([
            {'game': 1, 'rating': 92.1, 'uncertainty': 5.2},
            {'game': 2, 'rating': 93.5, 'uncertainty': 4.8},
            # ...
        ])
```

**Phase 1C: Validation**
- Test: Does Week 8 LEAF predict Week 9-17 better than season-end LEAF?
- Expected improvement: +1-2% correlation for in-season prediction

**Priority**: **HIGH** - Relatively easy, clear value

---

## Enhancement #2: Injury Tracking

### Data Sources

**nflfastR injury data** (if available):
- Check if `pbp` has injury flags or status columns
- May need to fetch from external source (ESPN API, NFL injury reports)

**Alternative sources**:
- ESPN API: `/sports/football/nfl/teams/{team}/injuries`
- Pro Football Reference injury data
- Manual tracking from weekly injury reports

### Injury Status Levels

1. **Healthy**: No designation
2. **Questionable**: Increased uncertainty (±10%)
3. **Probable**: Slight uncertainty increase (±5%)
4. **Out**: Rating = 0 (don't predict for missed games)
5. **IR/PUP**: Exclude from predictions entirely
6. **Returning from injury**: Gradual ramp-up over 2-4 games

### Implementation

**Phase 2A: Data Collection**
```python
class InjuryTracker:
    def fetch_weekly_injuries(self, week: int, season: int) -> pd.DataFrame:
        """Fetch injury reports for week."""
        # ESPN API or manual data entry
        return pd.DataFrame([
            {'player_id': 'P.Mahomes', 'status': 'Questionable', 'injury': 'Ankle'},
            {'player_id': 'J.Allen', 'status': 'Out', 'injury': 'Shoulder'}
        ])

    def is_returning_from_injury(self, player_id: str, game_id: str) -> bool:
        """Check if player just returned from multi-week absence."""
        # Logic to detect first 1-2 games back
```

**Phase 2B: Uncertainty Adjustment**
```python
def adjust_for_injury(rating: float, uncertainty: float, status: str) -> tuple:
    """Adjust rating and uncertainty for injury status."""
    if status == 'Out':
        return 0.0, 0.0
    elif status == 'Questionable':
        return rating, uncertainty * 1.10  # Increase uncertainty 10%
    elif status == 'Probable':
        return rating, uncertainty * 1.05  # Increase uncertainty 5%
    else:
        return rating, uncertainty
```

**Phase 2C: Recovery Model**
```python
def model_injury_recovery(games_since_return: int) -> float:
    """Model gradual return to full performance."""
    # Game 1 back: ~85% of pre-injury rating
    # Game 2 back: ~92% of pre-injury rating
    # Game 3+: Full rating
    recovery_factors = {1: 0.85, 2: 0.92, 3: 1.00}
    return recovery_factors.get(games_since_return, 1.00)
```

**Priority**: **MEDIUM** - Requires external data source, but high value for predictions

---

## Enhancement #3: Context Adjustments

### 3A: Weather Adjustments

**Data Source**: nflfastR includes weather data
- `temp`: Temperature (F)
- `wind`: Wind speed (mph)
- `weather`: Description (e.g., "Clear", "Rain", "Snow")

**Adjustment Approach**:
```python
def calculate_weather_adjustment(temp: float, wind: float, weather: str) -> float:
    """Calculate EPA adjustment for weather conditions."""
    adjustment = 0.0

    # Cold weather penalty (passing EPA drops in cold)
    if temp < 32:
        adjustment -= 0.02  # -0.02 EPA/play in freezing
    elif temp < 45:
        adjustment -= 0.01  # -0.01 EPA/play in cold

    # Wind penalty (especially for passing)
    if wind > 20:
        adjustment -= 0.03  # Severe wind
    elif wind > 15:
        adjustment -= 0.015  # Moderate wind

    # Precipitation
    if 'rain' in weather.lower():
        adjustment -= 0.01
    if 'snow' in weather.lower():
        adjustment -= 0.02

    return adjustment
```

**Validation**: Test on historical data - do weather-adjusted ratings predict better?

### 3B: Home/Away Adjustment

**Current**: Not explicitly modeled
**Proposal**: Track home vs away splits

```python
def calculate_home_field_advantage() -> float:
    """Historical average home field advantage."""
    # NFL average: ~2.5 point spread = ~0.01-0.02 EPA/play
    return 0.015  # Home team gets +0.015 EPA/play
```

**Refinement**: Team-specific home field advantage (some stadiums matter more)

### 3C: Rest Days Adjustment

**Short week penalty** (Thursday night):
- Both teams on short rest: Offense suffers more than defense
- Typical penalty: -0.01 to -0.02 EPA/play

**Extra rest bonus** (bye week):
- Coming off bye: +0.01 EPA/play (first game back)

```python
def calculate_rest_adjustment(days_since_last_game: int) -> float:
    """Adjust for rest between games."""
    if days_since_last_game <= 4:
        return -0.015  # Thursday night penalty
    elif days_since_last_game >= 14:
        return 0.01  # Bye week bonus (first game)
    else:
        return 0.0  # Normal week
```

### 3D: Game Script Adjustment

**Garbage time filtering**:
- Current approach: Include all plays
- Better approach: Weight/exclude plays in extreme score differentials

```python
def is_garbage_time(score_diff: int, time_remaining: int, quarter: int) -> bool:
    """Identify garbage time plays."""
    if quarter >= 4:
        # 4th quarter, 3+ score lead, < 5 minutes left
        if abs(score_diff) >= 17 and time_remaining < 300:
            return True
        # 4th quarter, 4+ score lead
        if abs(score_diff) >= 24:
            return True
    return False

def calculate_garbage_time_weight(is_garbage: bool) -> float:
    """Weight garbage time plays less heavily."""
    return 0.3 if is_garbage else 1.0  # 30% weight for garbage time
```

**Validation**: Test if garbage-time-adjusted EPA predicts better

### 3E: Time of Season

**Early season** (Weeks 1-4):
- Higher uncertainty (small sample)
- More regression to prior

**Mid-season** (Weeks 5-12):
- Peak confidence
- Full weight to current season data

**Late season** (Weeks 13-18):
- Playoff implications
- Possible resting of starters (if team out of contention)

```python
def calculate_sample_size_weight(week: int) -> float:
    """Weight based on sample size."""
    if week <= 4:
        return 0.7  # Heavy prior regression early
    elif week <= 8:
        return 0.85
    else:
        return 1.0  # Full weight mid-late season
```

**Priority**: **MEDIUM-HIGH** - Weather and home/away are easy wins

---

## Implementation Priority

### Phase 1 (High Priority, Quick Wins)
1. **In-game updates** (1-2 weeks)
   - Modify Kalman filter for game-by-game output
   - Store trajectories
   - Basic API for current rating

2. **Weather adjustments** (3-5 days)
   - nflfastR already has weather data
   - Simple adjustment model
   - Validate on historical data

3. **Home/away adjustment** (2-3 days)
   - Easy to implement
   - Fixed +0.015 EPA/play for home team
   - Can refine later with team-specific values

### Phase 2 (Medium Priority, 2-4 weeks)
4. **Injury tracking infrastructure** (1-2 weeks)
   - Set up data pipeline (ESPN API or manual)
   - Basic status tracking

5. **Injury uncertainty adjustment** (3-5 days)
   - Adjust confidence intervals for injury status
   - Flag questionable players

6. **Rest days adjustment** (2-3 days)
   - Calculate days since last game
   - Apply Thursday/bye adjustments

### Phase 3 (Lower Priority, Refinement)
7. **Garbage time filtering** (1 week)
   - Identify garbage time plays
   - Re-aggregate with weights
   - Validate improvement

8. **Recovery modeling** (1-2 weeks)
   - Track games since injury return
   - Model gradual recovery curve

9. **Time-of-season weighting** (3-5 days)
   - Dynamic prior regression based on week

---

## Expected Impact

### Conservative Estimates

| Enhancement | Expected Improvement | Confidence |
|-------------|---------------------|------------|
| In-game updates | +0.01-0.02 correlation | High |
| Weather adjustment | +0.005-0.01 | Medium |
| Home/away | +0.003-0.005 | High |
| Injury tracking | Better uncertainty | High |
| Rest days | +0.002-0.005 | Medium |
| Garbage time | +0.005-0.01 | Medium |
| **TOTAL** | **+0.025-0.05** | **Medium** |

**Target**: Improve from 0.8855 → 0.91-0.93 correlation

---

## Data Requirements

### Already Available (nflfastR)
- ✅ Temperature, wind, weather
- ✅ Home/away indicator
- ✅ Score differential, time remaining
- ✅ Game dates (for rest calculation)

### Need to Add
- ⚠️ Injury status (weekly reports)
- ⚠️ Days since last game (calculable from dates)

### Optional Enhancements
- 🔮 Stadium-specific factors (altitude, dome vs outdoor)
- 🔮 Referee assignments (some refs call tighter games)
- 🔮 Offensive coordinator changes

---

## Validation Framework

For each enhancement, test:

1. **Historical validation**: Does it improve 2020-2022 → 2023 prediction?
2. **In-season validation**: Does Week N rating predict Week N+1 better?
3. **Ablation study**: Marginal improvement vs existing baseline

**Acceptance criteria**: +0.005 correlation improvement (0.5%)

---

## Next Steps

1. ✅ **DONE**: Validate teammate adjustments (all failed)
2. **NOW**: Implement Phase 1 enhancements
   - Start with in-game updates (highest value)
   - Add weather adjustments (easy win)
3. **Week 2**: Validate Phase 1 improvements
4. **Week 3-4**: Implement Phase 2 (injury tracking)

---

## Lessons Learned from Teammate Adjustments

**What we learned**:
1. Don't fight circular dependencies - work with the system
2. Kalman+Opponent adjustments already capture most of the signal
3. Explicit decomposition often removes signal with noise
4. Focus on orthogonal improvements (context, uncertainty, temporal)

**Applied to new enhancements**:
- Weather is orthogonal to QB talent (no circular dependency)
- Home/away is independent of player quality
- Injuries are observable events, not latent factors
- In-game updates work WITH Kalman, not against it

---

## Summary

We're pivoting from multi-position LEAF (failed due to circular dependencies) to **context-aware, temporally-dynamic LEAF**:

- **Game-by-game ratings** (not season aggregates)
- **Injury-adjusted uncertainty** (flag risk)
- **Weather, home/away, rest adjustments** (context matters)

These enhancements avoid circular dependencies and work WITH the existing excellent baseline (0.8855).

**Next Action**: Start Phase 1 - implement in-game updates!
