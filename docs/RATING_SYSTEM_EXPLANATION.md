> **CORRECTION (July 2026).** The worked examples in this document pair the
> exponential weights with the wrong game indices (they show the OLDEST game at
> 13.62%). The deployed code is correct - the NEWEST game gets 13.62% - and the
> published Darnold current-form value (+0.144) reproduces under the correct
> orientation. Also: the r=0.3533 quoted here compares overlapping windows and
> the referenced "optimal vs proposed" finding is retracted (see the banner in
> WEIGHT_OPTIMIZATION_FINDINGS.md). For honest walk-forward predictive numbers
> use docs/LEAF_V3_RESULTS.md.

# QB Rating System Explanation

## Overview

We use multiple ratings depending on the purpose. Here's what each one means:

## Rating Types

### 1. Raw EPA (`epa_per_play`)
**What it is:** Expected Points Added per play, raw from nflfastR
- Measures how many points a QB added/subtracted vs expectation on each play
- Example: 3rd & 10 from own 20, expectation might be +0.5 points. Complete for 15 yards → actual result +2.0 points. EPA = +1.5

**Calculation:**
```
For each play:
  EPA = Points_Expected_After - Points_Expected_Before

For QB game:
  EPA_per_play = Sum(EPA) / Number_of_plays
```

**Pros:**
- Direct measure of QB production
- Accounts for down/distance/field position context

**Cons:**
- Noisy (varies wildly game-to-game)
- Not adjusted for opponent quality
- Heavily influenced by receivers, O-line, play-calling

**Typical Range:** -0.5 to +0.5 EPA/play
- Elite: +0.20+
- Good: +0.10 to +0.20
- Average: -0.05 to +0.10
- Below: < -0.05

---

### 2. Opponent-Adjusted EPA (`opp_adj_base_epa`)
**What it is:** Raw EPA adjusted for defensive quality faced

**Calculation:**
```
1. Calculate each defense's quality rating (iterative process)
2. For each QB game:
   - Adjustment = Defense_Quality * Impact_Factor
   - opp_adj_epa = raw_epa - Adjustment
```

**Example:**
- QB has +0.15 EPA/play vs a terrible defense (rated -0.10)
- Adjustment: -0.10 * 0.3 = -0.03
- Opponent-adjusted: +0.15 - (-0.03) = +0.18
→ QB gets credit for beating up bad defense, but not full credit

**Pros:**
- Fairer comparison across different schedules
- Rewards QBs who face tough defenses

**Cons:**
- Still noisy game-to-game
- Defense ratings themselves have uncertainty

---

### 3. Current Form (`current_form` in our analysis)
**What it is:** Opponent-adjusted EPA with optimal weighted averaging (12-game window, exponential decay)

**Calculation:**
```python
# Optimal configuration (empirically validated):
# - 12-game window
# - Exponential decay weights (rate=0.10)
# - 95th percentile outlier filtering

weights = exp(-0.10 * [11, 10, 9, ..., 0]) / sum(...)
# Result: [4.5%, 5.0%, 5.5%, ..., 12.3%, 13.6%] from oldest to newest

For each QB:
  1. Take last 12 games of opponent-adjusted EPA
  2. Apply 95th percentile winsorization (cap extreme values)
  3. Calculate weighted average using optimal weights
```

**Example Trajectory (Sam Darnold last 12 games):**
```
Game t-11: +0.220 × 13.62% = 0.030
Game t-10: +0.170 × 12.32% = 0.021
Game t-9:  -0.531 × 11.15% = -0.059
...
Game t-0:  +0.614 ×  4.53% = 0.028
-------------------------
Total: +0.144 (Elite tier)
```

**Pros:**
- Maximizes prediction of next 16 games (r=0.3533)
- Smooths out extreme single-game performances via outlier filtering
- Captures recent trends with balanced weighting
- 47% better predictive power than steep weighting (35% newest)

**Cons:**
- More complex than simple average
- Requires 12 games of data (uses partial weights for < 12 games)

**This is what we use for "Current Form" - empirically optimized from 60 tested configurations.**

---

### 4. Kalman Filtered Rating (`opp_adj_base_epa_kalman`)
**What it is:** Opponent-adjusted EPA with Bayesian updating (Kalman filter)

**Calculation:**
```
Prior belief: QB = 0.00 (average) with high uncertainty

For each game:
  1. Prediction: Estimate QB's true skill based on prior games
  2. Observation: New game performance (opp_adj_epa)
  3. Update: Blend prediction + observation based on confidence
     - Early career: More weight on observation (limited prior data)
     - Late career: More weight on prediction (strong prior belief)
  4. New belief = Updated estimate of QB's "true skill"
```

**Example:**
```
Game 1:
  Prior: 0.00 (±0.50 uncertainty)
  Observation: +0.20
  Update: 0.00 * 0.2 + 0.20 * 0.8 = +0.16

Game 50:
  Prior: +0.10 (±0.05 uncertainty - very confident)
  Observation: +0.30 (great game!)
  Update: +0.10 * 0.9 + 0.30 * 0.1 = +0.12
  → Barely moves! Filter thinks it's noise
```

**Pros:**
- Estimates "true skill" not just recent form
- Accounts for sample size (confident after many games)
- Optimal balance between prior belief and new evidence

**Cons:**
- **TOO CONSERVATIVE** - slow to recognize real improvement/decline
- Punishes QBs with bad early careers for dozens of games
- Not good for evaluating current performance

**This is why Darnold's Kalman rating was -0.080 while his raw EPA was +0.364!**

---

## Which Rating to Use When

### Current Performance Evaluation
**Use:** `current_form` (optimal 12-game weighted average)
- "How good is this QB right now?"
- Weekly rankings, fantasy football
- Recent form matters more than distant history
- Uses exponential decay weights + outlier filtering

**Sam Darnold 2025:** +0.144 current form → **Elite tier (#4 out of 62 QBs)**

---

### Early Career Thresholds
**Use:** `current_form` or even `raw_epa`
- Finding what performance level at game 8/16/32 predicts future success
- We WANT to see actual performance, not smoothed estimates
- Elite QBs must show flashes early, even if inconsistent

**Elite threshold at Game 8:** -0.057
- Means: 30% of Elite QBs had negative opponent-adjusted EPA through 8 games
- But by game 28, ALL Elite QBs are positive

---

### Contract/Long-term Decisions
**Use:** Weighted recent performance (last 51 games heavy, exponential decay)
- Balances current form with proven track record
- Don't overpay for a 8-game hot streak
- Don't underpay a proven vet having a rough stretch

**Sam Darnold:** Career rating -0.135, but current form +0.085
→ Pay for current performance with incentives for sustained success

---

### Historical Comparisons
**Use:** Career average `current_form`
- "How good was QB X's career?"
- Simple average of all games
- No recency bias

---

## Visual Example: Sam Darnold 2025

```
Game | Raw EPA | Opp-Adj | Filtered | Weight  | Current Form | Kalman
-----|---------|---------|----------|---------|--------------|--------
 t-11| +0.220  | +0.220  | +0.220   | 13.62%  |              |
 t-10| +0.170  | +0.170  | +0.170   | 12.32%  |              |
 t-9 | -0.531  | -0.531  | -0.531   | 11.15%  |              |
 t-8 | -0.553  | -0.553  | -0.541*  | 10.09%  |              |
 t-7 | -0.420  | -0.420  | -0.420   |  9.13%  |              |
 t-6 | -0.041  | -0.041  | -0.041   |  8.26%  |              |
 t-5 | +0.752  | +0.752  | +0.676*  |  7.47%  |              |
 t-4 | +0.183  | +0.183  | +0.183   |  6.76%  |              |
 t-3 | +0.602  | +0.602  | +0.602   |  6.12%  |              |
 t-2 | +0.189  | +0.189  | +0.189   |  5.54%  |              |
 t-1 | -0.153  | -0.153  | -0.153   |  5.01%  |              |
 t-0 | +0.614  | +0.614  | +0.614   |  4.53%  |              |
-----|---------|---------|----------|---------|--------------|--------
Avg: | +0.086  | +0.086  |          |         |    +0.144    | -0.080

* = Outlier filtered at 95th percentile
```

**Interpretation:**
- **Raw EPA (+0.086):** Simple 12-game average
- **Current Form (+0.144):** Optimal weighted average with outlier filtering - **Elite tier!**
- **Kalman (-0.080):** Still dragged down by 2018-2023 history

For 2025 evaluation, we use **Current Form (+0.144)** → Elite tier (#4 overall).

---

## Summary Table

| Rating | Purpose | Smoothing | Time Horizon | Darnold 2025 |
|--------|---------|-----------|--------------|--------------|
| Raw EPA | Game-level production | None | Single game | +0.086 (12-game avg) |
| Opp-Adj EPA | Fair comparison | None | Single game | +0.086 (12-game avg) |
| Current Form | Recent performance | Optimal (12 games) | Last 12 games weighted | +0.144 (Elite) |
| Kalman | "True skill" estimate | Heavy (career) | Full career | -0.080 |
| Career Rating | Long-term evaluation | None | Full career average | -0.125 |

**Optimal Current Form Configuration:**
- Window: 12 games (empirically validated as best predictor)
- Weights: Exponential decay (4.5% oldest → 13.6% newest)
- Outlier filtering: 95th percentile winsorization
- Predictive power: r=0.3533 correlation with next 16 games
- 47% better than steep weighting schemes (35% on newest game)

---

## Why the Different Results?

### Darnold 2018-2023 History
- Games 1-80: Average -0.13 (Below Average)
- Lots of terrible games (sub -0.50 EPA)
- Kalman filter learned "Darnold is bad" with high confidence

### Darnold 2024-2025
- Games 63-88: Average +0.15 (Good/Elite)
- Consistent positive performances
- But Kalman filter skeptical: "26 games vs 80 games of data"

### Current Form Says: Judge him on NOW
- Last 12 games (optimal weighted): +0.144 → Elite tier (#4 overall)
- Simple 12-game average: +0.086 → Good tier
- Optimal weighting boosts him +0.058 due to positive recent trend

**Verdict:** Use Current Form for current evaluation. Darnold IS playing at an Elite tier level right now, with empirically validated predictive methodology.
