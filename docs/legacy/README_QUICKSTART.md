# LEAF Rating System - Quick Start Guide

**Date**: 2025-11-05
**Version**: LEAF v2.0 + Visualization App

---

## What is LEAF?

**LEAF (Layered EPA Adaptive Framework)** is an advanced QB rating system that achieves **0.895 correlation** with future performance (170% improvement over raw EPA).

**Components**:
1. ✅ Context adjustments (weather, field position, etc.)
2. ✅ Opponent adjustments (defense quality)
3. ✅ Kalman filtering (adaptive learning)
4. ✅ Injury tracking (optional)
5. ✅ Game-by-game ratings
6. ✅ **NEW: Interactive visualization app**

---

## Quick Start

### 1. Deploy QB Ratings (2016-2024)

Generate production ratings for all available seasons:

```bash
python deploy_leaf_v2.py --seasons 2016 2017 2018 2019 2020 2021 2022 2023 2024
```

**Output** (in `data/production/`):
- `leaf_v2_game_by_game_[DATE].csv` - 6,089 game records
- `leaf_v2_season_ratings_[DATE].csv` - 1,010 QB-season ratings
- `leaf_v2_current_ratings_[DATE].csv` - 373 current QB ratings
- `leaf_v2_metadata_[DATE].json` - Deployment metadata

**Time**: ~2-3 minutes

---

### 2. Install Visualization Dependencies

```bash
pip install -r requirements_viz.txt
```

Installs: `dash`, `dash-bootstrap-components`, `plotly`

---

### 3. Launch Visualization App

```bash
python visualize_leaf_ratings.py
```

**Output**:
```
INFO:__main__:Loading LEAF ratings data...
INFO:__main__:  Loaded 6,089 game records
INFO:__main__:  Loaded 373 player ratings
INFO:__main__:  Filtered to 145 QBs with 50+ attempts
...
Dash is running on http://0.0.0.0:8050/
```

**Open browser**: http://localhost:8050

---

## Using the Visualization App

### Features

**1. Player Selection**
- Dropdown with 145 QBs (50+ career attempts)
- Sorted by current LEAF rating

**2. Current Rating Display**
- LEAF rating ± uncertainty
- Color-coded interpretation (Elite, Above Average, Average, Below Average)

**3. Career Statistics**
- Total games, attempts, seasons

**4. Future Predictions**
- 1, 2, 3 years forward
- Based on current trajectory + age decline
- Expanding uncertainty over time

**5. Interactive Trajectory Chart**
- **Historical**: Game-by-game ratings (2016-2024) with confidence bands
- **Predictions**: Future ratings with uncertainty
- **Reference lines**: Average, Elite, Poor QBs
- **Hover details**: Season, week, opponent, rating
- **Export**: Save chart as PNG

---

## Example: Analyzing Patrick Mahomes

### Step 1: Select Player
1. Open app: http://localhost:8050
2. Select "Patrick Mahomes (0.185)" from dropdown

### Step 2: View Current Rating
- **Rating**: 0.185 ± 0.021 (Elite QB) 🟢
- **Games**: 126 career games
- **Seasons**: 8 (2018-2024)

### Step 3: Explore Trajectory
- Chart shows all 126 games from 2018-2024
- Rating ranges from ~0.10 to ~0.25
- Tight confidence bands = very consistent performance
- Slight upward trend over career

### Step 4: Review Predictions
- **2025**: 0.178 ± 0.026 (slight decline from peak)
- **2026**: 0.171 ± 0.039 (continued slight decline)
- **2027**: 0.164 ± 0.052 (age 32, entering decline phase)

### Interpretation
Mahomes remains elite through 2027, but predictions show gradual decline as he ages. Uncertainty increases for longer-term predictions.

---

## Data Files

### Production Outputs

**`leaf_v2_game_by_game_[DATE].csv`**
- Full game-by-game ratings for all QBs
- 6,089 records (2016-2024)
- Key columns:
  - `passer_player_id`, `passer_player_name`
  - `season`, `week`, `game_number`
  - `opp_adj_base_epa_kalman` (LEAF rating)
  - `opp_adj_base_epa_uncertainty`
  - Context-adjusted metrics, opponent info

**`leaf_v2_season_ratings_[DATE].csv`**
- Final rating for each QB-season
- 1,010 QB-season records
- Key columns:
  - `player_id`, `season`, `player_name`
  - `leaf_rating` (final rating for season)
  - `leaf_uncertainty`, `games_played`, `attempts`

**`leaf_v2_current_ratings_[DATE].csv`**
- Most recent rating for each QB
- 373 QBs total (145 with 50+ attempts)
- Sorted by rating (best to worst)
- Key columns:
  - `player_id`, `player_name`
  - `leaf_rating`, `leaf_uncertainty`
  - `last_season`, `last_week`
  - `total_games`, `total_attempts`

**`leaf_v2_metadata_[DATE].json`**
- Deployment metadata
- Performance metrics (correlation, RMSE)
- Component flags (context, opponent, Kalman, injury)

---

## System Performance

### Validation Results

**Holdout Correlation**: 0.8951
- 2023 predictions based on 2016-2022 training data
- 170% improvement over raw EPA (0.331 → 0.895)

**RMSE**: 0.0812
- Average prediction error: 0.081 EPA per play

**Components**:
- Context adjustments: +1.08% improvement
- Opponent adjustments: +168% improvement (0.331 → 0.886)
- Kalman filtering: +3.5x improvement vs. simple average
- Injury tracking: Risk-adjusted uncertainty

---

## Customization

### Change Seasons

Deploy different seasons:
```bash
python deploy_leaf_v2.py --seasons 2022 2023 2024
```

### Enable Injury Tracking

Requires manual CSV injury data in `data/injuries/`:
```bash
python deploy_leaf_v2.py --seasons 2024 --enable-injury-tracking
```

See [data/injuries/README.md](data/injuries/README.md) for CSV format.

### Change QB Filter

Edit [visualize_leaf_ratings.py](visualize_leaf_ratings.py):
```python
# Line 329: Change minimum attempts threshold
qb_data = filter_to_qbs(current_data, min_attempts=100)  # Stricter filter
```

### Adjust Prediction Model

Edit [visualize_leaf_ratings.py](visualize_leaf_ratings.py) in `calculate_predictions()`:
```python
# Change age decline rates
age_decline_per_year = 0.020  # More aggressive decline

# Change regression to mean
regression_factor = 0.15 * years  # Stronger regression
```

---

## Documentation

### Full Documentation

- **[LEAF_V2_DOCUMENTATION.md](docs/LEAF_V2_DOCUMENTATION.md)** - Complete system documentation
- **[VISUALIZATION_APP.md](docs/VISUALIZATION_APP.md)** - Visualization app guide
- **[POSITION_SPECIFIC_RATINGS_ROADMAP.md](docs/POSITION_SPECIFIC_RATINGS_ROADMAP.md)** - Future WR/RB ratings

### Technical Details

- **[PHASE_2_INJURY_TRACKING.md](docs/PHASE_2_INJURY_TRACKING.md)** - Injury system
- **[POSITION_METRICS.md](docs/POSITION_METRICS.md)** - Position-specific metrics
- **[IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md)** - Project status

---

## Troubleshooting

### Missing Data Error

**Error**: `FileNotFoundError: No game-by-game ratings file found`

**Solution**: Deploy ratings first:
```bash
python deploy_leaf_v2.py --seasons 2016 2017 2018 2019 2020 2021 2022 2023 2024
```

---

### Import Error

**Error**: `ModuleNotFoundError: No module named 'dash'`

**Solution**: Install visualization dependencies:
```bash
pip install -r requirements_viz.txt
```

---

### Port Already in Use

**Error**: `Address already in use: Port 8050`

**Solution**: Kill existing process or change port:
```python
# In visualize_leaf_ratings.py, line 480
app.run(debug=True, host='0.0.0.0', port=8051)  # Use 8051
```

---

## Next Steps

### Current Status (Complete)
- ✅ QB LEAF v2.0 deployed (2016-2024)
- ✅ Visualization app with predictions
- ✅ 145 QBs with game-by-game data

### Future Enhancements (Roadmap)
- ⏳ WR/TE LEAF rating system
- ⏳ RB LEAF rating system
- ⏳ Defensive unit ratings
- ⏳ Player comparison mode in app
- ⏳ Export/sharing functionality

See [POSITION_SPECIFIC_RATINGS_ROADMAP.md](docs/POSITION_SPECIFIC_RATINGS_ROADMAP.md) for details.

---

## Summary

**LEAF v2.0** is a production-ready QB rating system with:
- 0.895 correlation (state-of-the-art performance)
- Game-by-game ratings from 2016-2024
- Interactive visualization app
- Future predictions (1-3 years)

**To Get Started**:
1. Deploy ratings: `python deploy_leaf_v2.py --seasons 2016 2017 2018 2019 2020 2021 2022 2023 2024`
2. Install viz deps: `pip install -r requirements_viz.txt`
3. Launch app: `python visualize_leaf_ratings.py`
4. Open browser: http://localhost:8050

**Documentation**: See `docs/` folder for complete guides.

---

**Built with nflfastR data | LEAF v2.0 | 2016-2024 NFL seasons**
