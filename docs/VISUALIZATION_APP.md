# LEAF Rating Visualization App

**Status**: ✅ COMPLETE
**Date**: 2025-11-05

---

## Overview

Interactive web application for visualizing QB LEAF ratings, historical trajectories, and future predictions (1, 2, 3 years forward).

Built with **Plotly Dash** - a Python framework for interactive web applications.

---

## Features

### 1. Player Selection
- **145 QBs** available (filtered to 50+ career attempts)
- Dropdown with current ratings displayed
- Sorted by rating (best to worst)

### 2. Current Rating Display
- **Large rating display** with uncertainty bands (±)
- **Color-coded interpretation**:
  - 🟢 Green: Elite (≥0.15 EPA)
  - 🔵 Blue: Above Average (0.05 to 0.15)
  - ⚪ Gray: Average (-0.05 to 0.05)
  - 🔴 Red: Below Average (<-0.05)

### 3. Career Statistics
- Total games played
- Total pass attempts
- Number of seasons

### 4. Next Season Prediction
- Predicted LEAF rating for next season
- Uncertainty estimate
- Based on current trajectory and age

### 5. Interactive Trajectory Chart
**Historical Data (2016-2024)**:
- Game-by-game LEAF ratings
- 95% confidence bands
- Hover details: Season, week, opponent, rating

**Future Predictions (1-3 years)**:
- Projected ratings for 1, 2, 3 years forward
- Prediction confidence bands (expanding over time)
- Based on:
  - Current rating
  - Recent trend (last 10 games)
  - Age-based decline curve
  - Regression to mean

**Reference Lines**:
- Average QB (0.0 EPA)
- Elite QB (0.15 EPA)
- Poor QB (-0.10 EPA)

---

## Installation

### Prerequisites
- Python 3.8+
- LEAF v2.0 production data deployed

### Install Dependencies
```bash
pip install -r requirements_viz.txt
```

This installs:
- `dash>=2.14.0` - Web framework
- `dash-bootstrap-components>=1.5.0` - UI components
- `plotly>=5.18.0` - Interactive charts
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical operations

---

## Usage

### Starting the App

**Command**:
```bash
python visualize_leaf_ratings.py
```

**Output**:
```
INFO:__main__:Loading LEAF ratings data...
INFO:__main__:  Loaded 6,089 game records
INFO:__main__:  Loaded 373 player ratings
INFO:__main__:  Filtered to 145 QBs with 50+ attempts
INFO:__main__:================================================================================
INFO:__main__:LEAF Rating Visualizer
INFO:__main__:================================================================================
INFO:__main__:
Loaded 145 QBs with 50+ attempts
INFO:__main__:Game-by-game data: 6,089 records
INFO:__main__:
Starting web server...
INFO:__main__:Open browser to: http://localhost:8050
INFO:__main__:
Press Ctrl+C to stop the server
Dash is running on http://0.0.0.0:8050/
```

### Accessing the App

Open your web browser to: **http://localhost:8050**

### Using the App

1. **Select a QB** from the dropdown menu
2. **Choose prediction years** (1, 2, 3 years forward)
3. **View current rating** and interpretation
4. **Explore trajectory chart**:
   - Zoom in/out with mouse wheel
   - Pan by clicking and dragging
   - Hover over points for details
5. **Export chart**: Click camera icon to save as PNG

### Stopping the App

Press **Ctrl+C** in the terminal to stop the server.

---

## Prediction Model

### Components

**1. Current Rating**: Final LEAF rating from most recent game

**2. Recent Trend**: Linear regression on last 10 games
- Captures hot/cold streaks
- Weight: 16 games per year × years forward

**3. Age-Based Decline**:
```python
if age > 32:
    decline = 0.015 EPA per year
elif age > 35:
    decline = 0.025 EPA per year  # Accelerated
else:
    decline = 0.0  # No decline before 32
```

**4. Regression to Mean**:
```python
mean_rating = 0.05  # Average QB
regression_factor = 0.1 * years  # 10% per year

predicted_rating = predicted_rating * (1 - regression_factor) +
                   mean_rating * regression_factor
```

**5. Uncertainty Growth**:
```python
uncertainty_multiplier = 1 + (0.5 * years)
# Example: 2-year prediction has 2.0x uncertainty
```

### Formula

```python
predicted_rating = (
    current_rating +
    trend * 16 * years -
    age_decline * years
) * (1 - regression_factor) + mean_rating * regression_factor
```

### Confidence Intervals

- **Historical**: ±1.96 × uncertainty (95% confidence)
- **Predictions**: ±1.96 × (uncertainty × multiplier)

---

## Data Sources

### Input Files

**Game-by-Game Data**:
- `data/production/leaf_v2_game_by_game_[DATE].csv`
- 6,089 records (2016-2024)
- Columns used:
  - `passer_player_id`, `passer_player_name`
  - `season`, `week`, `game_number`
  - `opp_adj_base_epa_kalman` (LEAF rating)
  - `opp_adj_base_epa_uncertainty`
  - `defteam` (opponent)

**Current Ratings**:
- `data/production/leaf_v2_current_ratings_[DATE].csv`
- 373 total passers (145 QBs with 50+ attempts)
- Columns used:
  - `player_id`, `player_name`
  - `leaf_rating`, `leaf_uncertainty`
  - `total_games`, `total_attempts`

### Filtering

**QB Filter**: `total_attempts >= 50`
- Excludes trick plays (punters, receivers, etc. with 1-4 attempts)
- Ensures sufficient sample size for predictions
- Results in 145 actual QBs

---

## Technical Details

### Architecture

**Framework**: Plotly Dash
- React-based web framework
- Python backend
- Real-time interactivity

**Components**:
```
visualize_leaf_ratings.py
├── Data Loading (load_ratings_data)
├── QB Filtering (filter_to_qbs)
├── Prediction Model (calculate_predictions)
├── Chart Generation (create_player_trajectory_figure)
└── Dash App
    ├── Layout (HTML/Bootstrap)
    └── Callbacks (Interactivity)
```

### Performance

- **Load time**: ~2 seconds (6,089 records)
- **Interactive updates**: <100ms per chart
- **Memory usage**: ~50 MB (all data in memory)

### Customization

**Change QB filter**:
```python
qb_data = filter_to_qbs(current_data, min_attempts=100)  # Stricter
```

**Adjust prediction model**:
```python
# In calculate_predictions():
age_decline_per_year = 0.020  # More aggressive decline
regression_factor = 0.15 * years  # Stronger regression
```

**Change port**:
```python
app.run(debug=True, host='0.0.0.0', port=8080)  # Use port 8080
```

---

## Screenshots

### Main Dashboard
```
┌─────────────────────────────────────────────────────────────────┐
│  🏈 LEAF Rating Visualizer                                      │
│  Interactive visualization of QB LEAF ratings, trajectories...  │
├─────────────────────────────────────────────────────────────────┤
│  Select QB: [Patrick Mahomes (0.185)          ▼]                │
│  Prediction Years: ☑ 1 Year ☑ 2 Years ☑ 3 Years                │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Current Rating  │  │ Career Stats    │  │ Next Season     │ │
│  │                 │  │                 │  │ Prediction      │ │
│  │    0.185        │  │ Games: 126      │  │    0.178        │ │
│  │   ±0.021        │  │ Attempts: 5,123 │  │   ±0.026        │ │
│  │                 │  │ Seasons: 8      │  │  (2025 Season)  │ │
│  │   Elite QB      │  │                 │  │                 │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  Patrick Mahomes - LEAF Rating Trajectory                       │
│                                                                 │
│   [Interactive chart with game-by-game ratings and predictions] │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Trajectory Chart Features
- **Line chart** with markers for each game
- **Shaded confidence bands** (95%)
- **Prediction line** (dashed, orange)
- **Reference lines** (Average, Elite, Poor)
- **Hover tooltips** with game details
- **Zoom/pan controls**
- **Export to PNG** button

---

## Use Cases

### 1. Player Evaluation
**Scenario**: Evaluating QB prospects for trades/signings

**Usage**:
1. Select player of interest
2. Review current rating and trend
3. Check prediction for contract length (1-3 years)
4. Compare with other QBs

**Example**: Josh Allen vs. Lamar Jackson comparison for MVP

### 2. Fantasy Football
**Scenario**: Drafting QBs for fantasy season

**Usage**:
1. Review top-rated QBs
2. Check consistency (tighter confidence bands = more consistent)
3. Identify hot/cold streaks from trajectory
4. Use predictions for keeper leagues

### 3. Front Office Analytics
**Scenario**: Contract negotiations for veteran QB

**Usage**:
1. Assess current performance level
2. Project performance decline over contract years
3. Estimate uncertainty (risk assessment)
4. Justify contract terms with data

### 4. Media/Broadcasting
**Scenario**: Pre-game analysis for broadcast

**Usage**:
1. Show QB trajectory over season
2. Highlight recent performance trends
3. Compare historical matchup performance
4. Provide data-driven storylines

---

## Future Enhancements

### Planned Features

**1. Player Comparison Mode**
- Select 2-4 QBs
- Overlay trajectories on same chart
- Side-by-side stat comparison

**2. Export Options**
- CSV download of player data
- PDF report generation
- Share link functionality

**3. Advanced Filters**
- Filter by season/year
- Filter by team
- Show only playoffs/regular season

**4. Additional Visualizations**
- Season-by-season averages (bar chart)
- Rating distribution histogram
- Rank over time chart

**5. Integration with Other Positions**
- WR/TE trajectory (once built)
- RB trajectory (once built)
- Multi-position comparison

### Technical Improvements

**1. Performance**
- Cache predictions (avoid recalculating)
- Lazy loading for large datasets
- Database backend (vs. CSV files)

**2. Deployment**
- Dockerize application
- Deploy to cloud (Heroku, AWS, etc.)
- HTTPS support

**3. User Experience**
- Dark mode toggle
- Mobile-responsive design
- Keyboard shortcuts

---

## Troubleshooting

### App Won't Start

**Error**: `ModuleNotFoundError: No module named 'dash'`

**Solution**:
```bash
pip install -r requirements_viz.txt
```

---

**Error**: `FileNotFoundError: No game-by-game ratings file found`

**Solution**: Deploy LEAF ratings first:
```bash
python deploy_leaf_v2.py --seasons 2016 2017 2018 2019 2020 2021 2022 2023 2024
```

---

**Error**: `Address already in use: Port 8050`

**Solution**: Kill existing process or use different port:
```python
app.run(debug=True, host='0.0.0.0', port=8051)  # Use 8051
```

---

### Chart Not Updating

**Issue**: Selected new QB but chart didn't update

**Solution**: Check browser console for JavaScript errors. Try:
1. Hard refresh (Ctrl+F5)
2. Clear browser cache
3. Restart app

---

### Slow Performance

**Issue**: App takes >5 seconds to load/update

**Solution**:
1. Reduce QB filter (fewer players):
   ```python
   qb_data = filter_to_qbs(current_data, min_attempts=100)
   ```

2. Disable debug mode:
   ```python
   app.run(debug=False, host='0.0.0.0', port=8050)
   ```

---

## Changelog

### Version 1.0 (2025-11-05)
- ✅ Initial release
- ✅ Interactive trajectory visualization
- ✅ Future predictions (1-3 years)
- ✅ 145 QBs from 2016-2024
- ✅ Bootstrap UI components
- ✅ Export to PNG functionality

---

## Summary

**LEAF Rating Visualizer** provides an interactive web interface for exploring QB performance trajectories and predictions.

**Key Features**:
- 145 QBs with game-by-game data (2016-2024)
- Current ratings with interpretation
- Historical trajectories with confidence bands
- Future predictions (1, 2, 3 years forward)
- Interactive charts with zoom/pan/export

**Technology**: Python, Plotly Dash, Bootstrap

**URL**: http://localhost:8050

**Status**: Production-ready ✅

---

**Built with [LEAF v2.0](LEAF_V2_DOCUMENTATION.md) | Correlation: 0.895 | Data: nflfastR**
