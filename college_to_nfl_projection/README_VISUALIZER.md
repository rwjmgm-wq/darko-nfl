# College QB to NFL Projection Visualizer

Interactive web application to explore college QB statistics and NFL projection models for 395 quarterbacks from the 2007-2026 draft classes.

## Quick Start

### Option 1: Double-click the batch file (Windows)
Simply double-click **`start_server.bat`** and your browser will open automatically.

### Option 2: Run the Python script (All platforms)
```bash
python start_server.py
```

### Option 3: Manual server start
```bash
python -m http.server 8000
```
Then open your browser to: http://localhost:8000/college_qb_explorer.html

## What's Included

### QB Database (395 Total)
- **2007-2022**: 120 QBs with complete NFL career outcomes
- **2023**: 13 QBs (2nd year players with actual data)
- **2024**: 9 QBs (rookies with actual 2024 season performance)
  - Jayden Daniels, Bo Nix, Caleb Williams, Drake Maye, Spencer Rattler, etc.
- **2025**: 2 QBs (college prospects)
- **2026**: 240 QBs (college prospects including Carson Beck, Garrett Nussmeier, etc.)

### Features
- **Search**: Find any QB by name
- **College Stats**: EPA/play, attempts, big play rate, etc.
- **NFL Projections**: Outcome probabilities (Elite, Solid Starter, Journeyman, Bust)
- **Career Trajectories**: Projected year-by-year LEAF performance
- **Actual Performance**: Real NFL data for historical QBs and 2024 rookies
- **Comparison**: See projected vs. actual rookie performance for 2024 class

### 2024 Rookie Highlights
- **Jayden Daniels**: Projected -0.400 → **Actual +4.492** (Rookie of the Year performance!)
- **Bo Nix**: Projected -0.194 → **Actual +1.553** (Exceeded expectations)
- **Caleb Williams**: Projected -0.655 → Actual -2.577 (Below projection)
- **Drake Maye**: Projected -0.667 → Actual -2.124 (Below projection)

### Top 2026 Prospects Available
- Carson Beck (Georgia)
- Garrett Nussmeier (LSU)
- Cade Klubnik (Clemson)
- Jaxson Dart (Ole Miss)
- Drew Allar (Penn State)
- Quinn Ewers (Texas)
- Shedeur Sanders (Colorado)
- Dylan Raiola (Nebraska)
- And 232 more...

## Data Files

The visualizer loads data from:
- **`data/projections/all_projections.json`** - All QB projections and outcomes
- **`data/processed/nfl_outcomes_comprehensive.csv`** - Actual NFL performance data

## Updating Data

To update with the latest NFL performance data:

1. Update NFL outcomes:
```bash
python src/update_nfl_outcomes.py
```

2. Rebuild projections:
```bash
python src/build_projection_database.py
```

3. Refresh your browser to see the updated data

## Technical Details

- Built with vanilla JavaScript and Chart.js
- No backend required - runs entirely in the browser
- Data stored in JSON format for fast loading
- Responsive design works on mobile and desktop

## Troubleshooting

**Error: "Error loading projection data"**
- Make sure you're running the local server (not opening the HTML file directly)
- Check that `data/projections/all_projections.json` exists
- Verify the server is running on port 8000

**Port 8000 already in use**
- Close any other applications using port 8000
- Or edit the port number in `start_server.py` or `start_server.bat`

## Model Information

The projection model uses:
- **College EPA/play** (competition-adjusted with SP+)
- **Attempt volume**
- **Big play rate** (15+ yard completions)
- **High leverage EPA** (3rd/4th down and red zone)
- **Success rate**

Trained on 120+ QBs from 2007-2022 with complete NFL careers.

Model Performance:
- **Outcome Classification**: 75% AUC
- **Rookie LEAF Prediction**: 18.3% R²

---

Created with data from:
- **College**: sportsdataverse & CFBD API
- **NFL**: nfl_data_py
- **Competition Adjustments**: Bill Connelly's SP+ ratings
