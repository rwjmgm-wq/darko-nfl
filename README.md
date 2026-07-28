# DARKO NFL - Position-Specific Composite Rating System

A comprehensive NFL player evaluation system with game-by-game composite ratings for all major positions. Named after basketball's DARKO system (Darko Miličić), applying similar predictive modeling to football.

## Overview

This system generates **data-driven composite ratings** for NFL players across 7 positions:
- **Offense**: QB (LEAF), WR, RB, TE
- **Defense**: EDGE, DT (Interior DL), LB

Each position uses **exhaustively validated metrics** (3-5 per position) optimized to predict future performance.

## Key Features

### Position-Specific Composite Ratings
Each position has its own validated metric combination:

| Position | Metrics | Predictive Power (r) |
|----------|---------|---------------------|
| **QB (LEAF v3)** | EPA, CPOE, Success Rate | r=0.47 next-season, r=0.47 next-16-games (walk-forward, frozen 2019-2025 test era; see docs/LEAF_V3_RESULTS.md) |
| **WR*** | Receiving EPA, Targets, Air Yards, Receptions | r=0.692 |
| **RB** | Rushing EPA, Attempts, Receiving Targets | r=0.554 |
| **TE** | Receiving EPA, Targets, Receptions, Air Yards | r=0.621 |
| **EDGE** | Sacks, QB Hits, Pressures, Pass Rush Plays | r=0.714 |
| **DT** | Pressures, QB Hits, Impact Plays, Pass Rush Plays, EPA Against | r=0.637 |
| **LB*** | Sacks, QB Hits, EPA Against (Overall/Pass/Run) | r=0.705 |

*Non-QB correlations were produced by the same validation style as the retracted
QB r=0.906 and have not been re-audited; treat them as upper bounds until the
walk-forward harness (scripts/v3_honest/) is extended to those positions.*

### Interactive Visualizations
Each position includes an interactive web dashboard with:
- **Game-by-game trajectory tracking** (entire career)
- **EWMA smoothed ratings** with uncertainty bands
- **Data-driven predictions** (1, 2, 3 years forward)
- **Career stage adjustments** based on actual trajectory analysis

### Career Trajectory Modeling
- Position-specific improvement rates (early career, mid career, peak)
- Decline rates for veteran players
- Peak performance windows identified from historical data
- Uncertainty quantification with confidence intervals

## Quick Start

### Prerequisites
- Python 3.8+
- pip or conda package manager

### Installation
```bash
# Clone or download repository
cd DARKO_NFL

# Install dependencies
pip install -r requirements.txt
```

### Running Visualizers

Each position has its own interactive visualizer running on a different port:

```bash
# Quarterbacks (QB LEAF)
python scripts/visualization/visualize_leaf_ratings.py
# → http://localhost:8050

# Wide Receivers
python scripts/visualization/visualize_wr_ratings.py
# → http://localhost:8051

# Running Backs
python scripts/visualization/visualize_rb_ratings.py
# → http://localhost:8052

# Tight Ends
python scripts/visualization/visualize_te_ratings.py
# → http://localhost:8053

# EDGE Rushers
python scripts/visualization/visualize_edge_ratings.py
# → http://localhost:8054

# Defensive Tackles
python scripts/visualization/visualize_dt_ratings.py
# → http://localhost:8055

# Linebackers
python scripts/visualization/visualize_lb_ratings.py
# → http://localhost:8056
```

## Project Structure

```
DARKO_NFL/
├── scripts/
│   ├── generation/          # Generate composite ratings
│   │   ├── generate_qb_composite.py
│   │   ├── generate_wr_game_by_game.py
│   │   ├── generate_rb_game_by_game.py
│   │   ├── generate_te_game_by_game.py
│   │   ├── generate_edge_game_by_game.py
│   │   ├── generate_dt_game_by_game.py
│   │   └── generate_lb_game_by_game.py
│   ├── analysis/            # Career trajectory analysis
│   │   ├── analyze_qb_career_trajectories.py
│   │   ├── analyze_edge_career_trajectories.py
│   │   ├── analyze_dt_career_trajectories.py
│   │   └── analyze_lb_career_trajectories.py
│   ├── research/            # Metric validation & optimization
│   │   ├── qb_exhaustive_composite_search.py
│   │   ├── rb_exhaustive_composite_search.py
│   │   ├── te_exhaustive_composite_search.py
│   │   ├── edge_exhaustive_composite_search.py
│   │   ├── dt_exhaustive_composite_search.py
│   │   └── lb_exhaustive_composite_search.py
│   ├── visualization/       # Interactive dashboards
│   │   ├── visualize_leaf_ratings.py      # QB (port 8050)
│   │   ├── visualize_wr_ratings.py        # WR (port 8051)
│   │   ├── visualize_rb_ratings.py        # RB (port 8052)
│   │   ├── visualize_te_ratings.py        # TE (port 8053)
│   │   ├── visualize_edge_ratings.py      # EDGE (port 8054)
│   │   ├── visualize_dt_ratings.py        # DT (port 8055)
│   │   └── visualize_lb_ratings.py        # LB (port 8056)
│   ├── tests/               # Unit tests
│   └── legacy/              # Historical/deprecated scripts
├── src/
│   ├── data_pipeline/
│   │   ├── nfl_data_fetcher.py          # nflfastR data loading
│   │   ├── qb_stats_aggregator.py       # QB statistics
│   │   ├── offensive_stats_aggregator.py # WR/RB/TE statistics
│   │   └── defensive_stats_aggregator.py # EDGE/DT/LB statistics
│   └── utils/
│       └── config_loader.py
├── data/
│   ├── production/          # Final composite ratings (CSV)
│   ├── processed/           # Intermediate data
│   └── raw/                 # Cached nflfastR data
├── docs/
│   └── legacy/              # Historical documentation
├── config/
│   └── config.yaml          # Configuration settings
└── README.md                # This file
```

## Methodology

### Exhaustive Metric Search
For each position, we:
1. Identify all available metrics (10-20+ candidates per position)
2. Test all 3-5 metric combinations (thousands of combinations)
3. Measure predictive power: correlation between Year N composite → Year N+1 performance
4. Select the optimal combination that best predicts future success

### Composite Rating Calculation
For each game:
1. **Standardize** metrics within season (z-scores)
2. **Combine** via equal weighting (no overfitting)
3. **Smooth** with EWMA (span varies by position)
4. **Track** uncertainty (rolling standard deviation)

### Career Trajectory Adjustments
Position-specific adjustments based on career stage:
- **Early career** (Games 0-48): Linear improvement rates
- **Mid career** (Games 48-80): Peak performance period
- **Late career** (Games 80+): Decline rates

Example (EDGE rushers):
- Early: +0.128/season improvement
- Mid: +0.131/season improvement
- Peak: Games 64-71
- Decline: -0.15/season (assumed, limited data)

### Prediction Algorithm
Future ratings predicted using:
1. **Current rating** (EWMA smoothed)
2. **Career stage adjustment** (data-driven rates)
3. **Recent trend** (dampened 20%, max ±0.3/year)
4. **Regression to mean** (20% per year)
5. **Uncertainty growth** (40% per year)

## Data Sources

- **Primary**: nflfastR play-by-play data (2020-2024)
  - EPA, CPOE, Success Rate
  - Box score statistics
  - Defensive player tracking
- **Coverage limitation**: No coverage data available for DBs (CBs/Safeties)
  - Would require manual charting or ML from All-22 film
  - System focuses on front-7 positions where public data works

## Generating New Ratings

To regenerate composite ratings with latest data:

```bash
# Offense
python scripts/generation/generate_qb_composite.py
python scripts/generation/generate_wr_game_by_game.py
python scripts/generation/generate_rb_game_by_game.py
python scripts/generation/generate_te_game_by_game.py

# Defense
python scripts/generation/generate_edge_game_by_game.py
python scripts/generation/generate_dt_game_by_game.py
python scripts/generation/generate_lb_game_by_game.py
```

Output: `data/production/{position}_composite_game_by_game_{date}.csv`

## Research & Validation

All metric combinations were exhaustively tested. To re-run research:

```bash
# Run exhaustive search for a position
python scripts/research/qb_exhaustive_composite_search.py
python scripts/research/edge_exhaustive_composite_search.py
# etc.

# Analyze career trajectories
python scripts/analysis/analyze_qb_career_trajectories.py
python scripts/analysis/analyze_edge_career_trajectories.py
# etc.
```

## Position-Specific Notes

### QB (LEAF - Layered EPA Adaptive Framework)
- Honest walk-forward predictive power: r=0.47 next-season EPA (the previously
  quoted r=0.906 came from a validation where the Kalman predictor had already
  seen the test games - see docs/LEAF_V3_RESULTS.md)
- EPA dominates (65% weight in composite)
- Named after Ryan Leaf (QB bust, playful nod to DARKO)

### EDGE Rushers
- Highest defensive predictability (r=0.714)
- Pass rush metrics strongly stable year-to-year
- Clear career trajectory (peak ~64 games)

### DT (Interior DL)
- Lower predictability (r=0.637) due to variance
- Interior pressure harder to sustain
- Requires pass rush opportunity metrics

### LB (Linebackers)
- EPA-based composite captures versatility (r=0.705)
- Balances pass rush, run defense, coverage
- Tackles-only approach (r=0.729) too narrow

### Coverage Limitations
- **No CB/Safety systems** - public data lacks coverage metrics
- Only discrete events (PBU, INT), not targets allowed or coverage quality
- Would require proprietary data or ML from All-22 film

## Output Files

Each position generates:
- **{position}_composite_game_by_game_{date}.csv**: Game-by-game ratings
  - Columns: season, week, player_id, player_name, raw_composite, smoothed_composite, career_game_number, uncertainty

Example:
```csv
season,week,player_id,player_name,position,raw_composite,smoothed_composite,career_game_number,uncertainty
2024,1,00-0036355,Nick Bosa,EDGE,+1.856,+1.542,89,0.156
```

## Technologies

- **Python 3.8+**
- **nfl_data_py**: nflfastR data access
- **pandas/numpy**: Data processing
- **scipy**: Statistical analysis
- **plotly/dash**: Interactive visualizations
- **dash-bootstrap-components**: UI styling

## Future Enhancements

- [ ] Real-time updates during season
- [ ] Historical trend charts (multi-season)
- [ ] Player comparison tools
- [ ] Team-level aggregations
- [ ] Rookie projections (college→NFL translation)
- [ ] API for external access

## Contributing

This is a research project. Suggestions and feedback welcome via GitHub issues.

## References

- **DARKO (NBA)**: Original inspiration for methodology
- **nflfastR**: Primary data source
- **Open Source Football**: Analytics community
- Research on EPA, predictive modeling, career trajectories

## License

MIT License

## Acknowledgments

- **nflfastR team** for excellent open-source data
- **DARKO creators** for original methodology
- **NFL analytics community** for research insights

---

**Note**: For research and educational purposes. Not financial or betting advice.
