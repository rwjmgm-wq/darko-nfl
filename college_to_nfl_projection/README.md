# College-to-NFL QB Projection Feasibility Study

## Project Overview

This project investigates whether college football performance data can predict NFL quarterback success, specifically:

1. **Rookie Season Performance**: Can college stats predict first-year NFL LEAF ratings?
2. **Career Longevity**: Can we predict which QBs will reach 85+ NFL starts?
3. **Peak Performance**: Can we project sustained elite/above-average play?

## Key Innovation

Leveraging an **existing ML-derived CFB competition strength model** to adjust college QB stats for opponent quality - solving the biggest challenge in college-to-NFL translation.

## Data Sources

### College Data (cfbfastr / CFBD API)
- Play-by-play data (EPA, success rate)
- Team statistics
- Schedule and opponent information

### Competition Adjustment (CFB Betting Model)
- ML-derived power rankings (-10 to +10 scale)
- Week-by-week team strength ratings (2020-2025)
- Strength of Schedule (SOS) calculations

### NFL Data (nflfastR / existing LEAF model)
- Rookie season LEAF ratings
- Career starts and longevity
- Peak performance metrics

## Project Structure

```
college_to_nfl_projection/
├── data/
│   ├── raw/               # Raw data from APIs
│   │   ├── drafted_qbs.csv
│   │   ├── college_pbp/   # College play-by-play
│   │   └── nfl_rookies/   # NFL rookie data
│   └── processed/         # Processed datasets
│       ├── college_qb_stats.csv
│       ├── nfl_rookie_ratings.csv
│       └── merged_dataset.csv
├── src/
│   ├── collect_draft_data.py      # Identify QBs drafted 2015-2024
│   ├── fetch_college_stats.py     # Pull cfbd data
│   ├── extract_nfl_ratings.py     # Get rookie LEAF ratings
│   ├── apply_competition_adj.py   # Use CFB power ratings
│   └── feasibility_analysis.py    # Main correlation study
├── notebooks/
│   └── exploratory_analysis.ipynb
├── results/
│   └── feasibility_report.md
└── README.md
```

## Methodology

### Phase 1: Feasibility Study (Current)

1. **Sample Identification**
   - All QBs drafted rounds 1-7 (2015-2024)
   - Filter: Minimum 300 college pass attempts

2. **Baseline Correlation**
   - Raw college EPA → Rookie NFL LEAF rating
   - Target: r > 0.30 to proceed

3. **Competition-Adjusted Metrics**
   - Weight college performance by opponent strength
   - Apply power ratings from CFB betting model
   - Calculate adjusted EPA, success rate

4. **Comparison**
   - Raw correlation vs. adjusted correlation
   - Does competition adjustment improve prediction?

### Phase 2: College LEAF Model (If feasible)

Adapt NFL LEAF framework for college data:
- Opponent adjustments (using power rankings)
- Context adjustments (weather, game script, situations)
- Kalman filtering (higher initial uncertainty)
- Game-by-game tracking

### Phase 3: NFL Translation Model

Build regression models:
- **Input**: College LEAF + draft capital + physical traits
- **Target**: NFL outcomes (rookie rating, career starts, peak rating)

## Success Criteria

- **Minimum**: Raw college EPA correlation r > 0.30 with rookie NFL LEAF
- **Target**: Competition-adjusted correlation r > 0.45
- **Stretch**: Prediction better than draft position alone

## Timeline

- **Week 1-2**: Feasibility study (correlation analysis)
- **Week 3-4**: Build College LEAF v1.0
- **Week 5-6**: NFL translation model
- **Week 7**: Validation and reporting

## Dependencies

```
pandas
numpy
scikit-learn
cfbd  # College Football Data API
nfl-data-py  # NFL data (already installed)
matplotlib
seaborn
```

## Usage

```bash
# Step 1: Collect draft data
python src/collect_draft_data.py

# Step 2: Fetch college stats
python src/fetch_college_stats.py --seasons 2014-2023

# Step 3: Extract NFL rookie ratings
python src/extract_nfl_ratings.py

# Step 4: Run feasibility analysis
python src/feasibility_analysis.py --output results/feasibility_report.md
```

## Notes

- College PBP data availability: 2014+ (via CFBD)
- CFB power rankings availability: 2020-2025 (may need training for earlier years)
- QBs drafted 2015-2024 gives us ~10 draft classes (~30-40 QBs)
