# DARKO NFL - Quick Start Guide

Get up and running with DARKO NFL in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Internet connection (to download NFL data)

## Installation

### 1. Set up Python environment

```bash
# Navigate to the project directory
cd DARKO_NFL

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

This will install:
- nfl_data_py (nflfastR data)
- pandas, numpy (data processing)
- scikit-learn, statsmodels (modeling)
- matplotlib, seaborn (visualization)
- and more...

## Running Your First Projection

### Option 1: Command Line (Fastest)

Run the complete system with default settings:

```bash
python main.py
```

This will:
1. Download 2022-2024 NFL data (~2-3 minutes first time, instant after caching)
2. Calculate QB statistics
3. Generate projections with confidence intervals
4. Output rankings to `results/` directory

**View results:**
```bash
# Open the output files in results/
# - epa_per_play_rankings.csv
# - cpoe_rankings.csv
# - success_rate_rankings.csv
```

### Option 2: Jupyter Notebook (Interactive)

For interactive exploration:

```bash
# Install Jupyter if not already installed
pip install jupyter

# Launch Jupyter
jupyter notebook

# Open notebooks/example_usage.ipynb
```

Follow the notebook cells to:
- Fetch and explore data
- Calculate time-weighted metrics
- Fit age curves
- Generate and visualize projections

## Understanding the Output

### Rankings Files

Each ranking file contains:

| Column | Description |
|--------|-------------|
| rank | Overall rank (1 = best) |
| player_name | QB name |
| age | Current age |
| total_attempts | Total pass attempts (sample size) |
| projection | Projected metric value |
| lower_ci | Lower 95% confidence bound |
| upper_ci | Upper 95% confidence bound |
| uncertainty | low/medium/high |

### Key Metrics

- **EPA per play**: Expected Points Added per play (0.25+ is elite)
- **CPOE**: Completion % Over Expected (3+ is very good)
- **Success Rate**: % of positive EPA plays (52%+ is good)

## Customization

### Analyze Different Years

```bash
python main.py --years 2023 2024
```

### Change Minimum Attempts Threshold

```bash
python main.py --min-attempts 200
```

### Force Data Refresh

```bash
python main.py --refresh
```

## Configuration

Edit `config/config.yaml` to customize:

```yaml
# Exponential decay rates (higher = slower decay)
decay:
  epa_beta: 0.996    # Try 0.99 for more emphasis on recent games

# Bayesian prior strength (lower = less regression to mean)
priors:
  prior_weight: 200  # Try 100 for less regression

# Age curve settings
age_curves:
  peak_age: 28       # When QBs typically peak
```

## Common Issues

### Installation Problems

**Issue**: `pip install` fails
```bash
# Try upgrading pip first
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Issue**: Missing dependencies
```bash
# Install build tools (Windows)
pip install wheel setuptools

# Try again
pip install -r requirements.txt
```

### Data Download Issues

**Issue**: nflfastR data download fails
- Check internet connection
- Try again (sometimes server is temporarily unavailable)
- Clear cache: delete files in `data/cache/` and rerun

### Memory Issues

**Issue**: Out of memory with many years of data
```bash
# Analyze fewer years at a time
python main.py --years 2024
```

## Next Steps

### 1. Explore the Data

```python
# In Python or Jupyter
from src.data_pipeline.nfl_data_fetcher import NFLDataFetcher

fetcher = NFLDataFetcher()
qb_pbp = fetcher.get_qb_play_by_play([2024])

# Explore available columns
print(qb_pbp.columns.tolist())

# Look at a specific QB
mahomes = qb_pbp[qb_pbp['passer_player_name'] == 'P.Mahomes']
print(mahomes[['game_date', 'qb_epa', 'cpoe', 'complete_pass']].head())
```

### 2. Customize Projections

See [README.md](README.md) for:
- Adding college football data for rookie projections
- Integrating PFF data
- Customizing age curves
- Adding context adjustments

### 3. Build Dashboards

Use the projection output to:
- Create visualizations
- Build interactive dashboards (Streamlit, Dash)
- Track projections over time
- Compare to Vegas lines

## Example Workflow

Here's a typical analysis workflow:

```bash
# 1. Get latest data
python main.py --years 2024 --refresh

# 2. Review top QBs
cat results/epa_per_play_rankings.csv | head -20

# 3. Explore interactively
jupyter notebook notebooks/example_usage.ipynb

# 4. Customize config
nano config/config.yaml

# 5. Rerun with custom settings
python main.py --years 2023 2024
```

## Getting Help

- **Documentation**: See [README.md](README.md) for detailed information
- **Examples**: Check [notebooks/example_usage.ipynb](notebooks/example_usage.ipynb)
- **Configuration**: Review [config/config.yaml](config/config.yaml)

## Performance Tips

1. **Use caching**: Don't use `--refresh` unless necessary
2. **Filter by attempts**: Use `--min-attempts` to focus on qualified QBs
3. **Analyze recent years**: Fewer years = faster processing
4. **Save intermediate results**: The system saves game stats to `data/processed/`

## What's Next?

Now that you have basic projections running, consider:

1. **Phase 2 Features** (from README):
   - Add Kalman filtering for smoother updates
   - Integrate opponent adjustments
   - Add weather/context factors

2. **Rookie Projections**:
   - Add your College Football Data API key to `.env`
   - Build college-to-NFL translation model

3. **Custom Analysis**:
   - Situational splits (3rd down, red zone)
   - Playoff projections
   - Contract value analysis

Happy projecting! 🏈📊
