# Injury Data Format

## CSV Format

The injury tracking system supports manual CSV uploads with the following format:

### Required Columns

- `player_id`: Player identifier (e.g., '00-0033873' for Patrick Mahomes)
- `player_name`: Player name (e.g., 'P.Mahomes')
- `team`: Team abbreviation (e.g., 'KC')
- `status`: Injury status (Healthy, Questionable, Doubtful, Out, IR, PUP)
- `injury_type`: Type of injury (e.g., 'Ankle', 'Shoulder', 'Concussion')
- `week`: NFL week number (1-18)
- `season`: Season year (e.g., 2024)

### Example CSV

```csv
player_id,player_name,team,status,injury_type,week,season
00-0033873,P.Mahomes,KC,Questionable,Ankle,10,2024
00-0036442,J.Allen,BUF,Healthy,,10,2024
00-0033077,J.Hurts,PHI,Out,Shoulder,10,2024
```

### Injury Status Levels

1. **Healthy**: No injury concern
2. **Questionable**: Uncertainty multiplier of 1.10x
3. **Doubtful**: Uncertainty multiplier of 1.25x
4. **Out**: Rating set to 0.0 (not playing)
5. **IR** (Injured Reserve): Rating set to 0.0
6. **PUP** (Physically Unable to Perform): Rating set to 0.0

### File Naming Convention

- Format: `injuries_{season}_week_{week}.csv`
- Example: `injuries_2024_week_10.csv`

## Data Sources

If you want to populate this manually, check:

1. **NFL.com Injury Report**: https://www.nfl.com/injuries/
2. **ESPN Injury Report**: https://www.espn.com/nfl/injuries
3. **Pro Football Reference**: https://www.pro-football-reference.com/
4. **Team websites**: Each team publishes official injury reports

## Usage in LEAF Pipeline

```python
from src.features.injury_tracking import InjuryTrackingSystem

# Initialize with manual data source
injury_system = InjuryTrackingSystem(data_source="manual")

# Fetch injuries for a specific week
injuries = injury_system.fetcher.fetch_weekly_injuries(week=10, season=2024)

# Adjust ratings for injuries
adjusted_ratings = injury_system.adjust_ratings_with_injuries(
    ratings=leaf_ratings,
    week=10,
    season=2024
)
```
