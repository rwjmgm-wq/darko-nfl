# Active CFB QB Projection System

This system identifies current college football QBs with 6+ starts, collects their full college careers, and generates NFL projections.

## Workflow

### Step 1: Identify Active QBs
```bash
python src/identify_active_cfb_qbs.py
```

This script:
- Loads 2024 CFB season play-by-play data
- Finds all QBs with 6+ games and 50+ attempts
- Calculates their 2024 stats (EPA/play, completion%, etc.)
- Saves list to `data/processed/active_cfb_qbs_2024.csv`
- Displays top 20 QBs by EPA/play

**Output**: List of ~50-100 active CFB QBs

### Step 2: Collect Full College Careers
```bash
python src/collect_active_qb_careers.py
```

This script:
- Reads the active QB list
- For each QB, downloads play-by-play data for last 5 seasons (2020-2024)
- Saves individual career files to `data/processed/active_qb_careers/`
- Skips QBs already collected (incremental updates)

**Output**: Career CSVs for each active QB

### Step 3: Generate NFL Projections
```bash
python src/project_active_qbs.py
```

This script:
- Loads historical training data (drafted QBs 2007-2023)
- Trains outcome and rookie LEAF models
- Processes each active QB's career data
- Calculates aggregated stats (EPA/play, success rate, etc.)
- Generates NFL projections
- Saves to `data/projections/active_cfb_projections.json`
- Displays top 10 prospects by Elite probability

**Output**: JSON file with projections for all active QBs

### Step 4: Add to Explorer (Optional)
To add active QBs to the web explorer, merge the projections:

```python
import json

# Load both projection files
with open('data/projections/all_projections.json') as f:
    drafted = json.load(f)

with open('data/projections/active_cfb_projections.json') as f:
    active = json.load(f)

# Combine
combined = drafted + active

# Save
with open('data/projections/all_projections.json', 'w') as f:
    json.dump(combined, f, indent=2)
```

Then restart the server and view active prospects in the explorer!

## Data Structure

### Active QB CSV Format
Each QB's career file contains play-by-play data:
- `season`: Year (2020-2024)
- `player_name`: QB name
- `college`: School
- `draft_year`: "Active" (not yet drafted)
- `epa`: Expected Points Added per play
- `success`: Success rate (binary)
- `yards_gained`: Yards on play
- `completion`: Completion (1/0)
- `touchdown`: TD (1/0)
- `interception`: INT (1/0)

### Projection JSON Format
```json
{
  "player_name": "QB Name",
  "draft_year": "Active 2024",
  "college": "School Name",
  "college_epa": 0.234,
  "college_attempts": 1349,
  "projected_outcome_probs": {
    "Elite": 0.058,
    "Solid Starter": 0.346,
    "Journeyman": 0.309,
    "Bust": 0.286
  },
  "projected_rookie_leaf": -0.170,
  "trajectories": { ... }
}
```

## Use Cases

1. **Pre-Draft Scouting**: Identify top QB prospects before they declare
2. **Transfer Portal**: Track QBs changing schools
3. **Season Updates**: Re-run monthly to update projections with latest stats
4. **Prospect Comparisons**: Compare active QBs to historical draft classes

## Notes

- QBs must have 6+ games and 50+ attempts to be included
- FCS schools may not have complete data in sportsdataverse
- Projections based on college stats only (no combine/measurables)
- Re-run Step 2-3 periodically to update with latest games
