"""
Quick test to verify sportsdataverse collection works for a single QB
"""

from sportsdataverse.cfb import load_cfb_pbp
import pandas as pd

# Test with Kevin Kolb (Houston, 2007 draft)
player_name = "Kevin Kolb"
college = "Houston"
draft_year = 2007

# Test his final season (2006)
test_season = 2006

print(f"Testing: {player_name} ({college}, {draft_year})")
print(f"Loading {test_season} season data...")

# Load play-by-play
pbp_data = load_cfb_pbp(seasons=[test_season], return_as_pandas=True)
print(f"Loaded {len(pbp_data):,} total plays")

# Filter to Houston's games
team_plays = pbp_data[
    (pbp_data['pos_team'] == college) |
    (pbp_data['def_pos_team'] == college)
]
print(f"Houston plays: {len(team_plays):,}")

# Filter to passing plays
passing_plays = team_plays[team_plays['pass'] == 1]
print(f"Houston passing plays: {len(passing_plays):,}")

# Check for Kolb
if len(passing_plays) > 0:
    qb_names = passing_plays['passer_player_name'].dropna().unique()
    print(f"\nQBs found: {list(qb_names)[:10]}")

    # Filter to Kolb
    kolb_plays = passing_plays[
        passing_plays['passer_player_name'].str.contains('Kolb', case=False, na=False)
    ]

    if len(kolb_plays) > 0:
        print(f"\n✓ Found {len(kolb_plays)} plays by Kolb!")

        # Show sample
        sample = kolb_plays[['passer_player_name', 'statYardage', 'EPA', 'scoringPlay']].head(3)
        print("\nSample plays:")
        print(sample)
    else:
        print("\n✗ No plays found for Kolb")
else:
    print("\n✗ No passing plays found")
