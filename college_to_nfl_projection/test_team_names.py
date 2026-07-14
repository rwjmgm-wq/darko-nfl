"""Test to find Houston team name in sportsdataverse data"""

from sportsdataverse.cfb import load_cfb_pbp
import pandas as pd

print("Loading 2006 season...")
df = load_cfb_pbp(seasons=[2006], return_as_pandas=True)
print(f"Total plays: {len(df):,}")

# Check Houston
houston_home = df[df['homeTeamName'].str.contains('Houston', case=False, na=False)]
houston_away = df[df['awayTeamName'].str.contains('Houston', case=False, na=False)]

print(f"\nHouston as home: {len(houston_home)} plays")
print(f"Houston as away: {len(houston_away)} plays")

if len(houston_home) > 0:
    print(f"Sample home team name: '{houston_home['homeTeamName'].iloc[0]}'")

if len(houston_away) > 0:
    print(f"Sample away team name: '{houston_away['awayTeamName'].iloc[0]}'")

# Check what passing plays look like
if len(houston_home) > 0 or len(houston_away) > 0:
    houston_plays = pd.concat([houston_home, houston_away])
    houston_passes = houston_plays[houston_plays['pass'] == 1]

    print(f"\nHouston passing plays: {len(houston_passes)}")

    if len(houston_passes) > 0:
        qbs = houston_passes['passer_player_name'].dropna().unique()
        print(f"QBs found: {list(qbs)}")
