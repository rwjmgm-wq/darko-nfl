"""Test full collection workflow for Kevin Kolb"""

import sys
sys.path.append('src')

from collect_college_careers_sportsdataverse import collect_qb_career

# Test Kevin Kolb
player_name = "Kevin Kolb"
draft_year = 2007
college = "Houston"

print(f"Testing full collection for: {player_name}")
print(f"College: {college}, Draft: {draft_year}")
print("="*60)

career_data = collect_qb_career(player_name, draft_year, college)

if len(career_data) > 0:
    print(f"\nSUCCESS! Collected {len(career_data)} plays")
    print(f"\nSeasons: {sorted(career_data['season'].unique())}")
    print(f"Total passing plays: {len(career_data)}")

    # Show summary by season
    print("\nBreakdown by season:")
    for season in sorted(career_data['season'].unique()):
        season_plays = career_data[career_data['season'] == season]
        avg_epa = season_plays['epa'].mean()
        print(f"  {season}: {len(season_plays)} plays, EPA/play = {avg_epa:+.3f}")

    # Show sample plays
    print("\nSample plays:")
    print(career_data[['season', 'down', 'distance', 'yards_gained', 'epa']].head(5).to_string(index=False))
else:
    print("\nFAILED - No plays collected")
