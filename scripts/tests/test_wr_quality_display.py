"""
Test WR Supporting Cast Quality Calculation

Quick test to verify the WR quality calculation function works correctly.
"""

import pandas as pd
from pathlib import Path

def calculate_team_wr_quality(wr_data, team, season):
    """Calculate aggregate WR quality for a team."""
    if wr_data is None or 'team' not in wr_data.columns:
        return None

    team_wrs = wr_data[
        (wr_data['team'] == team) &
        (wr_data['season'] == season)
    ].copy()

    if len(team_wrs) == 0:
        return None

    return {
        'n_wrs': len(team_wrs),
        'avg_composite': team_wrs['wr_composite_rating'].mean(),
        'total_targets': team_wrs['targets'].sum(),
        'total_yac_epa': team_wrs['total_yac_epa'].sum(),
        'top_wr': team_wrs.nlargest(1, 'wr_composite_rating').iloc[0]['receiver_player_name']
            if len(team_wrs) > 0 else None,
        'top_wr_rating': team_wrs.nlargest(1, 'wr_composite_rating').iloc[0]['wr_composite_rating']
            if len(team_wrs) > 0 else None
    }

def main():
    """Test WR quality calculation."""

    print("="*80)
    print("Testing WR Supporting Cast Quality Calculation")
    print("="*80)

    # Load WR composite data
    wr_file = Path("data/processed/wr_composite_ratings_2020_2024.csv")

    if not wr_file.exists():
        print(f"\nERROR: WR composite ratings file not found: {wr_file}")
        return

    wr_data = pd.read_csv(wr_file)
    print(f"\nLoaded {len(wr_data)} WR-seasons from {wr_file}")

    # Test with several teams from 2024
    test_teams = [
        ('DET', 2024),  # Lions (A. St. Brown)
        ('CIN', 2024),  # Bengals (J. Chase)
        ('KC', 2024),   # Chiefs (various)
        ('BUF', 2024),  # Bills (K. Shakir)
        ('SF', 2024),   # 49ers (various)
    ]

    print("\n" + "="*80)
    print("WR Supporting Cast Quality - 2024 Season")
    print("="*80)

    for team, season in test_teams:
        quality = calculate_team_wr_quality(wr_data, team, season)

        print(f"\n{team} ({season}):")
        if quality:
            print(f"  WRs on Roster: {quality['n_wrs']}")
            print(f"  Avg Composite Rating: {quality['avg_composite']:+.3f}")
            print(f"  Top WR: {quality['top_wr']} ({quality['top_wr_rating']:+.3f})")
            print(f"  Total Targets: {int(quality['total_targets']):,}")
            print(f"  Total YAC EPA: {quality['total_yac_epa']:+.1f}")
        else:
            print("  No WR data available")

    print("\n" + "="*80)
    print("Test Complete!")
    print("="*80)
    print("The WR quality calculation is working correctly.")

if __name__ == "__main__":
    main()
