"""
Generate Historical Power Rankings for QB Projection

This script batch-generates power rankings for 2020-2022 seasons
to enable competition adjustment analysis for QB college-to-NFL projections.
"""

import sys
from pathlib import Path

# Add CFB model to path
CFB_MODEL_PATH = Path(r"C:\Users\rwjmg\OneDrive\Pictures\Writing\new_cfb_betting_model")
sys.path.insert(0, str(CFB_MODEL_PATH))

from power_rankings.team_ratings import calculate_power_rankings
import json


def generate_season_rankings(season, final_week=15):
    """
    Generate power rankings for all weeks of a season

    Args:
        season: Season year (e.g., 2020)
        final_week: Last week to generate (default 15 for end of regular season)
    """
    print(f"\n{'='*80}")
    print(f"GENERATING POWER RANKINGS FOR {season} SEASON")
    print(f"{'='*80}")

    rankings_dir = CFB_MODEL_PATH / 'power_rankings' / 'historical_ratings'
    rankings_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0

    for week in range(1, final_week + 1):
        print(f"\n  Week {week:2d}... ", end='')

        try:
            # Calculate rankings
            results = calculate_power_rankings(season=season, week=week)

            # Prepare data for saving
            ratings = results['ratings']
            sos = results['sos']
            conferences = results['conferences']

            # Sort by rating
            sorted_teams = sorted(ratings.items(), key=lambda x: x[1], reverse=True)

            # Create output structure
            output = {
                'season': season,
                'week': week,
                'generated_at': results.get('generated_at', 'unknown'),
                'rankings': []
            }

            for rank, (team, rating) in enumerate(sorted_teams, 1):
                output['rankings'].append({
                    'rank': rank,
                    'team': team,
                    'rating': round(rating, 2),
                    'sos': round(sos.get(team, 0), 2),
                    'conference': conferences.get(team, 'Unknown')
                })

            # Save to JSON
            output_file = rankings_dir / f'week_{week}_{season}_ratings.json'
            with open(output_file, 'w') as f:
                json.dump(output, f, indent=2)

            print(f"[OK] ({len(sorted_teams)} teams)")
            success_count += 1

        except Exception as e:
            print(f"[ERROR] {str(e)[:50]}")

    return success_count


def main():
    """Generate historical rankings for all needed seasons"""
    print("="*80)
    print("HISTORICAL POWER RANKINGS GENERATION")
    print("="*80)
    print("\nGenerating power rankings for 2020-2022 seasons")
    print("(Coverage for QBs drafted 2021-2023)")

    # Seasons to generate (those with CFB model data)
    seasons = [2020, 2021, 2022]

    total_weeks = 0
    for season in seasons:
        count = generate_season_rankings(season, final_week=15)
        total_weeks += count

    # Summary
    print("\n" + "="*80)
    print("GENERATION COMPLETE")
    print("="*80)
    print(f"\n[OK] Generated {total_weeks} weeks of power rankings")
    print(f"     Seasons: {seasons}")
    print(f"     Output directory: {CFB_MODEL_PATH / 'power_rankings' / 'historical_ratings'}")

    print(f"\nNext Step: Re-run apply_competition_adjustment.py to calculate SOS")


if __name__ == "__main__":
    main()
