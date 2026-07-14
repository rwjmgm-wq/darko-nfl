"""
Collect College Career Data using sportsdataverse

Alternative to cfbd Python library - sportsdataverse is from the same team
as nflfastR/cfbfastR and may have better API handling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sportsdataverse.cfb import load_cfb_pbp
import time
from datetime import datetime


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent


def get_qb_seasons(draft_year):
    """Get potential college seasons for a QB"""
    final_season = draft_year - 1
    # Look back 5 years (covers redshirt seniors)
    return list(range(final_season - 4, final_season + 1))


def collect_qb_career(player_name, draft_year, college):
    """
    Collect all play-by-play data for a QB's college career

    Returns: DataFrame of all career plays
    """
    print(f"\n  Processing: {player_name} ({college}, {draft_year})")

    all_plays = []
    seasons = get_qb_seasons(draft_year)

    for season in seasons:
        try:
            print(f"    Fetching {season}...", end=' ', flush=True)

            # Load play-by-play for this season
            pbp_data = load_cfb_pbp(seasons=[season], return_as_pandas=True)

            if pbp_data is None or len(pbp_data) == 0:
                print("[No data]")
                continue

            # Filter to this team's games (use homeTeamName/awayTeamName)
            # Check if columns exist (2003 data may have different structure)
            if 'homeTeamName' not in pbp_data.columns or 'awayTeamName' not in pbp_data.columns:
                print("[Missing team columns]")
                continue

            team_plays = pbp_data[
                (pbp_data['homeTeamName'] == college) |
                (pbp_data['awayTeamName'] == college)
            ]

            if len(team_plays) == 0:
                print("[No team data]")
                continue

            # Filter to passing plays by this QB
            qb_plays = team_plays[
                (team_plays['pass'] == 1) &
                (team_plays['passer_player_name'].notna()) &
                (team_plays['passer_player_name'].str.contains(player_name, case=False, na=False))
            ]

            if len(qb_plays) > 0:
                # Extract relevant columns
                season_plays = pd.DataFrame({
                    'season': season,
                    'player_name': player_name,
                    'college': college,
                    'draft_year': draft_year,
                    'down': qb_plays['down'].values,
                    'distance': qb_plays['distance'].values,
                    'yards_gained': qb_plays['statYardage'].values,
                    'play_type': 'pass',
                    'epa': qb_plays['EPA'].values,
                    'scoring': qb_plays['scoringPlay'].values,
                    'yards_to_goal': qb_plays['start.yardsToEndzone'].values
                })

                all_plays.append(season_plays)
                print(f"[OK] {len(qb_plays)} plays")
            else:
                print("[No QB plays]")

            time.sleep(0.5)  # Rate limiting

        except Exception as e:
            print(f"[ERROR] {str(e)[:50]}")
            continue

    if all_plays:
        return pd.concat(all_plays, ignore_index=True)
    else:
        return pd.DataFrame()


def main():
    """Main execution"""
    print("="*80)
    print("COLLECTING COLLEGE CAREERS USING SPORTSDATAVERSE")
    print("="*80)
    print("\nAlternative to cfbd library - from nflfastR/cfbfastR team")

    root = get_project_root()

    # Load QB list
    print("\n[1] Loading QB list...")
    qb_list_path = root / 'data' / 'processed' / 'qb_list_expanded.csv'

    if not qb_list_path.exists():
        print("    [ERROR] QB list not found")
        return

    df_qbs = pd.read_csv(qb_list_path)
    print(f"    [OK] {len(df_qbs)} QBs loaded")

    # Create output directory
    output_dir = root / 'data' / 'processed' / 'full_college_careers'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect career data for each QB
    print(f"\n[2] Collecting career data for {len(df_qbs)} QBs...")
    print("-"*80)
    print("Note: sportsdataverse loads full seasons, so this will take several minutes per QB")

    career_summary = []
    success_count = 0
    fail_count = 0

    for idx, row in df_qbs.iterrows():
        player_name = row['player_name']
        draft_year = row['draft_year']
        college = row['college']

        print(f"[{idx+1}/{len(df_qbs)}]", end='')

        career_data = collect_qb_career(
            player_name,
            draft_year,
            college
        )

        if len(career_data) > 0:
            # Save this QB's career
            filename = f"{player_name.replace(' ', '_')}_{draft_year}.csv"
            filepath = output_dir / filename
            career_data.to_csv(filepath, index=False)

            # Update summary
            seasons = sorted(career_data['season'].unique())
            career_summary.append({
                'player_name': player_name,
                'draft_year': draft_year,
                'college': college,
                'seasons_played': len(seasons),
                'first_season': min(seasons),
                'final_season': max(seasons),
                'total_plays': len(career_data),
                'file': filename
            })

            success_count += 1
            print(f"    [SUCCESS] {len(career_data)} total plays saved")
        else:
            fail_count += 1
            print(f"    [FAILED] No plays found")

    # Save career summary
    print("\n[3] Saving career summary...")

    if career_summary:
        df_summary = pd.DataFrame(career_summary)
        summary_path = output_dir / 'career_summary.csv'
        df_summary.to_csv(summary_path, index=False)

        print(f"    [OK] Summary saved: {len(df_summary)} QBs")
        print(f"    Average seasons per QB: {df_summary['seasons_played'].mean():.1f}")
        print(f"    Total plays collected: {df_summary['total_plays'].sum():,}")

    # Final summary
    print("\n" + "="*80)
    print("COLLECTION SUMMARY")
    print("="*80)

    print(f"\n  Successfully collected: {success_count} QBs")
    print(f"  Failed: {fail_count} QBs")

    if career_summary:
        df_summary = pd.DataFrame(career_summary)

        print(f"\n  QBs by number of seasons:")
        season_counts = df_summary['seasons_played'].value_counts().sort_index()
        for seasons, count in season_counts.items():
            print(f"    {seasons} seasons: {count} QBs")

        print(f"\n  Data saved to: {output_dir}")
        print(f"\n  Next step: python src/create_multi_year_aggregations.py")
    else:
        print("\n  [ERROR] No data collected")
        print("\n  Troubleshooting:")
        print("    - Check internet connection")
        print("    - Verify sportsdataverse package is installed: pip install sportsdataverse")
        print("    - Try cfbfastR (R) if sportsdataverse also fails")


if __name__ == "__main__":
    main()
