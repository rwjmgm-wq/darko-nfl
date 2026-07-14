"""
Collect Remaining QB College Careers using sportsdataverse

Skips QBs that have already been collected to save time.
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
    print("COLLECTING REMAINING QB COLLEGE CAREERS")
    print("="*80)

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

    # Check which QBs already have data
    print("\n[2] Checking for already-collected QBs...")
    already_collected = set()
    for file in output_dir.glob('*_*.csv'):
        if file.name != 'career_summary.csv':
            already_collected.add(file.stem)

    print(f"    Already collected: {len(already_collected)} QBs")

    # Filter to QBs that need collection
    qbs_to_collect = []
    for idx, row in df_qbs.iterrows():
        filename_stem = f"{row['player_name'].replace(' ', '_')}_{row['draft_year']}"
        if filename_stem not in already_collected:
            qbs_to_collect.append((idx, row))

    print(f"    Remaining to collect: {len(qbs_to_collect)} QBs")

    if len(qbs_to_collect) == 0:
        print("\n    All QBs already collected!")
        return

    # Collect remaining QBs
    print(f"\n[3] Collecting data for {len(qbs_to_collect)} remaining QBs...")
    print("-"*80)

    success_count = 0
    fail_count = 0

    for i, (idx, row) in enumerate(qbs_to_collect):
        player_name = row['player_name']
        draft_year = row['draft_year']
        college = row['college']

        print(f"[{i+1}/{len(qbs_to_collect)}]", end='')

        career_data = collect_qb_career(player_name, draft_year, college)

        if len(career_data) > 0:
            # Save this QB's career
            filename = f"{player_name.replace(' ', '_')}_{draft_year}.csv"
            filepath = output_dir / filename
            career_data.to_csv(filepath, index=False)

            seasons = sorted(career_data['season'].unique())
            success_count += 1
            print(f"    [SUCCESS] {len(career_data)} plays, {len(seasons)} seasons saved")
        else:
            fail_count += 1
            print(f"    [FAILED] No plays found")

    # Generate summary
    print("\n[4] Generating career summary...")

    all_qb_files = list(output_dir.glob('*_*.csv'))
    all_qb_files = [f for f in all_qb_files if f.name != 'career_summary.csv']

    career_summary = []
    for file in all_qb_files:
        try:
            df = pd.read_csv(file)
            if len(df) > 0:
                seasons = sorted(df['season'].unique())
                career_summary.append({
                    'player_name': df['player_name'].iloc[0],
                    'draft_year': df['draft_year'].iloc[0],
                    'college': df['college'].iloc[0],
                    'seasons_played': len(seasons),
                    'first_season': min(seasons),
                    'final_season': max(seasons),
                    'total_plays': len(df),
                    'file': file.name
                })
        except:
            pass

    if career_summary:
        df_summary = pd.DataFrame(career_summary)
        df_summary = df_summary.sort_values(['draft_year', 'player_name'])
        summary_path = output_dir / 'career_summary.csv'
        df_summary.to_csv(summary_path, index=False)

        print(f"    [OK] Summary saved: {len(df_summary)} QBs total")
        print(f"    Average seasons per QB: {df_summary['seasons_played'].mean():.1f}")
        print(f"    Total plays collected: {df_summary['total_plays'].sum():,}")

    # Final summary
    print("\n" + "="*80)
    print("COLLECTION SUMMARY")
    print("="*80)

    print(f"\n  This run:")
    print(f"    Successfully collected: {success_count} QBs")
    print(f"    Failed: {fail_count} QBs")

    print(f"\n  Overall:")
    print(f"    Total QBs with data: {len(career_summary)} / {len(df_qbs)}")

    if len(career_summary) == len(df_qbs):
        print("\n  [SUCCESS] All QBs collected!")
        print("\n  Next step: python src/create_multi_year_aggregations.py")
    else:
        missing = len(df_qbs) - len(career_summary)
        print(f"\n  [INCOMPLETE] {missing} QBs still missing")


if __name__ == "__main__":
    main()
