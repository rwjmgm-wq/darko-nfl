"""
Collect Full College Careers for Active CFB QBs

Reads the list of active QBs and collects their full college career data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sportsdataverse.cfb import load_cfb_pbp
import time


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent


def collect_qb_career(player_name, team, final_season=2024):
    """
    Collect full college career for a QB

    Returns: DataFrame of career plays, or None if no data found
    """
    print(f"\n  Processing: {player_name} ({team})")

    all_plays = []
    seasons = list(range(final_season - 4, final_season + 1))  # 5 years back

    for season in seasons:
        try:
            print(f"    Fetching {season}...", end=' ', flush=True)

            pbp_data = load_cfb_pbp(seasons=[season], return_as_pandas=True)

            if pbp_data is None or len(pbp_data) == 0:
                print("[No data]")
                continue

            # Filter to this team's games
            if 'homeTeamName' not in pbp_data.columns or 'awayTeamName' not in pbp_data.columns:
                print("[Missing team columns]")
                continue

            team_plays = pbp_data[
                (pbp_data['homeTeamName'] == team) |
                (pbp_data['awayTeamName'] == team)
            ]

            if len(team_plays) == 0:
                print("[No team data]")
                continue

            # Filter to this QB's passing plays
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
                    'college': team,
                    'draft_year': 'Active',  # Mark as active
                    'down': qb_plays['down'].values,
                    'distance': qb_plays['distance'].values,
                    'yards_gained': qb_plays['statYardage'].values,
                    'epa': qb_plays['EPA'].values,
                    'success': qb_plays['success'].values,
                    'pass': 1,
                    'completion': qb_plays.get('completion', pd.Series([np.nan] * len(qb_plays))).values,
                    'interception': qb_plays.get('int', pd.Series([0] * len(qb_plays))).values,
                    'touchdown': qb_plays.get('scoring', pd.Series([0] * len(qb_plays))).values,
                })

                all_plays.append(season_plays)
                print(f"[OK] {len(qb_plays)} plays")
            else:
                print("[No QB plays]")

            time.sleep(0.5)  # Rate limiting

        except Exception as e:
            print(f"[Error: {str(e)[:40]}]")
            continue

    if len(all_plays) == 0:
        print("    [FAILED] No plays found")
        return None

    df_career = pd.concat(all_plays, ignore_index=True)
    print(f"    [SUCCESS] {len(df_career)} plays across {len(all_plays)} seasons")

    return df_career


def main():
    """Main execution"""
    print("=" * 80)
    print("COLLECTING ACTIVE CFB QB COLLEGE CAREERS")
    print("=" * 80)

    root = get_project_root()

    # Load list of active QBs
    print("\n[1] Loading active QB list...")
    active_qbs_path = root / 'data' / 'processed' / 'active_cfb_qbs_2024.csv'

    if not active_qbs_path.exists():
        print("    [ERROR] Active QBs list not found")
        print("    Run identify_active_cfb_qbs.py first")
        return

    df_active = pd.read_csv(active_qbs_path)
    print(f"    [OK] Loaded {len(df_active)} active QBs")

    # Check which ones already have career data
    careers_dir = root / 'data' / 'processed' / 'active_qb_careers'
    careers_dir.mkdir(parents=True, exist_ok=True)

    already_collected = set()
    for file in careers_dir.glob('*.csv'):
        already_collected.add(file.stem.replace('_', ' '))

    qbs_to_collect = []
    for _, row in df_active.iterrows():
        if row['player_name'] not in already_collected:
            qbs_to_collect.append(row)

    print(f"\n[2] Already collected: {len(already_collected)} QBs")
    print(f"    Remaining to collect: {len(qbs_to_collect)} QBs")

    if len(qbs_to_collect) == 0:
        print("\n[OK] All QBs already collected!")
        return

    # Collect careers
    print(f"\n[3] Collecting careers for {len(qbs_to_collect)} QBs...")
    print("-" * 80)

    collected = 0
    failed = 0

    for idx, qb in enumerate(qbs_to_collect, 1):
        print(f"\n[{idx}/{len(qbs_to_collect)}]")

        df_career = collect_qb_career(qb['player_name'], qb['team'], final_season=2024)

        if df_career is not None and len(df_career) > 0:
            # Save career data
            filename = qb['player_name'].replace(' ', '_') + '.csv'
            career_path = careers_dir / filename
            df_career.to_csv(career_path, index=False)
            collected += 1
        else:
            failed += 1

        # Progress update every 5 QBs
        if idx % 5 == 0:
            print(f"\n    Progress: {collected} collected, {failed} failed")

    # Summary
    print("\n" + "=" * 80)
    print("COLLECTION SUMMARY")
    print("=" * 80)
    print(f"\nSuccessfully collected: {collected} QBs")
    print(f"Failed: {failed} QBs")
    print(f"\nCareer data saved to: {careers_dir}")


if __name__ == "__main__":
    main()
