"""
Identify Active CFB QBs with 6+ Starts

Finds all current college QBs with at least 6 games started in the 2024 season,
collects their full college careers, and generates NFL projections.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sportsdataverse.cfb import load_cfb_pbp
from collections import defaultdict
import time


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent


def find_active_qbs_2024(min_starts=6):
    """
    Find all QBs with 6+ starts in the 2024 CFB season

    Returns: DataFrame with columns [player_name, team, games, attempts, epa_per_play]
    """
    print(f"\n[1] Loading 2024 CFB season data...")

    # Load 2024 season play-by-play data
    pbp_2024 = load_cfb_pbp(seasons=[2024], return_as_pandas=True)

    if pbp_2024 is None or len(pbp_2024) == 0:
        print("    [ERROR] No 2024 data found")
        return None

    print(f"    [OK] Loaded {len(pbp_2024):,} plays from 2024 season")

    # Filter to passing plays
    passing_plays = pbp_2024[
        (pbp_2024['pass'] == 1) &
        (pbp_2024['passer_player_name'].notna())
    ].copy()

    print(f"    [OK] Found {len(passing_plays):,} passing plays")

    # Aggregate QB stats by player and team
    print("\n[2] Aggregating QB statistics...")

    qb_stats = defaultdict(lambda: {
        'team': None,
        'games': set(),
        'attempts': 0,
        'completions': 0,
        'yards': 0,
        'tds': 0,
        'ints': 0,
        'epa_total': 0,
        'success': 0
    })

    for _, play in passing_plays.iterrows():
        qb_name = play['passer_player_name']
        game_id = play.get('game_id', play.get('id'))
        team = play.get('pos_team', play.get('offense', 'Unknown'))

        # Track game
        qb_stats[qb_name]['games'].add(game_id)
        qb_stats[qb_name]['team'] = team

        # Track stats
        qb_stats[qb_name]['attempts'] += 1

        if play.get('completion') == 1:
            qb_stats[qb_name]['completions'] += 1

        yards = play.get('statYardage', 0)
        if not pd.isna(yards):
            qb_stats[qb_name]['yards'] += yards

        if play.get('scoring') == 1 and play.get('type_text', '').lower().find('touchdown') >= 0:
            qb_stats[qb_name]['tds'] += 1

        if play.get('int') == 1:
            qb_stats[qb_name]['ints'] += 1

        epa = play.get('EPA', 0)
        if not pd.isna(epa):
            qb_stats[qb_name]['epa_total'] += epa

        if play.get('success') == 1:
            qb_stats[qb_name]['success'] += 1

    # Convert to DataFrame
    qb_data = []
    for qb_name, stats in qb_stats.items():
        games_played = len(stats['games'])

        # Filter to QBs with 6+ games (using as proxy for starts)
        if games_played >= min_starts and stats['attempts'] >= 50:
            qb_data.append({
                'player_name': qb_name,
                'team': stats['team'],
                'games': games_played,
                'attempts': stats['attempts'],
                'completions': stats['completions'],
                'yards': stats['yards'],
                'tds': stats['tds'],
                'ints': stats['ints'],
                'completion_pct': stats['completions'] / stats['attempts'] if stats['attempts'] > 0 else 0,
                'yards_per_attempt': stats['yards'] / stats['attempts'] if stats['attempts'] > 0 else 0,
                'epa_per_play': stats['epa_total'] / stats['attempts'] if stats['attempts'] > 0 else 0,
                'success_rate': stats['success'] / stats['attempts'] if stats['attempts'] > 0 else 0,
            })

    df_qbs = pd.DataFrame(qb_data)
    df_qbs = df_qbs.sort_values('epa_per_play', ascending=False)

    print(f"\n    [OK] Found {len(df_qbs)} QBs with {min_starts}+ games and 50+ attempts")

    return df_qbs


def collect_qb_career(player_name, team, final_season=2024):
    """
    Collect full college career for a QB

    Returns: DataFrame of career plays, or None if no data found
    """
    print(f"\n  Collecting career for {player_name} ({team})...")

    all_plays = []
    seasons = list(range(final_season - 4, final_season + 1))  # 5 years back

    for season in seasons:
        try:
            print(f"    {season}...", end=' ', flush=True)

            pbp_data = load_cfb_pbp(seasons=[season], return_as_pandas=True)

            if pbp_data is None or len(pbp_data) == 0:
                print("[No data]", end='')
                continue

            # Filter to this team's games
            team_plays = pbp_data[
                (pbp_data['homeTeamName'] == team) |
                (pbp_data['awayTeamName'] == team)
            ]

            if len(team_plays) == 0:
                print("[No team]", end='')
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
                print(f"[{len(qb_plays)} plays]", end='')
            else:
                print("[No QB]", end='')

            time.sleep(0.5)  # Rate limiting

        except Exception as e:
            print(f"[Error: {str(e)[:30]}]", end='')
            continue

    print()  # New line

    if len(all_plays) == 0:
        print("    [FAILED] No plays found")
        return None

    df_career = pd.concat(all_plays, ignore_index=True)
    print(f"    [SUCCESS] {len(df_career)} career plays across {len(all_plays)} seasons")

    return df_career


def main():
    """Main execution"""
    print("=" * 80)
    print("IDENTIFYING ACTIVE CFB QBs FOR NFL PROJECTION")
    print("=" * 80)

    root = get_project_root()

    # Find active QBs
    df_active = find_active_qbs_2024(min_starts=6)

    if df_active is None or len(df_active) == 0:
        print("\n[ERROR] No active QBs found")
        return

    # Display top QBs
    print("\n" + "=" * 80)
    print("TOP 20 ACTIVE CFB QBs (by EPA/play)")
    print("=" * 80)
    print(df_active[['player_name', 'team', 'games', 'attempts', 'epa_per_play', 'completion_pct']].head(20).to_string(index=False))

    # Save list of active QBs
    output_dir = root / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)

    active_qbs_path = output_dir / 'active_cfb_qbs_2024.csv'
    df_active.to_csv(active_qbs_path, index=False)
    print(f"\n[OK] Saved active QBs list to: {active_qbs_path}")

    # Ask if user wants to collect full careers
    print("\n" + "=" * 80)
    print(f"Found {len(df_active)} active CFB QBs")
    print("Next step: Run collect_active_qb_careers.py to download their full college careers")
    print("=" * 80)


if __name__ == "__main__":
    main()
