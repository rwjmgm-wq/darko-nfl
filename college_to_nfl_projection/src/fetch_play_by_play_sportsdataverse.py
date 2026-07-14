"""
Fetch Play-by-Play Data using sportsdataverse (no API limits!)

Uses sportsdataverse package which provides cached CFB data similar to nfl_data_py.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sportsdataverse.cfb import load_cfb_pbp
import time


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent


def load_qb_data():
    """Load the merged QB dataset"""
    root = get_project_root()
    path = root / 'data' / 'processed' / 'merged_with_sp_plus.csv'
    df = pd.read_csv(path)
    print(f"[OK] Loaded {len(df)} QBs with merged data")
    return df


def normalize_team_name(team_name):
    """Normalize team names for sportsdataverse"""
    replacements = {
        'Florida St.': 'Florida State',
        'North Carolina St.': 'NC State',
        'North Dakota St.': 'North Dakota State',
        'Mississippi St.': 'Mississippi State',
        'Colorado St.': 'Colorado State',
        'Oregon St.': 'Oregon State',
        'Iowa St.': 'Iowa State',
        'Ohio St.': 'Ohio State',
        'Penn St.': 'Penn State',
        'Michigan St.': 'Michigan State',
        'Boise St.': 'Boise State',
        'San Diego St.': 'San Diego State',
        'Fresno St.': 'Fresno State',
        'San Jose St.': 'San Jose State',
        'Miami (FL)': 'Miami',
        'Louisiana State': 'LSU',
        'Washington St.': 'Washington State',
        'Kansas St.': 'Kansas State',
    }
    return replacements.get(team_name, team_name)


def fetch_season_pbp(season):
    """Fetch all play-by-play data for a season using sportsdataverse"""
    try:
        print(f"    Loading {season} season data...", end=' ', flush=True)
        # Returns Polars DataFrame - convert to Pandas
        df_polars = load_cfb_pbp(seasons=[season])
        df = df_polars.to_pandas()
        print(f"[OK] {len(df):,} plays")
        return df
    except Exception as e:
        print(f"[ERROR] {str(e)[:100]}")
        return pd.DataFrame()


def extract_qb_plays(df_pbp, team_name, season, player_name):
    """
    Extract QB passing plays for a specific team from season play-by-play

    Returns DataFrame with QB plays and EPA
    """
    if df_pbp is None or len(df_pbp) == 0:
        return pd.DataFrame()

    # Filter to passing plays with QB identification
    pass_plays = df_pbp[
        (df_pbp['pass'] == True) &
        (df_pbp['passer_player_name'].notna())
    ].copy()

    if len(pass_plays) == 0:
        return pd.DataFrame()

    # Filter to this team's plays using team name
    # Team name appears in either homeTeamName or awayTeamName
    team_plays = pass_plays[
        (pass_plays['homeTeamName'] == team_name) |
        (pass_plays['awayTeamName'] == team_name)
    ].copy()

    if len(team_plays) == 0:
        return pd.DataFrame()

    # Further filter to plays where this team had possession
    # Use pos_team to match the team's ID (homeTeamId or awayTeamId)
    def is_offense_play(row):
        """Check if the team on offense matches our team"""
        if row['homeTeamName'] == team_name:
            return row['pos_team'] == row['homeTeamId']
        else:
            return row['pos_team'] == row['awayTeamId']

    team_plays = team_plays[team_plays.apply(is_offense_play, axis=1)].copy()

    if len(team_plays) == 0:
        return pd.DataFrame()

    # Extract relevant data
    qb_plays_list = []

    for _, play in team_plays.iterrows():
        # Determine opponent
        if play.get('homeTeamName') == team_name:
            opponent = play.get('awayTeamName', '')
        else:
            opponent = play.get('homeTeamName', '')

        # Extract scores
        home_score = play.get('homeScore', 0)
        away_score = play.get('awayScore', 0)

        # Determine offense/defense score based on home/away
        if play.get('homeTeamName') == team_name:
            offense_score = home_score
            defense_score = away_score
        else:
            offense_score = away_score
            defense_score = home_score

        # Extract clock info
        clock_str = play.get('clock.displayValue', '0:00')
        try:
            parts = clock_str.split(':')
            clock_minutes = int(parts[0]) if len(parts) > 0 else 0
            clock_seconds = int(parts[1]) if len(parts) > 1 else 0
        except:
            clock_minutes = 0
            clock_seconds = 0

        qb_play = {
            'play_id': play.get('id'),
            'play_type': play.get('type.text', 'Pass'),
            'down': play.get('down'),
            'distance': play.get('distance'),
            'yards_to_goal': play.get('start.yardsToEndzone'),
            'yard_line': play.get('start.yardLine'),
            'ppa': play.get('EPA'),  # Expected Points Added
            'scoring': play.get('scoringPlay', False),
            'home': play.get('homeTeamName'),
            'away': play.get('awayTeamName'),
            'offense_score': offense_score,
            'defense_score': defense_score,
            'period': play.get('period.number'),
            'clock_minutes': clock_minutes,
            'clock_seconds': clock_seconds,
            'week': play.get('week'),
            'opponent': opponent,
            'season': season,
            'team': team_name,
            'player_name': player_name,
        }
        qb_plays_list.append(qb_play)

    return pd.DataFrame(qb_plays_list)


def fetch_qb_season_plays(qb_row, season_pbp_cache):
    """
    Extract QB plays from cached season play-by-play data

    Returns DataFrame of plays with context and EPA
    """
    player_name = qb_row['player_name']
    college = normalize_team_name(qb_row.get('college_college', qb_row.get('college', '')))
    season = qb_row['final_season']

    print(f"\n  [{player_name}] {college}, {season}")

    # Get or load season data
    if season not in season_pbp_cache:
        df_season = fetch_season_pbp(season)
        if len(df_season) == 0:
            print(f"    [--] No season data available")
            return pd.DataFrame()
        season_pbp_cache[season] = df_season
    else:
        print(f"    Using cached {season} data")

    df_season = season_pbp_cache[season]

    # Extract QB plays (pass team name, not ID)
    df_qb_plays = extract_qb_plays(df_season, college, season, player_name)

    if len(df_qb_plays) == 0:
        print(f"    [--] No QB plays found for {college}")
        return pd.DataFrame()

    # Add draft year
    df_qb_plays['draft_year'] = qb_row['draft_year']

    print(f"    [OK] Found {len(df_qb_plays)} pass plays")

    return df_qb_plays


def main():
    """Main execution"""
    print("="*80)
    print("FETCHING PLAY-BY-PLAY DATA - SPORTSDATAVERSE")
    print("="*80)
    print("\nUsing sportsdataverse package (no API limits!)")

    root = get_project_root()
    output_dir = root / 'data' / 'processed' / 'play_by_play'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load QB data
    print("\n[1] Loading QB dataset...")
    df_qbs = load_qb_data()

    print(f"\n[2] Fetching play-by-play data for {len(df_qbs)} QBs...")

    # Cache season data to avoid re-loading
    season_pbp_cache = {}

    all_plays_list = []
    success_count = 0
    skipped_count = 0

    for idx, (_, qb) in enumerate(df_qbs.iterrows(), 1):
        print(f"\n[{idx}/{len(df_qbs)}]")

        # Check if we already have data for this QB
        qb_name_clean = qb['player_name'].replace(' ', '_')
        qb_file = output_dir / f"{qb_name_clean}_{qb['draft_year']}_plays.csv"

        if qb_file.exists():
            existing_df = pd.read_csv(qb_file)
            # Only skip if we have substantial data (>100 plays indicates complete collection)
            if len(existing_df) > 100:
                print(f"  [{qb['player_name']}] - ALREADY COLLECTED ({len(existing_df)} plays)")
                all_plays_list.append(existing_df)
                success_count += 1
                skipped_count += 1
                continue

        try:
            df_plays = fetch_qb_season_plays(qb, season_pbp_cache)

            if len(df_plays) > 0:
                all_plays_list.append(df_plays)
                success_count += 1

                # Save individual QB file
                df_plays.to_csv(qb_file, index=False)
                print(f"    [SAVED] {qb_file.name}")

        except Exception as e:
            print(f"    [ERROR] {str(e)[:100]}")

    # Combine all plays
    if all_plays_list:
        df_all_plays = pd.concat(all_plays_list, ignore_index=True)

        # Save combined file
        output_path = output_dir / 'all_qb_plays_full.csv'
        df_all_plays.to_csv(output_path, index=False)

        # Summary
        print("\n" + "="*80)
        print("PLAY-BY-PLAY COLLECTION COMPLETE")
        print("="*80)
        print(f"\n[OK] Successfully collected data for {success_count}/{len(df_qbs)} QBs")
        print(f"     Already had: {skipped_count} QBs")
        print(f"     Newly collected: {success_count - skipped_count} QBs")
        print(f"     Total plays: {len(df_all_plays):,}")
        print(f"     Avg plays per QB: {len(df_all_plays)/success_count:.0f}")
        print(f"\n[OK] Saved combined file: {output_path}")

        # Show sample
        print(f"\nSample plays:")
        print(df_all_plays[['player_name', 'week', 'opponent', 'down', 'distance', 'ppa']].head(10))

        print(f"\nPPA (EPA) Distribution:")
        print(df_all_plays['ppa'].describe())

    else:
        print("\n[ERROR] No play-by-play data collected")

    print(f"\nNext Step: Calculate College LEAF from play-by-play data")


if __name__ == "__main__":
    main()
