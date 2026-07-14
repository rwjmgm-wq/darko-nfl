"""
Fetch Play-by-Play Data for College QBs

Collects detailed play-by-play data including EPA (PPA) from CFBD API
for calculating College LEAF ratings.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import requests
import os
from dotenv import load_dotenv
import time
import json
from datetime import datetime


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent


def load_api_key():
    """Load CFBD API key from .env file"""
    root = get_project_root()
    env_path = root / '.env'
    load_dotenv(env_path)
    return os.getenv('CFBD_API_KEY')


def load_qb_data():
    """Load the merged QB dataset"""
    root = get_project_root()
    path = root / 'data' / 'processed' / 'merged_with_sp_plus.csv'
    df = pd.read_csv(path)
    print(f"[OK] Loaded {len(df)} QBs with merged data")
    return df


def normalize_team_name(team_name):
    """Normalize team names for CFBD API"""
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
    }
    return replacements.get(team_name, team_name)


def fetch_team_games(api_key, team, season, max_retries=3):
    """Fetch all games for a team in a season with retry logic"""
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Accept': 'application/json'
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(
                f'https://api.collegefootballdata.com/games',
                headers=headers,
                params={'year': season, 'team': team, 'seasonType': 'regular'},
                timeout=10
            )

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                # Rate limited - wait and retry with exponential backoff
                wait_time = (attempt + 1) * 10
                print(f"    [RATE LIMIT] Waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                return []

        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return []

    return []


def fetch_plays_for_game(api_key, season, week, team, opponent, max_retries=3):
    """
    Fetch play-by-play data for a specific game with retry logic

    Returns plays with PPA (EPA), down, distance, etc.
    """
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Accept': 'application/json'
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(
                f'https://api.collegefootballdata.com/plays',
                headers=headers,
                params={
                    'seasonType': 'regular',
                    'year': season,
                    'week': week,
                    'team': team
                },
                timeout=15
            )

            if response.status_code == 200:
                plays = response.json()
                return plays
            elif response.status_code == 429:
                # Rate limited - wait and retry
                wait_time = (attempt + 1) * 10
                print(f"      [RATE LIMIT] Waiting {wait_time}s... ", end='')
                time.sleep(wait_time)
                print(f"Retrying...")
                continue
            else:
                print(f"      [ERROR] Status {response.status_code}")
                return []

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            print(f"      [ERROR] {str(e)[:50]}")
            return []

    print(f"      [FAIL] Max retries reached")
    return []


def extract_qb_plays(all_plays, team):
    """
    Extract QB passing plays from all plays

    Returns list of QB plays with EPA and context
    """
    qb_plays = []

    for play in all_plays:
        # Only passing plays (API returns camelCase)
        play_type = play.get('playType', '')
        if 'pass' not in play_type.lower():
            continue

        # Only plays by our team (offensive plays)
        offense = play.get('offense')
        if offense != team:
            continue

        # Extract relevant data (API uses camelCase)
        qb_play = {
            'play_id': play.get('id'),
            'play_type': play_type,
            'down': play.get('down'),
            'distance': play.get('distance'),
            'yards_to_goal': play.get('yardsToGoal'),  # camelCase
            'yard_line': play.get('yardline'),  # lowercase
            'ppa': play.get('ppa'),  # Predicted Points Added (EPA)
            'scoring': play.get('scoring', False),
            'home': play.get('home'),
            'away': play.get('away'),
            'offense_score': play.get('offenseScore'),  # camelCase
            'defense_score': play.get('defenseScore'),  # camelCase
            'period': play.get('period'),
            'clock_minutes': play.get('clock', {}).get('minutes', 0) if isinstance(play.get('clock'), dict) else 0,
            'clock_seconds': play.get('clock', {}).get('seconds', 0) if isinstance(play.get('clock'), dict) else 0,
        }

        qb_plays.append(qb_play)

    return qb_plays


def fetch_qb_season_plays(api_key, qb_row):
    """
    Fetch all QB plays for a QB's final college season

    Returns DataFrame of plays with context and EPA
    """
    player_name = qb_row['player_name']
    college = normalize_team_name(qb_row.get('college', qb_row.get('college_college', '')))
    season = qb_row['final_season']

    print(f"\n  [{player_name}] {college}, {season}")

    # Get team's schedule
    games = fetch_team_games(api_key, college, season)

    if not games:
        print(f"    [--] No games found")
        return pd.DataFrame()

    print(f"    Found {len(games)} games")

    # Fetch plays for each game
    all_qb_plays = []
    games_with_data = 0

    for game in games:
        week = game.get('week')
        home_team = game.get('homeTeam')  # API returns camelCase
        away_team = game.get('awayTeam')  # API returns camelCase

        # Skip if missing critical data
        if not week or not home_team or not away_team:
            continue

        # Determine opponent
        opponent = away_team if home_team == college else home_team

        print(f"      Week {week:2d} vs {opponent:20s} ", end='')

        # Fetch plays
        plays = fetch_plays_for_game(api_key, season, week, college, opponent)

        if plays:
            # Extract QB plays
            qb_plays = extract_qb_plays(plays, college)

            if qb_plays:
                # Add game context
                for play in qb_plays:
                    play['week'] = week
                    play['opponent'] = opponent
                    play['season'] = season
                    play['team'] = college

                all_qb_plays.extend(qb_plays)
                print(f"[OK] {len(qb_plays)} pass plays")
                games_with_data += 1
            else:
                print(f"[--] No QB plays")
        else:
            print(f"[FAIL]")

        # Rate limit - increased to avoid 429 errors
        time.sleep(2.0)

    print(f"    Total: {len(all_qb_plays)} QB plays from {games_with_data}/{len(games)} games")

    if not all_qb_plays:
        return pd.DataFrame()

    # Convert to DataFrame
    df_plays = pd.DataFrame(all_qb_plays)

    # Add QB identifier
    df_plays['player_name'] = player_name
    df_plays['draft_year'] = qb_row['draft_year']

    return df_plays


def main():
    """Main execution"""
    print("="*80)
    print("FETCHING PLAY-BY-PLAY DATA FOR COLLEGE LEAF")
    print("="*80)

    root = get_project_root()
    output_dir = root / 'data' / 'processed' / 'play_by_play'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load API key
    print("\n[1] Loading CFBD API credentials...")
    api_key = load_api_key()
    print("    [OK] API key loaded")

    # Load QB data
    print("\n[2] Loading QB dataset...")
    df_qbs = load_qb_data()

    # Process all QBs in the dataset
    print("\n[3] Fetching play-by-play data...")
    print(f"    Processing all {len(df_qbs)} QBs (this will take ~45-60 minutes)")

    sample_qbs = df_qbs

    all_plays_list = []
    success_count = 0
    skipped_count = 0

    for idx, (_, qb) in enumerate(sample_qbs.iterrows(), 1):
        print(f"\n[{idx}/{len(sample_qbs)}]")

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
            df_plays = fetch_qb_season_plays(api_key, qb)

            if len(df_plays) > 0:
                all_plays_list.append(df_plays)
                success_count += 1

                # Save individual QB file
                df_plays.to_csv(qb_file, index=False)
                print(f"    [OK] Saved to {qb_file.name}")

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
        print(f"\n[OK] Successfully collected data for {success_count}/{len(sample_qbs)} QBs")
        print(f"     Already had: {skipped_count} QBs")
        print(f"     Newly collected: {success_count - skipped_count} QBs")
        print(f"     Total plays: {len(df_all_plays)}")
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
