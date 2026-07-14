"""
Fetch Opponent SP+ Ratings for Competition Adjustment

Collects defensive SP+ ratings for all opponents in play-by-play data.
Uses sportsdataverse to avoid API rate limits.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import requests
import os
from dotenv import load_dotenv
import time


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent


def load_api_key():
    """Load CFBD API key from .env file"""
    root = get_project_root()
    env_path = root / '.env'
    load_dotenv(env_path)
    return os.getenv('CFBD_API_KEY')


def load_play_by_play_data():
    """Load the play-by-play data"""
    root = get_project_root()
    path = root / 'data' / 'processed' / 'play_by_play' / 'all_qb_plays_full.csv'
    df = pd.read_csv(path)
    print(f"[OK] Loaded {len(df):,} plays")
    return df


def load_qb_metadata():
    """Load QB metadata with team SP+ data"""
    root = get_project_root()
    path = root / 'data' / 'processed' / 'merged_with_sp_plus.csv'
    df = pd.read_csv(path)
    return df


def fetch_sp_plus_ratings(api_key, season, max_retries=3):
    """
    Fetch SP+ ratings for a season from CFBD API

    Returns dict mapping team name to SP+ components
    """
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Accept': 'application/json'
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(
                'https://api.collegefootballdata.com/ratings/sp',
                headers=headers,
                params={'year': season},
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()

                # Create lookup dict: team name -> ratings
                ratings_dict = {}
                for team_rating in data:
                    team = team_rating.get('team')
                    if team:
                        ratings_dict[team] = {
                            'overall': team_rating.get('rating'),
                            'offense': team_rating.get('offense', {}).get('rating'),
                            'defense': team_rating.get('defense', {}).get('rating'),
                        }

                return ratings_dict

            elif response.status_code == 429:
                # Rate limited
                wait_time = (attempt + 1) * 10
                print(f"      [RATE LIMIT] Waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                return {}

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return {}

    return {}


def normalize_team_name(team_name):
    """Normalize team names for SP+ matching"""
    if pd.isna(team_name):
        return None

    # Common abbreviations and variations
    replacements = {
        'Florida St.': 'Florida State',
        'North Carolina St.': 'NC State',
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
        'Washington St.': 'Washington State',
        'Kansas St.': 'Kansas State',
        'Miami (FL)': 'Miami',
        'Louisiana State': 'LSU',
        'USC': 'Southern California',
        'UCF': 'Central Florida',
        'Southern Miss': 'Southern Mississippi',
        'UConn': 'Connecticut',
        'UMass': 'Massachusetts',
    }

    return replacements.get(team_name, team_name)


def main():
    """Main execution"""
    print("="*80)
    print("FETCHING OPPONENT SP+ RATINGS FOR COMPETITION ADJUSTMENT")
    print("="*80)

    root = get_project_root()

    # Load play-by-play data
    print("\n[1] Loading play-by-play data...")
    df_plays = load_play_by_play_data()

    # Load QB metadata for team SP+
    print("\n[2] Loading QB metadata...")
    df_metadata = load_qb_metadata()

    # Get unique (opponent, season) pairs
    print("\n[3] Identifying unique opponents...")
    df_plays['opponent_normalized'] = df_plays['opponent'].apply(normalize_team_name)
    df_plays['team_normalized'] = df_plays['team'].apply(normalize_team_name)

    unique_opponents = df_plays[['opponent_normalized', 'season']].drop_duplicates()
    unique_opponents = unique_opponents[unique_opponents['opponent_normalized'].notna()]
    print(f"    Found {len(unique_opponents)} unique (opponent, season) pairs")

    # Also get unique teams for team SP+
    unique_teams = df_plays[['team_normalized', 'season']].drop_duplicates()
    unique_teams = unique_teams[unique_teams['team_normalized'].notna()]
    print(f"    Found {len(unique_teams)} unique (team, season) pairs")

    # Combine for all needed SP+ data
    all_needed = pd.concat([
        unique_opponents.rename(columns={'opponent_normalized': 'team'}),
        unique_teams.rename(columns={'team_normalized': 'team'})
    ]).drop_duplicates()

    # Load API key
    print("\n[4] Loading CFBD API credentials...")
    api_key = load_api_key()
    if not api_key:
        print("    [WARNING] No API key found - will use cached data only")

    # Fetch SP+ ratings by season
    print("\n[5] Fetching SP+ ratings by season...")
    sp_plus_cache = {}

    seasons = sorted(all_needed['season'].unique())
    print(f"    Need data for {len(seasons)} seasons: {min(seasons)}-{max(seasons)}")

    for season in seasons:
        print(f"\n    [{season}]")

        if api_key:
            ratings = fetch_sp_plus_ratings(api_key, season)
            if ratings:
                sp_plus_cache[season] = ratings
                print(f"        [OK] Fetched {len(ratings)} teams")
                time.sleep(1.0)  # Rate limiting
            else:
                print(f"        [FAIL] Could not fetch ratings")
                sp_plus_cache[season] = {}
        else:
            sp_plus_cache[season] = {}

    # Merge opponent SP+ into plays
    print("\n[6] Merging opponent SP+ into plays...")

    def get_opponent_defense_sp(row):
        """Get opponent defensive SP+ for a play"""
        season = row['season']
        opponent = row['opponent_normalized']

        if season not in sp_plus_cache or opponent not in sp_plus_cache[season]:
            return np.nan

        return sp_plus_cache[season][opponent].get('defense', np.nan)

    def get_team_offense_sp(row):
        """Get team offensive SP+ for a play"""
        season = row['season']
        team = row['team_normalized']

        if season not in sp_plus_cache or team not in sp_plus_cache[season]:
            return np.nan

        return sp_plus_cache[season][team].get('offense', np.nan)

    print("    Adding opponent defensive SP+...")
    df_plays['opponent_defense_sp'] = df_plays.apply(get_opponent_defense_sp, axis=1)

    print("    Adding team offensive SP+...")
    df_plays['team_offense_sp'] = df_plays.apply(get_team_offense_sp, axis=1)

    # Calculate coverage
    total_plays = len(df_plays)
    plays_with_opp_sp = df_plays['opponent_defense_sp'].notna().sum()
    plays_with_team_sp = df_plays['team_offense_sp'].notna().sum()

    print(f"\n    Coverage:")
    print(f"        Opponent defense SP+: {plays_with_opp_sp:,}/{total_plays:,} ({plays_with_opp_sp/total_plays*100:.1f}%)")
    print(f"        Team offense SP+:     {plays_with_team_sp:,}/{total_plays:,} ({plays_with_team_sp/total_plays*100:.1f}%)")

    # Save enhanced dataset
    print("\n[7] Saving enhanced play-by-play data...")
    output_path = root / 'data' / 'processed' / 'play_by_play' / 'all_qb_plays_with_sp_plus.csv'
    df_plays.to_csv(output_path, index=False)
    print(f"    [OK] Saved to: {output_path}")

    # Summary statistics
    print("\n" + "="*80)
    print("OPPONENT SP+ COLLECTION COMPLETE")
    print("="*80)
    print(f"\nDataset Summary:")
    print(f"    Total plays: {len(df_plays):,}")
    print(f"    Plays with competition data: {plays_with_opp_sp:,} ({plays_with_opp_sp/len(df_plays)*100:.1f}%)")
    print(f"    Unique opponents: {len(unique_opponents)}")
    print(f"    Seasons covered: {len(seasons)}")

    print(f"\nOpponent Defense SP+ Distribution:")
    print(df_plays['opponent_defense_sp'].describe())

    print(f"\nTeam Offense SP+ Distribution:")
    print(df_plays['team_offense_sp'].describe())

    print(f"\nNext Step: Calculate College LEAF with full competition adjustment")


if __name__ == "__main__":
    main()
