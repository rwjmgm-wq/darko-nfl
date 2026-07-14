"""
Fetch College Stats for Drafted QBs (Direct HTTP Method)

Uses direct HTTP requests to CFBD API since the cfbd library has authentication issues.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import requests
import os
from dotenv import load_dotenv
import time
import warnings
warnings.filterwarnings('ignore')


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent


def load_api_key():
    """Load CFBD API key from .env file"""
    root = get_project_root()
    env_path = root / '.env'

    load_dotenv(env_path)
    api_key = os.getenv('CFBD_API_KEY')

    if not api_key:
        raise ValueError("CFBD_API_KEY not found in .env file")

    return api_key


def load_draft_data():
    """Load the drafted QBs data"""
    root = get_project_root()
    path = root / 'data' / 'raw' / 'drafted_qbs.csv'

    if not path.exists():
        raise FileNotFoundError(f"Draft data not found at {path}. Run collect_draft_data.py first.")

    df = pd.read_csv(path)
    print(f"[OK] Loaded {len(df)} drafted QBs")
    return df


def test_api_connection(api_key):
    """Test the API connection"""
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Accept': 'application/json'
    }

    try:
        response = requests.get(
            'https://api.collegefootballdata.com/games?year=2023&week=1',
            headers=headers,
            timeout=10
        )

        if response.status_code == 200:
            games = response.json()
            print(f"    [OK] API connection verified ({len(games)} games retrieved)")
            return True
        else:
            print(f"    [ERROR] API returned status {response.status_code}: {response.text}")
            return False

    except Exception as e:
        print(f"    [ERROR] Connection failed: {e}")
        return False


def get_final_college_season(draft_year):
    """
    Determine the final college season for a QB

    QBs drafted in spring 2024 played their last college season in fall 2023
    """
    return draft_year - 1


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
    }

    return replacements.get(team_name, team_name)


def fetch_team_games(api_key, team, season):
    """Fetch all games for a team in a season"""
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Accept': 'application/json'
    }

    try:
        response = requests.get(
            f'https://api.collegefootballdata.com/games',
            headers=headers,
            params={'year': season, 'team': team},
            timeout=10
        )

        if response.status_code == 200:
            return response.json()
        else:
            return []

    except Exception:
        return []


def fetch_player_stats(api_key, team, season, player_name):
    """Fetch player season stats"""
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Accept': 'application/json'
    }

    try:
        # Get player season stats
        response = requests.get(
            f'https://api.collegefootballdata.com/stats/player/season',
            headers=headers,
            params={
                'year': season,
                'team': team,
                'category': 'passing'
            },
            timeout=10
        )

        if response.status_code != 200:
            return None

        stats = response.json()

        # Find matching player (fuzzy match)
        player_name_lower = player_name.lower()
        for player in stats:
            if player.get('player') and player_name_lower in player.get('player', '').lower():
                return player

        return None

    except Exception:
        return None


def fetch_college_stats_for_qb(api_key, qb_row):
    """
    Fetch all college stats for a single QB

    Args:
        api_key: CFBD API key
        qb_row: Row from draft dataframe

    Returns:
        Dictionary of college stats
    """
    player_name = qb_row['player_name']
    college = normalize_team_name(qb_row['college'])
    draft_year = qb_row['draft_year']
    final_season = get_final_college_season(draft_year)

    # Initialize stats dictionary
    stats = {
        'player_name': player_name,
        'college': college,
        'draft_year': draft_year,
        'final_season': final_season,
        'draft_round': qb_row['round'],
        'draft_pick': qb_row['pick'],
    }

    # Fetch player stats
    player_stats = fetch_player_stats(api_key, college, final_season, player_name)
    if player_stats:
        stats.update({
            'pass_attempts': player_stats.get('stat') if player_stats.get('category') == 'passingAttempts' else None,
            'completions': player_stats.get('stat') if player_stats.get('category') == 'completions' else None,
            'passing_yards': player_stats.get('stat') if player_stats.get('category') == 'passingYards' else None,
            'passing_tds': player_stats.get('stat') if player_stats.get('category') == 'passingTDs' else None,
            'interceptions': player_stats.get('stat') if player_stats.get('category') == 'interceptions' else None,
        })

    # Fetch team schedule
    games = fetch_team_games(api_key, college, final_season)
    stats['games_played'] = len(games) if games else 0

    if games:
        opponents = []
        for game in games:
            # Determine opponent
            if game.get('home_team') == college:
                opp = game.get('away_team', '')
            else:
                opp = game.get('home_team', '')
            if opp:
                opponents.append(opp)
        stats['opponents'] = '|'.join(opponents) if opponents else ''
    else:
        stats['opponents'] = ''

    # Rate limit
    time.sleep(0.3)

    return stats


def main():
    """Main execution"""
    print("="*80)
    print("FETCHING COLLEGE STATS FOR DRAFTED QBs (v2 - Direct HTTP)")
    print("="*80)

    root = get_project_root()
    output_path = root / 'data' / 'processed' / 'college_qb_stats.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load API key
    print("\n[1] Loading CFBD API credentials...")
    api_key = load_api_key()
    print("    [OK] API key loaded")

    # Test connection
    print("    Testing API connection...")
    if not test_api_connection(api_key):
        raise ValueError("API connection test failed")

    # Load draft data
    print("\n[2] Loading draft data...")
    draft_qbs = load_draft_data()

    # Filter to QBs drafted 2015-2023
    draft_qbs_filtered = draft_qbs[draft_qbs['draft_year'] <= 2023].copy()
    print(f"    Focusing on {len(draft_qbs_filtered)} QBs drafted 2015-2023")
    print(f"    (College data needed: 2014-2022 seasons)")

    # Fetch stats for each QB
    print("\n[3] Fetching college stats from CFBD API...")
    print("    (This may take several minutes due to API rate limits)")

    all_stats = []
    success_count = 0

    for idx, (_, qb) in enumerate(draft_qbs_filtered.iterrows(), 1):
        print(f"\n  [{idx}/{len(draft_qbs_filtered)}] {qb['player_name']:<25}", end='')

        try:
            stats = fetch_college_stats_for_qb(api_key, qb)
            all_stats.append(stats)

            # Show if we got data
            if stats.get('games_played', 0) > 0:
                print(f" [OK] ({stats['games_played']} games)")
                success_count += 1
            else:
                print(f" [--] (No data)")

        except Exception as e:
            print(f" [ERROR] {str(e)[:50]}")
            # Still add basic info
            all_stats.append({
                'player_name': qb['player_name'],
                'college': qb['college'],
                'draft_year': qb['draft_year'],
                'final_season': get_final_college_season(qb['draft_year']),
                'draft_round': qb['round'],
                'draft_pick': qb['pick'],
            })

    # Create DataFrame
    college_df = pd.DataFrame(all_stats)

    # Save
    college_df.to_csv(output_path, index=False)

    # Summary
    print("\n" + "="*80)
    print("COLLEGE STATS COLLECTION COMPLETE")
    print("="*80)
    print(f"\n[OK] Saved college stats to: {output_path}")
    print(f"     Total QBs: {len(college_df)}")
    print(f"     QBs with game data: {success_count}")
    print(f"     Success rate: {success_count/len(college_df)*100:.1f}%")

    return college_df


if __name__ == "__main__":
    main()
