"""
Fetch College Stats for Drafted QBs

This script fetches college football statistics for all drafted QBs from CFBD API.
Focuses on their final college season (typically draft_year - 1).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import cfbd
from cfbd.rest import ApiException
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


def initialize_api_client(api_key):
    """Initialize CFBD API client"""
    configuration = cfbd.Configuration()
    # The API expects "Bearer <key>" format
    configuration.api_key['Authorization'] = api_key.strip()
    configuration.api_key_prefix['Authorization'] = 'Bearer'

    # Create client
    client = cfbd.ApiClient(configuration)

    # Test the API connection
    try:
        games_api = cfbd.GamesApi(client)
        test = games_api.get_games(year=2023, week=1, limit=1)
        print(f"    [OK] API connection verified (found {len(test)} game(s))")
    except Exception as e:
        print(f"    [ERROR] API test failed: {e}")
        raise ValueError("CFBD API key appears to be invalid or expired")

    return client


def get_final_college_season(draft_year):
    """
    Determine the final college season for a QB

    QBs drafted in spring 2024 played their last college season in fall 2023
    """
    return draft_year - 1


def normalize_team_name(team_name):
    """Normalize team names for CFBD API"""
    # Handle common variations
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
        'Texas Tech': 'Texas Tech',
    }

    return replacements.get(team_name, team_name)


def fetch_player_season_stats(api_client, player_name, team, season):
    """
    Fetch season aggregate stats for a player

    Args:
        api_client: CFBD API client
        player_name: Player name
        team: Team name
        season: Season year

    Returns:
        Dictionary of season stats
    """
    stats_api = cfbd.StatsApi(api_client)

    try:
        # Fetch player season stats
        stats = stats_api.get_player_season_stats(
            year=season,
            team=team,
            category='passing'
        )

        # Find matching player
        for player in stats:
            if player.player and player_name.lower() in player.player.lower():
                return {
                    'player': player.player,
                    'team': player.team,
                    'season': player.season,
                    'completions': player.stat if player.category == 'completions' else None,
                    'attempts': player.stat if player.category == 'attempts' else None,
                    'passing_yards': player.stat if player.category == 'passing_yards' else None,
                    'passing_tds': player.stat if player.category == 'passing_tds' else None,
                    'interceptions': player.stat if player.category == 'interceptions' else None,
                }

        return None

    except ApiException as e:
        print(f"  Warning: Could not fetch season stats for {player_name} ({team}, {season}): {e}")
        return None


def fetch_player_usage(api_client, player_name, team, season):
    """Fetch player usage stats (attempts, etc.)"""
    player_api = cfbd.PlayersApi(api_client)

    try:
        usage = player_api.get_player_usage(
            year=season,
            team=team
        )

        for player in usage:
            if player.name and player_name.lower() in player.name.lower():
                # Extract passing stats
                if hasattr(player, 'usage') and player.usage:
                    passing = player.usage.passing if hasattr(player.usage, 'passing') else None
                    if passing:
                        return {
                            'attempts': passing.attempts if hasattr(passing, 'attempts') else None,
                            'completions': passing.completions if hasattr(passing, 'completions') else None,
                            'completion_pct': passing.pct if hasattr(passing, 'pct') else None,
                        }

        return None

    except ApiException as e:
        # Usage API might not have data for all players
        return None


def fetch_ppa_stats(api_client, player_name, team, season):
    """
    Fetch PPA (Predicted Points Added) stats - college version of EPA

    Args:
        api_client: CFBD API client
        player_name: Player name
        team: Team name
        season: Season year

    Returns:
        Dictionary of PPA stats
    """
    # Try multiple API endpoints to get advanced stats
    stats_api = cfbd.StatsApi(api_client)

    try:
        # Use advanced stats endpoint
        stats = stats_api.get_advanced_team_season_stats(
            year=season,
            team=team
        )

        # For now, return None - we'll get PPA from different source
        # The team-level stats don't have player-level PPA
        return None

    except Exception:
        return None


def fetch_team_games(api_client, team, season):
    """Fetch all games for a team in a season"""
    games_api = cfbd.GamesApi(api_client)

    try:
        games = games_api.get_games(
            year=season,
            team=team
        )
        return games

    except ApiException as e:
        print(f"  Warning: Could not fetch games for {team} ({season}): {e}")
        return []


def fetch_college_stats_for_qb(api_client, qb_row):
    """
    Fetch all college stats for a single QB

    Args:
        api_client: CFBD API client
        qb_row: Row from draft dataframe

    Returns:
        Dictionary of college stats
    """
    player_name = qb_row['player_name']
    college = normalize_team_name(qb_row['college'])
    draft_year = qb_row['draft_year']
    final_season = get_final_college_season(draft_year)

    print(f"\n  Fetching: {player_name} ({college}, {final_season})...")

    # Initialize stats dictionary
    stats = {
        'player_name': player_name,
        'college': college,
        'draft_year': draft_year,
        'final_season': final_season,
        'draft_round': qb_row['round'],
        'draft_pick': qb_row['pick'],
    }

    # Fetch season usage stats
    usage = fetch_player_usage(api_client, player_name, college, final_season)
    if usage:
        stats.update({
            'pass_attempts': usage.get('attempts'),
            'completions': usage.get('completions'),
            'completion_pct': usage.get('completion_pct'),
        })

    # Fetch PPA stats (college EPA equivalent)
    ppa = fetch_ppa_stats(api_client, player_name, college, final_season)
    if ppa:
        stats.update({
            'avg_ppa': ppa.get('avg_ppa'),
            'avg_passing_ppa': ppa.get('avg_passing_ppa'),
            'total_ppa': ppa.get('total_ppa'),
            'total_passing_ppa': ppa.get('total_passing_ppa'),
        })

    # Fetch team schedule (for opponent tracking)
    games = fetch_team_games(api_client, college, final_season)
    stats['games_played'] = len(games) if games else 0

    if games:
        opponents = []
        for game in games:
            opp = game.away_team if game.home_team == college else game.home_team
            opponents.append(opp)
        stats['opponents'] = '|'.join(opponents) if opponents else ''
    else:
        stats['opponents'] = ''

    # Rate limit: pause between API calls
    time.sleep(0.5)

    return stats


def main():
    """Main execution"""
    print("="*80)
    print("FETCHING COLLEGE STATS FOR DRAFTED QBs")
    print("="*80)

    root = get_project_root()
    output_path = root / 'data' / 'processed' / 'college_qb_stats.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load API key
    print("\n[1] Loading CFBD API credentials...")
    api_key = load_api_key()
    print("    [OK] API key loaded")

    # Initialize API client
    api_client = initialize_api_client(api_key)
    print("    [OK] API client initialized")

    # Load draft data
    print("\n[2] Loading draft data...")
    draft_qbs = load_draft_data()

    # Filter to QBs drafted 2015-2023 (2024 draft class still in rookie year)
    # Also need college data from 2014-2022 (final_season = draft_year - 1)
    draft_qbs_filtered = draft_qbs[draft_qbs['draft_year'] <= 2023].copy()
    print(f"    Focusing on {len(draft_qbs_filtered)} QBs drafted 2015-2023")
    print(f"    (College data needed: 2014-2022 seasons)")

    # Fetch stats for each QB
    print("\n[3] Fetching college stats from CFBD API...")
    print("    (This may take several minutes due to API rate limits)")

    all_stats = []

    for idx, (_, qb) in enumerate(draft_qbs_filtered.iterrows(), 1):
        print(f"\n  [{idx}/{len(draft_qbs_filtered)}] {qb['player_name']}", end='')

        try:
            stats = fetch_college_stats_for_qb(api_client, qb)
            all_stats.append(stats)

            # Show if we got pass attempts
            if stats.get('pass_attempts') is not None:
                print(f" [OK] ({int(stats['pass_attempts'])} att)")
            else:
                print(f" [--] (No data)")

        except Exception as e:
            print(f" [ERROR] {str(e)}")
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
    print(f"     QBs with PPA data: {college_df['avg_ppa'].notna().sum()}")
    print(f"     QBs with pass attempts: {college_df['pass_attempts'].notna().sum()}")

    # Show top performers by PPA
    if college_df['avg_ppa'].notna().sum() > 0:
        print("\n" + "="*80)
        print("TOP 10 COLLEGE QBs BY PPA (Final Season):")
        print("="*80)
        top_10 = college_df.nlargest(10, 'avg_ppa')[
            ['player_name', 'college', 'final_season', 'avg_ppa', 'pass_attempts', 'draft_round']
        ]
        print(top_10.to_string(index=False))

    return college_df


if __name__ == "__main__":
    main()
