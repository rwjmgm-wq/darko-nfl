"""
Apply Competition Adjustment Using SP+ Ratings

SP+ is a tempo-free college football rating system that accounts for opponent strength.
This script uses SP+ ratings from CFBD API to calculate competition-adjusted metrics.
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


def load_merged_data():
    """Load the merged college-NFL dataset"""
    root = get_project_root()
    path = root / 'data' / 'processed' / 'merged_college_nfl.csv'
    df = pd.read_csv(path)
    print(f"[OK] Loaded {len(df)} QBs with merged data")
    return df


def fetch_sp_plus_ratings(api_key, season):
    """
    Fetch SP+ ratings for all teams in a season

    Args:
        api_key: CFBD API key
        season: Season year

    Returns:
        Dictionary of {team: sp_plus_rating}
    """
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Accept': 'application/json'
    }

    try:
        response = requests.get(
            f'https://api.collegefootballdata.com/ratings/sp',
            headers=headers,
            params={'year': season},
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            ratings = {}
            for team_data in data:
                team = team_data.get('team')
                rating = team_data.get('rating')
                if team and rating is not None:
                    ratings[team] = rating
            return ratings
        else:
            print(f"    [ERROR] API returned {response.status_code} for season {season}")
            return {}

    except Exception as e:
        print(f"    [ERROR] Failed to fetch SP+ for {season}: {e}")
        return {}


def normalize_team_name(team_name):
    """Normalize team names to match CFBD API format"""
    replacements = {
        'Florida St.': 'Florida State',
        'Florida State': 'Florida State',
        'North Carolina St.': 'NC State',
        'North Carolina State': 'NC State',
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
        'Miami (OH)': 'Miami (OH)',
        'Louisiana State': 'LSU',
    }
    return replacements.get(team_name, team_name)


def calculate_sos_from_sp_plus(opponents_str, season, sp_plus_cache):
    """
    Calculate Strength of Schedule from opponent SP+ ratings

    Args:
        opponents_str: Pipe-separated list of opponents
        season: Season year
        sp_plus_cache: Dictionary mapping season -> {team: rating}

    Returns:
        Average opponent SP+ rating or None
    """
    if pd.isna(opponents_str) or not opponents_str:
        return None

    # Get SP+ ratings for this season
    if season not in sp_plus_cache:
        return None

    ratings = sp_plus_cache[season]
    opponents = opponents_str.split('|')

    # Get ratings for each opponent
    opponent_ratings = []
    for opp in opponents:
        opp_normalized = normalize_team_name(opp.strip())
        if opp_normalized in ratings:
            opponent_ratings.append(ratings[opp_normalized])

    if not opponent_ratings:
        return None

    # Return average opponent rating (SOS)
    return np.mean(opponent_ratings)


def get_team_sp_plus(team_name, season, sp_plus_cache):
    """Get a team's SP+ rating for a given season"""
    if season not in sp_plus_cache:
        return None

    team_normalized = normalize_team_name(team_name)
    return sp_plus_cache[season].get(team_normalized)


def apply_competition_adjustments(df, api_key):
    """
    Apply competition adjustments using SP+ ratings

    Calculates:
    - Strength of Schedule (SOS) from opponent SP+ ratings
    - Team quality (own team's SP+ rating)
    - Competition-adjusted metrics
    """
    print("\n" + "="*80)
    print("APPLYING COMPETITION ADJUSTMENTS (SP+ RATINGS)")
    print("="*80)

    # Fetch SP+ ratings for all needed seasons
    print("\nFetching SP+ ratings from CFBD API...")
    seasons_needed = sorted(df['final_season'].unique())
    print(f"  Seasons needed: {list(seasons_needed)}")

    sp_plus_cache = {}
    for season in seasons_needed:
        print(f"  Fetching {season}... ", end='')
        ratings = fetch_sp_plus_ratings(api_key, season)
        if ratings:
            sp_plus_cache[season] = ratings
            print(f"[OK] ({len(ratings)} teams)")
        else:
            print(f"[--] (No data)")
        time.sleep(0.3)  # Rate limit

    # Calculate SOS and team ratings for each QB
    print("\nCalculating Strength of Schedule from SP+ ratings...")

    sos_values = []
    team_ratings = []

    for idx, row in df.iterrows():
        season = row['final_season']
        opponents = row.get('opponents', '')
        # Handle both 'college' and 'college_college' column names
        team = row.get('college', row.get('college_college', ''))

        # Calculate SOS
        sos = calculate_sos_from_sp_plus(opponents, season, sp_plus_cache)
        sos_values.append(sos)

        # Get team rating
        team_rating = get_team_sp_plus(team, season, sp_plus_cache)
        team_ratings.append(team_rating)

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(df)} QBs...")

    df['sos_sp_plus'] = sos_values
    df['team_sp_plus'] = team_ratings

    # Summary
    sos_available = df['sos_sp_plus'].notna().sum()
    team_rating_available = df['team_sp_plus'].notna().sum()

    print(f"\n[OK] Competition adjustments calculated")
    print(f"     QBs with SOS data: {sos_available}/{len(df)} ({sos_available/len(df)*100:.1f}%)")
    print(f"     QBs with team rating: {team_rating_available}/{len(df)} ({team_rating_available/len(df)*100:.1f}%)")

    if sos_available > 0:
        print(f"\n  SOS Distribution (SP+):")
        print(f"    Mean: {df['sos_sp_plus'].mean():.2f}")
        print(f"    Std:  {df['sos_sp_plus'].std():.2f}")
        print(f"    Min:  {df['sos_sp_plus'].min():.2f}")
        print(f"    Max:  {df['sos_sp_plus'].max():.2f}")

    if team_rating_available > 0:
        print(f"\n  Team SP+ Distribution:")
        print(f"    Mean: {df['team_sp_plus'].mean():.2f}")
        print(f"    Std:  {df['team_sp_plus'].std():.2f}")
        print(f"    Min:  {df['team_sp_plus'].min():.2f}")
        print(f"    Max:  {df['team_sp_plus'].max():.2f}")

    return df


def analyze_competition_effects(df):
    """Analyze the effect of competition adjustments"""
    print("\n" + "="*80)
    print("COMPETITION ADJUSTMENT ANALYSIS")
    print("="*80)

    # Filter to QBs with competition data
    df_comp = df[df['sos_sp_plus'].notna()].copy()

    print(f"\nSample with competition data: {len(df_comp)} QBs")

    if len(df_comp) == 0:
        print("\n[WARNING] No QBs with SP+ data available")
        return df_comp, {}

    # Correlations
    print(f"\n--- CORRELATIONS WITH ROOKIE NFL LEAF ---")

    correlations = {}

    # SOS correlation
    corr_sos = df_comp['sos_sp_plus'].corr(df_comp['rookie_leaf_mean'])
    correlations['sos_sp_plus'] = corr_sos
    print(f"  Strength of Schedule (SP+):  r = {corr_sos:+.3f}")

    # Team rating correlation
    if df_comp['team_sp_plus'].notna().sum() > 0:
        corr_team = df_comp['team_sp_plus'].corr(df_comp['rookie_leaf_mean'])
        correlations['team_sp_plus'] = corr_team
        print(f"  Team SP+ Rating:             r = {corr_team:+.3f}")

    # Draft pick (baseline)
    corr_pick = df_comp['draft_pick'].corr(df_comp['rookie_leaf_mean'])
    correlations['draft_pick'] = corr_pick
    print(f"  Draft Pick (baseline):       r = {corr_pick:+.3f}")

    # Games played
    corr_games = df_comp['games_played'].corr(df_comp['rookie_leaf_mean'])
    correlations['games'] = corr_games
    print(f"  College Games:               r = {corr_games:+.3f}")

    # SOS Tiers
    if len(df_comp) > 5 and df_comp['sos_sp_plus'].notna().sum() > 5:
        print(f"\n--- ROOKIE PERFORMANCE BY SOS TIER (SP+) ---")

        df_comp['sos_tier'] = pd.qcut(
            df_comp['sos_sp_plus'],
            q=3,
            labels=['Weak Schedule', 'Medium Schedule', 'Strong Schedule'],
            duplicates='drop'
        )

        tier_stats = df_comp.groupby('sos_tier', observed=True).agg({
            'rookie_leaf_mean': ['mean', 'count', 'std'],
            'sos_sp_plus': 'mean'
        })

        print(tier_stats.to_string())

    # Team SP+ Tiers
    if len(df_comp) > 5 and df_comp['team_sp_plus'].notna().sum() > 5:
        print(f"\n--- ROOKIE PERFORMANCE BY TEAM STRENGTH (SP+) ---")

        df_comp['team_tier'] = pd.qcut(
            df_comp['team_sp_plus'],
            q=3,
            labels=['Weak Team', 'Medium Team', 'Strong Team'],
            duplicates='drop'
        )

        tier_stats = df_comp.groupby('team_tier', observed=True).agg({
            'rookie_leaf_mean': ['mean', 'count', 'std'],
            'team_sp_plus': 'mean'
        })

        print(tier_stats.to_string())

    return df_comp, correlations


def save_adjusted_data(df):
    """Save the competition-adjusted dataset"""
    root = get_project_root()
    output_path = root / 'data' / 'processed' / 'merged_with_sp_plus.csv'

    df.to_csv(output_path, index=False)
    print(f"\n[OK] Saved adjusted dataset to: {output_path}")


def main():
    """Main execution"""
    print("="*80)
    print("COMPETITION ADJUSTMENT USING SP+ RATINGS")
    print("="*80)
    print("\nSP+ is a tempo-free rating system by Bill Connelly")
    print("Accounts for opponent strength, pace, and efficiency")
    print("Available for 2014-present from CFBD API")

    # Load API key
    print("\n[1] Loading CFBD API credentials...")
    api_key = load_api_key()
    print("    [OK] API key loaded")

    # Load merged data
    print("\n[2] Loading merged dataset...")
    df = load_merged_data()

    # Apply competition adjustments
    print("\n[3] Applying competition adjustments...")
    df = apply_competition_adjustments(df, api_key)

    # Analyze effects
    print("\n[4] Analyzing competition effects...")
    df_comp, correlations = analyze_competition_effects(df)

    # Save
    print("\n[5] Saving adjusted dataset...")
    save_adjusted_data(df)

    # Summary
    print("\n" + "="*80)
    print("COMPETITION ADJUSTMENT COMPLETE")
    print("="*80)

    print(f"\nKEY FINDINGS:")
    if correlations:
        print(f"  Draft Pick (baseline):       r = {correlations.get('draft_pick', 0):+.3f}")
        if 'sos_sp_plus' in correlations:
            print(f"  Strength of Schedule (SP+):  r = {correlations['sos_sp_plus']:+.3f}")
            improvement = abs(correlations['sos_sp_plus']) - abs(correlations.get('draft_pick', 0))
            print(f"  Improvement over baseline:   {improvement:+.3f}")
        if 'team_sp_plus' in correlations:
            print(f"  Team SP+ Rating:             r = {correlations['team_sp_plus']:+.3f}")

    print(f"\nNext Step: Generate final feasibility report")


if __name__ == "__main__":
    main()
