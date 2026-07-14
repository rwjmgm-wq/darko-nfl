"""
Apply Competition Adjustment Using CFB Power Rankings

This script integrates the CFB betting model's power rankings to calculate
competition-adjusted metrics for college QBs.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys

# Add path to CFB betting model
CFB_MODEL_PATH = Path(r"C:\Users\rwjmg\OneDrive\Pictures\Writing\new_cfb_betting_model")
sys.path.insert(0, str(CFB_MODEL_PATH))


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent


def load_merged_data():
    """Load the merged college-NFL dataset"""
    root = get_project_root()
    path = root / 'data' / 'processed' / 'merged_college_nfl.csv'

    df = pd.read_csv(path)
    print(f"[OK] Loaded {len(df)} QBs with merged data")
    return df


def load_cfb_power_rankings(season):
    """
    Load CFB power rankings from the betting model

    Args:
        season: Season year

    Returns:
        Dictionary of {team: rating} or None if not found
    """
    rankings_dir = CFB_MODEL_PATH / 'power_rankings' / 'historical_ratings'

    # Look for any week from that season (use average across available weeks)
    season_files = list(rankings_dir.glob(f'week_*_{season}_ratings.json'))

    if not season_files:
        return None

    # Use the latest week available (highest week number)
    latest_file = sorted(season_files, key=lambda x: int(x.name.split('_')[1]), reverse=True)[0]

    with open(latest_file, 'r') as f:
        data = json.load(f)

    # Extract ratings as dictionary
    ratings = {}
    for team_data in data['rankings']:
        ratings[team_data['team']] = team_data['rating']

    return ratings


def normalize_team_name_for_ranking(team_name):
    """Normalize team names to match power rankings format"""
    # Common variations
    replacements = {
        'Florida State': 'Florida State',
        'NC State': 'NC State',
        'North Carolina State': 'NC State',
        'North Dakota State': 'North Dakota State',
        'Mississippi State': 'Mississippi State',
        'Colorado State': 'Colorado State',
        'Oregon State': 'Oregon State',
        'Iowa State': 'Iowa State',
        'Ohio State': 'Ohio State',
        'Penn State': 'Penn State',
        'Michigan State': 'Michigan State',
        'Boise State': 'Boise State',
        'San Diego State': 'San Diego State',
        'Fresno State': 'Fresno State',
        'San Jose State': 'San Jose State',
    }

    return replacements.get(team_name, team_name)


def calculate_sos_from_opponents(opponents_str, season):
    """
    Calculate Strength of Schedule from opponent list

    Args:
        opponents_str: Pipe-separated list of opponents
        season: Season year

    Returns:
        Average opponent rating or None
    """
    if pd.isna(opponents_str) or not opponents_str:
        return None

    # Load power rankings for this season
    ratings = load_cfb_power_rankings(season)

    if not ratings:
        return None

    # Parse opponents
    opponents = opponents_str.split('|')

    # Get ratings for each opponent
    opponent_ratings = []
    for opp in opponents:
        opp_normalized = normalize_team_name_for_ranking(opp.strip())
        if opp_normalized in ratings:
            opponent_ratings.append(ratings[opp_normalized])

    if not opponent_ratings:
        return None

    # Return average opponent rating (SOS)
    return np.mean(opponent_ratings)


def get_team_rating(team_name, season):
    """Get a team's power rating for a given season"""
    ratings = load_cfb_power_rankings(season)

    if not ratings:
        return None

    team_normalized = normalize_team_name_for_ranking(team_name)
    return ratings.get(team_normalized)


def apply_competition_adjustments(df):
    """
    Apply competition adjustments to the dataset

    Calculates:
    - Strength of Schedule (SOS) from opponent ratings
    - Team quality rating
    - Competition-adjusted metrics
    """
    print("\n" + "="*80)
    print("APPLYING COMPETITION ADJUSTMENTS")
    print("="*80)

    # Calculate SOS for each QB
    print("\nCalculating Strength of Schedule from opponent ratings...")

    sos_values = []
    team_ratings = []

    for idx, row in df.iterrows():
        season = row['final_season']
        opponents = row.get('opponents', '')
        # Handle both 'college' and 'college_college' column names from merge
        team = row.get('college', row.get('college_college', ''))

        # Calculate SOS
        sos = calculate_sos_from_opponents(opponents, season)
        sos_values.append(sos)

        # Get team rating
        team_rating = get_team_rating(team, season)
        team_ratings.append(team_rating)

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(df)} QBs...")

    df['sos_rating'] = sos_values
    df['team_rating'] = team_ratings

    # Summary
    sos_available = df['sos_rating'].notna().sum()
    team_rating_available = df['team_rating'].notna().sum()

    print(f"\n[OK] Competition adjustments calculated")
    print(f"     QBs with SOS data: {sos_available}/{len(df)} ({sos_available/len(df)*100:.1f}%)")
    print(f"     QBs with team rating: {team_rating_available}/{len(df)} ({team_rating_available/len(df)*100:.1f}%)")

    if sos_available > 0:
        print(f"\n  SOS Distribution:")
        print(f"    Mean: {df['sos_rating'].mean():.2f}")
        print(f"    Std:  {df['sos_rating'].std():.2f}")
        print(f"    Min:  {df['sos_rating'].min():.2f}")
        print(f"    Max:  {df['sos_rating'].max():.2f}")

    return df


def analyze_competition_effects(df):
    """Analyze the effect of competition adjustments"""
    print("\n" + "="*80)
    print("COMPETITION ADJUSTMENT ANALYSIS")
    print("="*80)

    # Filter to QBs with competition data
    df_comp = df[df['sos_rating'].notna()].copy()

    print(f"\nSample with competition data: {len(df_comp)} QBs")

    # Correlations
    print(f"\n--- CORRELATIONS WITH ROOKIE NFL LEAF ---")

    correlations = {}

    if len(df_comp) > 0:
        # SOS correlation
        corr_sos = df_comp['sos_rating'].corr(df_comp['rookie_leaf_mean'])
        correlations['sos'] = corr_sos
        print(f"  Strength of Schedule:     r = {corr_sos:+.3f}")

        # Team rating correlation
        if df_comp['team_rating'].notna().sum() > 0:
            corr_team = df_comp['team_rating'].corr(df_comp['rookie_leaf_mean'])
            correlations['team_rating'] = corr_team
            print(f"  Team Rating:              r = {corr_team:+.3f}")

        # Draft pick (baseline)
        corr_pick = df_comp['draft_pick'].corr(df_comp['rookie_leaf_mean'])
        correlations['draft_pick'] = corr_pick
        print(f"  Draft Pick (baseline):    r = {corr_pick:+.3f}")

        # Games played
        corr_games = df_comp['games_played'].corr(df_comp['rookie_leaf_mean'])
        correlations['games'] = corr_games
        print(f"  College Games:            r = {corr_games:+.3f}")

    # SOS Tiers
    if len(df_comp) > 0 and df_comp['sos_rating'].notna().sum() > 5:
        print(f"\n--- ROOKIE PERFORMANCE BY SOS TIER ---")

        df_comp['sos_tier'] = pd.qcut(
            df_comp['sos_rating'],
            q=3,
            labels=['Weak Schedule', 'Medium Schedule', 'Strong Schedule'],
            duplicates='drop'
        )

        tier_stats = df_comp.groupby('sos_tier', observed=True).agg({
            'rookie_leaf_mean': ['mean', 'count', 'std'],
            'sos_rating': 'mean'
        })

        print(tier_stats.to_string())

    return df_comp, correlations


def save_adjusted_data(df):
    """Save the competition-adjusted dataset"""
    root = get_project_root()
    output_path = root / 'data' / 'processed' / 'merged_with_competition.csv'

    df.to_csv(output_path, index=False)
    print(f"\n[OK] Saved adjusted dataset to: {output_path}")


def main():
    """Main execution"""
    print("="*80)
    print("COMPETITION ADJUSTMENT USING CFB POWER RANKINGS")
    print("="*80)

    # Check if CFB model exists
    if not CFB_MODEL_PATH.exists():
        print(f"\n[ERROR] CFB betting model not found at: {CFB_MODEL_PATH}")
        print("         Please update CFB_MODEL_PATH in the script")
        return

    rankings_dir = CFB_MODEL_PATH / 'power_rankings' / 'historical_ratings'
    if not rankings_dir.exists():
        print(f"\n[ERROR] Power rankings directory not found at: {rankings_dir}")
        return

    # Count available ratings
    rating_files = list(rankings_dir.glob('week_*_*_ratings.json'))
    print(f"\n[OK] Found {len(rating_files)} power ranking files")

    # Show available seasons
    seasons = set()
    for f in rating_files:
        season = int(f.name.split('_')[2])
        seasons.add(season)
    print(f"     Seasons covered: {sorted(seasons)}")

    # Load merged data
    print("\n[1] Loading merged dataset...")
    df = load_merged_data()

    # Apply competition adjustments
    print("\n[2] Applying competition adjustments...")
    df = apply_competition_adjustments(df)

    # Analyze effects
    print("\n[3] Analyzing competition effects...")
    df_comp, correlations = analyze_competition_effects(df)

    # Save
    print("\n[4] Saving adjusted dataset...")
    save_adjusted_data(df)

    # Summary
    print("\n" + "="*80)
    print("COMPETITION ADJUSTMENT COMPLETE")
    print("="*80)

    print(f"\nKEY FINDINGS:")
    if correlations:
        print(f"  Draft Pick (baseline):    r = {correlations.get('draft_pick', 0):+.3f}")
        if 'sos' in correlations:
            print(f"  Strength of Schedule:     r = {correlations['sos']:+.3f}")
            improvement = abs(correlations['sos']) - abs(correlations.get('draft_pick', 0))
            print(f"  Improvement over baseline: {improvement:+.3f}")

    print(f"\nNext Step: Run final feasibility report to compare all approaches")


if __name__ == "__main__":
    main()
