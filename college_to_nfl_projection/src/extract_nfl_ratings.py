"""
Extract NFL Rookie LEAF Ratings for Drafted QBs

This script extracts rookie season LEAF ratings from the existing NFL LEAF model
for all QBs in our draft database.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from fuzzywuzzy import fuzz, process
import warnings
warnings.filterwarnings('ignore')


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent


def get_darko_root():
    """Get the main DARKO_NFL root directory"""
    return Path(__file__).parent.parent.parent


def load_draft_data():
    """Load the drafted QBs data"""
    root = get_project_root()
    path = root / 'data' / 'raw' / 'drafted_qbs.csv'

    if not path.exists():
        raise FileNotFoundError(f"Draft data not found at {path}. Run collect_draft_data.py first.")

    df = pd.read_csv(path)
    print(f"[OK] Loaded {len(df)} drafted QBs")
    return df


def load_leaf_ratings():
    """Load all LEAF game-by-game ratings"""
    darko_root = get_darko_root()

    # Try to load the most recent game-by-game ratings
    data_dir = darko_root / 'data' / 'production'

    # Find most recent file
    leaf_files = sorted(data_dir.glob('leaf_v2_game_by_game_*.csv'), reverse=True)

    if not leaf_files:
        # Try processed directory
        data_dir = darko_root / 'data' / 'processed'
        leaf_files = sorted(data_dir.glob('leaf_v2_game_by_game*.csv'), reverse=True)

    if not leaf_files:
        raise FileNotFoundError("Could not find LEAF game-by-game ratings file.")

    leaf_path = leaf_files[0]
    print(f"\n[OK] Loading LEAF ratings from: {leaf_path.name}")

    df = pd.read_csv(leaf_path)

    # Rename columns for consistency
    if 'passer_player_name' in df.columns:
        df = df.rename(columns={'passer_player_name': 'player_name'})
    if 'passer_player_id' in df.columns:
        df = df.rename(columns={'passer_player_id': 'player_id'})
    if 'pass_attempt_sum' in df.columns:
        df = df.rename(columns={'pass_attempt_sum': 'attempts'})

    # Display available columns
    print(f"     Total records: {len(df)}")
    print(f"     Unique players: {df['player_name'].nunique() if 'player_name' in df.columns else 'N/A'}")
    print(f"     Seasons: {df['season'].min() if 'season' in df.columns else 'N/A'} - "
          f"{df['season'].max() if 'season' in df.columns else 'N/A'}")

    return df


def normalize_name(name):
    """Normalize player name for matching"""
    if pd.isna(name):
        return ""

    # Convert to lowercase
    name = str(name).lower()

    # Remove common suffixes
    name = name.replace(' jr.', '').replace(' jr', '')
    name = name.replace(' sr.', '').replace(' sr', '')
    name = name.replace(' iii', '').replace(' ii', '').replace(' iv', '')

    # Remove periods and special characters
    name = name.replace('.', '').replace("'", '').replace('-', ' ')

    # Remove extra spaces
    name = ' '.join(name.split())

    return name


def create_name_variations(name):
    """
    Create name variations for matching

    Handles formats like:
    - "Josh Allen" -> ["josh allen", "j allen", "jallen"]
    - "J.Allen" -> ["j allen", "jallen"]
    """
    normalized = normalize_name(name)
    variations = [normalized]

    parts = normalized.split()
    if len(parts) >= 2:
        first, last = parts[0], ' '.join(parts[1:])
        # Add first initial + last name
        variations.append(f"{first[0]} {last}")
        variations.append(f"{first[0]}{last}")  # No space

    return variations


def match_player_names(draft_qbs, leaf_ratings):
    """
    Match player names from draft data to LEAF ratings

    Uses fuzzy matching to handle name variations.

    Args:
        draft_qbs: DataFrame with drafted QBs
        leaf_ratings: DataFrame with LEAF ratings

    Returns:
        DataFrame with matches
    """
    print("\n" + "="*80)
    print("MATCHING PLAYER NAMES")
    print("="*80)

    # Get unique player names from LEAF
    leaf_players = leaf_ratings['player_name'].unique()
    leaf_players_norm = {normalize_name(p): p for p in leaf_players}

    # Debug: Show some sample names from each dataset
    print(f"\nSample draft names (first 5):")
    for name in draft_qbs['player_name'].head(5):
        print(f"  '{name}' -> normalized: '{normalize_name(name)}'")

    print(f"\nSample LEAF names (first 5 matching recent QBs):")
    recent_leaf = sorted([p for p in leaf_players if any(x in str(p).lower() for x in ['mahomes', 'burrow', 'allen', 'herbert', 'jackson'])])[:5]
    for name in recent_leaf:
        print(f"  '{name}' -> normalized: '{normalize_name(name)}'")

    matches = []
    unmatched = []

    for _, qb in draft_qbs.iterrows():
        draft_name = qb['player_name']
        draft_year = qb['draft_year']

        # Create all possible name variations
        name_variations = create_name_variations(draft_name)

        # Try each variation for exact match
        leaf_name = None
        match_score = 0
        match_type = 'unmatched'

        for variation in name_variations:
            if variation in leaf_players_norm:
                leaf_name = leaf_players_norm[variation]
                match_score = 100
                match_type = 'exact'
                break

        # If no exact match, try fuzzy matching on all variations
        if leaf_name is None:
            best_result = None
            best_score = 0

            for variation in name_variations:
                result = process.extractOne(variation, leaf_players_norm.keys(),
                                            scorer=fuzz.ratio)
                if result and result[1] > best_score:
                    best_result = result
                    best_score = result[1]

            if best_result and best_score >= 80:  # 80% threshold
                matched_norm = best_result[0]
                leaf_name = leaf_players_norm[matched_norm]
                match_score = best_score
                match_type = 'fuzzy'
            else:
                unmatched.append({
                    'draft_name': draft_name,
                    'draft_year': draft_year,
                    'best_match': best_result[0] if best_result else None,
                    'score': best_score
                })

        matches.append({
            'draft_name': draft_name,
            'leaf_name': leaf_name,
            'draft_year': draft_year,
            'match_score': match_score,
            'match_type': match_type,
            'college': qb['college'],
            'nfl_team': qb['nfl_team'],
            'round': qb['round'],
            'pick': qb['pick']
        })

    matches_df = pd.DataFrame(matches)

    # Summary
    print(f"\nMatching Results:")
    print(f"  Exact matches: {len(matches_df[matches_df['match_type'] == 'exact'])}")
    print(f"  Fuzzy matches: {len(matches_df[matches_df['match_type'] == 'fuzzy'])}")
    print(f"  Unmatched: {len(matches_df[matches_df['match_type'] == 'unmatched'])}")

    if unmatched:
        print(f"\nUnmatched QBs (likely haven't played in NFL yet or limited data):")
        for um in unmatched[:10]:  # Show first 10
            print(f"  {um['draft_year']} - {um['draft_name']}")
        if len(unmatched) > 10:
            print(f"  ... and {len(unmatched) - 10} more")

    return matches_df


def extract_rookie_ratings(matches_df, leaf_ratings):
    """
    Extract rookie season LEAF ratings

    Args:
        matches_df: DataFrame with matched players
        leaf_ratings: DataFrame with all LEAF ratings

    Returns:
        DataFrame with rookie season statistics
    """
    print("\n" + "="*80)
    print("EXTRACTING ROOKIE SEASON RATINGS")
    print("="*80)

    rookie_stats = []

    for _, match in matches_df[matches_df['match_type'] != 'unmatched'].iterrows():
        leaf_name = match['leaf_name']
        draft_year = match['draft_year']
        rookie_season = draft_year  # Rookie season is same as draft year

        # Get all games for this player in their rookie season
        player_games = leaf_ratings[
            (leaf_ratings['player_name'] == leaf_name) &
            (leaf_ratings['season'] == rookie_season)
        ].copy()

        if len(player_games) == 0:
            # No rookie season data
            continue

        # Calculate rookie season statistics
        rookie_stats.append({
            'player_name': match['draft_name'],
            'leaf_name': leaf_name,
            'draft_year': draft_year,
            'college': match['college'],
            'nfl_team': match['nfl_team'],
            'round': match['round'],
            'pick': match['pick'],

            # Rookie season stats
            'rookie_games': len(player_games),
            'rookie_attempts': player_games['attempts'].sum() if 'attempts' in player_games.columns else 0,
            'rookie_leaf_mean': player_games['leaf_rating'].mean(),
            'rookie_leaf_final': player_games['leaf_rating'].iloc[-1],  # End-of-season rating
            'rookie_leaf_std': player_games['leaf_rating'].std(),
            'rookie_leaf_min': player_games['leaf_rating'].min(),
            'rookie_leaf_max': player_games['leaf_rating'].max(),

            # Additional metrics if available
            'rookie_epa_mean': player_games['epa'].mean() if 'epa' in player_games.columns else np.nan,
            'rookie_cpoe_mean': player_games['cpoe'].mean() if 'cpoe' in player_games.columns else np.nan,
        })

    rookie_df = pd.DataFrame(rookie_stats)

    print(f"\n[OK] Extracted rookie stats for {len(rookie_df)} QBs")
    if len(rookie_df) > 0:
        print(f"     Draft years: {rookie_df['draft_year'].min()}-{rookie_df['draft_year'].max()}")
        print(f"     Avg rookie games: {rookie_df['rookie_games'].mean():.1f}")
        rookie_df['rookie_attempts'] = pd.to_numeric(rookie_df['rookie_attempts'], errors='coerce').fillna(0)
        print(f"     QBs with 100+ attempts: {len(rookie_df[rookie_df['rookie_attempts'] >= 100])}")

    return rookie_df


def save_rookie_ratings(df, output_path):
    """Save rookie ratings to CSV"""
    df = df.sort_values(['draft_year', 'pick'])
    df.to_csv(output_path, index=False)

    print(f"\n[OK] Saved rookie ratings to: {output_path}")
    print(f"     Total QBs: {len(df)}")


def main():
    """Main execution"""
    print("="*80)
    print("EXTRACTING NFL ROOKIE LEAF RATINGS")
    print("="*80)

    root = get_project_root()
    output_path = root / 'data' / 'processed' / 'nfl_rookie_ratings.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    draft_qbs = load_draft_data()
    leaf_ratings = load_leaf_ratings()

    # Match player names
    matches_df = match_player_names(draft_qbs, leaf_ratings)

    # Extract rookie ratings
    rookie_df = extract_rookie_ratings(matches_df, leaf_ratings)

    # Save results
    save_rookie_ratings(rookie_df, output_path)

    # Display summary
    print("\n" + "="*80)
    print("TOP 10 ROOKIE SEASONS (by LEAF rating):")
    print("="*80)
    top_10 = rookie_df.nlargest(10, 'rookie_leaf_mean')[
        ['draft_year', 'player_name', 'college', 'rookie_games',
         'rookie_leaf_mean', 'rookie_attempts']
    ]
    print(top_10.to_string(index=False))

    print("\n" + "="*80)
    print("ROOKIE RATING EXTRACTION COMPLETE")
    print("="*80)

    return rookie_df


if __name__ == "__main__":
    main()
