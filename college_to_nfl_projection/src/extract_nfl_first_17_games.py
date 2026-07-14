"""
Extract NFL First 17 Career Games Performance

Instead of rookie season LEAF, calculate performance over first 17 career games
with progressive weighting (later games weighted more heavily).

This addresses issues like Drew Lock's early hot streak that didn't predict future performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import nfl_data_py as nfl


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent


def load_qb_list():
    """Load list of QBs to analyze"""
    root = get_project_root()
    path = root / 'data' / 'processed' / 'merged_with_sp_plus.csv'
    df = pd.read_csv(path)
    return df[['player_name', 'draft_year', 'draft_pick']].copy()


def convert_to_nfl_name(full_name):
    """
    Convert full name to nfl_data_py abbreviated format

    Examples:
    - "Patrick Mahomes" -> "P.Mahomes"
    - "Baker Mayfield" -> "B.Mayfield"
    """
    parts = full_name.split()
    if len(parts) < 2:
        return full_name

    first_initial = parts[0][0]
    last_name = parts[-1]

    return f"{first_initial}.{last_name}"


def get_first_17_games(player_name, draft_year):
    """
    Get first 17 career games for a QB

    Returns DataFrame of plays sorted by game sequence
    """
    # Convert name to nfl_data_py format
    nfl_name = convert_to_nfl_name(player_name)

    # Load multiple seasons of play-by-play data
    # Start from rookie year and go forward
    seasons = list(range(draft_year, min(draft_year + 5, 2024)))

    try:
        print(f"    Loading seasons {seasons}...")
        df_pbp = nfl.import_pbp_data(seasons, downcast=False)
    except Exception as e:
        print(f"    [ERROR] Failed to load data: {str(e)[:100]}")
        return pd.DataFrame()

    # Filter to this QB's plays (try both full name and abbreviated)
    qb_plays = df_pbp[
        (
            (df_pbp['passer_player_name'] == nfl_name) |
            (df_pbp['passer_player_name'] == player_name)
        ) &
        (df_pbp['pass'] == 1) &
        (df_pbp['qb_epa'].notna())
    ].copy()

    if len(qb_plays) == 0:
        print(f"    [--] No plays found")
        return pd.DataFrame()

    # Sort by game date
    qb_plays = qb_plays.sort_values(['season', 'week', 'game_id'])

    # Identify unique games
    games = qb_plays[['season', 'week', 'game_id']].drop_duplicates()

    # Take first 17 games
    first_17_games = games.head(17)

    if len(first_17_games) < 17:
        print(f"    [--] Only {len(first_17_games)} games found (need 17)")
        return pd.DataFrame()

    # Filter plays to first 17 games
    game_ids = first_17_games['game_id'].unique()
    qb_plays_17 = qb_plays[qb_plays['game_id'].isin(game_ids)].copy()

    # Add game sequence number
    game_id_to_seq = {gid: i+1 for i, gid in enumerate(first_17_games['game_id'])}
    qb_plays_17['game_seq'] = qb_plays_17['game_id'].map(game_id_to_seq)

    print(f"    [OK] {len(qb_plays_17)} plays across 17 games")
    return qb_plays_17


def calculate_weighted_leaf(df_plays, weighting='uniform'):
    """
    Calculate LEAF with different weighting schemes

    weighting options:
    - 'uniform': All games equally weighted (baseline)
    - 'linear_2x': Game 1 = 1.0x, Game 17 = 2.0x
    - 'linear_3x': Game 1 = 1.0x, Game 17 = 3.0x
    - 'exponential': Exponential growth (game 17 >> game 1)
    - 'back_half': Only last 9 games (games 9-17)
    - 'final_third': Only last 6 games (games 12-17)
    """
    if len(df_plays) == 0:
        return np.nan

    # Calculate game-level EPA
    game_stats = df_plays.groupby('game_seq').agg({
        'qb_epa': 'sum',
        'pass': 'count'
    }).reset_index()

    game_stats.columns = ['game_seq', 'total_epa', 'attempts']

    # Apply weighting
    if weighting == 'uniform':
        game_stats['weight'] = 1.0

    elif weighting == 'linear_2x':
        # Linear from 1.0x to 2.0x
        game_stats['weight'] = 1.0 + (game_stats['game_seq'] - 1) / 16

    elif weighting == 'linear_3x':
        # Linear from 1.0x to 3.0x
        game_stats['weight'] = 1.0 + 2.0 * (game_stats['game_seq'] - 1) / 16

    elif weighting == 'exponential':
        # Exponential: e^(game_seq/8) normalized
        weights = np.exp(game_stats['game_seq'] / 8)
        game_stats['weight'] = weights / weights.min()

    elif weighting == 'back_half':
        # Only games 9-17
        game_stats['weight'] = (game_stats['game_seq'] >= 9).astype(float)

    elif weighting == 'final_third':
        # Only games 12-17
        game_stats['weight'] = (game_stats['game_seq'] >= 12).astype(float)

    else:
        raise ValueError(f"Unknown weighting: {weighting}")

    # Calculate weighted LEAF
    game_stats['weighted_epa'] = game_stats['total_epa'] * game_stats['weight']
    game_stats['weighted_attempts'] = game_stats['attempts'] * game_stats['weight']

    total_weighted_epa = game_stats['weighted_epa'].sum()
    total_weighted_attempts = game_stats['weighted_attempts'].sum()

    if total_weighted_attempts > 0:
        return total_weighted_epa / total_weighted_attempts
    else:
        return np.nan


def main():
    """Main execution"""
    print("="*80)
    print("EXTRACTING NFL FIRST 17 CAREER GAMES")
    print("="*80)
    print("\nCalculating performance with progressive weighting")
    print("(Later games weighted more heavily to capture true talent)")

    # Load QB list
    print("\n[1] Loading QB list...")
    df_qbs = load_qb_list()
    print(f"    [OK] {len(df_qbs)} QBs to analyze")

    # Define weighting schemes to test
    weightings = [
        'uniform',
        'linear_2x',
        'linear_3x',
        'exponential',
        'back_half',
        'final_third'
    ]

    # Extract first 17 games for each QB
    print("\n[2] Extracting first 17 career games for each QB...")
    results = []

    for _, qb in df_qbs.iterrows():
        player_name = qb['player_name']
        draft_year = qb['draft_year']

        print(f"\n  {player_name} (drafted {draft_year}):")

        df_plays = get_first_17_games(player_name, draft_year)

        if len(df_plays) == 0:
            continue

        # Calculate all weighting schemes
        qb_result = {
            'player_name': player_name,
            'draft_year': draft_year,
            'draft_pick': qb['draft_pick'],
            'total_plays': len(df_plays),
            'total_games': 17
        }

        for weighting in weightings:
            leaf = calculate_weighted_leaf(df_plays, weighting)
            qb_result[f'leaf_{weighting}'] = leaf

        results.append(qb_result)
        print(f"    Calculated {len(weightings)} weighting schemes")

    df_results = pd.DataFrame(results)
    print(f"\n[OK] Extracted data for {len(df_results)} QBs with 17+ games")

    # Save results
    print("\n[3] Saving NFL first 17 games performance...")
    output_path = get_project_root() / 'data' / 'processed' / 'nfl_first_17_games.csv'
    df_results.to_csv(output_path, index=False)
    print(f"    [OK] Saved to: {output_path}")

    # Show summary statistics
    print("\n" + "="*80)
    print("WEIGHTING SCHEME COMPARISON")
    print("="*80)

    for weighting in weightings:
        col = f'leaf_{weighting}'
        if col in df_results.columns:
            mean = df_results[col].mean()
            std = df_results[col].std()
            print(f"\n{weighting:20s}:")
            print(f"  Mean: {mean:+.3f}")
            print(f"  Std:  {std:.3f}")

    # Show specific examples
    if len(df_results) > 0:
        print("\n" + "="*80)
        print("EXAMPLE: DREW LOCK")
        print("="*80)

        drew_lock = df_results[df_results['player_name'] == 'Drew Lock']
        if len(drew_lock) > 0:
            print("\nDrew Lock's performance by weighting scheme:")
            for weighting in weightings:
                col = f'leaf_{weighting}'
                if col in drew_lock.columns:
                    value = drew_lock[col].iloc[0]
                    print(f"  {weighting:20s}: {value:+.3f}")
        else:
            print("  Drew Lock not found in dataset")
    else:
        print("\n[WARNING] No QBs with 17+ games found - cannot show examples")

    print("\n" + "="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)
    print(f"\nNext Step: Correlate these weightings with college model predictions")
    print(f"to find optimal weighting scheme.")


if __name__ == "__main__":
    main()
