"""
Extract Comprehensive NFL Outcomes (5 Metrics)

Calculates all NFL outcome measures for testing college stat predictions:
1. Rookie LEAF performance (continuous - existing metric)
2. 85+ career starts (binary - longevity, exclude 2020-2023 draftees)
3. Bust classification (binary - never exceeded 40th percentile OR ≤35 starts)
4. Elite peak (binary - achieved 10-game rolling stretch > 0.15 EPA/play)
5. Sustained starter (binary - 30+ consecutive starts)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import nfl_data_py as nfl


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent


def convert_to_nfl_name(full_name):
    """
    Convert full name to nfl_data_py abbreviated format
    Examples: "Patrick Mahomes" -> "P.Mahomes"
    """
    parts = full_name.split()
    if len(parts) < 2:
        return full_name

    first_initial = parts[0][0]
    last_name = parts[-1]

    return f"{first_initial}.{last_name}"


def calculate_leaf_rating(df_games):
    """
    Calculate LEAF (Layered EPA Adaptive Framework) rating

    Uses EPA components with appropriate weighting
    """
    if len(df_games) == 0 or df_games['attempts'].sum() == 0:
        return np.nan

    # Weight games by attempts (more weight to games with more action)
    total_attempts = df_games['attempts'].sum()
    weights = df_games['attempts'] / total_attempts

    # Weighted average EPA per play
    leaf = (df_games['epa'] * weights).sum()

    return leaf


def extract_rookie_leaf(df_pbp, player_name, draft_year):
    """
    Outcome 1: Rookie season LEAF performance
    """
    rookie_season = draft_year

    # Filter to rookie season
    df_rookie = df_pbp[
        (df_pbp['season'] == rookie_season) &
        (df_pbp['passer_player_name'] == player_name)
    ]

    if len(df_rookie) == 0:
        return np.nan

    # Aggregate by game
    game_stats = df_rookie.groupby('game_id').agg({
        'epa': 'mean',
        'passer_player_name': 'count'
    }).rename(columns={'passer_player_name': 'attempts'})

    # Calculate LEAF
    leaf = calculate_leaf_rating(game_stats)

    return leaf


def extract_career_starts(df_pbp, player_name):
    """
    Count total career starts

    A "start" = game where QB had 10+ attempts
    """
    # Filter to this player
    df_player = df_pbp[df_pbp['passer_player_name'] == player_name]

    if len(df_player) == 0:
        return 0

    # Count attempts per game
    game_attempts = df_player.groupby(['season', 'week']).size()

    # Count games with 10+ attempts as starts
    starts = (game_attempts >= 10).sum()

    return starts


def extract_85_starts_outcome(df_pbp, player_name, draft_year):
    """
    Outcome 2: Reached 85+ career starts (binary)

    Exclude QBs drafted 2020-2023 (insufficient time)
    """
    if draft_year >= 2020:
        return np.nan  # Not enough time to reach 85 starts

    starts = extract_career_starts(df_pbp, player_name)

    return 1 if starts >= 85 else 0


def calculate_rolling_epa(df_pbp, player_name, window=10):
    """
    Calculate rolling window EPA per play

    Returns max and min rolling EPA achieved
    """
    # Filter to this player
    df_player = df_pbp[df_pbp['passer_player_name'] == player_name].copy()

    if len(df_player) == 0:
        return np.nan, np.nan

    # Sort by season and week
    df_player = df_player.sort_values(['season', 'week'])

    # Aggregate by game
    game_stats = df_player.groupby(['season', 'week']).agg({
        'epa': 'mean'
    }).reset_index()

    if len(game_stats) < window:
        return np.nan, np.nan

    # Calculate rolling mean
    game_stats['rolling_epa'] = game_stats['epa'].rolling(window=window, min_periods=window).mean()

    max_rolling = game_stats['rolling_epa'].max()
    min_rolling = game_stats['rolling_epa'].min()

    return max_rolling, min_rolling


def extract_bust_outcome(df_pbp, player_name):
    """
    Outcome 3: Bust classification (binary)

    Bust = NEVER had 10-game rolling stretch > -0.035 EPA/play OR ≤35 starts

    User specified: 40th percentile = -0.035 EPA/play
    """
    starts = extract_career_starts(df_pbp, player_name)

    # Automatic bust if ≤35 starts
    if starts <= 35:
        return 1

    # Check rolling performance
    max_rolling, _ = calculate_rolling_epa(df_pbp, player_name, window=10)

    if pd.isna(max_rolling):
        return 1  # Bust if insufficient data

    # Bust if never exceeded 40th percentile
    if max_rolling <= -0.035:
        return 1

    return 0


def extract_elite_peak_outcome(df_pbp, player_name):
    """
    Outcome 4: Elite peak (binary)

    Elite = Achieved 10-game rolling stretch > 0.15 EPA/play

    User specified: 0.15 EPA/play threshold
    """
    max_rolling, _ = calculate_rolling_epa(df_pbp, player_name, window=10)

    if pd.isna(max_rolling):
        return 0

    return 1 if max_rolling > 0.15 else 0


def extract_sustained_starter_outcome(df_pbp, player_name):
    """
    Outcome 5: Sustained starter (binary)

    Sustained starter = 30+ consecutive starts (games with 10+ attempts)
    """
    # Filter to this player
    df_player = df_pbp[df_pbp['passer_player_name'] == player_name]

    if len(df_player) == 0:
        return 0

    # Count attempts per game
    game_attempts = df_player.groupby(['season', 'week']).size().reset_index(name='attempts')

    # Mark games as starts (10+ attempts)
    game_attempts['is_start'] = (game_attempts['attempts'] >= 10).astype(int)

    # Find longest consecutive streak
    max_streak = 0
    current_streak = 0

    for is_start in game_attempts['is_start']:
        if is_start == 1:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    return 1 if max_streak >= 30 else 0


def main():
    """Main execution"""
    print("="*80)
    print("EXTRACTING COMPREHENSIVE NFL OUTCOMES (5 METRICS)")
    print("="*80)
    print("\n1. Rookie LEAF (continuous)")
    print("2. 85+ starts (binary, exclude 2020-2023)")
    print("3. Bust (binary, never > -0.035 EPA OR <=35 starts)")
    print("4. Elite peak (binary, 10-game rolling > 0.15 EPA)")
    print("5. Sustained starter (binary, 30+ consecutive starts)")

    root = get_project_root()

    # Load expanded QB list
    print("\n[1] Loading QB list...")
    qb_list_path = root / 'data' / 'processed' / 'qb_list_expanded.csv'

    if not qb_list_path.exists():
        print("    [ERROR] QB list not found")
        return

    df_qbs = pd.read_csv(qb_list_path)
    print(f"    [OK] {len(df_qbs)} QBs ({df_qbs['draft_year'].min()}-{df_qbs['draft_year'].max()})")

    # Load NFL play-by-play data
    print("\n[2] Loading NFL play-by-play data...")
    print("    This may take several minutes...")

    min_year = df_qbs['draft_year'].min()
    max_year = 2025  # Through current season

    years = list(range(min_year, max_year + 1))

    try:
        df_pbp = nfl.import_pbp_data(years, downcast=False)
        print(f"    [OK] Loaded {len(df_pbp):,} plays from {min_year}-{max_year}")
    except Exception as e:
        print(f"    [ERROR] Failed to load NFL data: {str(e)}")
        return

    # Filter to passing plays with EPA
    df_pbp = df_pbp[
        (df_pbp['play_type'] == 'pass') &
        (df_pbp['epa'].notna()) &
        (df_pbp['passer_player_name'].notna())
    ]
    print(f"    [OK] {len(df_pbp):,} passing plays with EPA")

    # Extract outcomes for each QB
    print(f"\n[3] Calculating outcomes for {len(df_qbs)} QBs...")
    print("-"*80)

    results = []

    for idx, row in df_qbs.iterrows():
        player_name = row['player_name']
        draft_year = row['draft_year']

        # Try both full name and abbreviated name
        nfl_name = convert_to_nfl_name(player_name)

        print(f"  [{idx+1}/{len(df_qbs)}] {player_name} ({draft_year})", end=' ')

        # Check which name format exists in data
        if player_name in df_pbp['passer_player_name'].values:
            search_name = player_name
        elif nfl_name in df_pbp['passer_player_name'].values:
            search_name = nfl_name
        else:
            print("[NOT FOUND]")
            results.append({
                'player_name': player_name,
                'draft_year': draft_year,
                'nfl_name_used': None,
                'rookie_leaf': np.nan,
                'reached_85_starts': np.nan,
                'is_bust': np.nan,
                'is_elite': np.nan,
                'is_sustained_starter': np.nan,
                'total_starts': 0,
                'max_rolling_epa': np.nan,
                'min_rolling_epa': np.nan
            })
            continue

        # Extract all 5 outcomes
        try:
            # Outcome 1: Rookie LEAF
            rookie_leaf = extract_rookie_leaf(df_pbp, search_name, draft_year)

            # Outcome 2: 85+ starts
            reached_85 = extract_85_starts_outcome(df_pbp, search_name, draft_year)

            # Outcome 3: Bust
            is_bust = extract_bust_outcome(df_pbp, search_name)

            # Outcome 4: Elite peak
            is_elite = extract_elite_peak_outcome(df_pbp, search_name)

            # Outcome 5: Sustained starter
            is_sustained = extract_sustained_starter_outcome(df_pbp, search_name)

            # Additional metrics for analysis
            total_starts = extract_career_starts(df_pbp, search_name)
            max_rolling, min_rolling = calculate_rolling_epa(df_pbp, search_name, window=10)

            results.append({
                'player_name': player_name,
                'draft_year': draft_year,
                'nfl_name_used': search_name,
                'rookie_leaf': rookie_leaf,
                'reached_85_starts': reached_85,
                'is_bust': is_bust,
                'is_elite': is_elite,
                'is_sustained_starter': is_sustained,
                'total_starts': total_starts,
                'max_rolling_epa': max_rolling,
                'min_rolling_epa': min_rolling
            })

            # Status summary
            status = []
            if not pd.isna(rookie_leaf):
                status.append(f"LEAF={rookie_leaf:+.3f}")
            if reached_85 == 1:
                status.append("85+ starts")
            if is_elite == 1:
                status.append("ELITE")
            if is_bust == 1:
                status.append("BUST")
            if is_sustained == 1:
                status.append("SUSTAINED")

            print(f"[OK] {', '.join(status) if status else 'Extracted'}")

        except Exception as e:
            print(f"[ERROR] {str(e)[:50]}")
            results.append({
                'player_name': player_name,
                'draft_year': draft_year,
                'nfl_name_used': search_name,
                'rookie_leaf': np.nan,
                'reached_85_starts': np.nan,
                'is_bust': np.nan,
                'is_elite': np.nan,
                'is_sustained_starter': np.nan,
                'total_starts': 0,
                'max_rolling_epa': np.nan,
                'min_rolling_epa': np.nan
            })

    # Save results
    print("\n[4] Saving NFL outcomes...")
    df_outcomes = pd.DataFrame(results)
    output_path = root / 'data' / 'processed' / 'nfl_outcomes_comprehensive.csv'
    df_outcomes.to_csv(output_path, index=False)
    print(f"    [OK] Saved to: {output_path}")

    # Summary statistics
    print("\n" + "="*80)
    print("OUTCOME SUMMARY")
    print("="*80)

    # QBs with data
    qbs_with_data = df_outcomes[df_outcomes['nfl_name_used'].notna()]
    print(f"\n  QBs found in NFL data: {len(qbs_with_data)}/{len(df_outcomes)}")

    # Outcome 1: Rookie LEAF
    valid_leaf = df_outcomes['rookie_leaf'].dropna()
    if len(valid_leaf) > 0:
        print(f"\n  Outcome 1: Rookie LEAF")
        print(f"    Valid measurements: {len(valid_leaf)} QBs")
        print(f"    Mean: {valid_leaf.mean():+.3f}")
        print(f"    Std:  {valid_leaf.std():.3f}")
        print(f"    Range: {valid_leaf.min():+.3f} to {valid_leaf.max():+.3f}")

    # Outcome 2: 85+ starts
    valid_85 = df_outcomes['reached_85_starts'].dropna()
    if len(valid_85) > 0:
        reached = (valid_85 == 1).sum()
        print(f"\n  Outcome 2: 85+ Career Starts")
        print(f"    Eligible QBs: {len(valid_85)} (excluding 2020-2023)")
        print(f"    Reached 85+: {reached} ({reached/len(valid_85)*100:.1f}%)")

    # Outcome 3: Bust
    valid_bust = df_outcomes['is_bust'].dropna()
    if len(valid_bust) > 0:
        busts = (valid_bust == 1).sum()
        print(f"\n  Outcome 3: Bust Classification")
        print(f"    Valid measurements: {len(valid_bust)} QBs")
        print(f"    Busts: {busts} ({busts/len(valid_bust)*100:.1f}%)")

    # Outcome 4: Elite
    valid_elite = df_outcomes['is_elite'].dropna()
    if len(valid_elite) > 0:
        elites = (valid_elite == 1).sum()
        print(f"\n  Outcome 4: Elite Peak")
        print(f"    Valid measurements: {len(valid_elite)} QBs")
        print(f"    Elite peaks: {elites} ({elites/len(valid_elite)*100:.1f}%)")

    # Outcome 5: Sustained starter
    valid_sustained = df_outcomes['is_sustained_starter'].dropna()
    if len(valid_sustained) > 0:
        sustained = (valid_sustained == 1).sum()
        print(f"\n  Outcome 5: Sustained Starter")
        print(f"    Valid measurements: {len(valid_sustained)} QBs")
        print(f"    Sustained starters: {sustained} ({sustained/len(valid_sustained)*100:.1f}%)")

    # Cross-tabulations
    print("\n" + "="*80)
    print("OUTCOME RELATIONSHIPS")
    print("="*80)

    # Elite vs 85 starts
    both_metrics = df_outcomes[
        df_outcomes['is_elite'].notna() &
        df_outcomes['reached_85_starts'].notna()
    ]

    if len(both_metrics) > 0:
        elite_and_85 = ((both_metrics['is_elite'] == 1) & (both_metrics['reached_85_starts'] == 1)).sum()
        elite_no_85 = ((both_metrics['is_elite'] == 1) & (both_metrics['reached_85_starts'] == 0)).sum()

        print(f"\n  Elite QBs who reached 85 starts: {elite_and_85}")
        print(f"  Elite QBs who didn't reach 85: {elite_no_85}")

    # Bust vs other outcomes
    valid_all = df_outcomes[
        df_outcomes['is_bust'].notna() &
        df_outcomes['is_elite'].notna()
    ]

    if len(valid_all) > 0:
        bust_count = (valid_all['is_bust'] == 1).sum()
        elite_count = (valid_all['is_elite'] == 1).sum()
        neither = len(valid_all) - bust_count - elite_count

        print(f"\n  Distribution:")
        print(f"    Busts: {bust_count}")
        print(f"    Elite: {elite_count}")
        print(f"    Middle tier: {neither}")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n  1. Test college stats from all 6 aggregation methods")
    print("  2. Predict all 5 NFL outcomes")
    print("  3. Find optimal stat combinations for each outcome")
    print("  4. Generate comprehensive comparison report")


if __name__ == "__main__":
    main()
