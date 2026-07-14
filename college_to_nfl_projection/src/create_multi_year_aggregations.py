"""
Create Multi-Year Aggregation Methods

Tests 6 different approaches to aggregating multi-year college careers:
1. Final season only (baseline - what we currently do)
2. Career average (mean across all seasons)
3. Recency-weighted (final=50%, junior=30%, sophomore=20%)
4. Cumulative career totals
5. Best single season (peak performance)
6. Volume-weighted (seasons weighted by attempts)

Each method will generate the full 7-feature ensemble plus additional stats.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent


def load_sp_plus_ratings():
    """Load SP+ ratings for opponent adjustment"""
    root = get_project_root()
    sp_plus_path = root / 'data' / 'processed' / 'sp_plus_historical' / 'sp_plus_all_years.csv'

    if sp_plus_path.exists():
        df_sp = pd.read_csv(sp_plus_path)
        return df_sp
    else:
        print("    [WARN] No SP+ data found - will use raw EPA")
        return None


def calculate_season_stats(df_season, sp_plus_df=None):
    """
    Calculate comprehensive stats for a single season

    Returns dict with all relevant statistics
    """
    if len(df_season) == 0:
        return None

    stats = {}

    # Basic volume
    stats['attempts'] = len(df_season)

    # EPA metrics
    epa_values = df_season['epa'].dropna()
    if len(epa_values) > 0:
        stats['epa_per_play'] = epa_values.mean()
        stats['total_epa'] = epa_values.sum()
    else:
        stats['epa_per_play'] = np.nan
        stats['total_epa'] = np.nan

    # Success rate (EPA > 0)
    stats['success_rate'] = (epa_values > 0).mean() if len(epa_values) > 0 else np.nan

    # Big plays (15+ yards)
    stats['big_play_rate'] = (df_season['yards_gained'] >= 15).mean()

    # Third down performance
    third_down = df_season[df_season['down'] == 3]
    if len(third_down) > 0:
        third_down_epa = third_down['epa'].dropna()
        stats['third_down_epa'] = third_down_epa.mean() if len(third_down_epa) > 0 else np.nan
        stats['third_down_attempts'] = len(third_down)
    else:
        stats['third_down_epa'] = np.nan
        stats['third_down_attempts'] = 0

    # Red zone performance (inside 20)
    red_zone = df_season[df_season['yards_to_goal'] <= 20]
    if len(red_zone) > 0:
        red_zone_epa = red_zone['epa'].dropna()
        stats['red_zone_epa'] = red_zone_epa.mean() if len(red_zone_epa) > 0 else np.nan
        stats['red_zone_attempts'] = len(red_zone)
    else:
        stats['red_zone_epa'] = np.nan
        stats['red_zone_attempts'] = 0

    # High leverage (3rd/4th down + red zone)
    high_leverage = df_season[
        (df_season['down'].isin([3, 4])) |
        (df_season['yards_to_goal'] <= 20)
    ]
    if len(high_leverage) > 0:
        high_leverage_epa = high_leverage['epa'].dropna()
        stats['high_leverage_epa'] = high_leverage_epa.mean() if len(high_leverage_epa) > 0 else np.nan
        stats['high_leverage_rate'] = len(high_leverage) / len(df_season)
    else:
        stats['high_leverage_epa'] = np.nan
        stats['high_leverage_rate'] = 0

    # Scoring plays
    stats['scoring_play_rate'] = df_season['scoring'].mean() if 'scoring' in df_season else 0

    return stats


def aggregate_final_season(career_data):
    """Method 1: Final season only (baseline)"""
    final_season = career_data['season'].max()
    df_final = career_data[career_data['season'] == final_season]
    return calculate_season_stats(df_final)


def aggregate_career_average(career_data):
    """Method 2: Average across all seasons"""
    seasons = career_data['season'].unique()

    # Calculate stats for each season
    season_stats = []
    for season in seasons:
        df_season = career_data[career_data['season'] == season]
        stats = calculate_season_stats(df_season)
        if stats:
            season_stats.append(stats)

    if not season_stats:
        return None

    # Average each stat across seasons
    avg_stats = {}
    for key in season_stats[0].keys():
        values = [s[key] for s in season_stats if not pd.isna(s[key])]
        avg_stats[key] = np.mean(values) if values else np.nan

    return avg_stats


def aggregate_recency_weighted(career_data):
    """Method 3: Recency-weighted (final=50%, junior=30%, sophomore=20%)"""
    seasons = sorted(career_data['season'].unique())
    n_seasons = len(seasons)

    # Define weights based on number of seasons
    if n_seasons == 1:
        weights = [1.0]
    elif n_seasons == 2:
        weights = [0.3, 0.7]  # sophomore, junior
    elif n_seasons == 3:
        weights = [0.2, 0.3, 0.5]  # sophomore, junior, senior
    elif n_seasons == 4:
        weights = [0.1, 0.2, 0.3, 0.4]  # freshman through senior
    else:
        # 5+ seasons (redshirt): distribute remaining 20% across early years
        weights = [0.05] * (n_seasons - 3) + [0.2, 0.3, 0.5]

    # Normalize weights to sum to 1.0
    weights = np.array(weights)
    weights = weights / weights.sum()

    # Calculate stats for each season
    season_stats = []
    for season in seasons:
        df_season = career_data[career_data['season'] == season]
        stats = calculate_season_stats(df_season)
        if stats:
            season_stats.append(stats)

    if not season_stats:
        return None

    # Weighted average
    weighted_stats = {}
    for key in season_stats[0].keys():
        values = [s[key] for s in season_stats]
        # Handle NaN values
        valid_mask = ~pd.isna(values)
        if valid_mask.any():
            valid_values = np.array(values)[valid_mask]
            valid_weights = weights[valid_mask]
            valid_weights = valid_weights / valid_weights.sum()  # Renormalize
            weighted_stats[key] = np.sum(valid_values * valid_weights)
        else:
            weighted_stats[key] = np.nan

    return weighted_stats


def aggregate_cumulative(career_data):
    """Method 4: Cumulative career totals"""
    stats = {}

    # Volume
    stats['attempts'] = len(career_data)

    # EPA metrics (cumulative makes sense)
    epa_values = career_data['epa'].dropna()
    if len(epa_values) > 0:
        stats['epa_per_play'] = epa_values.mean()  # Still per-play
        stats['total_epa'] = epa_values.sum()  # Cumulative total
    else:
        stats['epa_per_play'] = np.nan
        stats['total_epa'] = np.nan

    # Success rate (overall)
    stats['success_rate'] = (epa_values > 0).mean() if len(epa_values) > 0 else np.nan

    # Big plays (overall rate)
    stats['big_play_rate'] = (career_data['yards_gained'] >= 15).mean()

    # Third down (overall)
    third_down = career_data[career_data['down'] == 3]
    if len(third_down) > 0:
        third_down_epa = third_down['epa'].dropna()
        stats['third_down_epa'] = third_down_epa.mean() if len(third_down_epa) > 0 else np.nan
        stats['third_down_attempts'] = len(third_down)
    else:
        stats['third_down_epa'] = np.nan
        stats['third_down_attempts'] = 0

    # Red zone (overall)
    red_zone = career_data[career_data['yards_to_goal'] <= 20]
    if len(red_zone) > 0:
        red_zone_epa = red_zone['epa'].dropna()
        stats['red_zone_epa'] = red_zone_epa.mean() if len(red_zone_epa) > 0 else np.nan
        stats['red_zone_attempts'] = len(red_zone)
    else:
        stats['red_zone_epa'] = np.nan
        stats['red_zone_attempts'] = 0

    # High leverage (overall)
    high_leverage = career_data[
        (career_data['down'].isin([3, 4])) |
        (career_data['yards_to_goal'] <= 20)
    ]
    if len(high_leverage) > 0:
        high_leverage_epa = high_leverage['epa'].dropna()
        stats['high_leverage_epa'] = high_leverage_epa.mean() if len(high_leverage_epa) > 0 else np.nan
        stats['high_leverage_rate'] = len(high_leverage) / len(career_data)
    else:
        stats['high_leverage_epa'] = np.nan
        stats['high_leverage_rate'] = 0

    # Scoring plays
    stats['scoring_play_rate'] = career_data['scoring'].mean() if 'scoring' in career_data else 0

    return stats


def aggregate_best_season(career_data):
    """Method 5: Best single season (by EPA per play)"""
    seasons = career_data['season'].unique()

    # Calculate stats for each season
    season_stats = []
    for season in seasons:
        df_season = career_data[career_data['season'] == season]
        stats = calculate_season_stats(df_season)
        if stats and not pd.isna(stats.get('epa_per_play')):
            stats['season'] = season
            season_stats.append(stats)

    if not season_stats:
        return None

    # Find best season by EPA per play
    best_season = max(season_stats, key=lambda x: x['epa_per_play'])

    # Remove season marker
    del best_season['season']

    return best_season


def aggregate_volume_weighted(career_data):
    """Method 6: Volume-weighted (seasons weighted by attempts)"""
    seasons = career_data['season'].unique()

    # Calculate stats and volume for each season
    season_stats = []
    season_attempts = []

    for season in seasons:
        df_season = career_data[career_data['season'] == season]
        stats = calculate_season_stats(df_season)
        if stats:
            season_stats.append(stats)
            season_attempts.append(stats['attempts'])

    if not season_stats:
        return None

    # Calculate weights based on attempts
    total_attempts = sum(season_attempts)
    weights = [attempts / total_attempts for attempts in season_attempts]

    # Weighted average
    weighted_stats = {}
    for key in season_stats[0].keys():
        values = [s[key] for s in season_stats]
        # Handle NaN values
        valid_mask = ~pd.isna(values)
        if valid_mask.any():
            valid_values = np.array(values)[valid_mask]
            valid_weights = np.array(weights)[valid_mask]
            valid_weights = valid_weights / valid_weights.sum()  # Renormalize
            weighted_stats[key] = np.sum(valid_values * valid_weights)
        else:
            weighted_stats[key] = np.nan

    return weighted_stats


def main():
    """Main execution"""
    print("="*80)
    print("CREATING MULTI-YEAR AGGREGATION METHODS")
    print("="*80)
    print("\nTesting 6 approaches to aggregating college careers")

    root = get_project_root()

    # Load career summary to know which QBs we have
    print("\n[1] Loading career data...")
    career_dir = root / 'data' / 'processed' / 'full_college_careers'
    summary_path = career_dir / 'career_summary.csv'

    if not summary_path.exists():
        print("    [ERROR] Career data not found. Run collect_full_college_careers.py first")
        return

    df_summary = pd.read_csv(summary_path)
    print(f"    [OK] Found {len(df_summary)} QBs with career data")

    # Load SP+ if available
    print("\n[2] Loading SP+ ratings...")
    sp_plus_df = load_sp_plus_ratings()
    if sp_plus_df is not None:
        print(f"    [OK] Loaded SP+ data for {len(sp_plus_df)} team-seasons")
    else:
        print("    [WARN] Proceeding without SP+ adjustment")

    # Process each QB with all 6 aggregation methods
    print(f"\n[3] Computing stats for all 6 aggregation methods...")
    print("-"*80)

    all_results = {
        'final_season': [],
        'career_average': [],
        'recency_weighted': [],
        'cumulative': [],
        'best_season': [],
        'volume_weighted': []
    }

    for idx, row in df_summary.iterrows():
        player_name = row['player_name']
        draft_year = row['draft_year']
        career_file = career_dir / row['file']

        print(f"  [{idx+1}/{len(df_summary)}] {player_name} ({draft_year})", end=' ')

        # Load career data
        df_career = pd.read_csv(career_file)

        if len(df_career) == 0:
            print("[SKIP] No plays")
            continue

        # Apply all 6 aggregation methods
        methods = {
            'final_season': aggregate_final_season,
            'career_average': aggregate_career_average,
            'recency_weighted': aggregate_recency_weighted,
            'cumulative': aggregate_cumulative,
            'best_season': aggregate_best_season,
            'volume_weighted': aggregate_volume_weighted
        }

        method_results = {}
        for method_name, method_func in methods.items():
            stats = method_func(df_career)
            if stats:
                stats['player_name'] = player_name
                stats['draft_year'] = draft_year
                stats['college'] = row['college']
                stats['seasons_played'] = row['seasons_played']
                method_results[method_name] = stats

        if method_results:
            for method_name, stats in method_results.items():
                all_results[method_name].append(stats)
            print(f"[OK] {len(method_results)} methods")
        else:
            print("[FAIL] No stats computed")

    # Save results for each aggregation method
    print("\n[4] Saving aggregated datasets...")
    output_dir = root / 'data' / 'processed' / 'aggregated_stats'
    output_dir.mkdir(parents=True, exist_ok=True)

    for method_name, results in all_results.items():
        if results:
            df_method = pd.DataFrame(results)
            output_path = output_dir / f'stats_{method_name}.csv'
            df_method.to_csv(output_path, index=False)
            print(f"    [OK] {method_name}: {len(df_method)} QBs -> {output_path.name}")

    # Summary comparison
    print("\n" + "="*80)
    print("AGGREGATION METHOD COMPARISON")
    print("="*80)

    print("\n  QBs successfully processed per method:")
    for method_name, results in all_results.items():
        print(f"    {method_name}: {len(results)} QBs")

    # Compare key stats across methods
    if all(results for results in all_results.values()):
        print("\n  Average EPA/play by method:")
        for method_name, results in all_results.items():
            df_method = pd.DataFrame(results)
            avg_epa = df_method['epa_per_play'].mean()
            print(f"    {method_name}: {avg_epa:+.3f}")

        print("\n  Average attempts by method:")
        for method_name, results in all_results.items():
            df_method = pd.DataFrame(results)
            avg_attempts = df_method['attempts'].mean()
            print(f"    {method_name}: {avg_attempts:.0f}")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n  1. Extract 5 NFL outcome metrics")
    print("  2. Test which aggregation method predicts each outcome best")
    print("  3. Test which stats within each method predict best")
    print("  4. Generate comprehensive comparison report")


if __name__ == "__main__":
    main()
