"""
DEPRECATED - do not use. Superseded by v2_build_adjusted_stats.py.

BUG: this script's "adjustment" is a no-op. It computes avg_defense as the
national mean of ALL teams' defense ratings (never the QB's actual opponents),
then z-scores that mean against itself, so sos_percentile is always exactly 50
and epa_per_play_adj always equals epa_per_play_raw. Every "SP+ adjusted"
dataset it produced is identical to the raw data.

The v2 script computes real schedule strength from the games table and applies
an additive, empirically calibrated adjustment.

---
Apply SP+ Competition Adjustments to Full College Careers

Now that we have historical SP+ ratings (2003-2023), we can apply competition
adjustments at the play-by-play level, then re-aggregate with 6 different methods.

This provides a much more accurate assessment than just using final season SP+.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent


def load_sp_plus_ratings():
    """Load historical SP+ ratings"""
    root = get_project_root()
    sp_plus_path = root / 'data' / 'processed' / 'sp_plus_historical' / 'sp_plus_all_years.csv'

    if not sp_plus_path.exists():
        print("[ERROR] SP+ data not found")
        print(f"        Run fetch_historical_sp_plus.py first")
        return None

    df_sp = pd.read_csv(sp_plus_path)
    print(f"    [OK] Loaded {len(df_sp)} team-seasons of SP+ data")
    print(f"         Years: {df_sp['season'].min()}-{df_sp['season'].max()}")

    return df_sp


def calculate_season_stats_adjusted(df_season, df_sp_plus):
    """
    Calculate competition-adjusted stats for a single season

    Uses SP+ ratings to weight EPA against opponent strength:
    - Opponent defense rating adjusts offensive EPA
    - Adjust per-play EPA based on opponent quality

    Returns dict with adjusted and raw statistics
    """
    if len(df_season) == 0:
        return None

    season = df_season['season'].iloc[0]

    stats = {}

    # Basic volume
    stats['attempts'] = len(df_season)
    stats['season'] = season

    # Raw EPA metrics (unadjusted baseline)
    epa_values = df_season['epa'].dropna()
    if len(epa_values) > 0:
        stats['epa_per_play_raw'] = epa_values.mean()
        stats['total_epa_raw'] = epa_values.sum()
        stats['success_rate_raw'] = (epa_values > 0).mean()
    else:
        stats['epa_per_play_raw'] = np.nan
        stats['total_epa_raw'] = np.nan
        stats['success_rate_raw'] = np.nan

    # Get season average SP+ defensive rating (average opponent quality)
    season_sp = df_sp_plus[df_sp_plus['season'] == season]

    if len(season_sp) == 0:
        # No SP+ data for this season - use raw stats only
        stats['epa_per_play_adj'] = stats['epa_per_play_raw']
        stats['avg_opponent_sp_defense'] = np.nan
        stats['sos_percentile'] = np.nan
    else:
        # Calculate average opponent defensive strength
        avg_defense = season_sp['defense_rating'].mean()
        std_defense = season_sp['defense_rating'].std()

        stats['avg_opponent_sp_defense'] = avg_defense

        # Calculate SOS percentile (0-100, higher = tougher schedule)
        # Defensive rating: lower is better, so we flip it
        # A tougher defense (lower rating) should give higher SOS percentile
        if std_defense > 0:
            sos_zscore = -(avg_defense - season_sp['defense_rating'].mean()) / std_defense
            sos_percentile = 50 + (sos_zscore * 15)  # Convert to 0-100 scale (centered at 50)
            stats['sos_percentile'] = np.clip(sos_percentile, 0, 100)
        else:
            stats['sos_percentile'] = 50.0

        # Adjust EPA based on opponent strength
        # QBs facing tougher defenses get a bonus, easier defenses get a penalty
        # Formula: EPA_adj = EPA_raw * (1 + adjustment_factor)
        # adjustment_factor = (sos_percentile - 50) * 0.01  # ±0.5 max adjustment
        adjustment_factor = (stats['sos_percentile'] - 50) * 0.005  # ±0.25 max adjustment
        stats['epa_per_play_adj'] = stats['epa_per_play_raw'] * (1 + adjustment_factor)

    # Big plays (raw)
    if 'yards_gained' in df_season.columns:
        stats['big_play_rate'] = (df_season['yards_gained'] >= 15).mean()
    else:
        stats['big_play_rate'] = np.nan

    # Third down performance (raw)
    if 'down' in df_season.columns:
        third_down = df_season[df_season['down'] == 3]
        if len(third_down) > 0:
            third_down_epa = third_down['epa'].dropna()
            stats['third_down_epa'] = third_down_epa.mean() if len(third_down_epa) > 0 else np.nan
            stats['third_down_attempts'] = len(third_down)
        else:
            stats['third_down_epa'] = np.nan
            stats['third_down_attempts'] = 0
    else:
        stats['third_down_epa'] = np.nan
        stats['third_down_attempts'] = 0

    # Red zone performance (raw) - only if yards_to_goal is available
    if 'yards_to_goal' in df_season.columns:
        red_zone = df_season[df_season['yards_to_goal'] <= 20]
        if len(red_zone) > 0:
            red_zone_epa = red_zone['epa'].dropna()
            stats['red_zone_epa'] = red_zone_epa.mean() if len(red_zone_epa) > 0 else np.nan
            stats['red_zone_attempts'] = len(red_zone)
        else:
            stats['red_zone_epa'] = np.nan
            stats['red_zone_attempts'] = 0
    else:
        # For CFBD data without yards_to_goal, estimate from touchdown rate
        stats['red_zone_epa'] = np.nan
        stats['red_zone_attempts'] = 0

    # High leverage (raw) - modified to work without yards_to_goal
    if 'down' in df_season.columns and 'yards_to_goal' in df_season.columns:
        high_leverage = df_season[
            (df_season['down'].isin([3, 4])) |
            (df_season['yards_to_goal'] <= 20)
        ]
    elif 'down' in df_season.columns:
        # Just use third/fourth down as proxy
        high_leverage = df_season[df_season['down'].isin([3, 4])]
    else:
        high_leverage = pd.DataFrame()

    if len(high_leverage) > 0:
        high_leverage_epa = high_leverage['epa'].dropna()
        stats['high_leverage_epa'] = high_leverage_epa.mean() if len(high_leverage_epa) > 0 else np.nan
        stats['high_leverage_rate'] = len(high_leverage) / len(df_season)
    else:
        stats['high_leverage_epa'] = np.nan
        stats['high_leverage_rate'] = 0

    # Scoring plays - handle both formats
    if 'scoring' in df_season.columns:
        stats['scoring_play_rate'] = df_season['scoring'].mean()
    elif 'touchdown' in df_season.columns:
        # CFBD data uses 'touchdown' column
        stats['scoring_play_rate'] = df_season['touchdown'].mean()
    else:
        stats['scoring_play_rate'] = 0

    return stats


def aggregate_career_with_adjustments(career_data, df_sp_plus, aggregation_method):
    """
    Aggregate career with competition adjustments

    First calculates season-level stats with SP+ adjustments,
    then applies one of 6 aggregation methods
    """
    seasons = sorted(career_data['season'].unique())

    # Calculate adjusted stats for each season
    season_stats = []
    for season in seasons:
        df_season = career_data[career_data['season'] == season]
        stats = calculate_season_stats_adjusted(df_season, df_sp_plus)
        if stats:
            season_stats.append(stats)

    if not season_stats:
        return None

    # Apply aggregation method
    if aggregation_method == 'final_season':
        # Just return the final season
        return season_stats[-1]

    elif aggregation_method == 'career_average':
        # Average across all seasons
        avg_stats = {}
        for key in season_stats[0].keys():
            if key == 'season':
                continue
            values = [s[key] for s in season_stats if not pd.isna(s.get(key))]
            avg_stats[key] = np.mean(values) if values else np.nan
        return avg_stats

    elif aggregation_method == 'recency_weighted':
        # Weight recent seasons more heavily
        n_seasons = len(season_stats)
        if n_seasons == 1:
            weights = [1.0]
        elif n_seasons == 2:
            weights = [0.3, 0.7]
        elif n_seasons == 3:
            weights = [0.2, 0.3, 0.5]
        elif n_seasons == 4:
            weights = [0.1, 0.2, 0.3, 0.4]
        else:
            weights = [0.05] * (n_seasons - 3) + [0.2, 0.3, 0.5]

        weights = np.array(weights) / sum(weights)

        weighted_stats = {}
        for key in season_stats[0].keys():
            if key == 'season':
                continue
            values = [s.get(key) for s in season_stats]
            valid_mask = ~pd.isna(values)
            if valid_mask.any():
                valid_values = np.array(values)[valid_mask]
                valid_weights = weights[valid_mask]
                valid_weights = valid_weights / valid_weights.sum()
                weighted_stats[key] = np.sum(valid_values * valid_weights)
            else:
                weighted_stats[key] = np.nan
        return weighted_stats

    elif aggregation_method == 'cumulative':
        # Cumulative totals across career
        return calculate_season_stats_adjusted(career_data, df_sp_plus)

    elif aggregation_method == 'best_season':
        # Best season by adjusted EPA per play
        best_season = max(season_stats, key=lambda x: x.get('epa_per_play_adj', -999))
        return best_season

    elif aggregation_method == 'volume_weighted':
        # Weight by attempts per season
        attempts = [s['attempts'] for s in season_stats]
        total_attempts = sum(attempts)
        weights = [a / total_attempts for a in attempts]

        weighted_stats = {}
        for key in season_stats[0].keys():
            if key == 'season':
                continue
            values = [s.get(key) for s in season_stats]
            valid_mask = ~pd.isna(values)
            if valid_mask.any():
                valid_values = np.array(values)[valid_mask]
                valid_weights = np.array(weights)[valid_mask] / np.array(weights)[valid_mask].sum()
                weighted_stats[key] = np.sum(valid_values * valid_weights)
            else:
                weighted_stats[key] = np.nan
        return weighted_stats

    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")


def main():
    """Main execution"""
    print("="*80)
    print("APPLYING SP+ COMPETITION ADJUSTMENTS TO COLLEGE CAREERS")
    print("="*80)
    print("\nThis applies opponent-strength adjustments at the play-by-play level")
    print("and re-aggregates with all 6 methods using adjusted metrics.")

    root = get_project_root()

    # Load SP+ ratings
    print("\n[1] Loading SP+ ratings...")
    df_sp_plus = load_sp_plus_ratings()
    if df_sp_plus is None:
        return

    # Load career summary to know which QBs we have
    print("\n[2] Loading career data...")
    career_dir = root / 'data' / 'processed' / 'full_college_careers'
    summary_path = career_dir / 'career_summary.csv'

    if not summary_path.exists():
        print("    [ERROR] Career data not found")
        return

    df_summary = pd.read_csv(summary_path)
    print(f"    [OK] Found {len(df_summary)} QBs with career data")

    # Process each QB with all 6 aggregation methods
    print(f"\n[3] Computing SP+ adjusted stats for all 6 aggregation methods...")
    print("-"*80)

    aggregation_methods = [
        'final_season',
        'career_average',
        'recency_weighted',
        'cumulative',
        'best_season',
        'volume_weighted'
    ]

    all_results = {method: [] for method in aggregation_methods}

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

        # Apply all 6 aggregation methods with SP+ adjustments
        for method in aggregation_methods:
            stats = aggregate_career_with_adjustments(df_career, df_sp_plus, method)
            if stats:
                stats['player_name'] = player_name
                stats['draft_year'] = draft_year
                stats['college'] = row['college']
                stats['seasons_played'] = row['seasons_played']
                all_results[method].append(stats)

        print(f"[OK]")

    # Save results for each aggregation method
    print("\n[4] Saving SP+ adjusted aggregated datasets...")
    output_dir = root / 'data' / 'processed' / 'aggregated_stats_sp_adjusted'
    output_dir.mkdir(parents=True, exist_ok=True)

    for method_name, results in all_results.items():
        if results:
            df_method = pd.DataFrame(results)
            output_path = output_dir / f'stats_{method_name}_sp_adjusted.csv'
            df_method.to_csv(output_path, index=False)
            print(f"    [OK] {method_name}: {len(df_method)} QBs -> {output_path.name}")

    # Summary comparison
    print("\n" + "="*80)
    print("SP+ ADJUSTED AGGREGATION SUMMARY")
    print("="*80)

    print("\n  QBs successfully processed per method:")
    for method_name, results in all_results.items():
        print(f"    {method_name}: {len(results)} QBs")

    # Compare adjusted vs raw EPA
    if all(results for results in all_results.values()):
        print("\n  Average EPA/play comparison (Raw vs SP+ Adjusted):")
        for method_name, results in all_results.items():
            df_method = pd.DataFrame(results)
            if 'epa_per_play_raw' in df_method.columns and 'epa_per_play_adj' in df_method.columns:
                avg_raw = df_method['epa_per_play_raw'].mean()
                avg_adj = df_method['epa_per_play_adj'].mean()
                diff = avg_adj - avg_raw
                print(f"    {method_name}: Raw={avg_raw:+.3f} Adj={avg_adj:+.3f} (diff={diff:+.3f})")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n  1. Re-run test_comprehensive_models.py with SP+ adjusted stats")
    print("  2. Compare predictive power: Raw EPA vs SP+ Adjusted EPA")
    print("  3. Determine if competition adjustment improves NFL projection accuracy")


if __name__ == "__main__":
    main()
