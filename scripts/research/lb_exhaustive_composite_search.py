"""
Linebacker (LB) Exhaustive Composite Metric Search

Tests all possible 3-5 metric combinations for linebackers to find which
metrics best predict future season performance.

Tracks sacks, QB hits, TFLs, forced fumbles, tackles, and EPA against.
Expected to emphasize tackles and run defense more than EDGE/DT positions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_pipeline.nfl_data_fetcher import NFLDataFetcher
from data_pipeline.defensive_stats_aggregator import DefensiveStatsAggregator

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from itertools import combinations
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LB CANDIDATE METRICS
# Focus on tackles, run defense, gap filling, and coverage ability
LB_METRICS = [
    # Volume metrics (raw counts)
    'sacks', 'qb_hits', 'tfls', 'forced_fumbles',
    'solo_tackles', 'assist_tackles', 'total_tackles',
    'pressures', 'impact_plays',

    # Per-game efficiency metrics
    'sacks_per_game', 'qb_hits_per_game', 'tfls_per_game',
    'pressures_per_game', 'impact_plays_per_game', 'tackles_per_game',

    # EPA metrics (higher = better for defense)
    'epa_against', 'epa_against_pass', 'epa_against_run',

    # Play participation
    'total_plays', 'pass_rush_plays', 'run_defense_plays',
]


def load_lb_data(years=range(2020, 2025)):
    """Load linebacker defensive data."""

    logger.info("="*80)
    logger.info("LOADING LINEBACKER (LB) DATA")
    logger.info("="*80)

    fetcher = NFLDataFetcher()
    # Use edge_positions=['LB'] only to capture linebackers
    aggregator = DefensiveStatsAggregator(min_plays=1, edge_positions=['LB'])

    all_lb_stats = []

    for year in years:
        logger.info(f"\nLoading {year} season...")
        pbp = fetcher.fetch_pbp_data([year])
        roster = fetcher.fetch_rosters([year])

        # Get game stats
        game_stats = aggregator.aggregate_game_stats(pbp, roster)

        logger.info(f"  Found {len(game_stats)} LB games, {game_stats['player_id'].nunique()} unique LBs")

        # Aggregate to season level (50+ plays minimum)
        season_stats = aggregator.aggregate_season_stats(game_stats, min_plays=50)
        season_stats['season'] = year

        all_lb_stats.append(season_stats)

    lb_data = pd.concat(all_lb_stats, ignore_index=True)
    logger.info(f"\nLoaded {len(lb_data)} LB-seasons")
    logger.info(f"Unique LBs: {lb_data['player_id'].nunique()}")

    return lb_data


def test_metric_combination(df, metrics, target_col='next_season_tackles_per_game'):
    """Test a combination of metrics for predictive power."""

    # Check if all metrics exist
    missing = [m for m in metrics if m not in df.columns]
    if missing:
        return None

    # Get complete cases
    test_cols = metrics + [target_col]
    valid_data = df[test_cols].dropna()

    if len(valid_data) < 30:  # Need sufficient sample
        return None

    # Calculate equal-weighted composite (z-score normalization)
    composite = valid_data[metrics].apply(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    ).mean(axis=1)

    # Correlation with future performance
    corr, p_value = spearmanr(composite, valid_data[target_col])

    return {
        'metrics': metrics,
        'n_metrics': len(metrics),
        'correlation': corr,
        'p_value': p_value,
        'n_samples': len(valid_data),
        'metric_names': ', '.join(metrics)
    }


def exhaustive_search(df, target_col, min_metrics=3, max_metrics=5):
    """Test all combinations of metrics."""

    logger.info(f"\n{'='*80}")
    logger.info(f"EXHAUSTIVE COMPOSITE SEARCH")
    logger.info(f"{'='*80}")
    logger.info(f"Target: {target_col}")
    logger.info(f"Testing {min_metrics}-{max_metrics} metric combinations")

    # Get available metrics
    available_metrics = [m for m in LB_METRICS if m in df.columns]
    logger.info(f"\nAvailable metrics: {len(available_metrics)}")
    logger.info(f"{', '.join(available_metrics)}")

    results = []
    total_combinations = 0

    for n in range(min_metrics, max_metrics + 1):
        combos = list(combinations(available_metrics, n))
        total_combinations += len(combos)
        logger.info(f"\nTesting {len(combos):,} {n}-metric combinations...")

        for i, combo in enumerate(combos):
            if (i + 1) % 500 == 0:
                logger.info(f"  Progress: {i+1:,}/{len(combos):,}")

            result = test_metric_combination(df, list(combo), target_col)
            if result is not None:
                results.append(result)

    logger.info(f"\nTested {total_combinations:,} total combinations")
    logger.info(f"Valid results: {len(results):,}")

    return pd.DataFrame(results)


def main():
    # Load data
    lb_data = load_lb_data(years=range(2020, 2025))

    logger.info(f"\n{'='*80}")
    logger.info("DATA SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total LB-seasons: {len(lb_data)}")
    logger.info(f"Unique players: {lb_data['player_id'].nunique()}")
    logger.info(f"Seasons: {sorted(lb_data['season'].unique())}")
    logger.info(f"\nPosition breakdown:")
    logger.info(lb_data['position'].value_counts().to_string())

    # Create next season target variables
    lb_data = lb_data.sort_values(['player_id', 'season'])

    # Test multiple target variables to find best predictor
    # LBs may be better predicted by tackles/run defense than pass rush
    target_variables = {
        'next_season_tackles_per_game': 'tackles_per_game',
        'next_season_tfls_per_game': 'tfls_per_game',
        'next_season_impact_plays_per_game': 'impact_plays_per_game',
        'next_season_epa_against': 'epa_against',
        'next_season_epa_against_run': 'epa_against_run',
    }

    for target_name, source_col in target_variables.items():
        lb_data[target_name] = lb_data.groupby('player_id')[source_col].shift(-1)

    # Filter to valid samples (players with next season data)
    lb_data_filtered = lb_data[lb_data['next_season_tackles_per_game'].notna()].copy()

    logger.info(f"\nValid samples (with next season data): {len(lb_data_filtered)}")
    logger.info(f"Unique players: {lb_data_filtered['player_id'].nunique()}")

    # Run exhaustive search for each target variable
    all_results = {}

    for target_name in target_variables.keys():
        logger.info(f"\n{'='*80}")
        logger.info(f"SEARCHING FOR: {target_name}")
        logger.info(f"{'='*80}")

        results = exhaustive_search(
            lb_data_filtered,
            target_col=target_name,
            min_metrics=3,
            max_metrics=5
        )

        # Sort by correlation
        results = results.sort_values('correlation', ascending=False)
        all_results[target_name] = results

        # Display top results
        logger.info(f"\n{'='*80}")
        logger.info(f"TOP 10 COMBINATIONS FOR {target_name}")
        logger.info(f"{'='*80}")

        for idx, (i, row) in enumerate(results.head(10).iterrows()):
            logger.info(f"\n#{idx+1} (r={row['correlation']:.4f}, p={row['p_value']:.4f}, n={row['n_samples']}):")
            for metric in row['metrics']:
                logger.info(f"  - {metric}")

    # Find overall best combination across all targets
    logger.info(f"\n{'='*80}")
    logger.info("BEST COMBINATIONS BY TARGET VARIABLE")
    logger.info(f"{'='*80}")

    best_overall = None
    best_overall_corr = 0

    for target_name, results in all_results.items():
        if len(results) > 0:
            best = results.iloc[0]
            logger.info(f"\n{target_name}:")
            logger.info(f"  Correlation: {best['correlation']:.4f} (p={best['p_value']:.4f})")
            logger.info(f"  Metrics ({best['n_metrics']}): {best['metric_names']}")
            logger.info(f"  Sample size: {best['n_samples']}")

            if best['correlation'] > best_overall_corr:
                best_overall_corr = best['correlation']
                best_overall = (target_name, best)

    # Save results
    output_file = "data/processed/lb_exhaustive_search_results.csv"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Combine all results
    combined_results = []
    for target_name, results in all_results.items():
        results_copy = results.copy()
        results_copy['target_variable'] = target_name
        combined_results.append(results_copy)

    combined_df = pd.concat(combined_results, ignore_index=True)
    combined_df.to_csv(output_file, index=False)
    logger.info(f"\nSaved full results to: {output_file}")

    # Recommend best combination
    logger.info(f"\n{'='*80}")
    logger.info("RECOMMENDED COMPOSITE FOR LINEBACKERS")
    logger.info(f"{'='*80}")

    if best_overall:
        target_name, best = best_overall
        logger.info(f"\nBest predictor: {target_name}")
        logger.info(f"Correlation: {best['correlation']:.4f} (p={best['p_value']:.4f})")
        logger.info(f"Sample size: {best['n_samples']}")
        logger.info(f"\nRecommended metrics:")
        for metric in best['metrics']:
            logger.info(f"  - {metric}")

    logger.info(f"\n{'='*80}")
    logger.info("COMPLETE")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
