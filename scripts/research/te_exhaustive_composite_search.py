"""
TE Exhaustive Composite Metric Search

Tests all possible 3-5 metric combinations for tight ends to find which
metrics best predict future season performance.

Similar to WR/RB analysis but adapted for TE-specific metrics.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_pipeline.nfl_data_fetcher import NFLDataFetcher
from data_pipeline.receiver_stats_aggregator import ReceiverStatsAggregator

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from itertools import combinations
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TE-SPECIFIC CANDIDATE METRICS
# TEs have unique role - blocking + receiving, more intermediate routes
TE_METRICS = [
    # Volume metrics
    'targets', 'targets_per_game', 'receptions', 'receptions_per_game',

    # Efficiency metrics
    'yards_per_target', 'yards_per_reception', 'catch_rate',

    # EPA/value metrics
    'epa_per_target', 'total_epa', 'air_epa_per_target', 'yac_epa_per_target',

    # Yardage metrics
    'receiving_yards', 'yards_per_game', 'air_yards_share',
    'total_yac', 'yac_per_reception', 'total_yac_epa',

    # Route/target quality
    'racr', 'first_downs', 'first_down_rate',
    'touchdown_rate', 'touchdowns',

    # Depth of target (TEs more intermediate)
    'avg_depth_of_target', 'avg_air_yards',

    # Opportunity
    'target_share', 'wopr'  # Weighted Opportunity Rating
]

def load_te_data(years=range(2020, 2024)):
    """Load TE receiving data."""

    logger.info("="*80)
    logger.info("LOADING TE DATA")
    logger.info("="*80)

    fetcher = NFLDataFetcher()
    # Use ReceiverStatsAggregator but filter for TEs only
    aggregator = ReceiverStatsAggregator(filter_wr_only=False)  # Include all receivers

    all_te_stats = []

    for year in years:
        logger.info(f"\nLoading {year} season...")
        pbp = fetcher.fetch_pbp_data([year])
        roster = fetcher.fetch_rosters([year])

        # Get game stats (with position info merged)
        game_stats = aggregator.aggregate_game_stats(pbp, roster)

        # Filter to TEs only (position column is merged by aggregator)
        if 'position' in game_stats.columns:
            te_games = game_stats[game_stats['position'] == 'TE'].copy()
        else:
            # Fallback: match receiver IDs with TE roster
            roster_lookup = roster.groupby(['season', 'player_id'])['position'].agg(
                lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
            ).reset_index()
            game_stats_with_pos = game_stats.merge(
                roster_lookup,
                left_on=['season', 'receiver_player_id'],
                right_on=['season', 'player_id'],
                how='left'
            )
            te_games = game_stats_with_pos[game_stats_with_pos['position'] == 'TE'].copy()

        logger.info(f"  Found {len(te_games)} TE games, {te_games['receiver_player_id'].nunique()} unique TEs")

        # Aggregate to season level (with lower min_targets threshold)
        season_stats = aggregator.aggregate_season_stats(te_games, min_targets=15)
        season_stats['season'] = year

        all_te_stats.append(season_stats)

    te_data = pd.concat(all_te_stats, ignore_index=True)
    logger.info(f"\nLoaded {len(te_data)} TE-seasons")
    logger.info(f"TEs: {te_data['receiver_player_id'].nunique()}")

    return te_data

def calculate_available_metrics(df):
    """Calculate derived metrics from base stats."""

    # Ensure numeric
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Per-game metrics
    if 'games_played' in df.columns:
        if 'targets' in df.columns:
            df['targets_per_game'] = df['targets'] / df['games_played']
        if 'receptions' in df.columns:
            df['receptions_per_game'] = df['receptions'] / df['games_played']
        if 'receiving_yards' in df.columns:
            df['yards_per_game'] = df['receiving_yards'] / df['games_played']

    # Efficiency metrics
    if 'receptions' in df.columns and 'targets' in df.columns:
        df['catch_rate'] = df['receptions'] / df['targets'].replace(0, np.nan)

    if 'receiving_yards' in df.columns and 'targets' in df.columns:
        df['yards_per_target'] = df['receiving_yards'] / df['targets'].replace(0, np.nan)

    if 'receiving_yards' in df.columns and 'receptions' in df.columns:
        df['yards_per_reception'] = df['receiving_yards'] / df['receptions'].replace(0, np.nan)

    if 'first_downs' in df.columns and 'receptions' in df.columns:
        df['first_down_rate'] = df['first_downs'] / df['receptions'].replace(0, np.nan)

    if 'touchdowns' in df.columns and 'receptions' in df.columns:
        df['touchdown_rate'] = df['touchdowns'] / df['receptions'].replace(0, np.nan)

    # YAC metrics
    if 'total_yac' in df.columns and 'receptions' in df.columns:
        df['yac_per_reception'] = df['total_yac'] / df['receptions'].replace(0, np.nan)

    return df

def test_metric_combination(df, metrics, target_col='next_season_epa'):
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

    # Calculate equal-weighted composite
    composite = valid_data[metrics].apply(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0).mean(axis=1)

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
    available_metrics = [m for m in TE_METRICS if m in df.columns]
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
    te_data = load_te_data(years=range(2020, 2024))

    # Calculate derived metrics
    te_data = calculate_available_metrics(te_data)

    # Create next season target variable
    te_data = te_data.sort_values(['receiver_player_id', 'season'])
    te_data['next_season_epa'] = te_data.groupby('receiver_player_id')['epa_per_target'].shift(-1)

    # Filter to TEs with reasonable sample (15+ targets)
    te_data_filtered = te_data[te_data['targets'] >= 15].copy()
    logger.info(f"\nFiltered to {len(te_data_filtered)} TE-seasons with 15+ targets")

    # Run exhaustive search
    results = exhaustive_search(
        te_data_filtered,
        target_col='next_season_epa',
        min_metrics=3,
        max_metrics=5
    )

    # Sort by correlation
    results = results.sort_values('correlation', ascending=False)

    # Display top results
    logger.info(f"\n{'='*80}")
    logger.info("TOP 20 METRIC COMBINATIONS")
    logger.info(f"{'='*80}")

    for i, row in results.head(20).iterrows():
        logger.info(f"\n#{i+1} (r={row['correlation']:.4f}, p={row['p_value']:.4f}, n={row['n_samples']}):")
        for metric in row['metrics']:
            logger.info(f"  - {metric}")

    # Save results
    output_file = "data/processed/te_exhaustive_search_results.csv"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_file, index=False)
    logger.info(f"\nSaved full results to: {output_file}")

    # Summary by metric count
    logger.info(f"\n{'='*80}")
    logger.info("BEST COMBINATION BY METRIC COUNT")
    logger.info(f"{'='*80}")

    for n in sorted(results['n_metrics'].unique()):
        best = results[results['n_metrics'] == n].iloc[0]
        logger.info(f"\n{n} metrics (r={best['correlation']:.4f}):")
        logger.info(f"  {best['metric_names']}")

    logger.info(f"\n{'='*80}")
    logger.info("COMPLETE")
    logger.info(f"{'='*80}")

if __name__ == "__main__":
    main()
