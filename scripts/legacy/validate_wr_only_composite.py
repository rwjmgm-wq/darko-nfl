"""
Quick Validation: WR-Only Data with Recommended Composite

Tests the recommended 5-metric composite (Rank 4 from multi-year validation)
on clean WR-only data to ensure RB/TE contamination didn't affect results.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_pipeline.nfl_data_fetcher import NFLDataFetcher
from data_pipeline.receiver_stats_aggregator import ReceiverStatsAggregator

import pandas as pd
import numpy as np
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def standardize_metrics(data: pd.DataFrame, metrics: list) -> pd.DataFrame:
    """Standardize metrics to z-scores."""
    standardized = data.copy()
    for metric in metrics:
        if metric in data.columns:
            standardized[metric] = (data[metric] - data[metric].mean()) / data[metric].std()
    return standardized


def evaluate_composite(
    ratings_2023: pd.DataFrame,
    ratings_2024: pd.DataFrame,
    metrics: list,
    weights: list
) -> dict:
    """Evaluate composite on a year pair."""
    # Standardize
    ratings_2023_std = standardize_metrics(ratings_2023, metrics)
    ratings_2024_std = standardize_metrics(ratings_2024, metrics)

    # Get common players
    common_players = set(ratings_2023['receiver_player_id']) & set(ratings_2024['receiver_player_id'])

    # Filter and dedup
    train_2023 = ratings_2023_std[ratings_2023_std['receiver_player_id'].isin(common_players)]
    train_2024 = ratings_2024_std[ratings_2024_std['receiver_player_id'].isin(common_players)]

    train_2023 = train_2023.drop_duplicates('receiver_player_id', keep='first')
    train_2024 = train_2024.drop_duplicates('receiver_player_id', keep='first')

    train_2023 = train_2023.sort_values('receiver_player_id').reset_index(drop=True)
    train_2024 = train_2024.sort_values('receiver_player_id').reset_index(drop=True)

    # Filter to complete data
    train_2023_complete = train_2023[['receiver_player_id'] + metrics].dropna()
    train_2024_complete = train_2024[['receiver_player_id'] + metrics].dropna()

    valid_players = set(train_2023_complete['receiver_player_id']) & set(train_2024_complete['receiver_player_id'])

    train_2023_clean = train_2023[train_2023['receiver_player_id'].isin(valid_players)].copy()
    train_2024_clean = train_2024[train_2024['receiver_player_id'].isin(valid_players)].copy()

    train_2023_clean = train_2023_clean.sort_values('receiver_player_id').reset_index(drop=True)
    train_2024_clean = train_2024_clean.sort_values('receiver_player_id').reset_index(drop=True)

    # Create weighted composite
    composite_2023 = (train_2023_clean[metrics].values * weights).sum(axis=1)
    composite_2024 = (train_2024_clean[metrics].values * weights).sum(axis=1)

    # Calculate correlation
    correlation, p_value = stats.pearsonr(composite_2023, composite_2024)

    return {
        'correlation': correlation,
        'p_value': p_value,
        'n_players': len(valid_players)
    }


def main():
    """Quick validation of recommended composite on WR-only data."""

    logger.info("="*80)
    logger.info("WR-ONLY DATA VALIDATION")
    logger.info("="*80)

    # Recommended composite from multi-year validation (Rank 4)
    metrics = ['air_epa_per_target', 'air_yards', 'total_yac_epa', 'targets_per_game', 'total_air_epa']
    weights = [0.26087403, 0.3179893, 0.09457006, 0.20259696, 0.12396965]

    logger.info("\nRecommended Composite (Rank 4 - Most Stable):")
    logger.info("  Metrics & Weights:")
    for metric, weight in zip(metrics, weights):
        logger.info(f"    {weight:6.3f} × {metric}")

    # Fetch data with WR-only filtering
    logger.info("\nFetching data with WR-only filtering...")
    fetcher = NFLDataFetcher()

    years = [2023, 2024]
    pbp_2023 = fetcher.fetch_pbp_data([2023])
    pbp_2024 = fetcher.fetch_pbp_data([2024])
    rosters = fetcher.fetch_rosters(years)

    # Aggregate with WR-only filter
    aggregator = ReceiverStatsAggregator(min_targets=1, filter_wr_only=True)

    game_stats_2023 = aggregator.aggregate_game_stats(pbp_2023, roster_data=rosters)
    game_stats_2024 = aggregator.aggregate_game_stats(pbp_2024, roster_data=rosters)

    ratings_2023 = aggregator.aggregate_season_stats(game_stats_2023, min_targets=30)
    ratings_2024 = aggregator.aggregate_season_stats(game_stats_2024, min_targets=30)

    logger.info(f"\n2023: {len(ratings_2023)} WRs (30+ targets)")
    logger.info(f"2024: {len(ratings_2024)} WRs (30+ targets)")

    # Evaluate composite
    logger.info("\nEvaluating composite on WR-only data...")
    result = evaluate_composite(ratings_2023, ratings_2024, metrics, weights)

    # Display results
    print(f"\n{'='*80}")
    print("RESULTS: WR-ONLY DATA VALIDATION")
    print(f"{'='*80}")
    print(f"Correlation: r = {result['correlation']:.4f}")
    print(f"P-value: p = {result['p_value']:.4e}")
    print(f"Sample size: n = {result['n_players']} WRs")

    if result['p_value'] < 0.001:
        print("\nStatistical Significance: HIGHLY SIGNIFICANT (p < 0.001)")
    elif result['p_value'] < 0.01:
        print("\nStatistical Significance: VERY SIGNIFICANT (p < 0.01)")
    elif result['p_value'] < 0.05:
        print("\nStatistical Significance: SIGNIFICANT (p < 0.05)")
    else:
        print(f"\nStatistical Significance: NOT SIGNIFICANT (p = {result['p_value']:.4f})")

    print(f"\n{'='*80}")
    print("COMPARISON TO ORIGINAL RESULTS (Mixed WR/RB/TE)")
    print(f"{'='*80}")
    print("Original 2023->2024 (with RBs/TEs): r = 0.9010, n = 152")
    print(f"WR-only 2023->2024:                  r = {result['correlation']:.4f}, n = {result['n_players']}")

    diff = result['correlation'] - 0.9010
    print(f"\nDifference: {diff:+.4f}")

    if abs(diff) < 0.01:
        print("Result: STABLE (correlation changed < 0.01)")
    elif diff > 0:
        print("Result: IMPROVED (WR-only data shows stronger signal)")
    else:
        print("Result: SLIGHTLY LOWER (but still very strong)")

    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print(f"{'='*80}")

    if result['correlation'] > 0.85:
        print(f"The composite shows EXCELLENT predictive power on clean WR-only data.")
        print(f"  r = {result['correlation']:.4f} indicates strong year-over-year stability.")
    elif result['correlation'] > 0.75:
        print(f"The composite shows STRONG predictive power on clean WR-only data.")
        print(f"  r = {result['correlation']:.4f} is suitable for a performance rating system.")
    else:
        print(f"The composite shows MODERATE predictive power on clean WR-only data.")
        print(f"  r = {result['correlation']:.4f} suggests further refinement may help.")

    print("\nNext steps:")
    print("  1. Integrate this composite into WR LEAF pipeline")
    print("  2. Replace old base metric with this 5-metric composite")
    print("  3. Add to QB visualization dashboard as WR quality metric")


if __name__ == "__main__":
    main()
