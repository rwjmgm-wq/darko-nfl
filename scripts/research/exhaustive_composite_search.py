"""
Phase 2: Exhaustive 5-Metric Composite Search

Tests ALL possible 5-metric combinations from top candidates.
Uses equal weighting (z-score standardized) for fair comparison.

For top N metrics: Tests C(N,5) combinations
- N=15: 3,003 combinations
- N=20: 15,504 combinations
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_pipeline.nfl_data_fetcher import NFLDataFetcher
from data_pipeline.receiver_stats_aggregator import ReceiverStatsAggregator

import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
from tqdm import tqdm
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def standardize_metric(series: pd.Series) -> pd.Series:
    """Z-score standardization."""
    return (series - series.mean()) / series.std()


def create_composite(
    data: pd.DataFrame,
    metrics: list,
    weights: list = None
) -> pd.Series:
    """
    Create composite metric from multiple metrics.

    Args:
        data: DataFrame with metrics
        metrics: List of metric column names
        weights: Optional weights (default: equal)

    Returns:
        Composite series (z-score standardized)
    """
    if weights is None:
        weights = [1.0 / len(metrics)] * len(metrics)

    # Standardize each metric
    composite = pd.Series(0.0, index=data.index)
    for metric, weight in zip(metrics, weights):
        if metric in data.columns:
            standardized = standardize_metric(data[metric])
            composite += weight * standardized

    return composite


def evaluate_combination(
    ratings_2023: pd.DataFrame,
    ratings_2024: pd.DataFrame,
    metrics: tuple
) -> dict:
    """
    Evaluate a single 5-metric combination.

    Args:
        ratings_2023: 2023 season ratings
        ratings_2024: 2024 season ratings
        metrics: Tuple of 5 metric names

    Returns:
        Dictionary with evaluation statistics
    """
    # Create composite for each year
    composite_2023 = create_composite(ratings_2023, list(metrics))
    composite_2024 = create_composite(ratings_2024, list(metrics))

    # Add to dataframes
    ratings_2023_with_composite = ratings_2023.copy()
    ratings_2023_with_composite['composite'] = composite_2023

    ratings_2024_with_composite = ratings_2024.copy()
    ratings_2024_with_composite['composite'] = composite_2024

    # Merge on receiver_player_id
    merged = ratings_2023_with_composite[['receiver_player_id', 'composite']].merge(
        ratings_2024_with_composite[['receiver_player_id', 'composite']],
        on='receiver_player_id',
        suffixes=('_2023', '_2024')
    )

    # Remove NaN
    merged = merged.dropna()

    if len(merged) < 10:
        return None

    # Calculate correlation
    try:
        correlation, p_value = stats.pearsonr(
            merged['composite_2023'],
            merged['composite_2024']
        )

        # Calculate error metrics
        error = merged['composite_2024'] - merged['composite_2023']
        mae = np.abs(error).mean()
        rmse = np.sqrt((error ** 2).mean())

        return {
            'metrics': metrics,
            'correlation': correlation,
            'p_value': p_value,
            'n_players': len(merged),
            'mae': mae,
            'rmse': rmse
        }
    except Exception as e:
        return None


def main():
    """Run Phase 2: Exhaustive combination search."""

    logger.info("="*80)
    logger.info("PHASE 2: EXHAUSTIVE 5-METRIC COMPOSITE SEARCH")
    logger.info("="*80)

    # ==========================================
    # Load top metrics from Phase 1
    # ==========================================
    logger.info("\n[1/4] Loading top metrics from Phase 1...")

    top_metrics_file = Path('data/processed/top_20_metrics.txt')
    if not top_metrics_file.exists():
        logger.error("top_20_metrics.txt not found. Run analyze_metric_candidates.py first.")
        return

    with open(top_metrics_file, 'r') as f:
        top_metrics = [line.strip() for line in f.readlines() if line.strip()]

    # Use top 15 for manageable computation (3,003 combinations)
    # Can increase to 20 (15,504 combinations) if needed
    N_METRICS = 15
    top_metrics = top_metrics[:N_METRICS]

    logger.info(f"  Using top {len(top_metrics)} metrics")
    for i, metric in enumerate(top_metrics, 1):
        logger.info(f"    {i}. {metric}")

    # Calculate number of combinations
    n_combinations = len(list(combinations(range(len(top_metrics)), 5)))
    logger.info(f"\n  Total combinations to test: {n_combinations:,}")
    estimated_time = n_combinations * 0.002  # ~2ms per combination
    logger.info(f"  Estimated time: {estimated_time/60:.1f} minutes")

    # ==========================================
    # Load data
    # ==========================================
    logger.info("\n[2/4] Loading data...")

    fetcher = NFLDataFetcher()
    pbp_2023 = fetcher.fetch_pbp_data([2023])
    pbp_2024 = fetcher.fetch_pbp_data([2024])

    # Fetch roster data for WR-only filtering
    rosters = fetcher.fetch_rosters([2023, 2024])

    aggregator = ReceiverStatsAggregator(min_targets=1, filter_wr_only=True)
    game_stats_2023 = aggregator.aggregate_game_stats(pbp_2023, roster_data=rosters)
    game_stats_2024 = aggregator.aggregate_game_stats(pbp_2024, roster_data=rosters)

    ratings_2023 = aggregator.aggregate_season_stats(game_stats_2023, min_targets=30)
    ratings_2024 = aggregator.aggregate_season_stats(game_stats_2024, min_targets=30)

    logger.info(f"  2023: {len(ratings_2023)} WRs")
    logger.info(f"  2024: {len(ratings_2024)} WRs")

    # ==========================================
    # Test all combinations
    # ==========================================
    logger.info("\n[3/4] Testing all 5-metric combinations...")

    results = []
    start_time = time.time()

    for metric_combo in tqdm(combinations(top_metrics, 5), total=n_combinations, desc="Testing combinations"):
        result = evaluate_combination(ratings_2023, ratings_2024, metric_combo)
        if result is not None:
            results.append(result)

    elapsed_time = time.time() - start_time
    logger.info(f"\n  Completed {len(results):,} combinations in {elapsed_time:.1f} seconds")
    logger.info(f"  Average: {elapsed_time/len(results)*1000:.2f}ms per combination")

    # ==========================================
    # Rank and save results
    # ==========================================
    logger.info("\n[4/4] Ranking and saving results...")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('correlation', ascending=False)

    # Add rank
    results_df['rank'] = range(1, len(results_df) + 1)

    # Convert metrics tuple to string for CSV
    results_df['metrics_str'] = results_df['metrics'].apply(lambda x: ' + '.join(x))

    # Save full results
    output_dir = Path('data/processed')
    results_df.to_csv(output_dir / 'exhaustive_search_results.csv', index=False)

    # Save top 100
    top_100 = results_df.head(100)
    top_100.to_csv(output_dir / 'top_100_combinations.csv', index=False)

    logger.info(f"  Saved {len(results_df):,} results to exhaustive_search_results.csv")
    logger.info(f"  Saved top 100 to top_100_combinations.csv")

    # ==========================================
    # Display results
    # ==========================================
    print(f"\n{'='*80}")
    print("TOP 20 5-METRIC COMBINATIONS")
    print(f"{'='*80}")

    for _, row in top_100.head(20).iterrows():
        metrics_str = ' + '.join(row['metrics'])
        print(f"\nRank {int(row['rank'])}: r = {row['correlation']:.4f} (p = {row['p_value']:.4e})")
        print(f"  {metrics_str}")

    # ==========================================
    # Summary statistics
    # ==========================================
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Total combinations tested: {len(results_df):,}")
    print(f"Statistically significant (p<0.05): {(results_df['p_value'] < 0.05).sum():,}")
    print(f"\nCorrelation distribution:")
    print(f"  Mean: {results_df['correlation'].mean():.4f}")
    print(f"  Median: {results_df['correlation'].median():.4f}")
    print(f"  Std: {results_df['correlation'].std():.4f}")
    print(f"  Min: {results_df['correlation'].min():.4f}")
    print(f"  Max: {results_df['correlation'].max():.4f}")
    print(f"\nTop combinations:")
    print(f"  > 0.90: {(results_df['correlation'] > 0.90).sum():,}")
    print(f"  > 0.85: {(results_df['correlation'] > 0.85).sum():,}")
    print(f"  > 0.80: {(results_df['correlation'] > 0.80).sum():,}")

    # ==========================================
    # Best combination details
    # ==========================================
    best = results_df.iloc[0]
    print(f"\n{'='*80}")
    print("BEST 5-METRIC COMBINATION")
    print(f"{'='*80}")
    print(f"Metrics: {' + '.join(best['metrics'])}")
    print(f"Correlation: {best['correlation']:.4f}")
    print(f"P-value: {best['p_value']:.4e}")
    print(f"Sample size: {int(best['n_players'])} players")
    print(f"MAE: {best['mae']:.4f}")
    print(f"RMSE: {best['rmse']:.4f}")

    # Compare to single best metric
    single_metric_file = Path('data/processed/metric_candidate_analysis.csv')
    if single_metric_file.exists():
        single_metrics = pd.read_csv(single_metric_file)
        best_single = single_metrics.iloc[0]
        print(f"\n{'='*80}")
        print("COMPARISON TO BEST SINGLE METRIC")
        print(f"{'='*80}")
        print(f"Best single metric: {best_single['metric']}")
        print(f"  Correlation: {best_single['correlation']:.4f}")
        print(f"\nBest 5-metric composite:")
        print(f"  Correlation: {best['correlation']:.4f}")
        print(f"  Improvement: {(best['correlation'] - best_single['correlation']):.4f} ({(best['correlation']/best_single['correlation'] - 1)*100:+.2f}%)")


if __name__ == "__main__":
    main()
