"""
QB Composite Analysis - Phase 1: Metric Candidate Selection

Identifies the top QB metrics with highest predictive power for year-over-year performance.
Similar to WR analysis but for quarterbacks.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_pipeline.nfl_data_fetcher import NFLDataFetcher
from data_pipeline.qb_stats_aggregator import QBStatsAggregator

import pandas as pd
import numpy as np
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_year_over_year_correlation(metric_name, year1_data, year2_data):
    """Calculate correlation for a single metric across two years."""

    # Merge on player ID
    merged = year1_data[['passer_player_id', metric_name]].merge(
        year2_data[['passer_player_id', metric_name]],
        on='passer_player_id',
        suffixes=('_y1', '_y2')
    )

    # Remove NaN/inf values
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna()

    if len(merged) < 10:  # Need minimum sample size
        return None, None, 0

    # Calculate correlation
    r, p = stats.pearsonr(merged[f'{metric_name}_y1'], merged[f'{metric_name}_y2'])

    return r, p, len(merged)

def main():
    """Run Phase 1: Metric candidate selection for QBs."""

    logger.info("="*80)
    logger.info("QB COMPOSITE ANALYSIS - PHASE 1: METRIC CANDIDATES")
    logger.info("="*80)

    # ==================================================
    # Load Data
    # ==================================================
    logger.info("\n[1/3] Loading QB data for 2023-2024...")

    fetcher = NFLDataFetcher()

    # Load 2023 and 2024 data
    pbp_2023 = fetcher.fetch_pbp_data([2023])
    pbp_2024 = fetcher.fetch_pbp_data([2024])

    # Filter to pass plays
    pbp_2023_pass = pbp_2023[pbp_2023['play_type'] == 'pass'].copy()
    pbp_2024_pass = pbp_2024[pbp_2024['play_type'] == 'pass'].copy()

    logger.info(f"  2023: {len(pbp_2023_pass):,} pass plays")
    logger.info(f"  2024: {len(pbp_2024_pass):,} pass plays")

    # Aggregate to season level
    aggregator = QBStatsAggregator()

    game_stats_2023 = aggregator.aggregate_game_stats(pbp_2023_pass)
    season_stats_2023 = aggregator.aggregate_season_stats(game_stats_2023, min_attempts=100)

    game_stats_2024 = aggregator.aggregate_game_stats(pbp_2024_pass)
    season_stats_2024 = aggregator.aggregate_season_stats(game_stats_2024, min_attempts=100)

    logger.info(f"  2023: {len(season_stats_2023)} QBs with 100+ attempts")
    logger.info(f"  2024: {len(season_stats_2024)} QBs with 100+ attempts")

    # ==================================================
    # Define Candidate Metrics
    # ==================================================
    logger.info("\n[2/3] Testing candidate metrics...")

    # All numeric columns that might be predictive
    candidate_metrics = [
        'completion_pct',
        'yards_per_attempt',
        'td_rate',
        'int_rate',
        'sack_rate',
        'epa_per_play',
        'cpoe',
        'success_rate',
        'air_yards',
        'total_air_yards',
        'total_yac',
        'attempts_per_game',
        'yards_per_game',
        'attempts',
        'completions',
        'yards',
        'touchdowns',
        'interceptions',
        'scrambles',
        'sacks'
    ]

    # Check which columns actually exist
    available_metrics = [m for m in candidate_metrics if m in season_stats_2023.columns]

    logger.info(f"  Testing {len(available_metrics)} candidate metrics")

    # ==================================================
    # Calculate Correlations
    # ==================================================
    results = []

    for metric in available_metrics:
        r, p, n = calculate_year_over_year_correlation(
            metric, season_stats_2023, season_stats_2024
        )

        if r is not None:
            results.append({
                'metric': metric,
                'correlation': r,
                'p_value': p,
                'sample_size': n,
                'significant': p < 0.05
            })

            logger.info(f"  {metric:25s}: r = {r:6.3f}, p = {p:.4f}, n = {n}")

    # ==================================================
    # Analyze Results
    # ==================================================
    logger.info("\n[3/3] Analyzing results...")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('correlation', ascending=False)

    # Calculate composite score (correlation + significance)
    results_df['composite_score'] = results_df['correlation'] + (results_df['significant'].astype(int) * 0.1)
    results_df = results_df.sort_values('composite_score', ascending=False)

    # Save results
    output_file = Path("data/processed/qb_metric_candidates.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file, index=False)

    logger.info(f"\n  Saved results to: {output_file}")

    # Display top 15
    print("\n" + "="*80)
    print("TOP 15 QB METRIC CANDIDATES")
    print("="*80)
    print("\nBased on 2023→2024 year-over-year correlation:\n")

    top_15 = results_df.head(15)
    print(top_15[['metric', 'correlation', 'composite_score', 'p_value', 'sample_size']].to_string(index=False))

    # Summary stats
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"\nMetrics tested: {len(results_df)}")
    print(f"Significant correlations (p < 0.05): {results_df['significant'].sum()}")
    print(f"\nBest single metric: {results_df.iloc[0]['metric']}")
    print(f"  Correlation: {results_df.iloc[0]['correlation']:.4f}")
    print(f"  Sample size: {int(results_df.iloc[0]['sample_size'])} QBs")
    print(f"\nMedian correlation: {results_df['correlation'].median():.4f}")
    print(f"Mean correlation: {results_df['correlation'].mean():.4f}")

    logger.info("\n" + "="*80)
    logger.info("PHASE 1 COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nTop 15 metrics identified for Phase 2 exhaustive combination search.")
    logger.info(f"Next step: Run qb_exhaustive_composite_search.py")

if __name__ == "__main__":
    main()
