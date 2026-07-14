"""
Test WR Composite Rating Integration

Quick test to verify the new composite rating method works correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_pipeline.nfl_data_fetcher import NFLDataFetcher
from data_pipeline.receiver_stats_aggregator import ReceiverStatsAggregator
from features.wr_leaf_pipeline import WRLEAFPipeline

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Test composite rating integration."""

    logger.info("="*80)
    logger.info("Testing WR Composite Rating Integration")
    logger.info("="*80)

    # Fetch data
    logger.info("\nFetching data...")
    fetcher = NFLDataFetcher()
    pbp_2024 = fetcher.fetch_pbp_data([2024])
    rosters = fetcher.fetch_rosters([2024])

    # Aggregate WR-only stats
    logger.info("\nAggregating WR stats...")
    aggregator = ReceiverStatsAggregator(min_targets=1, filter_wr_only=True)
    game_stats_2024 = aggregator.aggregate_game_stats(pbp_2024, roster_data=rosters)
    season_stats_2024 = aggregator.aggregate_season_stats(game_stats_2024, min_targets=30)

    logger.info(f"  {len(season_stats_2024)} WRs with 30+ targets")

    # Initialize pipeline
    logger.info("\nInitializing WR LEAF pipeline...")
    pipeline = WRLEAFPipeline()

    # Calculate composite rating (equal weights)
    logger.info("\nCalculating composite rating (equal weights)...")
    season_with_composite = pipeline.calculate_composite_rating(
        season_stats_2024,
        use_optimized_weights=False
    )

    # Calculate composite rating (optimized weights)
    logger.info("\nCalculating composite rating (optimized weights)...")
    season_with_composite_opt = pipeline.calculate_composite_rating(
        season_stats_2024,
        use_optimized_weights=True
    )

    # Display results
    print(f"\n{'='*80}")
    print("Top 20 WRs - Composite Rating (Equal Weights)")
    print(f"{'='*80}")
    top_20 = season_with_composite.sort_values('wr_composite_rating', ascending=False).head(20)
    print(top_20[['receiver_player_name', 'wr_composite_rating', 'wr_composite_percentile',
                  'targets', 'receptions', 'total_yac_epa', 'first_downs']].to_string(index=False))

    print(f"\n{'='*80}")
    print("Top 20 WRs - Composite Rating (Optimized Weights)")
    print(f"{'='*80}")
    top_20_opt = season_with_composite_opt.sort_values('wr_composite_rating', ascending=False).head(20)
    print(top_20_opt[['receiver_player_name', 'wr_composite_rating', 'wr_composite_percentile']].to_string(index=False))

    print(f"\n{'='*80}")
    print("Comparison: Equal vs Optimized Weights")
    print(f"{'='*80}")

    # Merge both ratings
    comparison = season_with_composite[['receiver_player_id', 'receiver_player_name', 'wr_composite_rating']].merge(
        season_with_composite_opt[['receiver_player_id', 'wr_composite_rating']],
        on='receiver_player_id',
        suffixes=('_equal', '_opt')
    )

    # Calculate correlation
    from scipy import stats
    correlation, p_value = stats.pearsonr(
        comparison['wr_composite_rating_equal'],
        comparison['wr_composite_rating_opt']
    )

    print(f"Correlation between equal and optimized weights: r = {correlation:.4f} (p = {p_value:.4e})")
    print(f"Sample size: n = {len(comparison)} WRs")

    # Show biggest differences
    comparison['difference'] = comparison['wr_composite_rating_opt'] - comparison['wr_composite_rating_equal']
    comparison['abs_diff'] = comparison['difference'].abs()

    print(f"\n{'='*80}")
    print("Top 10 Biggest Rating Differences (Optimized - Equal)")
    print(f"{'='*80}")
    biggest_diff = comparison.sort_values('abs_diff', ascending=False).head(10)
    print(biggest_diff[['receiver_player_name', 'wr_composite_rating_equal',
                       'wr_composite_rating_opt', 'difference']].to_string(index=False))

    print(f"\n{'='*80}")
    print("Test Complete!")
    print(f"{'='*80}")
    print("The composite rating has been successfully integrated into the WR LEAF pipeline.")
    print(f"Use pipeline.calculate_composite_rating(season_stats) to calculate ratings.")


if __name__ == "__main__":
    main()
