"""
Generate WR Composite Ratings for Visualization

Calculates and saves WR composite ratings for use in the QB visualization dashboard.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_pipeline.nfl_data_fetcher import NFLDataFetcher
from data_pipeline.receiver_stats_aggregator import ReceiverStatsAggregator
from features.wr_leaf_pipeline import WRLEAFPipeline

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Generate WR composite ratings and save to file."""

    logger.info("="*80)
    logger.info("Generating WR Composite Ratings")
    logger.info("="*80)

    # Fetch data for multiple years
    years = [2020, 2021, 2022, 2023, 2024]

    logger.info(f"\nFetching data for years: {years}")
    fetcher = NFLDataFetcher()
    pbp = fetcher.fetch_pbp_data(years)
    rosters = fetcher.fetch_rosters(years)

    # Aggregate WR-only stats
    logger.info("\nAggregating WR stats (game level)...")
    aggregator = ReceiverStatsAggregator(min_targets=1, filter_wr_only=True)
    game_stats = aggregator.aggregate_game_stats(pbp, roster_data=rosters)

    logger.info(f"  {len(game_stats)} game-level records")

    # Aggregate to season level
    logger.info("\nAggregating to season level (30+ targets)...")
    season_stats = aggregator.aggregate_season_stats(game_stats, min_targets=30)

    logger.info(f"  {len(season_stats)} WR-seasons with 30+ targets")

    # Calculate composite rating
    logger.info("\nCalculating WR composite ratings...")
    pipeline = WRLEAFPipeline()
    season_with_composite = pipeline.calculate_composite_rating(
        season_stats,
        use_optimized_weights=False  # Use equal weighting (recommended)
    )

    # Save to file
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "wr_composite_ratings_2020_2024.csv"

    logger.info(f"\nSaving to {output_file}...")
    season_with_composite.to_csv(output_file, index=False)

    # Display sample results
    logger.info("\n" + "="*80)
    logger.info("Sample Results (Top 10 WRs from 2024)")
    logger.info("="*80)

    sample_2024 = season_with_composite[season_with_composite['season'] == 2024].copy()
    sample_2024 = sample_2024.sort_values('wr_composite_rating', ascending=False).head(10)

    print("\n")
    display_cols = ['receiver_player_name', 'wr_composite_rating',
                   'wr_composite_percentile', 'targets', 'receptions',
                   'total_yac_epa', 'first_downs']
    if 'team' in sample_2024.columns:
        display_cols.insert(1, 'team')
    print(sample_2024[display_cols].to_string(index=False))

    logger.info("\n" + "="*80)
    logger.info("Generation Complete!")
    logger.info("="*80)
    logger.info(f"Total WR-seasons: {len(season_with_composite)}")
    logger.info(f"Years covered: {sorted(season_with_composite['season'].unique())}")
    logger.info(f"Output file: {output_file}")
    logger.info("\nThe composite ratings are now ready for use in the visualization dashboard.")

if __name__ == "__main__":
    main()
