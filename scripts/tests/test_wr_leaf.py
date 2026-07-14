"""
Test WR LEAF Pipeline

Tests the complete WR LEAF v1.0 system end-to-end.
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
    """Test WR LEAF pipeline."""

    # Fetch data
    logger.info("Fetching nflfastR data...")
    fetcher = NFLDataFetcher()
    pbp = fetcher.fetch_pbp_data([2023, 2024])

    # Aggregate WR stats
    logger.info("Aggregating WR game statistics...")
    aggregator = ReceiverStatsAggregator(min_targets=1)
    game_stats = aggregator.aggregate_game_stats(pbp)

    logger.info(f"  ✓ {len(game_stats):,} receiver-games")

    # Initialize pipeline
    logger.info("Initializing WR LEAF pipeline...")
    pipeline = WRLEAFPipeline(
        use_context=True,
        use_opponent_adj=True,
        use_kalman=True
    )

    # Fit context model
    pipeline.fit_context_model(pbp, metrics=['epa', 'air_epa', 'yac_epa'])

    # Calculate secondary ratings
    secondary_ratings = pipeline.calculate_secondary_ratings(pbp)

    # Run full pipeline
    processed = pipeline.process_full_pipeline(
        game_stats=game_stats,
        pbp_data=pbp,
        secondary_ratings=secondary_ratings
    )

    # Get current ratings
    current = pipeline.get_current_ratings(processed, min_targets=30)

    # Display results
    print(f"\n{'='*80}")
    print("Top 20 WRs - LEAF Ratings (2023-2024)")
    print(f"{'='*80}")
    print(current.head(20)[['receiver_player_name', 'wr_leaf_rating',
                            'wr_leaf_uncertainty', 'total_targets']].to_string(index=False))

    print(f"\n{'='*80}")
    print("Bottom 10 WRs - LEAF Ratings (2023-2024)")
    print(f"{'='*80}")
    print(current.tail(10)[['receiver_player_name', 'wr_leaf_rating',
                            'wr_leaf_uncertainty', 'total_targets']].to_string(index=False))

    print(f"\n{'='*80}")
    print("WR LEAF Rating Statistics")
    print(f"{'='*80}")
    print(f"Total WRs: {len(current)}")
    print(f"Mean: {processed['wr_leaf_rating'].mean():.4f}")
    print(f"Std: {processed['wr_leaf_rating'].std():.4f}")
    print(f"Min: {processed['wr_leaf_rating'].min():.4f}")
    print(f"Max: {processed['wr_leaf_rating'].max():.4f}")

    # Save results
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)

    processed.to_csv(output_dir / 'wr_leaf_game_by_game.csv', index=False)
    current.to_csv(output_dir / 'wr_leaf_current_ratings.csv', index=False)

    logger.info(f"\n✓ Saved results to {output_dir}")


if __name__ == "__main__":
    main()
