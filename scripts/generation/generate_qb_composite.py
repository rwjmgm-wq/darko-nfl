"""
Generate QB Composite Ratings for 2020-2024

Creates QB composite ratings file for use in visualizations and analysis.

Output: data/processed/qb_composite_ratings_2020_2024.csv

Includes:
- QB composite rating (z-score based)
- Percentile ranking
- All component metrics
- Team information
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_pipeline.nfl_data_fetcher import NFLDataFetcher
from data_pipeline.qb_stats_aggregator import QBStatsAggregator
from features.qb_composite_pipeline import QBCompositePipeline

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Generate QB composite ratings for 2020-2024."""

    logger.info("="*80)
    logger.info("GENERATING QB COMPOSITE RATINGS (2020-2024)")
    logger.info("="*80)

    # ==================================================
    # Load Data
    # ==================================================
    logger.info("\n[1/3] Loading QB data for 2020-2024...")

    fetcher = NFLDataFetcher()
    aggregator = QBStatsAggregator()

    all_season_stats = []

    for year in range(2020, 2025):
        logger.info(f"  Loading {year}...")

        # Fetch play-by-play data
        pbp = fetcher.fetch_pbp_data([year])
        pbp_pass = pbp[pbp['play_type'] == 'pass'].copy()

        # Aggregate to season level
        game_stats = aggregator.aggregate_game_stats(pbp_pass)
        season_stats = aggregator.aggregate_season_stats(game_stats, min_attempts=100)

        all_season_stats.append(season_stats)

        logger.info(f"    {year}: {len(season_stats)} QBs")

    # Combine all seasons
    all_qbs = pd.concat(all_season_stats, ignore_index=True)

    logger.info(f"\n  Total: {len(all_qbs)} QB-seasons across 5 years")

    # ==================================================
    # Calculate Composite Ratings
    # ==================================================
    logger.info("\n[2/3] Calculating QB composite ratings...")

    pipeline = QBCompositePipeline(use_optimized_weights=False)
    qb_ratings = pipeline.calculate_multi_season_ratings(all_qbs, seasons=list(range(2020, 2025)))

    # ==================================================
    # Save Results
    # ==================================================
    logger.info("\n[3/3] Saving results...")

    output_file = Path("data/processed/qb_composite_ratings_2020_2024.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Select columns to save
    output_columns = [
        'season',
        'passer_player_id',
        'passer_player_name',
        'team',
        'qb_composite_rating',
        'qb_composite_percentile',
        'yards_per_game',
        'completion_pct',
        'yards_per_attempt',
        'attempts_per_game',
        'sack_rate',
        'attempts',
        'completions',
        'yards',
        'touchdowns',
        'interceptions',
        'sacks',
        'games',
        'epa_per_play',
        'cpoe',
        'success_rate'
    ]

    # Filter to only columns that exist
    available_columns = [col for col in output_columns if col in qb_ratings.columns]

    qb_ratings[available_columns].to_csv(output_file, index=False)

    logger.info(f"  ✓ Saved to: {output_file}")
    logger.info(f"  Columns: {len(available_columns)}")
    logger.info(f"  Rows: {len(qb_ratings)}")

    # ==================================================
    # Display Results
    # ==================================================
    print("\n" + "="*80)
    print("QB COMPOSITE RATINGS SUMMARY")
    print("="*80)

    # Top 10 QB-seasons overall
    print("\nTop 10 QB-Seasons (2020-2024):\n")
    top_10 = qb_ratings.nlargest(10, 'qb_composite_rating')[
        ['season', 'passer_player_name', 'team', 'qb_composite_rating',
         'qb_composite_percentile', 'yards_per_game', 'completion_pct', 'attempts']
    ]
    print(top_10.to_string(index=False))

    # Best by season
    print("\n" + "="*80)
    print("BEST QB BY SEASON")
    print("="*80)

    for year in range(2020, 2025):
        year_data = qb_ratings[qb_ratings['season'] == year]
        if len(year_data) > 0:
            best = year_data.nlargest(1, 'qb_composite_rating').iloc[0]
            print(f"\n{year}: {best['passer_player_name']} ({best['team']})")
            print(f"  Composite Rating: {best['qb_composite_rating']:.3f} "
                  f"({best['qb_composite_percentile']:.1f}th percentile)")
            print(f"  Stats: {best['yards_per_game']:.1f} YPG, "
                  f"{best['completion_pct']:.1f}% comp, "
                  f"{best['yards_per_attempt']:.2f} YPA")

    # Multi-year leaders
    print("\n" + "="*80)
    print("MULTI-YEAR CONSISTENCY LEADERS")
    print("="*80)
    print("\nQBs with 3+ seasons (avg composite rating):\n")

    qb_consistency = qb_ratings.groupby('passer_player_name').agg({
        'qb_composite_rating': ['mean', 'std', 'count'],
        'season': lambda x: f"{int(x.min())}-{int(x.max())}"
    }).reset_index()

    qb_consistency.columns = ['QB', 'Avg Rating', 'Std Dev', 'Seasons', 'Years']
    qb_consistency = qb_consistency[qb_consistency['Seasons'] >= 3]
    qb_consistency = qb_consistency.sort_values('Avg Rating', ascending=False).head(10)

    print(qb_consistency.to_string(index=False))

    print("\n" + "="*80)
    logger.info("COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nQB composite ratings ready for visualization and analysis.")
    logger.info(f"File: {output_file}")


if __name__ == "__main__":
    main()
