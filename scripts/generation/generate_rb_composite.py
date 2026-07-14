"""
Generate RB Composite Ratings for 2020-2024

Creates RB composite ratings file for use in visualizations and analysis.

Output: data/processed/rb_composite_ratings_2020_2024.csv

Includes:
- RB composite rating (z-score based, equal-weighted)
- Percentile ranking
- All component metrics (rushing + receiving)
- Team information
- RB role classification

Based on validated 5-metric composite:
- targets_per_game (20%)
- touches_per_game (20%)
- total_yards_per_game (20%)
- rec_yards_per_game (20%)
- rush_share (20%)

Multi-year stability: r = 0.7241 (BEST of all positions!)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_pipeline.nfl_data_fetcher import NFLDataFetcher
from data_pipeline.rb_stats_aggregator import RBStatsAggregator
from features.rb_composite_pipeline import RBCompositePipeline

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Generate RB composite ratings for 2020-2024."""

    logger.info("="*80)
    logger.info("GENERATING RB COMPOSITE RATINGS (2020-2024)")
    logger.info("="*80)

    # ==================================================
    # Load Data
    # ==================================================
    logger.info("\n[1/4] Loading RB data for 2020-2024...")

    fetcher = NFLDataFetcher()
    aggregator = RBStatsAggregator(filter_rb_only=True)

    all_season_stats = []

    for year in range(2020, 2025):
        logger.info(f"  Loading {year}...")

        # Fetch play-by-play and roster data
        pbp = fetcher.fetch_pbp_data([year])
        rosters = fetcher.fetch_rosters([year])

        # Aggregate to season level
        game_stats = aggregator.aggregate_game_stats(pbp, rosters)
        season_stats = aggregator.aggregate_season_stats(game_stats, min_touches=50)

        all_season_stats.append(season_stats)

        logger.info(f"    {year}: {len(season_stats)} RBs")

    # Combine all seasons
    all_rbs = pd.concat(all_season_stats, ignore_index=True)

    logger.info(f"\n  Total: {len(all_rbs)} RB-seasons across 5 years")

    # ==================================================
    # Calculate Composite Ratings
    # ==================================================
    logger.info("\n[2/4] Calculating RB composite ratings...")

    pipeline = RBCompositePipeline(use_optimized_weights=False)
    rb_ratings = pipeline.calculate_multi_season_ratings(all_rbs, seasons=list(range(2020, 2025)))

    # ==================================================
    # Classify RB Roles
    # ==================================================
    logger.info("\n[3/4] Classifying RB roles...")

    rb_ratings = pipeline.classify_rb_role(rb_ratings)

    role_dist = rb_ratings['rb_role'].value_counts()
    logger.info(f"  Role distribution:")
    for role, count in role_dist.items():
        logger.info(f"    {role}: {count} ({count/len(rb_ratings)*100:.1f}%)")

    # ==================================================
    # Save Results
    # ==================================================
    logger.info("\n[4/4] Saving results...")

    output_file = Path("data/processed/rb_composite_ratings_2020_2024.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Select columns to save
    output_columns = [
        'season',
        'player_id',
        'player_name',
        'team',
        'rb_composite_rating',
        'rb_composite_percentile',
        'rb_role',
        # Component metrics
        'targets_per_game',
        'touches_per_game',
        'total_yards_per_game',
        'rec_yards_per_game',
        'rush_share',
        # Rushing stats
        'rush_attempts',
        'rush_yards',
        'rush_attempts_per_game',
        'rush_yards_per_game',
        'yards_per_carry',
        'rush_tds',
        'rush_epa_per_attempt',
        'rush_success_rate',
        # Receiving stats
        'targets',
        'receptions',
        'rec_yards',
        'catch_rate',
        'yards_per_target',
        'yards_per_reception',
        'rec_tds',
        'rec_epa_per_target',
        'yac_epa_per_target',
        # Combined stats
        'total_touches',
        'total_yards',
        'total_tds',
        'total_first_downs',
        'epa_per_touch',
        'yards_per_touch',
        'games',
        'fumbles',
        'fumble_rate'
    ]

    # Filter to only columns that exist
    available_columns = [col for col in output_columns if col in rb_ratings.columns]

    rb_ratings[available_columns].to_csv(output_file, index=False)

    logger.info(f"  ✓ Saved to: {output_file}")
    logger.info(f"  Columns: {len(available_columns)}")
    logger.info(f"  Rows: {len(rb_ratings)}")

    # ==================================================
    # Display Results
    # ==================================================
    print("\n" + "="*80)
    print("RB COMPOSITE RATINGS SUMMARY")
    print("="*80)

    # Top 10 RB-seasons overall
    print("\nTop 10 RB-Seasons (2020-2024):\n")
    top_10 = rb_ratings.nlargest(10, 'rb_composite_rating')[
        ['season', 'player_name', 'team', 'rb_composite_rating',
         'rb_composite_percentile', 'rb_role', 'touches_per_game',
         'total_yards_per_game', 'total_touches']
    ]
    print(top_10.to_string(index=False))

    # Best by season
    print("\n" + "="*80)
    print("BEST RB BY SEASON")
    print("="*80)

    for year in range(2020, 2025):
        year_data = rb_ratings[rb_ratings['season'] == year]
        if len(year_data) > 0:
            best = year_data.nlargest(1, 'rb_composite_rating').iloc[0]
            print(f"\n{year}: {best['player_name']} ({best['team']})")
            print(f"  Composite Rating: {best['rb_composite_rating']:.3f} "
                  f"({best['rb_composite_percentile']:.1f}th percentile)")
            print(f"  Role: {best['rb_role']}")
            print(f"  Stats: {best['touches_per_game']:.1f} touches/game, "
                  f"{best['total_yards_per_game']:.1f} yards/game, "
                  f"{best['rush_share']:.1f}% rush share")

    # Multi-year leaders
    print("\n" + "="*80)
    print("MULTI-YEAR CONSISTENCY LEADERS")
    print("="*80)
    print("\nRBs with 3+ seasons (avg composite rating):\n")

    rb_consistency = rb_ratings.groupby('player_name').agg({
        'rb_composite_rating': ['mean', 'std', 'count'],
        'season': lambda x: f"{int(x.min())}-{int(x.max())}"
    }).reset_index()

    rb_consistency.columns = ['RB', 'Avg Rating', 'Std Dev', 'Seasons', 'Years']
    rb_consistency = rb_consistency[rb_consistency['Seasons'] >= 3]
    rb_consistency = rb_consistency.sort_values('Avg Rating', ascending=False).head(10)

    print(rb_consistency.to_string(index=False))

    # Role-specific leaders
    print("\n" + "="*80)
    print("BEST RB BY ROLE (2020-2024)")
    print("="*80)

    for role in ['three_down_back', 'pure_rusher', 'pass_catching_specialist']:
        role_rbs = rb_ratings[rb_ratings['rb_role'] == role]
        if len(role_rbs) > 0:
            best = role_rbs.nlargest(1, 'rb_composite_rating').iloc[0]
            print(f"\n{role.replace('_', ' ').title()}:")
            print(f"  {best['player_name']} ({best['season']} {best['team']})")
            print(f"  Rating: {best['rb_composite_rating']:.3f}")
            print(f"  Stats: {best['touches_per_game']:.1f} touches/g, "
                  f"{best['total_yards_per_game']:.1f} yards/g")

    # Cross-position comparison
    print("\n" + "="*80)
    print("CROSS-POSITION STABILITY COMPARISON")
    print("="*80)
    print("\nBased on multi-year validation (2020-2024):\n")
    print("Position  | Year-to-Year Stability | Relative Performance")
    print("-"*60)
    print("RB        | r = 0.7241 (±0.0556)  | BEST (baseline)")
    print("WR        | r = 0.6614 (±0.0441)  | -8.7% vs RB")
    print("QB        | r = 0.6194 (±0.0531)  | -14.5% vs RB")
    print("\nRBs show HIGHEST year-to-year stability of all positions!")
    print("Reason: Volume/usage metrics more consistent than efficiency.")

    print("\n" + "="*80)
    logger.info("COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nRB composite ratings ready for visualization and analysis.")
    logger.info(f"File: {output_file}")


if __name__ == "__main__":
    main()
