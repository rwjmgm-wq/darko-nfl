"""
Generate Linebacker (LB) Game-by-Game Composite Ratings

Creates game-level composite ratings for linebackers based on validated metrics:
- sacks (pass rush ability)
- qb_hits (pressure generation)
- epa_against (overall defensive impact)
- epa_against_pass (pass defense quality)
- epa_against_run (run defense quality)

These 5 metrics capture LB versatility (r=0.705): pass rush, run defense, and coverage combined.
More holistic than tackles-only approach, measuring complete LB performance.

Output: data/production/lb_composite_game_by_game_[date].csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_pipeline.nfl_data_fetcher import NFLDataFetcher
from data_pipeline.defensive_stats_aggregator import DefensiveStatsAggregator

import pandas as pd
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Validated LB composite metrics (from exhaustive search, r=0.705)
# Season-level search found: sacks_per_game, qb_hits_per_game, epa_against, epa_against_pass, epa_against_run
# Game-level equivalent: raw counts + EPA metrics
LB_METRICS = ['sacks', 'qb_hits', 'epa_against', 'epa_against_pass', 'epa_against_run']


def calculate_season_stats(game_stats_df):
    """
    Pre-calculate season-level statistics for standardization.

    Args:
        game_stats_df: DataFrame with all games in season

    Returns:
        Dictionary with means and stds for each metric
    """
    stats = {}

    for metric in LB_METRICS:
        if metric in game_stats_df.columns:
            values = game_stats_df[metric].replace([np.inf, -np.inf], np.nan).dropna()
            stats[metric] = {
                'mean': values.mean() if len(values) > 0 else 0,
                'std': values.std() if len(values) > 0 else 1
            }

    return stats


def calculate_game_composite_rating(game_stats, season_stats):
    """
    Calculate composite rating for a single game using pre-calculated season stats.

    Args:
        game_stats: Single game stats dict
        season_stats: Pre-calculated season means/stds

    Returns:
        Composite rating (z-score)
    """
    # Get metric values for this game
    game_values = []

    for metric in LB_METRICS:
        if metric not in game_stats or metric not in season_stats:
            return None  # Missing data

        # Get value for this game
        value = game_stats[metric]

        # Skip if NaN/None
        if pd.isna(value):
            value = 0  # Treat missing as 0 for that game

        # Standardize using pre-calculated season stats
        mean = season_stats[metric]['mean']
        std = season_stats[metric]['std']

        if std > 0:
            z_score = (value - mean) / std
        else:
            z_score = 0

        game_values.append(z_score)

    # Equal-weighted composite
    composite = np.mean(game_values)

    return composite


def apply_ewma_smoothing(series, span=8):
    """
    Apply exponential weighted moving average smoothing.

    Args:
        series: pandas Series to smooth
        span: EWMA span (games) - using 8 for LB (wider than offense due to smaller sample)

    Returns:
        Smoothed series
    """
    return series.ewm(span=span, adjust=False).mean()


def generate_lb_game_by_game_ratings(years=range(2020, 2025)):
    """Generate game-by-game composite ratings for all linebackers."""

    logger.info("="*80)
    logger.info("GENERATING LINEBACKER GAME-BY-GAME COMPOSITE RATINGS")
    logger.info("="*80)
    logger.info("Metrics: solo_tackles, assist_tackles, total_tackles")
    logger.info("Predictive power: r=0.729 for next season")

    fetcher = NFLDataFetcher()
    # Use edge_positions=['LB'] only to capture linebackers
    aggregator = DefensiveStatsAggregator(min_plays=1, edge_positions=['LB'])

    all_game_ratings = []

    for year in years:
        logger.info(f"\n[{year}] Processing season...")

        # Load data
        pbp = fetcher.fetch_pbp_data([year])
        roster = fetcher.fetch_rosters([year])

        # Get game stats
        game_stats = aggregator.aggregate_game_stats(pbp, roster)

        logger.info(f"  Found {len(game_stats)} LB games, {game_stats['player_id'].nunique()} unique LBs")

        # Sort by player and week for career tracking
        game_stats = game_stats.sort_values(['player_id', 'week'])

        # Pre-calculate season-level statistics once
        season_stats = calculate_season_stats(game_stats)

        # Calculate composite ratings
        for _, game in game_stats.iterrows():
            composite = calculate_game_composite_rating(
                game.to_dict(),
                season_stats
            )

            if composite is not None:
                all_game_ratings.append({
                    'season': year,
                    'week': game['week'],
                    'player_id': game['player_id'],
                    'player_name': game['player_name'],
                    'position': game['position'],
                    'sacks': game.get('sacks', 0),
                    'qb_hits': game.get('qb_hits', 0),
                    'tfls': game.get('tfls', 0),
                    'forced_fumbles': game.get('forced_fumbles', 0),
                    'solo_tackles': game.get('solo_tackles', 0),
                    'assist_tackles': game.get('assist_tackles', 0),
                    'total_tackles': game.get('total_tackles', 0),
                    'pressures': game.get('pressures', 0),
                    'impact_plays': game.get('impact_plays', 0),
                    'epa_against': game.get('epa_against', 0),
                    'raw_composite': composite
                })

        logger.info(f"  Calculated {len([r for r in all_game_ratings if r['season'] == year])} game ratings")

    # Convert to DataFrame
    ratings_df = pd.DataFrame(all_game_ratings)

    logger.info(f"\n{'='*80}")
    logger.info(f"SMOOTHING AND CAREER NUMBERING")
    logger.info(f"{'='*80}")

    # Add career game numbers and smoothing
    ratings_df = ratings_df.sort_values(['player_id', 'season', 'week'])

    # Career game number
    ratings_df['career_game_number'] = ratings_df.groupby('player_id').cumcount() + 1

    # Apply EWMA smoothing per player (span=8 for LB)
    ratings_df['smoothed_composite'] = ratings_df.groupby('player_id')['raw_composite'].transform(
        lambda x: apply_ewma_smoothing(x, span=8)
    )

    # Calculate uncertainty (rolling std dev)
    ratings_df['composite_uncertainty'] = ratings_df.groupby('player_id')['raw_composite'].transform(
        lambda x: x.rolling(window=10, min_periods=3).std()
    )

    logger.info(f"  Total game ratings: {len(ratings_df):,}")
    logger.info(f"  Unique LBs: {ratings_df['player_id'].nunique()}")
    logger.info(f"  Seasons covered: {sorted(ratings_df['season'].unique())}")

    # Position breakdown
    logger.info(f"\n  Position breakdown:")
    for pos, count in ratings_df['position'].value_counts().items():
        logger.info(f"    {pos}: {count:,} games")

    # Save to production folder
    output_dir = Path('data/production')
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d")
    output_file = output_dir / f'lb_composite_game_by_game_{timestamp}.csv'

    ratings_df.to_csv(output_file, index=False)

    logger.info(f"\n{'='*80}")
    logger.info(f"OUTPUT")
    logger.info(f"{'='*80}")
    logger.info(f"Saved to: {output_file}")

    # Display summary statistics
    logger.info(f"\n{'='*80}")
    logger.info("RATING DISTRIBUTION")
    logger.info(f"{'='*80}")
    logger.info(f"Raw Composite:")
    logger.info(f"  Mean:   {ratings_df['raw_composite'].mean():.3f}")
    logger.info(f"  Median: {ratings_df['raw_composite'].median():.3f}")
    logger.info(f"  Std:    {ratings_df['raw_composite'].std():.3f}")
    logger.info(f"  Min:    {ratings_df['raw_composite'].min():.3f}")
    logger.info(f"  Max:    {ratings_df['raw_composite'].max():.3f}")

    logger.info(f"\nSmoothed Composite:")
    logger.info(f"  Mean:   {ratings_df['smoothed_composite'].mean():.3f}")
    logger.info(f"  Median: {ratings_df['smoothed_composite'].median():.3f}")
    logger.info(f"  Std:    {ratings_df['smoothed_composite'].std():.3f}")
    logger.info(f"  Min:    {ratings_df['smoothed_composite'].min():.3f}")
    logger.info(f"  Max:    {ratings_df['smoothed_composite'].max():.3f}")

    # Show top LBs by average rating
    logger.info(f"\n{'='*80}")
    logger.info("TOP 10 LINEBACKERS (by average smoothed rating)")
    logger.info(f"{'='*80}")

    player_avg_ratings = ratings_df.groupby(['player_id', 'player_name']).agg({
        'smoothed_composite': 'mean',
        'career_game_number': 'max',
        'sacks': 'sum',
        'total_tackles': 'sum',
        'tfls': 'sum'
    }).reset_index()

    # Filter to players with 20+ games
    player_avg_ratings = player_avg_ratings[player_avg_ratings['career_game_number'] >= 20]

    top_lbs = player_avg_ratings.nlargest(10, 'smoothed_composite')
    for _, lb in top_lbs.iterrows():
        logger.info(f"{lb['player_name']:25s} | Rating: {lb['smoothed_composite']:+.3f} | "
                   f"Games: {int(lb['career_game_number']):3d} | "
                   f"Tackles: {int(lb['total_tackles']):3d} | "
                   f"TFLs: {lb['tfls']:.1f} | Sacks: {lb['sacks']:.1f}")

    logger.info(f"\n{'='*80}")
    logger.info("COMPLETE")
    logger.info(f"{'='*80}")

    return ratings_df


if __name__ == "__main__":
    ratings = generate_lb_game_by_game_ratings()
