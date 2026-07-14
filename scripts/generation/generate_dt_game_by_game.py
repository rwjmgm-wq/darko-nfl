"""
Generate DT (Interior Defensive Line) Game-by-Game Composite Ratings

Creates game-level composite ratings for DTs (interior DL) based on validated metrics:
- pressures (sacks + QB hits from the interior)
- qb_hits (interior pressure with contact)
- impact_plays (sacks + TFLs + forced fumbles)
- pass_rush_plays (opportunities to rush passer)
- epa_against (EPA prevented, higher = better for defense)

These 5 metrics showed strong predictive power (r=0.637) for next season performance.
Note: DT predictability is lower than EDGE rushers due to higher variance in interior play.

Output: data/production/dt_composite_game_by_game_[date].csv
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

# Validated DT composite metrics (from exhaustive search, r=0.637)
# Season-level search found: pressures, qb_hits_per_game, pressures_per_game, impact_plays_per_game, pass_rush_plays
# Game-level equivalent: raw counts + opportunity metrics, standardized via z-scores
DT_METRICS = ['pressures', 'qb_hits', 'impact_plays', 'pass_rush_plays', 'epa_against']


def calculate_season_stats(game_stats_df):
    """
    Pre-calculate season-level statistics for standardization.

    Args:
        game_stats_df: DataFrame with all games in season

    Returns:
        Dictionary with means and stds for each metric
    """
    stats = {}

    for metric in DT_METRICS:
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

    for metric in DT_METRICS:
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
        span: EWMA span (games) - using 8 for DT (wider than offense due to smaller sample)

    Returns:
        Smoothed series
    """
    return series.ewm(span=span, adjust=False).mean()


def generate_dt_game_by_game_ratings(years=range(2020, 2025)):
    """Generate game-by-game composite ratings for all DTs."""

    logger.info("="*80)
    logger.info("GENERATING DT (INTERIOR DL) GAME-BY-GAME COMPOSITE RATINGS")
    logger.info("="*80)
    logger.info("Metrics: qb_hits, pressures, impact_plays, epa_against")
    logger.info("Predictive power: r=0.637 for next season")

    fetcher = NFLDataFetcher()
    # Use edge_positions=['DL'] only to capture interior defensive linemen
    aggregator = DefensiveStatsAggregator(min_plays=1, edge_positions=['DL'])

    all_game_ratings = []

    for year in years:
        logger.info(f"\n[{year}] Processing season...")

        # Load data
        pbp = fetcher.fetch_pbp_data([year])
        roster = fetcher.fetch_rosters([year])

        # Get game stats
        game_stats = aggregator.aggregate_game_stats(pbp, roster)

        logger.info(f"  Found {len(game_stats)} DT games, {game_stats['player_id'].nunique()} unique DTs")

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

    # Apply EWMA smoothing per player (span=8 for DT)
    ratings_df['smoothed_composite'] = ratings_df.groupby('player_id')['raw_composite'].transform(
        lambda x: apply_ewma_smoothing(x, span=8)
    )

    # Calculate uncertainty (rolling std dev)
    ratings_df['composite_uncertainty'] = ratings_df.groupby('player_id')['raw_composite'].transform(
        lambda x: x.rolling(window=10, min_periods=3).std()
    )

    logger.info(f"  Total game ratings: {len(ratings_df):,}")
    logger.info(f"  Unique DTs: {ratings_df['player_id'].nunique()}")
    logger.info(f"  Seasons covered: {sorted(ratings_df['season'].unique())}")

    # Position breakdown
    logger.info(f"\n  Position breakdown:")
    for pos, count in ratings_df['position'].value_counts().items():
        logger.info(f"    {pos}: {count:,} games")

    # Save to production folder
    output_dir = Path('data/production')
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d")
    output_file = output_dir / f'dt_composite_game_by_game_{timestamp}.csv'

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

    # Show top DTs by average rating
    logger.info(f"\n{'='*80}")
    logger.info("TOP 10 DTs (by average smoothed rating)")
    logger.info(f"{'='*80}")

    player_avg_ratings = ratings_df.groupby(['player_id', 'player_name']).agg({
        'smoothed_composite': 'mean',
        'career_game_number': 'max',
        'sacks': 'sum',
        'pressures': 'sum',
        'total_tackles': 'sum'
    }).reset_index()

    # Filter to players with 20+ games
    player_avg_ratings = player_avg_ratings[player_avg_ratings['career_game_number'] >= 20]

    top_dts = player_avg_ratings.nlargest(10, 'smoothed_composite')
    for _, dt in top_dts.iterrows():
        logger.info(f"{dt['player_name']:25s} | Rating: {dt['smoothed_composite']:+.3f} | "
                   f"Games: {int(dt['career_game_number']):3d} | "
                   f"Sacks: {dt['sacks']:.1f} | Pressures: {int(dt['pressures'])} | "
                   f"Tackles: {int(dt['total_tackles'])}")

    logger.info(f"\n{'='*80}")
    logger.info("COMPLETE")
    logger.info(f"{'='*80}")

    return ratings_df


if __name__ == "__main__":
    ratings = generate_dt_game_by_game_ratings()
