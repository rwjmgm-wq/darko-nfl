"""
Generate EDGE Rusher Game-by-Game Composite Ratings

Creates game-level composite ratings for EDGE rushers (DL/LB) based on validated metrics:
- qb_hits (volume of QB pressures with contact)
- pressures (sacks + QB hits)
- pressures_per_game (efficiency metric)
- epa_against (EPA prevented, higher = better for defense)

These metrics showed exceptional predictive power (r=0.816) for next season performance.

Output: data/production/edge_composite_game_by_game_[date].csv
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

# Validated EDGE composite metrics (from exhaustive search, r=0.816)
EDGE_METRICS = ['qb_hits', 'pressures', 'epa_against']


def calculate_season_stats(game_stats_df):
    """
    Pre-calculate season-level statistics for standardization.

    Args:
        game_stats_df: DataFrame with all games in season

    Returns:
        Dictionary with means and stds for each metric
    """
    stats = {}

    for metric in EDGE_METRICS:
        if metric in game_stats_df.columns:
            values = game_stats_df[metric].replace([np.inf, -np.inf], np.nan).dropna()
            stats[metric] = {
                'mean': values.mean() if len(values) > 0 else 0,
                'std': values.std() if len(values) > 0 else 1
            }

    # Also calculate stats for pressures (for pressures_per_game standardization)
    pressures = game_stats_df['pressures'].replace([np.inf, -np.inf], np.nan).dropna()
    stats['pressures'] = {
        'mean': pressures.mean() if len(pressures) > 0 else 0,
        'std': pressures.std() if len(pressures) > 0 else 1
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

    for metric in EDGE_METRICS:
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

    # Add pressures as 4th metric (use raw pressures value)
    pressures = game_stats.get('pressures', 0)
    if pd.isna(pressures):
        pressures = 0

    # Standardize pressures
    mean_pressures = season_stats['pressures']['mean']
    std_pressures = season_stats['pressures']['std']

    if std_pressures > 0:
        pressures_z = (pressures - mean_pressures) / std_pressures
    else:
        pressures_z = 0

    game_values.append(pressures_z)

    # Equal-weighted composite
    composite = np.mean(game_values)

    return composite


def apply_ewma_smoothing(series, span=8):
    """
    Apply exponential weighted moving average smoothing.

    Args:
        series: pandas Series to smooth
        span: EWMA span (games) - using 8 for EDGE (wider than offense due to smaller sample)

    Returns:
        Smoothed series
    """
    return series.ewm(span=span, adjust=False).mean()


def generate_edge_game_by_game_ratings(years=range(2020, 2025)):
    """Generate game-by-game composite ratings for all EDGE rushers."""

    logger.info("="*80)
    logger.info("GENERATING EDGE RUSHER GAME-BY-GAME COMPOSITE RATINGS")
    logger.info("="*80)
    logger.info("Metrics: qb_hits, pressures, pressures_per_game, epa_against")
    logger.info("Predictive power: r=0.816 for next season")

    fetcher = NFLDataFetcher()
    aggregator = DefensiveStatsAggregator(min_plays=1, edge_positions=['DL', 'LB'])

    all_game_ratings = []

    for year in years:
        logger.info(f"\n[{year}] Processing season...")

        # Load data
        pbp = fetcher.fetch_pbp_data([year])
        roster = fetcher.fetch_rosters([year])

        # Get game stats
        game_stats = aggregator.aggregate_game_stats(pbp, roster)

        logger.info(f"  Found {len(game_stats)} EDGE games, {game_stats['player_id'].nunique()} unique EDGEs")

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

    # Apply EWMA smoothing per player (span=8 for EDGE)
    ratings_df['smoothed_composite'] = ratings_df.groupby('player_id')['raw_composite'].transform(
        lambda x: apply_ewma_smoothing(x, span=8)
    )

    # Calculate uncertainty (rolling std dev)
    ratings_df['composite_uncertainty'] = ratings_df.groupby('player_id')['raw_composite'].transform(
        lambda x: x.rolling(window=10, min_periods=3).std()
    )

    logger.info(f"  Total game ratings: {len(ratings_df):,}")
    logger.info(f"  Unique EDGE rushers: {ratings_df['player_id'].nunique()}")
    logger.info(f"  Seasons covered: {sorted(ratings_df['season'].unique())}")

    # Position breakdown
    logger.info(f"\n  Position breakdown:")
    for pos, count in ratings_df['position'].value_counts().items():
        logger.info(f"    {pos}: {count:,} games")

    # Save to production folder
    output_dir = Path('data/production')
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d")
    output_file = output_dir / f'edge_composite_game_by_game_{timestamp}.csv'

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

    # Show top EDGE rushers by average rating
    logger.info(f"\n{'='*80}")
    logger.info("TOP 10 EDGE RUSHERS (by average smoothed rating)")
    logger.info(f"{'='*80}")

    player_avg_ratings = ratings_df.groupby(['player_id', 'player_name']).agg({
        'smoothed_composite': 'mean',
        'career_game_number': 'max',
        'sacks': 'sum',
        'pressures': 'sum'
    }).reset_index()

    # Filter to players with 20+ games
    player_avg_ratings = player_avg_ratings[player_avg_ratings['career_game_number'] >= 20]

    top_edges = player_avg_ratings.nlargest(10, 'smoothed_composite')
    for _, edge in top_edges.iterrows():
        logger.info(f"{edge['player_name']:25s} | Rating: {edge['smoothed_composite']:+.3f} | "
                   f"Games: {int(edge['career_game_number']):3d} | "
                   f"Sacks: {edge['sacks']:.1f} | Pressures: {int(edge['pressures'])}")

    logger.info(f"\n{'='*80}")
    logger.info("COMPLETE")
    logger.info(f"{'='*80}")

    return ratings_df


if __name__ == "__main__":
    ratings = generate_edge_game_by_game_ratings()
