"""
Generate TE Game-by-Game Composite Ratings

Creates game-level composite ratings for tight ends based on validated metrics:
- yac_epa_per_target (efficiency creating value after catch)
- first_downs (volume of chain-moving plays)
- first_down_rate (efficiency converting targets to first downs)
- touchdowns (scoring production)

Note: This is a RECEIVING-ONLY rating. Blocking performance is not measured
due to lack of publicly available blocking metrics.

Output: data/production/te_composite_game_by_game_[date].csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_pipeline.nfl_data_fetcher import NFLDataFetcher
from data_pipeline.receiver_stats_aggregator import ReceiverStatsAggregator

import pandas as pd
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Validated TE composite metrics (game-level versions)
# Season metrics: yac_epa_per_target, first_downs, first_down_rate, touchdowns
# At game level, we calculate yac_epa and first_down_rate from raw counts
TE_METRICS = ['total_yac_epa', 'first_downs', 'touchdowns']

def calculate_game_composite_rating(game_stats, all_games_context):
    """
    Calculate composite rating for a single game using season context for z-scores.

    Args:
        game_stats: Single game stats dict
        all_games_context: All games in same season for standardization

    Returns:
        Composite rating (z-score)
    """
    # Get metric values for this game
    game_values = []

    for metric in TE_METRICS:
        if metric not in game_stats:
            return None  # Missing data

        # Get value for this game
        value = game_stats[metric]

        # Skip if NaN/None
        if pd.isna(value):
            value = 0  # Treat missing as 0 for that game

        # Standardize using all games in season as context
        season_values = all_games_context[metric].replace([np.inf, -np.inf], np.nan).dropna()

        if len(season_values) < 5:  # Need minimum sample
            return None

        mean = season_values.mean()
        std = season_values.std()

        if std > 0:
            z_score = (value - mean) / std
        else:
            z_score = 0

        game_values.append(z_score)

    # Add first_down_rate as 4th metric (calculate on the fly)
    receptions = game_stats.get('receptions', 0)
    first_downs = game_stats.get('first_downs', 0)
    if receptions > 0:
        first_down_rate = first_downs / receptions
    else:
        first_down_rate = 0

    # Standardize first_down_rate
    fd_rates = []
    for _, ctx_game in all_games_context.iterrows():
        ctx_recs = ctx_game.get('receptions', 0)
        ctx_fds = ctx_game.get('first_downs', 0)
        if ctx_recs > 0:
            fd_rates.append(ctx_fds / ctx_recs)

    if len(fd_rates) >= 5:
        mean_fd_rate = np.mean(fd_rates)
        std_fd_rate = np.std(fd_rates)
        if std_fd_rate > 0:
            fd_rate_z = (first_down_rate - mean_fd_rate) / std_fd_rate
        else:
            fd_rate_z = 0
        game_values.append(fd_rate_z)

    # Equal-weighted composite
    composite = np.mean(game_values)

    return composite

def apply_ewma_smoothing(series, span=5):
    """
    Apply exponential weighted moving average smoothing.

    Args:
        series: pandas Series to smooth
        span: EWMA span (games)

    Returns:
        Smoothed series
    """
    return series.ewm(span=span, adjust=False).mean()

def generate_te_game_by_game_ratings(years=range(2020, 2025)):
    """Generate game-by-game composite ratings for all TEs."""

    logger.info("="*80)
    logger.info("GENERATING TE GAME-BY-GAME COMPOSITE RATINGS")
    logger.info("="*80)
    logger.info("NOTE: This is a RECEIVING-ONLY rating (blocking not measured)")

    fetcher = NFLDataFetcher()
    aggregator = ReceiverStatsAggregator(filter_wr_only=False)  # Include all receivers

    all_game_ratings = []

    for year in years:
        logger.info(f"\n[{year}] Processing season...")

        # Load data
        pbp = fetcher.fetch_pbp_data([year])
        roster = fetcher.fetch_rosters([year])

        # Get game stats
        game_stats = aggregator.aggregate_game_stats(pbp, roster)

        # Filter to TEs only
        if 'position' in game_stats.columns:
            te_games = game_stats[game_stats['position'] == 'TE'].copy()
        else:
            # Fallback: match receiver IDs with TE roster
            roster_lookup = roster.groupby(['season', 'player_id'])['position'].agg(
                lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
            ).reset_index()
            game_stats_with_pos = game_stats.merge(
                roster_lookup,
                left_on=['season', 'receiver_player_id'],
                right_on=['season', 'player_id'],
                how='left'
            )
            te_games = game_stats_with_pos[game_stats_with_pos['position'] == 'TE'].copy()

        logger.info(f"  Found {len(te_games)} TE games, {te_games['receiver_player_id'].nunique()} unique TEs")

        # Calculate composite ratings
        for _, game in te_games.iterrows():
            # Use all TE games in this season for context
            season_context = te_games[te_games['season'] == year]

            composite = calculate_game_composite_rating(game.to_dict(), season_context)

            if composite is not None:
                all_game_ratings.append({
                    'season': year,
                    'week': game['week'],
                    'receiver_player_id': game['receiver_player_id'],
                    'receiver_player_name': game['receiver_player_name'],
                    'team': game['posteam'],
                    'opponent': game.get('defteam', ''),
                    'targets': game.get('targets', 0),
                    'receptions': game.get('receptions', 0),
                    'yards': game.get('yards', 0),
                    'touchdowns': game.get('touchdowns', 0),
                    'first_downs': game.get('first_downs', 0),
                    'total_yac_epa': game.get('total_yac_epa', 0),
                    'raw_composite': composite
                })

        logger.info(f"  Calculated {len([r for r in all_game_ratings if r['season'] == year])} game ratings")

    # Convert to DataFrame
    ratings_df = pd.DataFrame(all_game_ratings)

    logger.info(f"\n{'='*80}")
    logger.info(f"SMOOTHING AND CAREER NUMBERING")
    logger.info(f"{'='*80}")

    # Add career game numbers and smoothing
    ratings_df = ratings_df.sort_values(['receiver_player_id', 'season', 'week'])

    # Career game number
    ratings_df['career_game_number'] = ratings_df.groupby('receiver_player_id').cumcount() + 1

    # Apply EWMA smoothing per player
    def smooth_player(group):
        group = group.copy()
        group['smoothed_composite'] = apply_ewma_smoothing(group['raw_composite'], span=5)
        return group

    ratings_df = ratings_df.groupby('receiver_player_id', group_keys=False).apply(smooth_player)

    logger.info(f"  Applied EWMA smoothing (span=5 games)")

    # Save to file
    output_file = f"data/production/te_composite_game_by_game_{datetime.now().strftime('%Y%m%d')}.csv"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    ratings_df.to_csv(output_file, index=False)

    logger.info(f"\n{'='*80}")
    logger.info(f"SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total games: {len(ratings_df):,}")
    logger.info(f"Unique TEs: {ratings_df['receiver_player_id'].nunique()}")
    logger.info(f"Seasons: {sorted(ratings_df['season'].unique())}")
    logger.info(f"Composite rating range: {ratings_df['smoothed_composite'].min():.3f} to {ratings_df['smoothed_composite'].max():.3f}")
    logger.info(f"\nSaved to: {output_file}")
    logger.info(f"{'='*80}")

    return ratings_df

if __name__ == "__main__":
    generate_te_game_by_game_ratings()
