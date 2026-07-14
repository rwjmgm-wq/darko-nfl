"""
Generate WR Game-by-Game Composite Ratings

Creates smoothed game-by-game composite ratings for WRs with:
- Running composite calculation after each game
- Exponential weighted moving average smoothing
- Uncertainty bands
- Future performance predictions

Output: data/production/wr_composite_game_by_game_[date].csv
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

# Validated WR composite metrics (game-level versions)
# Season metrics: targets_per_game, receptions, total_yac_epa, first_downs, targets
# At game level, we use the raw counts
WR_METRICS = ['targets', 'receptions', 'total_yac_epa', 'first_downs']

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

    for metric in WR_METRICS:
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

    # Equal-weighted composite
    composite = np.mean(game_values)

    return composite

def apply_ewma_smoothing(series, span=5):
    """
    Apply exponential weighted moving average smoothing.

    Args:
        series: Pandas Series to smooth
        span: Smoothing window (default 5 games)

    Returns:
        Smoothed series
    """
    return series.ewm(span=span, min_periods=1).mean()

def calculate_uncertainty(series, window=10):
    """
    Calculate rolling uncertainty (standard error).

    Args:
        series: Rating series
        window: Window for uncertainty calculation

    Returns:
        Series of uncertainties
    """
    rolling_std = series.rolling(window=window, min_periods=3).std()
    # Start with higher uncertainty, decrease as more data available
    n_games = np.arange(1, len(series) + 1)
    sample_uncertainty = 1.0 / np.sqrt(n_games)

    # Combine rolling std with sample size uncertainty
    uncertainty = pd.Series(index=series.index)
    for i in range(len(series)):
        if pd.notna(rolling_std.iloc[i]):
            uncertainty.iloc[i] = max(rolling_std.iloc[i], sample_uncertainty[i])
        else:
            uncertainty.iloc[i] = sample_uncertainty[i]

    return uncertainty

def generate_wr_game_by_game_ratings(years=range(2020, 2025)):
    """Generate game-by-game composite ratings for all WRs."""

    logger.info("="*80)
    logger.info("GENERATING WR GAME-BY-GAME COMPOSITE RATINGS")
    logger.info("="*80)

    fetcher = NFLDataFetcher()
    aggregator = ReceiverStatsAggregator(filter_wr_only=True)

    all_game_ratings = []

    for year in years:
        logger.info(f"\n[{year}] Processing season...")

        # Load data
        pbp = fetcher.fetch_pbp_data([year])
        rosters = fetcher.fetch_rosters([year])

        # Get game-level stats
        game_stats = aggregator.aggregate_game_stats(pbp, rosters)

        logger.info(f"  {len(game_stats)} WR-game records")

        # Calculate composite for each game
        for idx, game in game_stats.iterrows():
            # Use all games in season for standardization context
            season_context = game_stats[game_stats['season'] == year]

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

        logger.info(f"  Calculated {len([r for r in all_game_ratings if r['season'] == year])} composite ratings")

    # Convert to DataFrame
    game_ratings_df = pd.DataFrame(all_game_ratings)

    logger.info(f"\nTotal game ratings: {len(game_ratings_df)}")
    logger.info(f"Unique WRs: {game_ratings_df['receiver_player_id'].nunique()}")

    # Apply smoothing and uncertainty for each player
    logger.info("\nApplying smoothing and calculating uncertainty...")

    smoothed_data = []

    for player_id in game_ratings_df['receiver_player_id'].unique():
        player_games = game_ratings_df[
            game_ratings_df['receiver_player_id'] == player_id
        ].copy()

        # Sort by season, week
        player_games = player_games.sort_values(['season', 'week']).reset_index(drop=True)

        if len(player_games) < 3:
            continue  # Skip players with very few games

        # Apply smoothing
        player_games['smoothed_composite'] = apply_ewma_smoothing(
            player_games['raw_composite'],
            span=5
        )

        # Calculate uncertainty
        player_games['composite_uncertainty'] = calculate_uncertainty(
            player_games['smoothed_composite'],
            window=10
        )

        # Add career game number
        player_games['career_game_number'] = range(len(player_games))

        smoothed_data.append(player_games)

    # Combine all players
    final_ratings = pd.concat(smoothed_data, ignore_index=True)

    logger.info(f"  Smoothed ratings for {final_ratings['receiver_player_id'].nunique()} WRs")

    # Save to production folder
    output_dir = Path("data/production")
    output_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d")
    output_file = output_dir / f"wr_composite_game_by_game_{date_str}.csv"

    final_ratings.to_csv(output_file, index=False)

    logger.info(f"\n✓ Saved to: {output_file}")
    logger.info(f"  Records: {len(final_ratings)}")
    logger.info(f"  Columns: {len(final_ratings.columns)}")

    # Display summary stats
    print("\n" + "="*80)
    print("WR GAME-BY-GAME COMPOSITE RATINGS SUMMARY")
    print("="*80)
    print(f"\nTotal game records: {len(final_ratings):,}")
    print(f"Unique WRs: {final_ratings['receiver_player_id'].nunique()}")
    print(f"Seasons: {final_ratings['season'].min()}-{final_ratings['season'].max()}")
    print(f"\nComposite Rating Range: [{final_ratings['smoothed_composite'].min():.3f}, {final_ratings['smoothed_composite'].max():.3f}]")
    print(f"Mean Composite: {final_ratings['smoothed_composite'].mean():.3f}")
    print(f"Mean Uncertainty: {final_ratings['composite_uncertainty'].mean():.3f}")

    # Top 10 individual games
    print("\n" + "="*80)
    print("TOP 10 INDIVIDUAL GAME PERFORMANCES")
    print("="*80)
    top_games = final_ratings.nlargest(10, 'raw_composite')[
        ['season', 'week', 'receiver_player_name', 'team', 'raw_composite', 'targets', 'receptions', 'yards', 'touchdowns']
    ]
    print(top_games.to_string(index=False))

    logger.info("\n" + "="*80)
    logger.info("COMPLETE!")
    logger.info("="*80)

    return final_ratings

if __name__ == "__main__":
    generate_wr_game_by_game_ratings()
