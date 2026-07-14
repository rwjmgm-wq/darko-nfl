"""
Generate RB Game-by-Game Composite Ratings

Creates smoothed game-by-game composite ratings for RBs with:
- Running composite calculation after each game
- Exponential weighted moving average smoothing
- Uncertainty bands
- Future performance predictions

Output: data/production/rb_composite_game_by_game_[date].csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_pipeline.nfl_data_fetcher import NFLDataFetcher
from data_pipeline.rb_stats_aggregator import RBStatsAggregator

import pandas as pd
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Validated RB composite metrics (game-level versions)
# Season metrics: targets_per_game, touches_per_game, total_yards_per_game, rec_yards_per_game, rush_share
# At game level, we use the raw counts
RB_METRICS = ['targets', 'total_touches', 'total_yards', 'rec_yards']

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

    for metric in RB_METRICS:
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

    # Add rush_share as 5th metric (calculate on the fly)
    total_touches = game_stats.get('total_touches', 0)
    if total_touches > 0:
        rush_share = (game_stats.get('rush_attempts', 0) / total_touches) * 100
    else:
        rush_share = 0

    # Standardize rush_share
    rush_shares = []
    for _, ctx_game in all_games_context.iterrows():
        ctx_touches = ctx_game.get('total_touches', 0)
        if ctx_touches > 0:
            ctx_rush_share = (ctx_game.get('rush_attempts', 0) / ctx_touches) * 100
            rush_shares.append(ctx_rush_share)

    if len(rush_shares) >= 5:
        mean_rush_share = np.mean(rush_shares)
        std_rush_share = np.std(rush_shares)
        if std_rush_share > 0:
            rush_share_z = (rush_share - mean_rush_share) / std_rush_share
        else:
            rush_share_z = 0
        game_values.append(rush_share_z)

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

def generate_rb_game_by_game_ratings(years=range(2020, 2025)):
    """Generate game-by-game composite ratings for all RBs."""

    logger.info("="*80)
    logger.info("GENERATING RB GAME-BY-GAME COMPOSITE RATINGS")
    logger.info("="*80)

    fetcher = NFLDataFetcher()
    aggregator = RBStatsAggregator(filter_rb_only=True)

    all_game_ratings = []

    for year in years:
        logger.info(f"\n[{year}] Processing season...")

        # Load data
        pbp = fetcher.fetch_pbp_data([year])
        rosters = fetcher.fetch_rosters([year])

        # Get game-level stats
        game_stats = aggregator.aggregate_game_stats(pbp, rosters)

        logger.info(f"  {len(game_stats)} RB-game records")

        # Calculate composite for each game
        for idx, game in game_stats.iterrows():
            # Use all games in season for standardization context
            season_context = game_stats[game_stats['season'] == year]

            composite = calculate_game_composite_rating(game.to_dict(), season_context)

            if composite is not None:
                all_game_ratings.append({
                    'season': year,
                    'week': game['week'],
                    'player_id': game['player_id'],
                    'player_name': game['player_name'],
                    'team': game['posteam'],
                    'opponent': game.get('defteam', ''),
                    'rush_attempts': game.get('rush_attempts', 0),
                    'rush_yards': game.get('rush_yards', 0),
                    'targets': game.get('targets', 0),
                    'receptions': game.get('receptions', 0),
                    'rec_yards': game.get('rec_yards', 0),
                    'total_touches': game.get('total_touches', 0),
                    'total_yards': game.get('total_yards', 0),
                    'total_tds': game.get('total_tds', 0),
                    'raw_composite': composite
                })

        logger.info(f"  Calculated {len([r for r in all_game_ratings if r['season'] == year])} composite ratings")

    # Convert to DataFrame
    game_ratings_df = pd.DataFrame(all_game_ratings)

    logger.info(f"\nTotal game ratings: {len(game_ratings_df)}")
    logger.info(f"Unique RBs: {game_ratings_df['player_id'].nunique()}")

    # Apply smoothing and uncertainty for each player
    logger.info("\nApplying smoothing and calculating uncertainty...")

    smoothed_data = []

    for player_id in game_ratings_df['player_id'].unique():
        player_games = game_ratings_df[
            game_ratings_df['player_id'] == player_id
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

    logger.info(f"  Smoothed ratings for {final_ratings['player_id'].nunique()} RBs")

    # Save to production folder
    output_dir = Path("data/production")
    output_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d")
    output_file = output_dir / f"rb_composite_game_by_game_{date_str}.csv"

    final_ratings.to_csv(output_file, index=False)

    logger.info(f"\n✓ Saved to: {output_file}")
    logger.info(f"  Records: {len(final_ratings)}")
    logger.info(f"  Columns: {len(final_ratings.columns)}")

    # Display summary stats
    print("\n" + "="*80)
    print("RB GAME-BY-GAME COMPOSITE RATINGS SUMMARY")
    print("="*80)
    print(f"\nTotal game records: {len(final_ratings):,}")
    print(f"Unique RBs: {final_ratings['player_id'].nunique()}")
    print(f"Seasons: {final_ratings['season'].min()}-{final_ratings['season'].max()}")
    print(f"\nComposite Rating Range: [{final_ratings['smoothed_composite'].min():.3f}, {final_ratings['smoothed_composite'].max():.3f}]")
    print(f"Mean Composite: {final_ratings['smoothed_composite'].mean():.3f}")
    print(f"Mean Uncertainty: {final_ratings['composite_uncertainty'].mean():.3f}")

    # Top 10 individual games
    print("\n" + "="*80)
    print("TOP 10 INDIVIDUAL GAME PERFORMANCES")
    print("="*80)
    top_games = final_ratings.nlargest(10, 'raw_composite')[
        ['season', 'week', 'player_name', 'team', 'raw_composite', 'total_touches', 'total_yards', 'total_tds']
    ]
    print(top_games.to_string(index=False))

    logger.info("\n" + "="*80)
    logger.info("COMPLETE!")
    logger.info("="*80)

    return final_ratings

if __name__ == "__main__":
    generate_rb_game_by_game_ratings()
