"""
Game-by-Game LEAF Ratings

Extends Kalman filtering to provide real-time ratings after each game,
enabling in-season predictions and trajectory tracking.

Key Features:
- Rating trajectory: Full season path of QB ratings
- Uncertainty evolution: How confidence improves game-by-game
- In-season prediction: Use Week N rating to predict Week N+1
- Trend detection: Identify hot/cold streaks
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from pathlib import Path

from .kalman_filter import QBKalmanFilter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GameByGameRatingSystem:
    """
    Manages game-by-game LEAF ratings with full trajectory tracking.
    """

    def __init__(self, kalman_filter: Optional[QBKalmanFilter] = None):
        """
        Initialize game-by-game rating system.

        Args:
            kalman_filter: QBKalmanFilter instance (creates default if None)
        """
        self.kalman_filter = kalman_filter or QBKalmanFilter(
            auto_tune=True,
            em_iterations=10
        )

    def process_game_by_game(
        self,
        game_stats: pd.DataFrame,
        metrics: List[str],
        player_col: str = 'passer_player_id',
        game_date_col: str = 'game_date',
        prior_means: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Process all players and return game-by-game trajectory.

        Unlike process_all_players which returns one row per player,
        this returns one row per player-game with rating after that game.

        Args:
            game_stats: Game-level statistics
            metrics: List of metric columns to filter
            player_col: Column identifying players
            game_date_col: Column with game dates
            prior_means: Dictionary of prior means for each metric

        Returns:
            DataFrame with one row per player-game containing:
            - All original game data
            - Kalman estimates after this game
            - Uncertainty after this game
            - Game number in sequence
        """
        if prior_means is None:
            prior_means = {metric: 0.0 for metric in metrics}

        # Ensure game_date is datetime
        if not pd.api.types.is_datetime64_any_dtype(game_stats[game_date_col]):
            game_stats = game_stats.copy()
            game_stats[game_date_col] = pd.to_datetime(game_stats[game_date_col])

        logger.info("Processing game-by-game ratings...")

        # Process each player
        all_trajectories = []

        players = game_stats[player_col].unique()
        logger.info(f"  Processing {len(players)} players across {len(metrics)} metrics")

        for i, player_id in enumerate(players):
            if (i + 1) % 50 == 0:
                logger.info(f"  Processed {i + 1}/{len(players)} players...")

            player_games = game_stats[game_stats[player_col] == player_id].copy()
            player_games = player_games.sort_values(game_date_col)

            # Filter each metric
            for metric in metrics:
                if metric not in player_games.columns:
                    continue

                prior_mean = prior_means.get(metric, 0.0)

                # Get full trajectory
                estimates, uncertainties = self.kalman_filter.filter_player_metric(
                    player_id=player_id,
                    player_games=player_games,
                    metric=metric,
                    prior_mean=prior_mean,
                    game_date_col=game_date_col
                )

                # Add to player games dataframe
                player_games[f'{metric}_kalman'] = estimates
                player_games[f'{metric}_uncertainty'] = uncertainties

            # Add game sequence number
            player_games['game_number'] = range(1, len(player_games) + 1)

            all_trajectories.append(player_games)

        # Combine all player trajectories
        result = pd.concat(all_trajectories, ignore_index=True)

        logger.info(f"  Complete! {len(result)} player-games processed")

        return result

    def get_rating_at_date(
        self,
        player_id: str,
        date: pd.Timestamp,
        game_by_game_data: pd.DataFrame,
        metric: str = 'opp_adj_epa_kalman'
    ) -> Optional[Dict]:
        """
        Get player rating as of a specific date.

        Args:
            player_id: Player identifier
            date: Date to get rating for
            game_by_game_data: Output from process_game_by_game
            metric: Metric to retrieve

        Returns:
            Dictionary with rating info, or None if no games before date
        """
        player_data = game_by_game_data[
            (game_by_game_data['passer_player_id'] == player_id) &
            (game_by_game_data['game_date'] <= date)
        ].sort_values('game_date')

        if len(player_data) == 0:
            return None

        # Get most recent game before date
        latest_game = player_data.iloc[-1]

        return {
            'player_id': player_id,
            'player_name': latest_game.get('passer_player_name', ''),
            'rating': latest_game[metric],
            'uncertainty': latest_game[f'{metric.replace("_kalman", "")}_uncertainty'],
            'games_played': latest_game['game_number'],
            'last_game_date': latest_game['game_date'],
            'season': latest_game['season'],
            'week': latest_game['week']
        }

    def get_player_trajectory(
        self,
        player_id: str,
        season: int,
        game_by_game_data: pd.DataFrame,
        metric: str = 'opp_adj_epa_kalman'
    ) -> pd.DataFrame:
        """
        Get full season trajectory for a player.

        Args:
            player_id: Player identifier
            season: Season year
            game_by_game_data: Output from process_game_by_game
            metric: Metric to retrieve

        Returns:
            DataFrame with game-by-game ratings for this player-season
        """
        trajectory = game_by_game_data[
            (game_by_game_data['passer_player_id'] == player_id) &
            (game_by_game_data['season'] == season)
        ].sort_values('game_date')

        if len(trajectory) == 0:
            return pd.DataFrame()

        # Return relevant columns
        cols = [
            'game_id', 'game_date', 'week', 'game_number',
            metric, f'{metric.replace("_kalman", "")}_uncertainty',
            'attempts', 'epa_per_play'
        ]
        cols = [c for c in cols if c in trajectory.columns]

        return trajectory[cols].copy()

    def detect_trends(
        self,
        player_id: str,
        season: int,
        game_by_game_data: pd.DataFrame,
        metric: str = 'opp_adj_epa_kalman',
        window: int = 4
    ) -> Optional[Dict]:
        """
        Detect if player is on hot/cold streak.

        Args:
            player_id: Player identifier
            season: Season year
            game_by_game_data: Output from process_game_by_game
            metric: Metric to analyze
            window: Number of recent games to analyze

        Returns:
            Dictionary with trend info
        """
        trajectory = self.get_player_trajectory(
            player_id, season, game_by_game_data, metric
        )

        if len(trajectory) < window:
            return None

        # Get last N games
        recent = trajectory.tail(window)

        # Calculate trend (linear regression slope)
        x = np.arange(len(recent))
        y = recent[metric].values

        # Simple linear regression
        slope = np.polyfit(x, y, 1)[0]

        # Classify trend
        if slope > 0.01:
            trend = 'improving'
        elif slope < -0.01:
            trend = 'declining'
        else:
            trend = 'stable'

        return {
            'player_id': player_id,
            'season': season,
            'games_analyzed': len(recent),
            'trend': trend,
            'slope_per_game': slope,
            'current_rating': y[-1],
            'rating_change': y[-1] - y[0]
        }


def validate_in_season_prediction(
    game_by_game_data: pd.DataFrame,
    train_years: List[int],
    test_years: List[int],
    metric: str = 'opp_adj_epa_kalman',
    prediction_weeks: List[int] = [8, 10, 12],
    min_attempts: int = 150
) -> pd.DataFrame:
    """
    Validate whether mid-season ratings predict better than end-of-season.

    Tests: Does Week 8 rating predict Weeks 9-17 better than season-end rating?

    Args:
        game_by_game_data: Output from process_game_by_game
        train_years: Years for training
        test_years: Years for testing
        metric: Rating metric to use
        prediction_weeks: Which weeks to test as prediction points
        min_attempts: Minimum season attempts

    Returns:
        DataFrame with validation results by prediction week
    """
    logger.info("\n" + "="*80)
    logger.info("Validating In-Season Prediction Power")
    logger.info("="*80)

    results = []

    for pred_week in prediction_weeks:
        logger.info(f"\nTesting Week {pred_week} predictions...")

        # For each QB in train years, get Week N rating
        train_ratings = []

        for season in train_years:
            season_data = game_by_game_data[game_by_game_data['season'] == season]

            # Get rating after pred_week
            week_n_data = season_data[season_data['week'] == pred_week]

            for player_id in week_n_data['passer_player_id'].unique():
                player_season = season_data[season_data['passer_player_id'] == player_id]

                # Check if they have enough attempts
                if player_season['attempts'].sum() < min_attempts:
                    continue

                # Get rating after pred_week
                week_n_rating = week_n_data[
                    week_n_data['passer_player_id'] == player_id
                ][metric].values

                if len(week_n_rating) == 0:
                    continue

                # Get next season raw EPA
                next_season_data = game_by_game_data[
                    (game_by_game_data['season'] == season + 1) &
                    (game_by_game_data['passer_player_id'] == player_id)
                ]

                if len(next_season_data) == 0:
                    continue

                next_season_attempts = next_season_data['attempts'].sum()
                if next_season_attempts < min_attempts:
                    continue

                next_season_epa = next_season_data['epa_per_play'].mean()

                train_ratings.append({
                    'player_id': player_id,
                    'season': season,
                    'week_n_rating': week_n_rating[0],
                    'next_season_epa': next_season_epa
                })

        train_df = pd.DataFrame(train_ratings)

        if len(train_df) < 10:
            logger.warning(f"  Not enough data for Week {pred_week}")
            continue

        # Calculate correlation
        corr = train_df['week_n_rating'].corr(train_df['next_season_epa'])
        rmse = np.sqrt(((train_df['week_n_rating'] - train_df['next_season_epa']) ** 2).mean())

        logger.info(f"  Week {pred_week} Rating → Next Season EPA:")
        logger.info(f"    Correlation: {corr:.4f}")
        logger.info(f"    RMSE: {rmse:.4f}")
        logger.info(f"    Sample: {len(train_df)} QB-seasons")

        results.append({
            'prediction_week': pred_week,
            'correlation': corr,
            'rmse': rmse,
            'n_qbs': len(train_df)
        })

    return pd.DataFrame(results)


def main():
    """Example usage of game-by-game rating system."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.data_pipeline.nfl_data_fetcher import NFLDataFetcher
    from src.data_pipeline.qb_stats_aggregator import QBStatsAggregator
    from src.features.opponent_adjustments import DefenseRatingSystem, OpponentAdjuster

    # Fetch data
    logger.info("Fetching data...")
    fetcher = NFLDataFetcher()
    pbp = fetcher.fetch_pbp_data([2020, 2021, 2022, 2023])

    # Aggregate QB stats
    logger.info("Aggregating QB stats...")
    qb_agg = QBStatsAggregator()
    qb_games = qb_agg.aggregate_game_stats(pbp)

    # Apply opponent adjustments
    logger.info("Calculating opponent adjustments...")
    defense_system = DefenseRatingSystem()
    defense_ratings = defense_system.calculate_defense_ratings(pbp)
    adjuster = OpponentAdjuster(defense_ratings)
    qb_adj = adjuster.adjust_qb_game_stats(qb_games)

    # Create game-by-game ratings
    logger.info("\nCreating game-by-game ratings...")
    rating_system = GameByGameRatingSystem()

    game_by_game = rating_system.process_game_by_game(
        game_stats=qb_adj,
        metrics=['epa_per_play', 'opp_adj_epa'],
        prior_means={'epa_per_play': 0.0, 'opp_adj_epa': 0.0}
    )

    # Save results
    output_dir = Path('data') / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / 'qb_game_by_game_ratings.csv'
    game_by_game.to_csv(output_path, index=False)
    logger.info(f"\nSaved to {output_path}")

    # Example: Get Patrick Mahomes 2023 trajectory
    logger.info("\nExample: Patrick Mahomes 2023 Trajectory")
    mahomes_trajectory = rating_system.get_player_trajectory(
        player_id='00-0033873',  # Mahomes ID
        season=2023,
        game_by_game_data=game_by_game,
        metric='opp_adj_epa_kalman'
    )

    if len(mahomes_trajectory) > 0:
        print(mahomes_trajectory.to_string(index=False))

    # Validate in-season prediction
    logger.info("\nValidating in-season prediction power...")
    validation_results = validate_in_season_prediction(
        game_by_game_data=game_by_game,
        train_years=[2020, 2021, 2022],
        test_years=[2023],
        metric='opp_adj_epa_kalman',
        prediction_weeks=[8, 10, 12]
    )

    print("\n" + "="*80)
    print("In-Season Prediction Validation")
    print("="*80)
    print(validation_results.to_string(index=False))

    # Save validation results
    val_path = Path('results') / 'in_season_prediction_validation.csv'
    val_path.parent.mkdir(exist_ok=True)
    validation_results.to_csv(val_path, index=False)
    logger.info(f"\nSaved validation results to {val_path}")


if __name__ == "__main__":
    main()
