"""
Kalman Filter for QB Performance Tracking

Implements adaptive Kalman filtering for QB metrics with:
- Auto-tuning of process and measurement noise
- Player-specific initial uncertainty based on experience
- Time-series filtering for optimal weight updating
- Uncertainty quantification for Bayesian projections
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pykalman import KalmanFilter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QBKalmanFilter:
    """
    Kalman filter for tracking QB performance over time.

    Adapts learning rates based on uncertainty, providing optimal
    weights for recent vs historical performance.
    """

    def __init__(
        self,
        auto_tune: bool = True,
        em_iterations: int = 10,
        initial_process_noise: float = 0.02,
        initial_observation_noise: float = 0.15
    ):
        """
        Initialize Kalman filter system.

        Args:
            auto_tune: Whether to use EM algorithm for parameter tuning
            em_iterations: Number of EM iterations for auto-tuning
            initial_process_noise: Starting process noise (skill change rate)
            initial_observation_noise: Starting observation noise (game-to-game variance)
        """
        self.auto_tune = auto_tune
        self.em_iterations = em_iterations
        self.initial_process_noise = initial_process_noise
        self.initial_observation_noise = initial_observation_noise

        # Store learned parameters per metric
        self.tuned_parameters = {}

    def _get_initial_uncertainty(
        self,
        player_id: str,
        player_games: pd.DataFrame,
        metric: str,
        attempts_col: str = 'attempts'
    ) -> float:
        """
        Determine initial uncertainty based on player experience.

        Args:
            player_id: Player identifier
            player_games: Historical games for this player
            metric: Metric being tracked
            attempts_col: Name of attempts column

        Returns:
            Initial state covariance (higher = more uncertainty)
        """
        total_attempts = player_games[attempts_col].sum()

        # Rookie or very limited experience: high uncertainty
        if total_attempts < 100:
            return 2.0
        # Moderate experience: medium uncertainty
        elif total_attempts < 500:
            return 1.0
        # Veteran: low uncertainty (we have good prior)
        else:
            return 0.5

    def _create_kalman_filter(
        self,
        initial_mean: float,
        initial_covariance: float,
        metric: str
    ) -> KalmanFilter:
        """
        Create Kalman filter instance for a specific metric.

        Args:
            initial_mean: Initial state estimate (prior belief)
            initial_covariance: Initial uncertainty
            metric: Metric being tracked

        Returns:
            Configured KalmanFilter instance
        """
        kf = KalmanFilter(
            transition_matrices=[1],  # Random walk model: state_t = state_{t-1} + noise
            observation_matrices=[1],  # Direct observation of state
            initial_state_mean=initial_mean,
            initial_state_covariance=initial_covariance,
            observation_covariance=self.initial_observation_noise**2,
            transition_covariance=self.initial_process_noise**2
        )

        return kf

    def _auto_tune_filter(
        self,
        kf: KalmanFilter,
        observations: np.ndarray,
        metric: str
    ) -> KalmanFilter:
        """
        Auto-tune Kalman filter parameters using EM algorithm.

        Args:
            kf: Kalman filter to tune
            observations: Observed values
            metric: Metric name for caching

        Returns:
            Tuned KalmanFilter instance
        """
        if not self.auto_tune or len(observations) < 5:
            return kf

        # Use EM algorithm to learn optimal noise parameters
        kf_tuned = kf.em(
            observations.reshape(-1, 1),
            n_iter=self.em_iterations,
            em_vars=['transition_covariance', 'observation_covariance']
        )

        # Cache learned parameters
        if metric not in self.tuned_parameters:
            self.tuned_parameters[metric] = {
                'observation_noise': np.sqrt(self._get_scalar(kf_tuned.observation_covariance)),
                'process_noise': np.sqrt(self._get_scalar(kf_tuned.transition_covariance)),
                'sample_size': len(observations)
            }

        return kf_tuned

    def _get_scalar(self, value):
        """Extract scalar from value that might be float or array."""
        if isinstance(value, (int, float)):
            return value
        return value.item() if hasattr(value, 'item') else value[0, 0]

    def filter_player_metric(
        self,
        player_id: str,
        player_games: pd.DataFrame,
        metric: str,
        prior_mean: float = 0.0,
        game_date_col: str = 'game_date',
        attempts_col: str = 'attempts'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Kalman filter to a player's game-by-game metric values.

        Args:
            player_id: Player identifier
            player_games: DataFrame with player's games (must be sorted by date)
            metric: Column name for metric to filter
            prior_mean: Prior belief about player skill (league average)
            game_date_col: Name of date column
            attempts_col: Name of attempts column

        Returns:
            Tuple of (filtered_estimates, uncertainties)
        """
        # Ensure sorted by date
        player_games = player_games.sort_values(game_date_col).copy()

        # Get observations
        observations = player_games[metric].values

        # Handle missing values
        valid_mask = ~np.isnan(observations)
        if not valid_mask.any():
            # No valid observations
            return np.full(len(observations), prior_mean), np.full(len(observations), 999.0)

        # Determine initial uncertainty
        initial_cov = self._get_initial_uncertainty(player_id, player_games, metric, attempts_col)

        # Create Kalman filter
        kf = self._create_kalman_filter(
            initial_mean=prior_mean,
            initial_covariance=initial_cov,
            metric=metric
        )

        # Auto-tune if enabled
        kf = self._auto_tune_filter(kf, observations[valid_mask], metric)

        # Apply Kalman filter
        filtered_states, filtered_covariances = kf.filter(observations.reshape(-1, 1))

        # Extract results
        estimates = filtered_states.flatten()
        uncertainties = np.sqrt(filtered_covariances[:, 0, 0])

        return estimates, uncertainties

    def process_all_players(
        self,
        game_stats: pd.DataFrame,
        metrics: List[str],
        player_col: str = 'passer_player_id',
        game_date_col: str = 'game_date',
        attempts_col: str = 'attempts',
        prior_means: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Apply Kalman filtering to all players for specified metrics.

        Args:
            game_stats: Game-level statistics
            metrics: List of metric columns to filter
            player_col: Column identifying players
            game_date_col: Column with game dates
            attempts_col: Column with attempts/targets count
            prior_means: Dictionary of prior means for each metric

        Returns:
            DataFrame with Kalman-filtered estimates and uncertainties
        """
        if prior_means is None:
            prior_means = {metric: 0.0 for metric in metrics}

        # Ensure game_date is datetime
        if not pd.api.types.is_datetime64_any_dtype(game_stats[game_date_col]):
            game_stats[game_date_col] = pd.to_datetime(game_stats[game_date_col])

        # Process each player
        results = []

        players = game_stats[player_col].unique()
        logger.info(f"Applying Kalman filter to {len(players)} players across {len(metrics)} metrics")

        for i, player_id in enumerate(players):
            if (i + 1) % 50 == 0:
                logger.info(f"  Processed {i + 1}/{len(players)} players...")

            player_games = game_stats[game_stats[player_col] == player_id].copy()

            # Get player info
            player_name = player_games['passer_player_name'].iloc[0] if 'passer_player_name' in player_games.columns else ''
            total_attempts = player_games[attempts_col].sum()

            player_result = {
                player_col: player_id,
                'player_name': player_name,
                'total_attempts': total_attempts,
                'games_played': len(player_games)
            }

            # Filter each metric
            for metric in metrics:
                if metric not in player_games.columns:
                    logger.warning(f"Metric {metric} not found for player {player_id}")
                    continue

                prior_mean = prior_means.get(metric, 0.0)

                estimates, uncertainties = self.filter_player_metric(
                    player_id=player_id,
                    player_games=player_games,
                    metric=metric,
                    prior_mean=prior_mean,
                    game_date_col=game_date_col,
                    attempts_col=attempts_col
                )

                # Use most recent estimate as current projection
                player_result[f'{metric}_kalman'] = estimates[-1]
                player_result[f'{metric}_uncertainty'] = uncertainties[-1]

                # Also store full time series if needed (optional)
                # player_result[f'{metric}_kalman_series'] = estimates

            results.append(player_result)

        results_df = pd.DataFrame(results)

        # Log tuned parameters
        if self.tuned_parameters:
            logger.info("\nAuto-tuned Kalman parameters:")
            for metric, params in self.tuned_parameters.items():
                logger.info(f"  {metric}:")
                logger.info(f"    Observation noise: {params['observation_noise']:.4f}")
                logger.info(f"    Process noise: {params['process_noise']:.4f}")
                logger.info(f"    Based on {params['sample_size']} observations")

        return results_df

    def get_tuned_parameters(self) -> Dict[str, Dict[str, float]]:
        """
        Get auto-tuned parameters for each metric.

        Returns:
            Dictionary of tuned parameters by metric
        """
        return self.tuned_parameters.copy()


class HybridKalmanDecay:
    """
    Hybrid approach combining Kalman filtering with exponential decay.

    Uses Kalman filter for adaptive learning, but incorporates
    exponential time-weighting within the Kalman framework.
    """

    def __init__(
        self,
        kalman_filter: QBKalmanFilter,
        decay_rate: float = 0.996
    ):
        """
        Initialize hybrid system.

        Args:
            kalman_filter: QBKalmanFilter instance
            decay_rate: Exponential decay rate for time weighting
        """
        self.kalman_filter = kalman_filter
        self.decay_rate = decay_rate

    def filter_with_time_weights(
        self,
        player_id: str,
        player_games: pd.DataFrame,
        metric: str,
        prior_mean: float = 0.0,
        game_date_col: str = 'game_date'
    ) -> Tuple[float, float]:
        """
        Apply Kalman filter with exponential time-weighting.

        Args:
            player_id: Player identifier
            player_games: Player's games sorted by date
            metric: Metric to filter
            prior_mean: Prior mean
            game_date_col: Date column name

        Returns:
            Tuple of (final_estimate, final_uncertainty)
        """
        # First apply Kalman filter
        estimates, uncertainties = self.kalman_filter.filter_player_metric(
            player_id=player_id,
            player_games=player_games,
            metric=metric,
            prior_mean=prior_mean,
            game_date_col=game_date_col
        )

        # Apply exponential time-weighting to the Kalman estimates
        if not pd.api.types.is_datetime64_any_dtype(player_games[game_date_col]):
            dates = pd.to_datetime(player_games[game_date_col])
        else:
            dates = player_games[game_date_col]

        reference_date = dates.max()
        days_elapsed = (reference_date - dates).dt.days.values
        time_weights = self.decay_rate ** days_elapsed

        # Weight Kalman estimates by recency
        # Use inverse variance weighting combined with time weights
        precision = 1.0 / (uncertainties ** 2 + 1e-6)  # Avoid division by zero
        combined_weights = time_weights * precision

        # Weighted average of Kalman estimates
        final_estimate = np.average(estimates, weights=combined_weights)

        # Uncertainty of weighted average
        # Use most recent uncertainty, scaled by time-weighting effect
        final_uncertainty = uncertainties[-1] * np.sqrt(1.0 / combined_weights.sum() * len(combined_weights))

        return final_estimate, final_uncertainty


def main():
    """Example usage of Kalman filter."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from data_pipeline.nfl_data_fetcher import NFLDataFetcher
    from data_pipeline.qb_stats_aggregator import QBStatsAggregator

    # Fetch data
    fetcher = NFLDataFetcher()
    pbp = fetcher.get_qb_play_by_play([2024])

    # Aggregate game stats
    aggregator = QBStatsAggregator()
    game_stats = aggregator.aggregate_game_stats(pbp)

    # Apply Kalman filter
    kf = QBKalmanFilter(auto_tune=True, em_iterations=10)

    metrics = ['epa_per_play', 'cpoe_mean', 'success_mean']
    prior_means = {
        'epa_per_play': 0.0,
        'cpoe_mean': 0.0,
        'success_mean': 0.5
    }

    kalman_results = kf.process_all_players(
        game_stats=game_stats,
        metrics=metrics,
        prior_means=prior_means
    )

    # Show top QBs by Kalman-filtered EPA
    print("\nTop 10 QBs by Kalman-filtered EPA:")
    top_qbs = kalman_results.nlargest(10, 'epa_per_play_kalman')
    print(top_qbs[['player_name', 'total_attempts', 'epa_per_play_kalman',
                   'epa_per_play_uncertainty']].to_string(index=False))

    # Show tuned parameters
    print("\nTuned Kalman parameters:")
    for metric, params in kf.get_tuned_parameters().items():
        print(f"\n{metric}:")
        print(f"  Observation noise: {params['observation_noise']:.4f}")
        print(f"  Process noise: {params['process_noise']:.4f}")


if __name__ == "__main__":
    main()
