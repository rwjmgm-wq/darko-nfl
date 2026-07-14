"""
Time-Weighted Metrics with Exponential Decay

Implements DARKO-style exponential decay weighting for QB statistics.
More recent games are weighted more heavily than older games.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeWeightedMetrics:
    """
    Calculates time-weighted metrics using exponential decay.

    The weight for each game is β^t where:
    - β is the decay rate (between 0 and 1)
    - t is the number of days elapsed since the game

    Example: β=0.996 means yesterday's game is weighted at 0.996,
    a week ago at 0.972, and a year ago at 0.208
    """

    def __init__(
        self,
        decay_rates: Optional[Dict[str, float]] = None,
        reference_date: Optional[datetime] = None
    ):
        """
        Initialize time-weighted metrics calculator.

        Args:
            decay_rates: Dictionary mapping metric names to decay rates (beta values)
                        If None, uses default rates
            reference_date: Reference date for calculating time weights
                           If None, uses the most recent game date
        """
        self.decay_rates = decay_rates or self._get_default_decay_rates()
        self.reference_date = reference_date

    def _get_default_decay_rates(self) -> Dict[str, float]:
        """Get default decay rates for different metrics."""
        return {
            'epa': 0.996,
            'cpoe': 0.995,
            'success_rate': 0.996,
            'completion_pct': 0.997,
            'yards_per_attempt': 0.996,
            'td_rate': 0.994,
            'int_rate': 0.994,
            'default': 0.996
        }

    def calculate_time_weights(
        self,
        dates: pd.Series,
        beta: float,
        reference_date: Optional[datetime] = None
    ) -> pd.Series:
        """
        Calculate exponential decay weights based on game dates.

        Args:
            dates: Series of game dates
            beta: Decay rate (between 0 and 1)
            reference_date: Reference date for weight calculation

        Returns:
            Series of weights
        """
        if reference_date is None:
            reference_date = dates.max()

        # Convert to datetime if needed
        dates = pd.to_datetime(dates)
        reference_date = pd.to_datetime(reference_date)

        # Calculate days elapsed
        days_elapsed = (reference_date - dates).dt.days

        # Calculate weights: β^t
        weights = beta ** days_elapsed

        return weights

    def calculate_weighted_metric(
        self,
        df: pd.DataFrame,
        metric_col: str,
        date_col: str = 'game_date',
        beta: Optional[float] = None,
        weight_by_attempts: bool = True,
        attempts_col: str = 'attempts'
    ) -> pd.Series:
        """
        Calculate time-weighted average for a metric across all games.

        Args:
            df: DataFrame with game-level data
            metric_col: Column name of metric to weight
            date_col: Column name of game dates
            beta: Decay rate (if None, uses metric-specific rate)
            weight_by_attempts: If True, also weight by number of attempts
            attempts_col: Column name for attempts

        Returns:
            Series with weighted metric for each row
        """
        # Get decay rate
        if beta is None:
            beta = self.decay_rates.get(metric_col, self.decay_rates['default'])

        # Calculate time weights
        time_weights = self.calculate_time_weights(
            df[date_col],
            beta,
            self.reference_date
        )

        # Combine with attempt weights if requested
        if weight_by_attempts and attempts_col in df.columns:
            total_weights = time_weights * df[attempts_col]
        else:
            total_weights = time_weights

        # Calculate weighted average
        weighted_metric = df[metric_col] * total_weights

        return weighted_metric

    def aggregate_player_metrics(
        self,
        df: pd.DataFrame,
        player_col: str = 'passer_player_id',
        metrics: Optional[List[str]] = None,
        date_col: str = 'game_date',
        weight_by_attempts: bool = True
    ) -> pd.DataFrame:
        """
        Aggregate time-weighted metrics for each player.

        Args:
            df: DataFrame with game-level stats
            player_col: Column identifying players
            metrics: List of metric columns to weight
            date_col: Column with game dates
            weight_by_attempts: If True, also weight by attempts

        Returns:
            DataFrame with one row per player and their weighted metrics
        """
        if metrics is None:
            metrics = ['epa_per_play', 'cpoe', 'success_rate',
                      'completion_pct', 'yards_per_attempt']

        # Ensure required columns exist
        available_metrics = [m for m in metrics if m in df.columns]
        if len(available_metrics) < len(metrics):
            missing = set(metrics) - set(available_metrics)
            logger.warning(f"Missing metrics: {missing}")

        results = []

        for player_id, player_df in df.groupby(player_col):
            player_stats = {player_col: player_id}

            # Get player name if available
            if 'passer_player_name' in df.columns:
                player_stats['player_name'] = player_df['passer_player_name'].iloc[0]

            # Get latest season
            if 'season' in df.columns:
                player_stats['latest_season'] = player_df['season'].max()

            # Total attempts
            if 'attempts' in player_df.columns:
                player_stats['total_attempts'] = player_df['attempts'].sum()

            # Calculate weighted metrics
            for metric in available_metrics:
                # Get decay rate for this metric
                beta = self.decay_rates.get(metric, self.decay_rates['default'])

                # Calculate time weights
                time_weights = self.calculate_time_weights(
                    player_df[date_col],
                    beta,
                    self.reference_date
                )

                # Weight by attempts if requested
                if weight_by_attempts and 'attempts' in player_df.columns:
                    weights = time_weights * player_df['attempts']
                else:
                    weights = time_weights

                # Calculate weighted average
                weighted_values = player_df[metric] * weights
                player_stats[f'{metric}_weighted'] = weighted_values.sum() / weights.sum()

            results.append(player_stats)

        result_df = pd.DataFrame(results)

        logger.info(f"Calculated weighted metrics for {len(result_df)} players")

        return result_df

    def get_decay_weights_summary(self, days: int = 365) -> pd.DataFrame:
        """
        Show how weights decay over time for different metrics.

        Args:
            days: Number of days to show

        Returns:
            DataFrame showing weights at different time points
        """
        time_points = [0, 1, 7, 30, 90, 180, 365]
        time_points = [t for t in time_points if t <= days]

        rows = []
        for metric, beta in self.decay_rates.items():
            row = {'metric': metric, 'beta': beta}
            for t in time_points:
                row[f'day_{t}'] = beta ** t
            rows.append(row)

        return pd.DataFrame(rows)


class RollingTimeWeightedMetrics:
    """
    Calculates rolling time-weighted metrics that update with each game.

    This is more efficient for real-time updates as it maintains state
    and only updates when new games are added.
    """

    def __init__(self, decay_rate: float = 0.996):
        """
        Initialize rolling calculator.

        Args:
            decay_rate: Beta value for exponential decay
        """
        self.decay_rate = decay_rate
        self.player_states = {}

    def update(
        self,
        player_id: str,
        metric_value: float,
        attempts: int = 1,
        days_since_last_update: int = 1
    ) -> float:
        """
        Update weighted metric for a player with new game data.

        Args:
            player_id: Player identifier
            metric_value: New metric value
            attempts: Number of attempts in this game
            days_since_last_update: Days since last game

        Returns:
            Updated weighted metric
        """
        # Decay existing weighted value
        if player_id in self.player_states:
            old_weighted_value, old_total_weight = self.player_states[player_id]

            # Apply time decay to old values
            decay_factor = self.decay_rate ** days_since_last_update
            decayed_value = old_weighted_value * decay_factor
            decayed_weight = old_total_weight * decay_factor
        else:
            decayed_value = 0
            decayed_weight = 0

        # Add new observation
        new_weighted_value = decayed_value + (metric_value * attempts)
        new_total_weight = decayed_weight + attempts

        # Store updated state
        self.player_states[player_id] = (new_weighted_value, new_total_weight)

        # Return weighted average
        return new_weighted_value / new_total_weight if new_total_weight > 0 else 0

    def get_current_value(self, player_id: str) -> float:
        """Get current weighted metric value for a player."""
        if player_id not in self.player_states:
            return 0

        weighted_value, total_weight = self.player_states[player_id]
        return weighted_value / total_weight if total_weight > 0 else 0


def main():
    """Example usage of time-weighted metrics."""
    # Create sample data
    dates = pd.date_range(start='2024-09-01', end='2024-12-01', freq='7D')
    sample_data = pd.DataFrame({
        'game_date': dates,
        'passer_player_id': ['player1'] * len(dates),
        'passer_player_name': ['Patrick Mahomes'] * len(dates),
        'epa_per_play': np.random.randn(len(dates)) * 0.1 + 0.15,
        'cpoe': np.random.randn(len(dates)) * 3 + 2,
        'attempts': np.random.randint(25, 45, len(dates)),
        'season': [2024] * len(dates)
    })

    # Calculate time-weighted metrics
    twm = TimeWeightedMetrics()

    # Show decay rates
    print("Decay Weight Summary:")
    print(twm.get_decay_weights_summary())

    # Calculate weighted metrics for player
    weighted_stats = twm.aggregate_player_metrics(sample_data)
    print("\nWeighted Stats:")
    print(weighted_stats)

    # Compare with simple average
    print("\nSimple averages:")
    print(f"EPA/play: {sample_data['epa_per_play'].mean():.4f}")
    print(f"CPOE: {sample_data['cpoe'].mean():.4f}")

    print("\nWeighted averages:")
    print(f"EPA/play: {weighted_stats['epa_per_play_weighted'].iloc[0]:.4f}")
    print(f"CPOE: {weighted_stats['cpoe_weighted'].iloc[0]:.4f}")


if __name__ == "__main__":
    main()
