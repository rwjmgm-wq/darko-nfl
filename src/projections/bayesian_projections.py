"""
Bayesian Projection System

Implements DARKO-style Bayesian projections that combine:
1. Prior beliefs (league average, historical performance)
2. Recent performance (time-weighted)
3. Age adjustments
4. Uncertainty quantification

The system updates projections as new data arrives using Bayesian inference.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BayesianProjection:
    """
    Bayesian projection system for QB performance.

    Uses Bayesian updating to combine prior beliefs with observed performance,
    properly accounting for uncertainty and sample size.
    """

    def __init__(
        self,
        prior_means: Optional[Dict[str, float]] = None,
        prior_weight: float = 200,
        confidence_level: float = 0.95
    ):
        """
        Initialize Bayesian projection system.

        Args:
            prior_means: Dictionary of league-average values for each metric
            prior_weight: Strength of prior (equivalent number of attempts)
            confidence_level: Confidence level for intervals (e.g., 0.95 for 95%)
        """
        self.prior_means = prior_means or self._get_default_priors()
        self.prior_weight = prior_weight
        self.confidence_level = confidence_level

    def _get_default_priors(self) -> Dict[str, float]:
        """Get default league-average priors."""
        return {
            'epa_per_play': 0.0,
            'cpoe': 0.0,
            'success_rate': 0.50,
            'completion_pct': 0.64,
            'yards_per_attempt': 7.0,
            'td_rate': 0.045,
            'int_rate': 0.025
        }

    def calculate_posterior(
        self,
        observed_mean: float,
        observed_n: float,
        observed_std: float,
        prior_mean: float,
        prior_weight: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Calculate Bayesian posterior mean and standard deviation.

        Uses normal-normal conjugate prior for analytical solution.

        Args:
            observed_mean: Mean of observed data
            observed_n: Number of observations (weight)
            observed_std: Standard deviation of observations
            prior_mean: Prior mean
            prior_weight: Weight of prior (if None, uses self.prior_weight)

        Returns:
            Tuple of (posterior_mean, posterior_std)
        """
        if prior_weight is None:
            prior_weight = self.prior_weight

        # Posterior mean is weighted average of prior and observed
        total_weight = prior_weight + observed_n
        posterior_mean = (
            (prior_weight * prior_mean + observed_n * observed_mean) /
            total_weight
        )

        # Posterior standard deviation
        # Assumes prior std is similar to observed std
        posterior_variance = (observed_std ** 2) / total_weight
        posterior_std = np.sqrt(posterior_variance)

        return posterior_mean, posterior_std

    def project_metric(
        self,
        player_data: pd.Series,
        metric: str,
        attempts_col: str = 'total_attempts',
        std_col: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Project a single metric for a player.

        Args:
            player_data: Series with player's statistics
            metric: Name of metric to project
            attempts_col: Column name for attempts (sample size)
            std_col: Column name for standard deviation (if available)

        Returns:
            Dictionary with projection, confidence intervals, and uncertainty
        """
        # Get weighted metric name
        metric_col = f'{metric}_weighted'
        if metric_col not in player_data:
            metric_col = metric

        if metric_col not in player_data or pd.isna(player_data[metric_col]):
            # No data, return prior
            return {
                'projection': self.prior_means.get(metric, 0.0),
                'lower_ci': None,
                'upper_ci': None,
                'uncertainty': 'high'
            }

        observed_mean = player_data[metric_col]
        observed_n = player_data.get(attempts_col, 100)

        # Estimate standard deviation if not provided
        if std_col and std_col in player_data:
            observed_std = player_data[std_col]
        else:
            # Use typical within-player standard deviations
            typical_stds = {
                'epa_per_play': 0.15,
                'cpoe': 3.0,
                'success_rate': 0.10,
                'completion_pct': 0.05,
                'yards_per_attempt': 1.0,
                'td_rate': 0.02,
                'int_rate': 0.015
            }
            observed_std = typical_stds.get(metric, 0.1)

        # Calculate posterior
        prior_mean = self.prior_means.get(metric, 0.0)
        posterior_mean, posterior_std = self.calculate_posterior(
            observed_mean, observed_n, observed_std, prior_mean
        )

        # Calculate confidence intervals
        z_score = stats.norm.ppf((1 + self.confidence_level) / 2)
        margin = z_score * posterior_std

        lower_ci = posterior_mean - margin
        upper_ci = posterior_mean + margin

        # Categorize uncertainty
        if observed_n < 50:
            uncertainty = 'high'
        elif observed_n < 200:
            uncertainty = 'medium'
        else:
            uncertainty = 'low'

        return {
            'projection': posterior_mean,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci,
            'std': posterior_std,
            'uncertainty': uncertainty,
            'sample_size': observed_n
        }

    def project_player(
        self,
        player_data: pd.Series,
        metrics: Optional[list] = None,
        age_adjustment: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Generate projections for all metrics for a player.

        Args:
            player_data: Series with player's current statistics
            metrics: List of metrics to project
            age_adjustment: Dictionary of age adjustments for each metric

        Returns:
            DataFrame with projections and confidence intervals
        """
        if metrics is None:
            metrics = ['epa_per_play', 'cpoe', 'success_rate',
                      'completion_pct', 'yards_per_attempt']

        projections = []

        for metric in metrics:
            proj = self.project_metric(player_data, metric)

            # Apply age adjustment if provided
            if age_adjustment and metric in age_adjustment:
                adj = age_adjustment[metric]
                proj['projection'] += adj
                if proj['lower_ci'] is not None:
                    proj['lower_ci'] += adj
                    proj['upper_ci'] += adj

            projections.append({
                'metric': metric,
                **proj
            })

        return pd.DataFrame(projections)

    def project_all_players(
        self,
        player_stats: pd.DataFrame,
        metrics: Optional[list] = None,
        player_col: str = 'passer_player_id',
        name_col: str = 'player_name'
    ) -> pd.DataFrame:
        """
        Generate projections for all players.

        Args:
            player_stats: DataFrame with current player statistics
            metrics: List of metrics to project
            player_col: Column name for player ID
            name_col: Column name for player name

        Returns:
            DataFrame with projections for all players
        """
        if metrics is None:
            metrics = ['epa_per_play', 'cpoe', 'success_rate']

        all_projections = []

        for idx, player_data in player_stats.iterrows():
            player_proj = self.project_player(player_data, metrics)

            # Add player identifying information
            player_proj[player_col] = player_data.get(player_col)
            if name_col in player_data:
                player_proj[name_col] = player_data.get(name_col)

            all_projections.append(player_proj)

        result = pd.concat(all_projections, ignore_index=True)

        logger.info(f"Generated projections for {len(player_stats)} players")

        return result

    def calculate_regression_amount(
        self,
        observed_n: float,
        metric: str = 'epa_per_play'
    ) -> float:
        """
        Calculate how much to regress toward the mean.

        Returns the weight given to observed data vs prior.

        Args:
            observed_n: Number of observations
            metric: Metric name (for metric-specific regression)

        Returns:
            Weight for observed data (0 to 1)
        """
        return observed_n / (observed_n + self.prior_weight)

    def update_projection(
        self,
        current_projection: float,
        new_observation: float,
        new_weight: float = 1.0
    ) -> float:
        """
        Update projection with new observation (for online/daily updates).

        Args:
            current_projection: Current projected value
            new_observation: New observed value
            new_weight: Weight of new observation (e.g., number of attempts)

        Returns:
            Updated projection
        """
        # This is simplified; full version would track full posterior distribution
        learning_rate = new_weight / (self.prior_weight + new_weight)
        updated = current_projection + learning_rate * (new_observation - current_projection)

        return updated


class DARKOProjectionSystem:
    """
    Complete DARKO-style projection system integrating all components.

    Combines:
    - Time-weighted recent performance
    - Age adjustments
    - Bayesian updating
    - Uncertainty quantification
    """

    def __init__(
        self,
        time_weighted_metrics,
        age_curve_model,
        bayesian_projector: Optional[BayesianProjection] = None
    ):
        """
        Initialize complete projection system.

        Args:
            time_weighted_metrics: TimeWeightedMetrics instance
            age_curve_model: AgeCurveModel instance
            bayesian_projector: BayesianProjection instance (optional)
        """
        self.twm = time_weighted_metrics
        self.age_model = age_curve_model
        self.bayesian = bayesian_projector or BayesianProjection()

    def generate_projections(
        self,
        game_stats: pd.DataFrame,
        roster_data: pd.DataFrame,
        metrics: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Generate complete projections for all QBs.

        Args:
            game_stats: DataFrame with game-level QB stats
            roster_data: DataFrame with roster information (including age)
            metrics: List of metrics to project

        Returns:
            DataFrame with projections for all QBs
        """
        if metrics is None:
            metrics = ['epa_per_play', 'cpoe', 'success_rate']

        # Step 1: Calculate time-weighted metrics
        logger.info("Calculating time-weighted metrics...")
        weighted_stats = self.twm.aggregate_player_metrics(
            game_stats,
            metrics=metrics
        )

        # Step 2: Merge with roster data to get ages
        if 'age' in roster_data.columns:
            weighted_stats = weighted_stats.merge(
                roster_data[['player_id', 'age']].drop_duplicates(),
                left_on='passer_player_id',
                right_on='player_id',
                how='left'
            )

            # Step 3: Apply age adjustments
            logger.info("Applying age adjustments...")
            for metric in metrics:
                metric_col = f'{metric}_weighted'
                if metric_col in weighted_stats.columns:
                    age_adjustments = []
                    for _, row in weighted_stats.iterrows():
                        if pd.notna(row.get('age')):
                            adj = self.age_model.get_age_adjustment_factor(
                                int(row['age']), metric
                            )
                            age_adjustments.append(adj)
                        else:
                            age_adjustments.append(0)

                    weighted_stats[f'{metric}_age_adjusted'] = (
                        weighted_stats[metric_col] - age_adjustments
                    )

        # Step 4: Generate Bayesian projections
        logger.info("Generating Bayesian projections...")
        projections = self.bayesian.project_all_players(
            weighted_stats,
            metrics=metrics
        )

        return projections

    def rank_players(
        self,
        projections: pd.DataFrame,
        metric: str = 'epa_per_play',
        min_sample_size: int = 50
    ) -> pd.DataFrame:
        """
        Rank players by projected performance.

        Args:
            projections: DataFrame with projections
            metric: Metric to rank by
            min_sample_size: Minimum sample size to include

        Returns:
            DataFrame with ranked players
        """
        # Filter to specific metric
        metric_proj = projections[projections['metric'] == metric].copy()

        # Filter by sample size if available
        if 'sample_size' in metric_proj.columns:
            metric_proj = metric_proj[
                metric_proj['sample_size'] >= min_sample_size
            ]

        # Sort by projection
        metric_proj = metric_proj.sort_values('projection', ascending=False)
        metric_proj['rank'] = range(1, len(metric_proj) + 1)

        return metric_proj


def main():
    """Example usage of Bayesian projection system."""
    # Create sample weighted stats
    sample_data = pd.DataFrame({
        'passer_player_id': ['player1', 'player2', 'player3'],
        'player_name': ['Patrick Mahomes', 'Josh Allen', 'Joe Burrow'],
        'epa_per_play_weighted': [0.25, 0.22, 0.20],
        'cpoe_weighted': [3.5, 2.8, 2.2],
        'success_rate_weighted': [0.55, 0.53, 0.51],
        'total_attempts': [400, 450, 300],
    })

    # Initialize projection system
    projector = BayesianProjection()

    # Generate projections
    projections = projector.project_all_players(sample_data)

    print("QB Projections:")
    print(projections)

    # Show regression amounts
    print("\nRegression to mean (weight on observed data):")
    for n in [50, 100, 200, 400]:
        weight = projector.calculate_regression_amount(n)
        print(f"  {n} attempts: {weight:.1%}")


if __name__ == "__main__":
    main()
