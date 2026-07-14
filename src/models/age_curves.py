"""
Age Curve Models

Models how QB performance changes with age using various approaches:
- Polynomial regression
- Generalized Additive Models (GAMs)
- Empirical age adjustments

Research shows QB peak around age 27-29 with decline beginning in early 30s.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgeCurveModel:
    """Models QB performance as a function of age."""

    def __init__(
        self,
        degree: int = 3,
        peak_age: int = 28,
        min_attempts: int = 100
    ):
        """
        Initialize age curve model.

        Args:
            degree: Polynomial degree for fitting
            peak_age: Expected peak age for QBs
            min_attempts: Minimum attempts to include a season
        """
        self.degree = degree
        self.peak_age = peak_age
        self.min_attempts = min_attempts
        self.models = {}
        self.age_adjustments = {}

    def fit(
        self,
        df: pd.DataFrame,
        metrics: list,
        age_col: str = 'age',
        player_col: str = 'passer_player_id'
    ):
        """
        Fit age curve models for specified metrics.

        Args:
            df: DataFrame with player-season data including age and metrics
            metrics: List of metric columns to model
            age_col: Column name for age
            player_col: Column name for player ID
        """
        # Filter to qualified seasons
        if 'attempts' in df.columns:
            df = df[df['attempts'] >= self.min_attempts].copy()

        for metric in metrics:
            if metric not in df.columns:
                logger.warning(f"Metric {metric} not found in DataFrame")
                continue

            # Remove missing values
            valid_data = df[[age_col, metric, player_col]].dropna()

            if len(valid_data) < 50:
                logger.warning(f"Insufficient data for {metric}: {len(valid_data)} observations")
                continue

            # Create polynomial features
            X = valid_data[[age_col]].values
            y = valid_data[metric].values

            # Fit polynomial regression with ridge regularization
            model = Pipeline([
                ('poly', PolynomialFeatures(degree=self.degree)),
                ('ridge', Ridge(alpha=1.0))
            ])

            model.fit(X, y)
            self.models[metric] = model

            # Calculate empirical age adjustments (alternative approach)
            age_adjustments = self._calculate_empirical_adjustments(
                valid_data, metric, age_col, player_col
            )
            self.age_adjustments[metric] = age_adjustments

            logger.info(f"Fitted age curve for {metric}")

    def _calculate_empirical_adjustments(
        self,
        df: pd.DataFrame,
        metric: str,
        age_col: str,
        player_col: str
    ) -> Dict[int, float]:
        """
        Calculate empirical age adjustments using career-average approach.

        This method looks at how players perform at each age relative to
        their career average, accounting for selection bias.

        Args:
            df: DataFrame with player-season data
            metric: Metric to adjust
            age_col: Column name for age
            player_col: Column name for player ID

        Returns:
            Dictionary mapping age to adjustment factor
        """
        # Calculate each player's career average
        career_avgs = df.groupby(player_col)[metric].mean()

        # Join back to get differences from career average
        df_with_career = df.copy()
        df_with_career['career_avg'] = df_with_career[player_col].map(career_avgs)
        df_with_career['diff_from_career'] = df_with_career[metric] - df_with_career['career_avg']

        # Calculate average difference by age
        age_effects = df_with_career.groupby(age_col)['diff_from_career'].mean()

        # Normalize so peak age has 0 adjustment
        age_effects = age_effects - age_effects.get(self.peak_age, 0)

        return age_effects.to_dict()

    def predict(
        self,
        ages: np.ndarray,
        metric: str,
        method: str = 'polynomial'
    ) -> np.ndarray:
        """
        Predict metric values for given ages.

        Args:
            ages: Array of ages
            metric: Metric to predict
            method: 'polynomial' or 'empirical'

        Returns:
            Predicted values
        """
        if method == 'polynomial':
            if metric not in self.models:
                raise ValueError(f"No model fitted for {metric}")

            ages_2d = np.array(ages).reshape(-1, 1)
            return self.models[metric].predict(ages_2d)

        elif method == 'empirical':
            if metric not in self.age_adjustments:
                raise ValueError(f"No empirical adjustments for {metric}")

            adjustments = self.age_adjustments[metric]

            # Get adjustments for each age (use nearest if not available)
            result = []
            available_ages = sorted(adjustments.keys())

            for age in ages:
                if age in adjustments:
                    result.append(adjustments[age])
                else:
                    # Use nearest age
                    nearest_age = min(available_ages, key=lambda x: abs(x - age))
                    result.append(adjustments[nearest_age])

            return np.array(result)

        else:
            raise ValueError(f"Unknown method: {method}")

    def get_age_adjustment_factor(
        self,
        age: int,
        metric: str,
        method: str = 'empirical'
    ) -> float:
        """
        Get adjustment factor for a specific age.

        Returns the expected difference from peak performance.

        Args:
            age: Player age
            metric: Metric name
            method: 'polynomial' or 'empirical'

        Returns:
            Adjustment factor (0 = peak, negative = below peak)
        """
        if method == 'empirical':
            if metric not in self.age_adjustments:
                return 0.0

            adjustments = self.age_adjustments[metric]
            return adjustments.get(age, 0.0)

        elif method == 'polynomial':
            if metric not in self.models:
                return 0.0

            peak_value = self.predict(np.array([self.peak_age]), metric, 'polynomial')[0]
            age_value = self.predict(np.array([age]), metric, 'polynomial')[0]

            return age_value - peak_value

        return 0.0

    def apply_age_adjustment(
        self,
        df: pd.DataFrame,
        metrics: list,
        age_col: str = 'age',
        method: str = 'empirical'
    ) -> pd.DataFrame:
        """
        Apply age adjustments to metrics in a DataFrame.

        Args:
            df: DataFrame with player data
            metrics: List of metrics to adjust
            age_col: Column name for age
            method: 'polynomial' or 'empirical'

        Returns:
            DataFrame with age-adjusted columns added
        """
        df = df.copy()

        for metric in metrics:
            if metric not in df.columns:
                continue

            adjustments = []
            for age in df[age_col]:
                adj = self.get_age_adjustment_factor(age, metric, method)
                adjustments.append(adj)

            df[f'{metric}_age_adj'] = df[metric] - adjustments

        return df

    def plot_age_curve(
        self,
        metric: str,
        age_range: Tuple[int, int] = (22, 40),
        method: str = 'empirical'
    ):
        """
        Plot the age curve for a metric.

        Args:
            metric: Metric to plot
            age_range: Tuple of (min_age, max_age)
            method: 'polynomial' or 'empirical'

        Returns:
            matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib required for plotting")
            return None

        ages = np.arange(age_range[0], age_range[1] + 1)
        values = self.predict(ages, metric, method)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(ages, values, linewidth=2, marker='o')
        ax.axvline(self.peak_age, color='r', linestyle='--', alpha=0.5, label='Peak Age')
        ax.set_xlabel('Age')
        ax.set_ylabel(metric)
        ax.set_title(f'Age Curve: {metric} ({method})')
        ax.grid(True, alpha=0.3)
        ax.legend()

        return fig


class SimpleAgeCurve:
    """
    Simplified age curve using predefined adjustment factors.

    Useful when there isn't enough data to fit a full model.
    Based on research showing typical QB aging patterns.
    """

    def __init__(self):
        """Initialize with typical QB age curve."""
        # Adjustment factors relative to peak (age 28)
        # Based on research: peak ~27-29, decline starts ~31
        self.base_curve = {
            22: -0.10,  # 10% below peak
            23: -0.07,
            24: -0.05,
            25: -0.02,
            26: -0.01,
            27: 0.00,   # Peak
            28: 0.00,   # Peak
            29: 0.00,   # Peak
            30: -0.01,
            31: -0.02,
            32: -0.04,
            33: -0.06,
            34: -0.08,
            35: -0.10,
            36: -0.13,
            37: -0.16,
            38: -0.20,
            39: -0.25,
            40: -0.30,
        }

    def get_adjustment_factor(self, age: int) -> float:
        """
        Get multiplicative adjustment factor for age.

        Args:
            age: Player age

        Returns:
            Multiplier (1.0 = peak, 0.9 = 10% below peak)
        """
        if age < 22:
            age = 22
        elif age > 40:
            age = 40

        return 1.0 + self.base_curve.get(age, 0.0)

    def adjust_metric(self, value: float, age: int) -> float:
        """
        Adjust a metric value for age.

        Args:
            value: Metric value
            age: Player age

        Returns:
            Age-adjusted value
        """
        factor = self.get_adjustment_factor(age)
        return value / factor  # Divide to get peak-equivalent performance


def main():
    """Example usage of age curve models."""
    # Create sample data
    np.random.seed(42)

    players = [f'player_{i}' for i in range(50)]
    seasons = []

    for player in players:
        # Each player has 3-8 seasons
        n_seasons = np.random.randint(3, 9)
        start_age = np.random.randint(23, 28)

        for i in range(n_seasons):
            age = start_age + i

            # Simulate performance with age curve
            peak_performance = np.random.randn() * 0.05 + 0.15
            age_factor = 1.0 - 0.02 * abs(age - 28)  # Peak at 28

            seasons.append({
                'passer_player_id': player,
                'age': age,
                'season': 2015 + i,
                'epa_per_play': peak_performance * age_factor + np.random.randn() * 0.03,
                'attempts': np.random.randint(200, 500)
            })

    df = pd.DataFrame(seasons)

    # Fit age curve model
    model = AgeCurveModel(degree=3, peak_age=28)
    model.fit(df, metrics=['epa_per_play'])

    # Show predictions
    print("Age curve predictions (EPA per play):")
    ages = np.arange(23, 38)
    for method in ['polynomial', 'empirical']:
        print(f"\n{method.capitalize()} method:")
        predictions = model.predict(ages, 'epa_per_play', method=method)
        for age, pred in zip(ages, predictions):
            print(f"  Age {age}: {pred:.4f}")

    # Simple age curve
    simple = SimpleAgeCurve()
    print("\nSimple age curve adjustment factors:")
    for age in range(24, 37, 2):
        factor = simple.get_adjustment_factor(age)
        print(f"  Age {age}: {factor:.3f}")


if __name__ == "__main__":
    main()
