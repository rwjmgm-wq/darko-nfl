"""
Context and Weather Adjustments for QB Performance

Uses regression modeling to quantify the impact of:
- Weather conditions (temperature, wind, precipitation, dome/outdoor)
- Game script (score differential, time remaining)
- Home/away advantage
- Situational factors (down, distance, field position)

Adjusts QB stats to neutral context for fair comparisons.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContextAdjustmentModel:
    """
    Regression-based model for context adjustments.

    Quantifies how weather, game script, and situation affect QB performance.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        use_ridge: bool = True
    ):
        """
        Initialize context adjustment model.

        Args:
            alpha: Regularization strength for Ridge regression
            fit_intercept: Whether to fit intercept
            use_ridge: Use Ridge regression (True) or OLS (False)
        """
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.use_ridge = use_ridge

        # Store models for each metric
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.coefficients = {}

    def _extract_context_features(
        self,
        pbp_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Extract context features from play-by-play data.

        Args:
            pbp_data: Play-by-play DataFrame

        Returns:
            Tuple of (features DataFrame, feature names)
        """
        features = pd.DataFrame(index=pbp_data.index)

        # Weather features
        if 'temp' in pbp_data.columns:
            temp = pd.to_numeric(pbp_data['temp'], errors='coerce')
            features['temp'] = temp.fillna(70)  # Neutral temperature
            features['temp_extreme_cold'] = (features['temp'] < 32).astype(float)
            features['temp_cold'] = ((features['temp'] >= 32) & (features['temp'] < 50)).astype(float)
            features['temp_hot'] = (features['temp'] > 85).astype(float)
        else:
            features['temp'] = 70
            features['temp_extreme_cold'] = 0
            features['temp_cold'] = 0
            features['temp_hot'] = 0

        if 'wind' in pbp_data.columns:
            wind = pd.to_numeric(pbp_data['wind'], errors='coerce')
            features['wind'] = wind.fillna(0)
            features['wind_high'] = (features['wind'] > 15).astype(float)
            features['wind_moderate'] = ((features['wind'] > 10) & (features['wind'] <= 15)).astype(float)
        else:
            features['wind'] = 0
            features['wind_high'] = 0
            features['wind_moderate'] = 0

        # Roof type (dome = ideal conditions)
        if 'roof' in pbp_data.columns:
            features['dome'] = (pbp_data['roof'].fillna('outdoors') == 'dome').astype(float)
            features['retractable'] = (pbp_data['roof'].fillna('outdoors') == 'retractable').astype(float)
            features['outdoors'] = (pbp_data['roof'].fillna('outdoors') == 'outdoors').astype(float)
        else:
            features['dome'] = 0
            features['retractable'] = 0
            features['outdoors'] = 1

        # Precipitation (if available in weather description)
        if 'weather' in pbp_data.columns:
            weather_str = pbp_data['weather'].fillna('').str.lower()
            features['precipitation'] = (
                weather_str.str.contains('rain') |
                weather_str.str.contains('snow') |
                weather_str.str.contains('sleet')
            ).astype(float)
        else:
            features['precipitation'] = 0

        # Game script features
        if 'score_differential' in pbp_data.columns:
            score_diff = pbp_data['score_differential'].fillna(0)
            features['score_diff'] = score_diff
            features['leading_big'] = (score_diff > 14).astype(float)
            features['trailing_big'] = (score_diff < -14).astype(float)
            features['close_game'] = (abs(score_diff) <= 7).astype(float)
        else:
            features['score_diff'] = 0
            features['leading_big'] = 0
            features['trailing_big'] = 0
            features['close_game'] = 1

        # Time features
        if 'qtr' in pbp_data.columns:
            features['qtr'] = pbp_data['qtr'].fillna(2)
            features['qtr_4'] = (features['qtr'] == 4).astype(float)
            features['qtr_1'] = (features['qtr'] == 1).astype(float)
        else:
            features['qtr'] = 2
            features['qtr_4'] = 0
            features['qtr_1'] = 0

        if 'half_seconds_remaining' in pbp_data.columns:
            features['seconds_remaining'] = pbp_data['half_seconds_remaining'].fillna(900)
            features['end_of_half'] = (features['seconds_remaining'] < 120).astype(float)
        else:
            features['seconds_remaining'] = 900
            features['end_of_half'] = 0

        # Home/away
        if 'posteam' in pbp_data.columns and 'home_team' in pbp_data.columns:
            features['home'] = (pbp_data['posteam'] == pbp_data['home_team']).astype(float)
        else:
            features['home'] = 0.5  # Neutral

        # Down and distance
        if 'down' in pbp_data.columns:
            features['down_1'] = (pbp_data['down'] == 1).astype(float)
            features['down_2'] = (pbp_data['down'] == 2).astype(float)
            features['down_3'] = (pbp_data['down'] == 3).astype(float)
            features['down_4'] = (pbp_data['down'] == 4).astype(float)
        else:
            features['down_1'] = 0.25
            features['down_2'] = 0.25
            features['down_3'] = 0.25
            features['down_4'] = 0.25

        if 'ydstogo' in pbp_data.columns:
            features['distance'] = pbp_data['ydstogo'].fillna(10)
            features['long_distance'] = (features['distance'] > 10).astype(float)
            features['short_distance'] = (features['distance'] <= 3).astype(float)
        else:
            features['distance'] = 10
            features['long_distance'] = 0
            features['short_distance'] = 0

        # Field position
        if 'yardline_100' in pbp_data.columns:
            features['field_position'] = pbp_data['yardline_100'].fillna(75)
            features['red_zone'] = (features['field_position'] <= 20).astype(float)
            features['own_territory'] = (features['field_position'] >= 80).astype(float)
        else:
            features['field_position'] = 75
            features['red_zone'] = 0
            features['own_territory'] = 0

        # Altitude (Denver effect)
        if 'home_team' in pbp_data.columns:
            features['altitude'] = pbp_data['home_team'].isin(['DEN']).astype(float)
        else:
            features['altitude'] = 0

        feature_names = features.columns.tolist()

        return features, feature_names

    def fit(
        self,
        pbp_data: pd.DataFrame,
        metrics: List[str] = ['qb_epa', 'cpoe', 'success']
    ):
        """
        Fit context adjustment models for specified metrics.

        Args:
            pbp_data: Play-by-play data with context variables
            metrics: List of metrics to model
        """
        logger.info(f"Fitting context adjustment models for {len(metrics)} metrics...")

        # Filter to passing plays
        pass_plays = pbp_data[pbp_data['play_type'] == 'pass'].copy()

        # Extract context features
        X, feature_names = self._extract_context_features(pass_plays)
        self.feature_names = feature_names

        # Handle missing values
        X = X.fillna(0)

        # Fit model for each metric
        for metric in metrics:
            if metric not in pass_plays.columns:
                logger.warning(f"Metric {metric} not found in data, skipping")
                continue

            # Get target variable
            y = pass_plays[metric].values

            # Remove rows with missing target
            valid_mask = ~np.isnan(y)
            X_valid = X[valid_mask]
            y_valid = y[valid_mask]

            if len(y_valid) < 100:
                logger.warning(f"Insufficient data for {metric}: {len(y_valid)} observations")
                continue

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_valid)

            # Fit model
            if self.use_ridge:
                model = Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept)
            else:
                model = LinearRegression(fit_intercept=self.fit_intercept)

            model.fit(X_scaled, y_valid)

            # Store model and scaler
            self.models[metric] = model
            self.scalers[metric] = scaler

            # Store coefficients for interpretation
            self.coefficients[metric] = pd.DataFrame({
                'feature': feature_names,
                'coefficient': model.coef_
            }).sort_values('coefficient', key=abs, ascending=False)

            logger.info(f"  ✓ Fitted {metric} model (R² = {model.score(X_scaled, y_valid):.3f})")

            # Show top positive and negative impacts
            top_positive = self.coefficients[metric].head(3)
            top_negative = self.coefficients[metric].tail(3)

            logger.info(f"    Top positive impacts: {', '.join(top_positive['feature'].values)}")
            logger.info(f"    Top negative impacts: {', '.join(top_negative['feature'].values)}")

    def predict_context_impact(
        self,
        pbp_data: pd.DataFrame,
        metric: str
    ) -> np.ndarray:
        """
        Predict the impact of context on a metric.

        Args:
            pbp_data: Play-by-play data
            metric: Metric to predict impact for

        Returns:
            Array of predicted context impacts
        """
        if metric not in self.models:
            raise ValueError(f"Model for {metric} has not been fitted")

        # Extract features
        X, _ = self._extract_context_features(pbp_data)
        X = X[self.feature_names].fillna(0)

        # Scale features
        X_scaled = self.scalers[metric].transform(X)

        # Predict impact
        impact = self.models[metric].predict(X_scaled)

        return impact

    def adjust_game_stats(
        self,
        game_stats: pd.DataFrame,
        pbp_data: pd.DataFrame,
        metrics_map: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Adjust game-level stats for context effects.

        Args:
            game_stats: Game-level QB statistics
            pbp_data: Play-by-play data for context
            metrics_map: Mapping of {pbp_metric: game_stat_column}

        Returns:
            DataFrame with context-adjusted columns added
        """
        adjusted = game_stats.copy()

        # Group play-by-play by game and QB
        pbp_games = pbp_data.groupby(['game_id', 'passer_player_id'], group_keys=False).apply(
            lambda x: self._calculate_avg_context_impact(x, metrics_map),
            include_groups=False
        ).reset_index()

        # Merge with game stats
        adjusted = adjusted.merge(
            pbp_games,
            on=['game_id', 'passer_player_id'],
            how='left'
        )

        # Create context-adjusted columns
        for pbp_metric, game_col in metrics_map.items():
            impact_col = f'{pbp_metric}_context_impact'
            if impact_col in adjusted.columns and game_col in adjusted.columns:
                # Adjusted = Raw - Context Impact
                # (remove the context effect to get "neutral" performance)
                adjusted[f'{game_col}_ctx_adj'] = adjusted[game_col] - adjusted[impact_col]

        return adjusted

    def _calculate_avg_context_impact(
        self,
        game_plays: pd.DataFrame,
        metrics_map: Dict[str, str]
    ) -> pd.Series:
        """Calculate average context impact for a single game."""
        result = {}

        for pbp_metric in metrics_map.keys():
            if pbp_metric in self.models:
                try:
                    impact = self.predict_context_impact(game_plays, pbp_metric)
                    result[f'{pbp_metric}_context_impact'] = np.mean(impact)
                except Exception as e:
                    logger.warning(f"Error calculating context impact for {pbp_metric}: {e}")
                    result[f'{pbp_metric}_context_impact'] = 0.0
            else:
                result[f'{pbp_metric}_context_impact'] = 0.0

        return pd.Series(result)

    def get_feature_importance(self, metric: str, top_n: int = 10) -> pd.DataFrame:
        """
        Get feature importance for a specific metric.

        Args:
            metric: Metric to analyze
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importances
        """
        if metric not in self.coefficients:
            raise ValueError(f"No coefficients available for {metric}")

        return self.coefficients[metric].head(top_n)

    def get_neutral_baseline(self) -> Dict[str, float]:
        """
        Get baseline context values representing neutral conditions.

        Returns:
            Dictionary of neutral feature values
        """
        return {
            'temp': 70,
            'wind': 5,
            'dome': 0,
            'precipitation': 0,
            'score_diff': 0,
            'close_game': 1,
            'home': 0.5,
            'qtr': 2,
            'seconds_remaining': 900,
            'down_2': 1,  # 2nd down is most "neutral"
            'distance': 10,
            'field_position': 75,
            'red_zone': 0,
            'altitude': 0
        }


def main():
    """Example usage of context adjustment system."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from data_pipeline.nfl_data_fetcher import NFLDataFetcher
    from data_pipeline.qb_stats_aggregator import QBStatsAggregator

    # Fetch data - use multiple years for better context modeling
    fetcher = NFLDataFetcher()
    pbp = fetcher.get_qb_play_by_play([2022, 2023, 2024])

    # Fit context model
    context_model = ContextAdjustmentModel(alpha=1.0, use_ridge=True)
    context_model.fit(pbp, metrics=['qb_epa', 'cpoe', 'success'])

    # Show feature importance
    print("\n" + "="*60)
    print("Top Context Effects on EPA")
    print("="*60)
    print(context_model.get_feature_importance('qb_epa', top_n=10).to_string(index=False))

    print("\n" + "="*60)
    print("Top Context Effects on CPOE")
    print("="*60)
    print(context_model.get_feature_importance('cpoe', top_n=10).to_string(index=False))

    # Apply to game stats
    aggregator = QBStatsAggregator()
    game_stats = aggregator.aggregate_game_stats(pbp)

    metrics_map = {
        'qb_epa': 'epa_per_play',
        'cpoe': 'cpoe_mean',
        'success': 'success_mean'
    }

    adjusted_stats = context_model.adjust_game_stats(game_stats, pbp, metrics_map)

    # Compare raw vs context-adjusted
    print("\n" + "="*60)
    print("Example: Context Adjustments")
    print("="*60)
    sample = adjusted_stats[['passer_player_name', 'game_id',
                             'epa_per_play', 'epa_per_play_ctx_adj',
                             'qb_epa_context_impact']].head(10)
    print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
