"""
Context Adjustments for WR Performance

Uses regression modeling to quantify the impact of:
- Weather conditions (temperature, wind, precipitation, dome/outdoor)
- Target depth (deep vs short)
- Game script (score differential, time remaining)
- Field position (red zone, midfield)
- Situational factors (down, distance)

Adjusts WR stats to neutral context for fair comparisons.
Designed for WR LEAF v1.0 system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WRContextAdjustmentModel:
    """
    Regression-based model for WR context adjustments.

    Quantifies how weather, target depth, game script, and situation affect WR performance.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        use_ridge: bool = True
    ):
        """
        Initialize WR context adjustment model.

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

    def _extract_wr_context_features(
        self,
        pbp_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Extract WR-specific context features from play-by-play data.

        Args:
            pbp_data: Play-by-play DataFrame with receiver info

        Returns:
            Tuple of (features DataFrame, feature names)
        """
        features = pd.DataFrame(index=pbp_data.index)

        # ==========================================
        # Weather Features (affects catching ability)
        # ==========================================

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
            features['wind_high'] = (features['wind'] > 15).astype(float)  # Affects deep balls
            features['wind_moderate'] = ((features['wind'] > 10) & (features['wind'] <= 15)).astype(float)
        else:
            features['wind'] = 0
            features['wind_high'] = 0
            features['wind_moderate'] = 0

        # Roof type (dome = ideal catching conditions)
        if 'roof' in pbp_data.columns:
            features['dome'] = (pbp_data['roof'].fillna('outdoors') == 'dome').astype(float)
            features['retractable'] = (pbp_data['roof'].fillna('outdoors') == 'retractable').astype(float)
            features['outdoors'] = (pbp_data['roof'].fillna('outdoors') == 'outdoors').astype(float)
        else:
            features['dome'] = 0
            features['retractable'] = 0
            features['outdoors'] = 1

        # Precipitation (harder to catch)
        if 'weather' in pbp_data.columns:
            weather_str = pbp_data['weather'].fillna('').str.lower()
            features['precipitation'] = (
                weather_str.str.contains('rain') |
                weather_str.str.contains('snow') |
                weather_str.str.contains('sleet')
            ).astype(float)
        else:
            features['precipitation'] = 0

        # ==========================================
        # Target Depth Features (WR-specific!)
        # ==========================================

        if 'air_yards' in pbp_data.columns:
            air_yards = pbp_data['air_yards'].fillna(0)
            features['air_yards'] = air_yards
            features['deep_target'] = (air_yards > 15).astype(float)  # Deep ball
            features['medium_target'] = ((air_yards >= 5) & (air_yards <= 15)).astype(float)
            features['short_target'] = (air_yards < 5).astype(float)  # Screen/short
            features['behind_los'] = (air_yards < 0).astype(float)  # Behind line of scrimmage
        else:
            features['air_yards'] = 0
            features['deep_target'] = 0
            features['medium_target'] = 0.5
            features['short_target'] = 0.5
            features['behind_los'] = 0

        # ==========================================
        # Game Script Features
        # ==========================================

        if 'score_differential' in pbp_data.columns:
            score_diff = pbp_data['score_differential'].fillna(0)
            features['score_diff'] = score_diff
            features['leading_big'] = (score_diff > 14).astype(float)  # Prevent defense
            features['trailing_big'] = (score_diff < -14).astype(float)  # Catch-up mode
            features['close_game'] = (abs(score_diff) <= 7).astype(float)
        else:
            features['score_diff'] = 0
            features['leading_big'] = 0
            features['trailing_big'] = 0
            features['close_game'] = 1

        # Time features
        if 'qtr' in pbp_data.columns:
            features['qtr'] = pbp_data['qtr'].fillna(2)
            features['qtr_4'] = (features['qtr'] == 4).astype(float)  # Crunch time
            features['qtr_1'] = (features['qtr'] == 1).astype(float)
        else:
            features['qtr'] = 2
            features['qtr_4'] = 0
            features['qtr_1'] = 0

        if 'half_seconds_remaining' in pbp_data.columns:
            features['seconds_remaining'] = pbp_data['half_seconds_remaining'].fillna(900)
            features['end_of_half'] = (features['seconds_remaining'] < 120).astype(float)  # 2-minute drill
        else:
            features['seconds_remaining'] = 900
            features['end_of_half'] = 0

        # Home/away
        if 'posteam' in pbp_data.columns and 'home_team' in pbp_data.columns:
            features['home'] = (pbp_data['posteam'] == pbp_data['home_team']).astype(float)
        else:
            features['home'] = 0.5  # Neutral

        # ==========================================
        # Down and Distance (affects target selection)
        # ==========================================

        if 'down' in pbp_data.columns:
            features['down_1'] = (pbp_data['down'] == 1).astype(float)
            features['down_2'] = (pbp_data['down'] == 2).astype(float)
            features['down_3'] = (pbp_data['down'] == 3).astype(float)  # Critical
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

        # ==========================================
        # Field Position (affects route concepts)
        # ==========================================

        if 'yardline_100' in pbp_data.columns:
            features['field_position'] = pbp_data['yardline_100'].fillna(75)
            features['red_zone'] = (features['field_position'] <= 20).astype(float)  # Red zone
            features['own_territory'] = (features['field_position'] >= 80).astype(float)
            features['midfield'] = ((features['field_position'] > 40) &
                                   (features['field_position'] < 60)).astype(float)
        else:
            features['field_position'] = 75
            features['red_zone'] = 0
            features['own_territory'] = 0
            features['midfield'] = 0.2

        # ==========================================
        # Special Situations
        # ==========================================

        # Pass location (left, middle, right) - if available
        if 'pass_location' in pbp_data.columns:
            features['pass_left'] = (pbp_data['pass_location'] == 'left').astype(float)
            features['pass_middle'] = (pbp_data['pass_location'] == 'middle').astype(float)
            features['pass_right'] = (pbp_data['pass_location'] == 'right').astype(float)
        else:
            features['pass_left'] = 0.33
            features['pass_middle'] = 0.33
            features['pass_right'] = 0.33

        feature_names = features.columns.tolist()

        return features, feature_names

    def fit(
        self,
        pbp_data: pd.DataFrame,
        metrics: List[str] = ['epa', 'air_epa', 'yac_epa']
    ):
        """
        Fit WR context adjustment models for specified metrics.

        Args:
            pbp_data: Play-by-play data with receiver and context variables
            metrics: List of metrics to model (default: WR LEAF metrics)
        """
        logger.info(f"Fitting WR context adjustment models for {len(metrics)} metrics...")

        # Filter to pass attempts with receiver info
        receiver_plays = pbp_data[
            (pbp_data['pass_attempt'] == 1) &
            (pbp_data['receiver_player_id'].notna())
        ].copy()

        logger.info(f"  Training on {len(receiver_plays):,} receiver targets")

        # Extract context features
        X, feature_names = self._extract_wr_context_features(receiver_plays)
        self.feature_names = feature_names

        # Handle missing values
        X = X.fillna(0)

        # Fit model for each metric
        for metric in metrics:
            if metric not in receiver_plays.columns:
                logger.warning(f"  Metric {metric} not found in data, skipping")
                continue

            # Get target variable
            y = receiver_plays[metric].values

            # Remove rows with missing target
            valid_mask = ~np.isnan(y)
            X_valid = X[valid_mask]
            y_valid = y[valid_mask]

            if len(y_valid) < 100:
                logger.warning(f"  Insufficient data for {metric}: {len(y_valid)} observations")
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
            self.coefficients[metric] = dict(zip(feature_names, model.coef_))

            # Calculate R² for validation
            r2 = model.score(X_scaled, y_valid)
            logger.info(f"  ✓ Fitted {metric} model (R² = {r2:.3f})")

            # Show top positive/negative impacts
            coef_df = pd.DataFrame({
                'feature': feature_names,
                'coefficient': model.coef_
            }).sort_values('coefficient', ascending=False)

            top_positive = coef_df.head(3)['feature'].tolist()
            top_negative = coef_df.tail(3)['feature'].tolist()

            logger.info(f"    Top positive impacts: {', '.join(top_positive)}")
            logger.info(f"    Top negative impacts: {', '.join(top_negative)}")

        logger.info(f"  Context model fitted!")

    def adjust(
        self,
        pbp_data: pd.DataFrame,
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Apply context adjustments to WR metrics.

        Args:
            pbp_data: Play-by-play data with receiver targets
            metrics: List of metrics to adjust (if None, adjust all fitted metrics)

        Returns:
            DataFrame with adjusted metrics (adds '_ctx_adj' suffix)
        """
        if not self.models:
            raise ValueError("Models not fitted. Call fit() first.")

        if metrics is None:
            metrics = list(self.models.keys())

        adjusted_data = pbp_data.copy()

        # Extract context features
        X, _ = self._extract_wr_context_features(pbp_data)
        X = X.fillna(0)

        # Apply adjustments for each metric
        for metric in metrics:
            if metric not in self.models:
                logger.warning(f"No model fitted for {metric}, skipping")
                continue

            # Get model and scaler
            model = self.models[metric]
            scaler = self.scalers[metric]

            # Scale features
            X_scaled = scaler.transform(X)

            # Predict context impact
            context_impact = model.predict(X_scaled)

            # Adjust metric (remove context impact)
            # adjusted = actual - predicted_context_effect
            adjusted_col = f"{metric}_ctx_adj"
            adjusted_data[adjusted_col] = pbp_data[metric] - context_impact

            # Also store the context impact itself
            impact_col = f"{metric}_context_impact"
            adjusted_data[impact_col] = context_impact

        return adjusted_data

    def get_feature_importance(
        self,
        metric: str,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Get feature importance for a specific metric.

        Args:
            metric: Metric to analyze
            top_n: Number of top features to return

        Returns:
            DataFrame with features and coefficients
        """
        if metric not in self.coefficients:
            raise ValueError(f"No model fitted for {metric}")

        coef_df = pd.DataFrame({
            'feature': list(self.coefficients[metric].keys()),
            'coefficient': list(self.coefficients[metric].values()),
            'abs_coefficient': [abs(c) for c in self.coefficients[metric].values()]
        }).sort_values('abs_coefficient', ascending=False).head(top_n)

        return coef_df[['feature', 'coefficient']]


def main():
    """Example usage of WRContextAdjustmentModel."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from data_pipeline.nfl_data_fetcher import NFLDataFetcher

    # Fetch data
    logger.info("Fetching nflfastR data...")
    fetcher = NFLDataFetcher()
    pbp = fetcher.fetch_pbp_data([2023, 2024])

    # Filter to receiver targets
    receiver_plays = pbp[
        (pbp['pass_attempt'] == 1) &
        (pbp['receiver_player_id'].notna())
    ].copy()

    logger.info(f"Found {len(receiver_plays):,} receiver targets")

    # Fit context model
    model = WRContextAdjustmentModel(alpha=1.0, use_ridge=True)
    model.fit(receiver_plays, metrics=['epa', 'air_epa', 'yac_epa'])

    # Apply adjustments
    adjusted = model.adjust(receiver_plays, metrics=['air_epa', 'yac_epa'])

    # Display results
    print(f"\n{'='*80}")
    print("WR Context Adjustment Results")
    print(f"{'='*80}")

    for metric in ['air_epa', 'yac_epa']:
        raw_col = metric
        adj_col = f"{metric}_ctx_adj"

        if adj_col in adjusted.columns:
            print(f"\n{metric.upper()}:")
            print(f"  Raw - Mean: {adjusted[raw_col].mean():.4f}, Std: {adjusted[raw_col].std():.4f}")
            print(f"  Adjusted - Mean: {adjusted[adj_col].mean():.4f}, Std: {adjusted[adj_col].std():.4f}")

            # Show correlation
            corr = adjusted[[raw_col, adj_col]].corr().iloc[0, 1]
            print(f"  Correlation: {corr:.4f}")

    # Show feature importance
    print(f"\n{'='*80}")
    print("Top 10 Most Important Features for air_epa")
    print(f"{'='*80}")
    print(model.get_feature_importance('air_epa', top_n=10).to_string(index=False))


if __name__ == "__main__":
    main()
