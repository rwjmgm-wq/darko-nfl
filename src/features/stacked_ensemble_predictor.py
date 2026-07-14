"""
Stacked Ensemble Predictor for QB Performance

Combines Random Forest + XGBoost + Baseline with Ridge meta-learner.
Achieves r=0.3946 correlation with next 16 games (best validated result).

Architecture:
- Level 1: Random Forest, XGBoost, Baseline (multi-feature)
- Level 2: Ridge regression meta-learner
- Features: 30 engineered features from recent 12-game window

Usage:
    predictor = StackedEnsemblePredictor.load('models/stacked_ensemble')
    prediction = predictor.predict(player_games)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Union
import joblib
from scipy import stats

from .multifeature_rating_calculator import MultiFeatureRatingCalculator


class StackedEnsemblePredictor:
    """
    Production-ready stacked ensemble predictor.

    Combines Random Forest, XGBoost, and baseline multi-feature rating
    using a Ridge meta-learner to achieve r=0.3946 predictive accuracy.
    """

    def __init__(
        self,
        rf_model=None,
        xgb_model=None,
        stacker_model=None,
        scaler=None,
        feature_names=None,
        baseline_calculator: Optional[MultiFeatureRatingCalculator] = None
    ):
        """
        Initialize stacked ensemble predictor.

        Args:
            rf_model: Trained RandomForestRegressor
            xgb_model: Trained XGBRegressor
            stacker_model: Trained Ridge meta-learner
            scaler: Fitted StandardScaler for features
            feature_names: List of feature names
            baseline_calculator: MultiFeatureRatingCalculator for baseline predictions
        """
        self.rf_model = rf_model
        self.xgb_model = xgb_model
        self.stacker_model = stacker_model
        self.scaler = scaler
        self.feature_names = feature_names

        # Default baseline calculator if not provided
        self.baseline_calculator = baseline_calculator or MultiFeatureRatingCalculator(
            window_size=12,
            decay_rate=0.10,
            outlier_percentile=95.0,
            volatility_penalty=0.3,
            apply_volatility_adjustment=True
        )

    @classmethod
    def load(cls, model_dir: Union[str, Path]) -> 'StackedEnsemblePredictor':
        """
        Load trained stacked ensemble from directory.

        Args:
            model_dir: Directory containing saved models

        Returns:
            Loaded StackedEnsemblePredictor instance
        """
        model_dir = Path(model_dir)

        rf_model = joblib.load(model_dir / 'random_forest_best.pkl')
        xgb_model = joblib.load(model_dir / 'xgboost_best.pkl')
        stacker_model = joblib.load(model_dir / 'stacker_ridge.pkl')
        scaler = joblib.load(model_dir / 'feature_scaler.pkl')
        feature_names = joblib.load(model_dir / 'feature_names.pkl')

        return cls(
            rf_model=rf_model,
            xgb_model=xgb_model,
            stacker_model=stacker_model,
            scaler=scaler,
            feature_names=feature_names
        )

    def save(self, model_dir: Union[str, Path]):
        """
        Save stacked ensemble to directory.

        Args:
            model_dir: Directory to save models
        """
        model_dir = Path(model_dir)
        model_dir.mkdir(exist_ok=True, parents=True)

        joblib.dump(self.rf_model, model_dir / 'random_forest_best.pkl')
        joblib.dump(self.xgb_model, model_dir / 'xgboost_best.pkl')
        joblib.dump(self.stacker_model, model_dir / 'stacker_ridge.pkl')
        joblib.dump(self.scaler, model_dir / 'feature_scaler.pkl')
        joblib.dump(self.feature_names, model_dir / 'feature_names.pkl')

    def engineer_features(self, player_games: pd.DataFrame, window: int = 12) -> Optional[Dict]:
        """
        Engineer features from player game history.

        Args:
            player_games: DataFrame with player's game-by-game stats
            window: Number of recent games to use (default 12)

        Returns:
            Dictionary of engineered features, or None if insufficient data
        """
        if len(player_games) < 3:
            return None

        recent = player_games.tail(window) if len(player_games) >= window else player_games
        features = {}

        # Base 5 features (avg, std, max, min for each) = 20 features
        base_features = ['epa_mean', 'opp_adj_success_rate', 'epa_per_play',
                         'qb_epa_mean', 'success_rate']

        for feat in base_features:
            if feat in recent.columns:
                features[f'{feat}_avg'] = recent[feat].mean()
                features[f'{feat}_std'] = recent[feat].std()
                features[f'{feat}_max'] = recent[feat].max()
                features[f'{feat}_min'] = recent[feat].min()

        # EPA trend features = 7 features
        if 'opp_adj_base_epa' in recent.columns:
            epa_values = recent['opp_adj_base_epa'].values

            # Linear trend
            if len(epa_values) >= 3:
                x = np.arange(len(epa_values))
                slope, intercept, r_val, p_val, std_err = stats.linregress(x, epa_values)
                features['epa_trend_slope'] = slope
                features['epa_trend_r2'] = r_val ** 2

            # Acceleration (change in trend)
            if len(epa_values) >= 6:
                mid = len(epa_values) // 2
                first_half = epa_values[:mid]
                second_half = epa_values[mid:]

                slope1, _, _, _, _ = stats.linregress(np.arange(len(first_half)), first_half)
                slope2, _, _, _, _ = stats.linregress(np.arange(len(second_half)), second_half)
                features['epa_acceleration'] = slope2 - slope1

            # Recent vs earlier
            if len(epa_values) >= 8:
                recent_4 = epa_values[-4:].mean()
                earlier_8 = epa_values[:-4].mean()
                features['recent_vs_earlier'] = recent_4 - earlier_8

            # Streak
            mean_epa = epa_values.mean()
            streak = 0
            for val in reversed(epa_values):
                if (val > mean_epa and streak >= 0) or (val <= mean_epa and streak <= 0):
                    streak += 1 if val > mean_epa else -1
                else:
                    break
            features['streak_length'] = streak

            # Consistency
            features['epa_consistency_cv'] = epa_values.std() / abs(epa_values.mean()) if abs(epa_values.mean()) > 0.01 else 0

        # Interaction features = 2 features
        if 'epa_mean_avg' in features and 'opp_adj_success_rate_avg' in features:
            features['epa_x_success'] = features['epa_mean_avg'] * features['opp_adj_success_rate_avg']

        if 'epa_mean_avg' in features and 'epa_per_play_avg' in features:
            features['epa_mean_x_per_play'] = features['epa_mean_avg'] * features['epa_per_play_avg']

        # Games played = 2 features
        features['games_played'] = len(player_games)
        features['recent_games'] = len(recent)

        return features

    def predict(
        self,
        player_games: pd.DataFrame,
        return_components: bool = False
    ) -> Union[float, Dict[str, float]]:
        """
        Predict next 16 games average performance for a player.

        Args:
            player_games: DataFrame with player's game-by-game stats (recent 12+ games)
            return_components: If True, return dict with individual model predictions

        Returns:
            Predicted opponent-adjusted EPA for next 16 games
            If return_components=True, returns dict with rf_pred, xgb_pred, baseline_pred, stacked_pred
        """
        # Engineer features
        features = self.engineer_features(player_games, window=12)

        if features is None:
            raise ValueError(f"Insufficient data: need at least 3 games, got {len(player_games)}")

        # Check for missing features
        missing = [f for f in self.feature_names if f not in features]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        # Prepare feature vector
        X = np.array([features[f] for f in self.feature_names]).reshape(1, -1)

        # Handle NaN/inf
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("Features contain NaN or inf values")

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Get base model predictions
        rf_pred = self.rf_model.predict(X_scaled)[0]
        xgb_pred = self.xgb_model.predict(X_scaled)[0]

        # Get baseline prediction
        baseline_pred = self.baseline_calculator.calculate_multifeature_rating(player_games)

        # Stack predictions
        X_stack = np.array([[rf_pred, xgb_pred, baseline_pred]])
        stacked_pred = self.stacker_model.predict(X_stack)[0]

        if return_components:
            return {
                'prediction': stacked_pred,
                'rf_pred': rf_pred,
                'xgb_pred': xgb_pred,
                'baseline_pred': baseline_pred
            }

        return stacked_pred

    def predict_batch(
        self,
        players_data: Dict[str, pd.DataFrame],
        return_components: bool = False
    ) -> pd.DataFrame:
        """
        Predict for multiple players.

        Args:
            players_data: Dict mapping player_id to game DataFrame
            return_components: If True, include individual model predictions

        Returns:
            DataFrame with predictions for all players
        """
        results = []

        for player_id, player_games in players_data.items():
            try:
                pred = self.predict(player_games, return_components=return_components)

                if return_components:
                    result = {
                        'player_id': player_id,
                        'prediction': pred['prediction'],
                        'rf_pred': pred['rf_pred'],
                        'xgb_pred': pred['xgb_pred'],
                        'baseline_pred': pred['baseline_pred']
                    }
                else:
                    result = {
                        'player_id': player_id,
                        'prediction': pred
                    }

                results.append(result)

            except Exception as e:
                print(f"Warning: Failed to predict for player {player_id}: {e}")
                continue

        return pd.DataFrame(results)

    def get_model_info(self) -> Dict:
        """
        Get information about the stacked ensemble.

        Returns:
            Dictionary with model configuration and metadata
        """
        return {
            'architecture': 'Stacked Ensemble (RF + XGBoost + Baseline -> Ridge)',
            'target': 'Next 16 games average opponent-adjusted EPA',
            'window': 12,
            'features': len(self.feature_names),
            'validated_correlation': 0.3946,
            'baseline_correlation': 0.3853,
            'improvement': '+2.4%',
            'rf_params': self.rf_model.get_params() if self.rf_model else None,
            'xgb_params': self.xgb_model.get_params() if self.xgb_model else None,
            'stacker_params': self.stacker_model.get_params() if self.stacker_model else None
        }


def main():
    """Example usage."""
    print("="*80)
    print("STACKED ENSEMBLE PREDICTOR")
    print("="*80)

    # Load model
    try:
        predictor = StackedEnsemblePredictor.load('models/stacked_ensemble')
        print("\n[+] Stacked ensemble loaded successfully!")

        # Show model info
        info = predictor.get_model_info()
        print(f"\nModel Information:")
        print(f"  Architecture: {info['architecture']}")
        print(f"  Target: {info['target']}")
        print(f"  Window: {info['window']} games")
        print(f"  Features: {info['features']}")
        print(f"  Validated r: {info['validated_correlation']:.4f}")
        print(f"  Improvement: {info['improvement']}")

    except FileNotFoundError:
        print("\n[-] Models not found. Run train_stacked_ensemble.py first.")
        print("    This will train and save the stacked ensemble models.")


if __name__ == "__main__":
    main()
