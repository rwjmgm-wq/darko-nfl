"""
Teammate Adjustment Validation - Proof of Concept

Tests whether adjusting QB performance for receiver quality improves
prediction of future performance.

Key Question:
  Does QB_EPA adjusted for WR quality predict future QB performance
  better than raw QB_EPA?

Methodology:
  - Train on 2020-2022, test on 2023
  - Calculate receiver corps quality for each team-game
  - Adjust QB EPA for receiver quality using ridge regression
  - Compare predictive power: Raw vs Adjusted

Usage:
    python validate_teammate_adjustments.py --train-years 2020 2021 2022 --test-years 2023
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
from sklearn.linear_model import Ridge
from typing import Dict, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_pipeline.nfl_data_fetcher import NFLDataFetcher
from data_pipeline.qb_stats_aggregator import QBStatsAggregator
from data_pipeline.receiver_stats_aggregator import ReceiverStatsAggregator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TeammateAdjustmentValidator:
    """
    Validates whether teammate adjustments improve predictive power.
    """

    def __init__(self, alpha: float = 1.0):
        """
        Initialize validator.

        Args:
            alpha: Ridge regression regularization strength
        """
        self.alpha = alpha
        self.model = None

    def prepare_qb_with_receiver_context(
        self,
        qb_games: pd.DataFrame,
        receiver_games: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge QB game stats with receiver corps quality.

        Args:
            qb_games: QB game-level stats
            receiver_games: Receiver game-level stats

        Returns:
            DataFrame with QB stats + receiver context
        """
        logger.info("Merging QB stats with receiver quality...")

        # Calculate team-level receiver quality for each game
        rec_agg = ReceiverStatsAggregator()
        team_receiver_quality = rec_agg.calculate_team_receiver_quality(receiver_games)

        # Merge with QB games
        qb_with_context = qb_games.merge(
            team_receiver_quality[['game_id', 'posteam', 'receiver_corps_epa',
                                   'receiver_corps_catch_rate', 'receiver_corps_yards_per_target']],
            on=['game_id', 'posteam'],
            how='left'
        )

        # Fill missing receiver data (e.g., games with no recorded targets)
        qb_with_context['receiver_corps_epa'] = qb_with_context['receiver_corps_epa'].fillna(0)
        qb_with_context['receiver_corps_catch_rate'] = qb_with_context['receiver_corps_catch_rate'].fillna(0.6)
        qb_with_context['receiver_corps_yards_per_target'] = qb_with_context['receiver_corps_yards_per_target'].fillna(6.5)

        logger.info(f"  Merged {len(qb_with_context)} QB-games with receiver context")

        return qb_with_context

    def adjust_qb_for_receivers(
        self,
        train_games: pd.DataFrame,
        test_games: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Adjust QB EPA for receiver quality using ridge regression.

        Args:
            train_games: Training QB games (with receiver context)
            test_games: Test QB games (with receiver context)

        Returns:
            Tuple of (train_adjusted, test_adjusted)
        """
        logger.info("Fitting receiver adjustment model...")

        # Features: receiver quality
        X_train = train_games[['receiver_corps_epa', 'receiver_corps_catch_rate',
                                'receiver_corps_yards_per_target']].values
        y_train = train_games['epa_per_play'].values

        # Fit ridge regression
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X_train, y_train)

        # Predict expected EPA given receiver quality
        train_expected = self.model.predict(X_train)
        X_test = test_games[['receiver_corps_epa', 'receiver_corps_catch_rate',
                              'receiver_corps_yards_per_target']].values
        test_expected = self.model.predict(X_test)

        # Residual = QB performance over expected given receivers
        train_adjusted = train_games.copy()
        train_adjusted['epa_receiver_adj'] = y_train - train_expected

        test_adjusted = test_games.copy()
        test_adjusted['epa_receiver_adj'] = test_games['epa_per_play'] - test_expected

        logger.info(f"  Receiver adjustment coefficients: {self.model.coef_}")

        return train_adjusted, test_adjusted

    def aggregate_to_season(
        self,
        game_stats: pd.DataFrame,
        min_attempts: int = 150
    ) -> pd.DataFrame:
        """
        Aggregate game stats to player-season level.

        Args:
            game_stats: Game-level statistics
            min_attempts: Minimum attempts to include

        Returns:
            Season-level aggregated stats
        """
        season_stats = game_stats.groupby(
            ['passer_player_id', 'season']
        ).agg({
            'passer_player_name': 'first',
            'attempts': 'sum',
            'epa_per_play': 'mean',  # Raw EPA
            'epa_receiver_adj': 'mean' if 'epa_receiver_adj' in game_stats.columns else lambda x: np.nan
        }).reset_index()

        season_stats = season_stats[season_stats['attempts'] >= min_attempts]

        return season_stats

    def test_predictive_power(
        self,
        train_seasons: pd.DataFrame,
        test_seasons: pd.DataFrame,
        metric_name: str
    ) -> Dict[str, float]:
        """
        Test how well a metric predicts future performance.

        Args:
            train_seasons: Training season data
            test_seasons: Test season data
            metric_name: Name of metric to test

        Returns:
            Dictionary with correlation, RMSE, MAE
        """
        # Merge train and test on player_id
        comparison = train_seasons[['passer_player_id', metric_name]].merge(
            test_seasons[['passer_player_id', 'epa_per_play']],
            on='passer_player_id',
            how='inner',
            suffixes=('_train', '_test')
        )

        if len(comparison) < 10:
            return {
                'metric': metric_name,
                'correlation': np.nan,
                'rmse': np.nan,
                'mae': np.nan,
                'n_qbs': len(comparison)
            }

        # Handle column names after merge (pandas adds suffixes)
        pred_col = f'{metric_name}_train' if f'{metric_name}_train' in comparison.columns else metric_name
        if 'epa_per_play_test' in comparison.columns:
            actual_col = 'epa_per_play_test'
        else:
            actual_col = 'epa_per_play'

        pred = comparison[pred_col]
        actual = comparison[actual_col]

        # Remove NaN
        valid = ~(pred.isna() | actual.isna())
        pred = pred[valid]
        actual = actual[valid]

        if len(pred) < 10:
            return {
                'metric': metric_name,
                'correlation': np.nan,
                'rmse': np.nan,
                'mae': np.nan,
                'n_qbs': len(pred)
            }

        correlation = pred.corr(actual)
        rmse = np.sqrt(((pred - actual) ** 2).mean())
        mae = (pred - actual).abs().mean()

        return {
            'metric': metric_name,
            'correlation': correlation,
            'rmse': rmse,
            'mae': mae,
            'n_qbs': len(pred)
        }


def main():
    """Run teammate adjustment validation."""
    parser = argparse.ArgumentParser(
        description='Validate teammate adjustments improve predictive power'
    )
    parser.add_argument(
        '--train-years',
        type=int,
        nargs='+',
        default=[2020, 2021, 2022],
        help='Years to use for training'
    )
    parser.add_argument(
        '--test-years',
        type=int,
        nargs='+',
        default=[2023],
        help='Years to use for testing'
    )
    parser.add_argument(
        '--min-attempts',
        type=int,
        default=150,
        help='Minimum attempts to include QB'
    )

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("Teammate Adjustment Validation - QB + Receivers")
    logger.info("="*80)

    # Fetch data
    logger.info(f"\nFetching data for training years: {args.train_years}")
    logger.info(f"Fetching data for test years: {args.test_years}")

    fetcher = NFLDataFetcher()
    all_years = args.train_years + args.test_years
    pbp = fetcher.fetch_pbp_data(all_years)

    # Aggregate QB stats
    logger.info("\nAggregating QB statistics...")
    qb_agg = QBStatsAggregator()
    qb_games = qb_agg.aggregate_game_stats(pbp)

    # Aggregate receiver stats
    logger.info("\nAggregating receiver statistics...")
    rec_agg = ReceiverStatsAggregator()
    rec_games = rec_agg.aggregate_game_stats(pbp)

    # Initialize validator
    validator = TeammateAdjustmentValidator(alpha=1.0)

    # Merge QB with receiver context
    qb_with_receivers = validator.prepare_qb_with_receiver_context(qb_games, rec_games)

    # Split train/test
    train_games = qb_with_receivers[qb_with_receivers['season'].isin(args.train_years)]
    test_games = qb_with_receivers[qb_with_receivers['season'].isin(args.test_years)]

    logger.info(f"\nTrain games: {len(train_games)}")
    logger.info(f"Test games: {len(test_games)}")

    # Adjust QB for receivers
    train_adj, test_adj = validator.adjust_qb_for_receivers(train_games, test_games)

    # Aggregate to season level
    train_seasons = validator.aggregate_to_season(train_adj, min_attempts=args.min_attempts)
    test_seasons = validator.aggregate_to_season(test_adj, min_attempts=args.min_attempts)

    logger.info(f"\nTrain QB-seasons: {len(train_seasons)}")
    logger.info(f"Test QB-seasons: {len(test_seasons)}")

    # Test predictive power
    logger.info("\n" + "="*80)
    logger.info("Testing Predictive Power")
    logger.info("="*80)

    # Baseline: Raw EPA
    baseline = validator.test_predictive_power(train_seasons, test_seasons, 'epa_per_play')

    # Adjusted: Receiver-adjusted EPA
    adjusted = validator.test_predictive_power(train_seasons, test_seasons, 'epa_receiver_adj')

    # Results
    print(f"\n{'='*80}")
    print("Predictive Power Comparison")
    print(f"{'='*80}")

    print(f"\nBaseline (Raw QB EPA):")
    print(f"  Correlation with future EPA: {baseline['correlation']:.4f}")
    print(f"  RMSE: {baseline['rmse']:.4f}")
    print(f"  MAE: {baseline['mae']:.4f}")
    print(f"  Sample size: {baseline['n_qbs']} QBs")

    print(f"\nReceiver-Adjusted QB EPA:")
    print(f"  Correlation with future EPA: {adjusted['correlation']:.4f}")
    print(f"  RMSE: {adjusted['rmse']:.4f}")
    print(f"  MAE: {adjusted['mae']:.4f}")
    print(f"  Sample size: {adjusted['n_qbs']} QBs")

    improvement = adjusted['correlation'] - baseline['correlation']
    pct_improvement = (improvement / baseline['correlation']) * 100

    print(f"\nImprovement:")
    print(f"  Absolute: {improvement:+.4f}")
    print(f"  Relative: {pct_improvement:+.2f}%")

    # Decision
    print(f"\n{'='*80}")
    if improvement > 0.01:  # 1% threshold
        print("[+] DECISION: Receiver adjustments IMPROVE prediction")
        print(f"  Improvement of {improvement:.4f} exceeds 0.01 threshold")
        print("  Recommendation: Proceed with full teammate adjustment infrastructure")
    else:
        print("[-] DECISION: Receiver adjustments DO NOT improve prediction")
        print(f"  Improvement of {improvement:.4f} does not meet 0.01 threshold")
        print("  Recommendation: Do not invest in full teammate adjustment system")
        if improvement < -0.005:
            print(f"  WARNING: Adjustments actually HURT prediction by {abs(improvement):.4f}")
            print("  Possible reasons:")
            print("    1. Receiver quality already captured in QB EPA")
            print("    2. Regression removing signal along with noise")
            print("    3. Need opponent+Kalman adjustments first")

    print("="*80)

    # Save results
    results = pd.DataFrame([baseline, adjusted])
    results.to_csv('results/teammate_adjustment_validation.csv', index=False)
    logger.info("\nSaved results to results/teammate_adjustment_validation.csv")

    return 0


if __name__ == "__main__":
    exit(main())
