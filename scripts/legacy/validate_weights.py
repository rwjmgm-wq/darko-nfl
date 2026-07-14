"""
DEPRECATED - do not use. Superseded by scripts/v3_honest/.

DATA LEAKAGE: prepare_data() pools train AND test years, runs the Kalman
filter over everything, and merges each player's FINAL estimate onto training
rows by player id - so the "predictor" has already seen the test games it is
scored on. Defense ratings are also fit on all years, and each QB's rows are
duplicated across train seasons. The r=0.906 this script produced (quoted in
composite_metric.py and the README) is invalid; the honest walk-forward
next-season number is r ~ 0.47 (docs/LEAF_V3_RESULTS.md).

---
LEAF Weight Validation and Optimization

Uses holdout validation to test how well different component weights
predict future QB performance. Optimizes weights based on correlation
with next-year EPA.

Usage:
    python validate_weights.py --train-years 2020 2021 2022 --test-years 2023
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
from typing import Dict, List, Tuple
from scipy.optimize import minimize
from itertools import product

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_pipeline.nfl_data_fetcher import NFLDataFetcher
from data_pipeline.qb_stats_aggregator import QBStatsAggregator
from features.opponent_adjustments import DefenseRatingSystem, OpponentAdjuster
from features.kalman_filter import QBKalmanFilter
from features.context_adjustments import ContextAdjustmentModel
from projections.composite_metric import LEAFCompositeMetric
from utils.config_loader import get_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LEAFWeightValidator:
    """
    Validates and optimizes LEAF component weights using holdout validation.
    """

    def __init__(self, min_attempts: int = 100):
        """
        Initialize validator.

        Args:
            min_attempts: Minimum attempts to include a QB in validation
        """
        self.min_attempts = min_attempts
        self.validation_results = []

    def prepare_data(
        self,
        train_years: List[int],
        test_years: List[int]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch and prepare training and test data.

        Args:
            train_years: Years to use for training/building projections
            test_years: Years to predict (holdout set)

        Returns:
            Tuple of (train_stats, test_stats)
        """
        logger.info(f"Preparing data: train={train_years}, test={test_years}")

        # Fetch all data
        fetcher = NFLDataFetcher()
        all_years = train_years + test_years
        qb_pbp = fetcher.get_qb_play_by_play(all_years)

        # Aggregate game stats
        aggregator = QBStatsAggregator()
        game_stats = aggregator.aggregate_game_stats(qb_pbp)

        # Calculate defense ratings on ALL data (for fair opponent adjustments)
        logger.info("Calculating defense ratings...")
        defense_system = DefenseRatingSystem(decay_rate=0.996, min_attempts=100, iterations=5)
        defense_ratings = defense_system.calculate_defense_ratings(qb_pbp)

        # Apply opponent adjustments
        adjuster = OpponentAdjuster(defense_ratings)
        game_stats = adjuster.adjust_qb_game_stats(game_stats)

        # Apply context adjustments
        logger.info("Applying context adjustments...")
        context_model = ContextAdjustmentModel(alpha=1.0, use_ridge=True)
        context_model.fit(qb_pbp, metrics=['qb_epa', 'cpoe', 'success'])

        metrics_map = {
            'qb_epa': 'epa_per_play',
            'cpoe': 'cpoe_mean',
            'success': 'success_mean'
        }
        game_stats = context_model.adjust_game_stats(game_stats, qb_pbp, metrics_map)

        # Apply Kalman filtering
        logger.info("Applying Kalman filtering...")
        kf = QBKalmanFilter(auto_tune=True, em_iterations=10)

        kalman_metrics = ['epa_per_play', 'cpoe_mean', 'success_mean',
                          'opp_adj_epa', 'opp_adj_cpoe', 'opp_adj_success_rate']

        kalman_prior_means = {
            'epa_per_play': 0.0,
            'cpoe_mean': 0.0,
            'success_mean': 0.5,
            'opp_adj_epa': 0.0,
            'opp_adj_cpoe': 0.0,
            'opp_adj_success_rate': 0.0
        }

        kalman_stats = kf.process_all_players(
            game_stats=game_stats,
            metrics=kalman_metrics,
            prior_means=kalman_prior_means
        )

        # Merge Kalman results with game stats
        player_season_stats = game_stats.groupby(['passer_player_id', 'season']).agg({
            'passer_player_name': 'first',
            'attempts': 'sum',
            'epa_per_play': 'mean',
            'cpoe_mean': 'mean',
            'success_mean': 'mean',
            'opp_adj_epa': 'mean',
            'opp_adj_cpoe': 'mean',
            'opp_adj_success_rate': 'mean'
        }).reset_index()

        # Add Kalman estimates (use most recent for each season)
        # For simplicity, merge on player_id only (Kalman uses all historical data)
        player_season_stats = player_season_stats.merge(
            kalman_stats,
            on='passer_player_id',
            how='left',
            suffixes=('', '_kalman_full')
        )

        # Split into train and test
        train_stats = player_season_stats[player_season_stats['season'].isin(train_years)].copy()
        test_stats = player_season_stats[player_season_stats['season'].isin(test_years)].copy()

        # Filter by minimum attempts
        train_stats = train_stats[train_stats['attempts'] >= self.min_attempts]
        test_stats = test_stats[test_stats['attempts'] >= self.min_attempts]

        logger.info(f"  Train: {len(train_stats)} QB-seasons")
        logger.info(f"  Test: {len(test_stats)} QB-seasons")

        return train_stats, test_stats

    def calculate_leaf_with_weights(
        self,
        player_stats: pd.DataFrame,
        weights: Dict[str, float],
        use_kalman: bool = True,
        use_opponent_adjusted: bool = True
    ) -> pd.DataFrame:
        """
        Calculate LEAF with custom weights.

        Args:
            player_stats: Player statistics
            weights: Component weights
            use_kalman: Use Kalman-filtered estimates
            use_opponent_adjusted: Use opponent-adjusted metrics

        Returns:
            DataFrame with LEAF scores
        """
        # Standardize components
        components = player_stats.copy()

        # Select appropriate metric versions
        if use_opponent_adjusted:
            epa_col = 'opp_adj_epa_kalman' if use_kalman else 'opp_adj_epa'
            cpoe_col = 'opp_adj_cpoe_kalman' if use_kalman else 'opp_adj_cpoe'
            success_col = 'opp_adj_success_rate_kalman' if use_kalman else 'opp_adj_success_rate'
        else:
            epa_col = 'epa_per_play_kalman' if use_kalman else 'epa_per_play'
            cpoe_col = 'cpoe_mean_kalman' if use_kalman else 'cpoe_mean'
            success_col = 'success_mean_kalman' if use_kalman else 'success_mean'

        # Extract components (with fallbacks)
        components['epa_component'] = components.get(epa_col, components.get('epa_per_play', 0))
        components['cpoe_component'] = components.get(cpoe_col, components.get('cpoe_mean', 0))
        components['success_component'] = components.get(success_col, components.get('success_mean', 0.5))
        components['situational_component'] = 0  # Not available in this validation

        # Standardize (z-scores)
        for col in ['epa_component', 'cpoe_component', 'success_component']:
            mean_val = components[col].mean()
            std_val = components[col].std()
            if std_val > 0:
                components[f'{col}_z'] = (components[col] - mean_val) / std_val
            else:
                components[f'{col}_z'] = 0

        components['situational_component_z'] = 0

        # Calculate weighted composite
        components['leaf_z'] = (
            weights['epa'] * components['epa_component_z'] +
            weights['cpoe'] * components['cpoe_component_z'] +
            weights['success_rate'] * components['success_component_z'] +
            weights.get('situational', 0.0) * components['situational_component_z']
        )

        # Convert to EPA scale
        epa_mean = components['epa_component'].mean()
        epa_std = components['epa_component'].std()
        components['leaf'] = epa_mean + (components['leaf_z'] * epa_std)

        return components

    def evaluate_weights(
        self,
        train_stats: pd.DataFrame,
        test_stats: pd.DataFrame,
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Evaluate how well weights predict future performance.

        Args:
            train_stats: Training season stats
            test_stats: Test season stats (future performance)
            weights: Component weights to test

        Returns:
            Dictionary of evaluation metrics
        """
        # Calculate LEAF on training data
        train_with_leaf = self.calculate_leaf_with_weights(train_stats, weights)

        # Merge with test data (same QBs, next year)
        # For each QB, get their train LEAF and test EPA
        comparison = train_with_leaf[['passer_player_id', 'leaf']].merge(
            test_stats[['passer_player_id', 'epa_per_play']],
            on='passer_player_id',
            how='inner',
            suffixes=('_train', '_test')
        )

        if len(comparison) < 10:
            logger.warning(f"Only {len(comparison)} QBs in overlap - results may be unstable")
            return {
                'correlation': 0.0,
                'rmse': 999.0,
                'mae': 999.0,
                'n_qbs': len(comparison)
            }

        # Calculate metrics
        correlation = comparison['leaf'].corr(comparison['epa_per_play'])
        rmse = np.sqrt(((comparison['leaf'] - comparison['epa_per_play']) ** 2).mean())
        mae = (comparison['leaf'] - comparison['epa_per_play']).abs().mean()

        return {
            'correlation': correlation,
            'rmse': rmse,
            'mae': mae,
            'n_qbs': len(comparison)
        }

    def grid_search_weights(
        self,
        train_stats: pd.DataFrame,
        test_stats: pd.DataFrame,
        grid_resolution: int = 5
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
        """
        Grid search over weight combinations.

        Args:
            train_stats: Training data
            test_stats: Test data
            grid_resolution: Number of values to try per weight

        Returns:
            Tuple of (best_weights, all_results_df)
        """
        logger.info(f"Running grid search with resolution {grid_resolution}...")

        # Create grid (weights must sum to 1.0)
        # We'll grid over epa, cpoe, success_rate and derive situational
        # epa: 0.3 to 0.7
        # cpoe: 0.1 to 0.4
        # success: 0.05 to 0.25
        # situational: remainder (0 to 0.15)

        epa_range = np.linspace(0.35, 0.65, grid_resolution)
        cpoe_range = np.linspace(0.15, 0.35, grid_resolution)
        success_range = np.linspace(0.10, 0.20, grid_resolution)

        results = []

        total_combos = len(epa_range) * len(cpoe_range) * len(success_range)
        logger.info(f"Testing {total_combos} weight combinations...")

        for i, (epa_w, cpoe_w, success_w) in enumerate(product(epa_range, cpoe_range, success_range)):
            # Ensure weights sum to 1.0
            situational_w = 1.0 - (epa_w + cpoe_w + success_w)

            if situational_w < 0 or situational_w > 0.2:
                continue  # Skip invalid combinations

            weights = {
                'epa': epa_w,
                'cpoe': cpoe_w,
                'success_rate': success_w,
                'situational': situational_w
            }

            metrics = self.evaluate_weights(train_stats, test_stats, weights)

            results.append({
                'epa_weight': epa_w,
                'cpoe_weight': cpoe_w,
                'success_weight': success_w,
                'situational_weight': situational_w,
                'correlation': metrics['correlation'],
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'n_qbs': metrics['n_qbs']
            })

            if (i + 1) % 50 == 0:
                logger.info(f"  Tested {i + 1}/{total_combos} combinations...")

        results_df = pd.DataFrame(results)

        # Find best by correlation
        best_idx = results_df['correlation'].idxmax()
        best_weights = {
            'epa': results_df.loc[best_idx, 'epa_weight'],
            'cpoe': results_df.loc[best_idx, 'cpoe_weight'],
            'success_rate': results_df.loc[best_idx, 'success_weight'],
            'situational': results_df.loc[best_idx, 'situational_weight']
        }

        logger.info(f"\nBest weights (correlation = {results_df.loc[best_idx, 'correlation']:.4f}):")
        for key, val in best_weights.items():
            logger.info(f"  {key}: {val:.3f}")

        return best_weights, results_df

    def cross_validate_weights(
        self,
        years: List[int],
        weights: Dict[str, float],
        cv_folds: int = 3
    ) -> pd.DataFrame:
        """
        Cross-validate weights across multiple train/test splits.

        Args:
            years: All years to use
            weights: Weights to validate
            cv_folds: Number of CV folds

        Returns:
            DataFrame with results for each fold
        """
        logger.info(f"Running {cv_folds}-fold cross-validation on years {years}...")

        results = []

        # Leave-one-year-out CV
        for test_year in years[-cv_folds:]:
            train_years = [y for y in years if y < test_year]
            if len(train_years) < 2:
                continue

            logger.info(f"\nFold: train={train_years}, test=[{test_year}]")

            train_stats, test_stats = self.prepare_data(train_years, [test_year])
            metrics = self.evaluate_weights(train_stats, test_stats, weights)

            results.append({
                'test_year': test_year,
                'train_years': str(train_years),
                'correlation': metrics['correlation'],
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'n_qbs': metrics['n_qbs']
            })

            logger.info(f"  Correlation: {metrics['correlation']:.4f}")
            logger.info(f"  RMSE: {metrics['rmse']:.4f}")
            logger.info(f"  N QBs: {metrics['n_qbs']}")

        return pd.DataFrame(results)


def main():
    """Run weight validation."""
    parser = argparse.ArgumentParser(
        description='Validate and optimize LEAF component weights'
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
        '--grid-search',
        action='store_true',
        help='Run grid search to find optimal weights'
    )
    parser.add_argument(
        '--grid-resolution',
        type=int,
        default=5,
        help='Grid search resolution (higher = more combinations)'
    )
    parser.add_argument(
        '--cross-validate',
        action='store_true',
        help='Run cross-validation'
    )
    parser.add_argument(
        '--min-attempts',
        type=int,
        default=100,
        help='Minimum attempts to include QB'
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("LEAF Weight Validation")
    logger.info("=" * 80)

    validator = LEAFWeightValidator(min_attempts=args.min_attempts)

    # Load config to get current weights
    config = get_config('config/config.yaml')
    current_weights = config.get('composite.weights', {
        'epa': 0.50,
        'cpoe': 0.25,
        'success_rate': 0.15,
        'situational': 0.10
    })

    # Test current weights
    logger.info("\n" + "=" * 80)
    logger.info("Testing Current Weights")
    logger.info("=" * 80)
    logger.info(f"Current weights: {current_weights}")

    train_stats, test_stats = validator.prepare_data(args.train_years, args.test_years)
    current_metrics = validator.evaluate_weights(train_stats, test_stats, current_weights)

    print(f"\n{'='*80}")
    print("Current Weights Performance")
    print(f"{'='*80}")
    print(f"  Correlation with future EPA: {current_metrics['correlation']:.4f}")
    print(f"  RMSE: {current_metrics['rmse']:.4f}")
    print(f"  MAE: {current_metrics['mae']:.4f}")
    print(f"  N QBs in overlap: {current_metrics['n_qbs']}")

    # Baseline: just EPA
    logger.info("\n" + "=" * 80)
    logger.info("Testing Baseline (EPA only)")
    logger.info("=" * 80)

    baseline_weights = {'epa': 1.0, 'cpoe': 0.0, 'success_rate': 0.0, 'situational': 0.0}
    baseline_metrics = validator.evaluate_weights(train_stats, test_stats, baseline_weights)

    print(f"\n{'='*80}")
    print("Baseline (EPA only) Performance")
    print(f"{'='*80}")
    print(f"  Correlation with future EPA: {baseline_metrics['correlation']:.4f}")
    print(f"  RMSE: {baseline_metrics['rmse']:.4f}")
    print(f"  MAE: {baseline_metrics['mae']:.4f}")

    improvement = current_metrics['correlation'] - baseline_metrics['correlation']
    print(f"\n  LEAF improvement over EPA-only: {improvement:+.4f}")

    # Grid search if requested
    if args.grid_search:
        logger.info("\n" + "=" * 80)
        logger.info("Running Grid Search for Optimal Weights")
        logger.info("=" * 80)

        best_weights, grid_results = validator.grid_search_weights(
            train_stats, test_stats, grid_resolution=args.grid_resolution
        )

        # Save grid results
        grid_results.to_csv('results/weight_grid_search.csv', index=False)
        logger.info(f"\nSaved grid search results to results/weight_grid_search.csv")

        # Compare best to current
        best_metrics = validator.evaluate_weights(train_stats, test_stats, best_weights)

        print(f"\n{'='*80}")
        print("Optimized Weights Performance")
        print(f"{'='*80}")
        print(f"  Correlation with future EPA: {best_metrics['correlation']:.4f}")
        print(f"  RMSE: {best_metrics['rmse']:.4f}")
        print(f"  MAE: {best_metrics['mae']:.4f}")

        print(f"\n  Improvement over current: {best_metrics['correlation'] - current_metrics['correlation']:+.4f}")

        print(f"\n{'='*80}")
        print("Optimal Weights")
        print(f"{'='*80}")
        for key, val in best_weights.items():
            current_val = current_weights.get(key, 0.0)
            diff = val - current_val
            print(f"  {key:15s}: {val:.3f}  (current: {current_val:.3f}, diff: {diff:+.3f})")

    # Cross-validation if requested
    if args.cross_validate:
        logger.info("\n" + "=" * 80)
        logger.info("Running Cross-Validation")
        logger.info("=" * 80)

        all_years = sorted(args.train_years + args.test_years)
        cv_results = validator.cross_validate_weights(all_years, current_weights)

        cv_results.to_csv('results/weight_cross_validation.csv', index=False)

        print(f"\n{'='*80}")
        print("Cross-Validation Results")
        print(f"{'='*80}")
        print(cv_results.to_string(index=False))

        print(f"\nAverage correlation: {cv_results['correlation'].mean():.4f} ± {cv_results['correlation'].std():.4f}")
        print(f"Average RMSE: {cv_results['rmse'].mean():.4f} ± {cv_results['rmse'].std():.4f}")

    logger.info("\n" + "=" * 80)
    logger.info("Validation Complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    exit(main())
