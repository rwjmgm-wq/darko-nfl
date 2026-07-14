"""
Comprehensive Predictive Power Analysis for LEAF Components

Tests all available metrics individually and in combinations to determine
which stats have the most predictive power for future QB performance.

Usage:
    python analyze_predictive_power.py --train-years 2020 2021 2022 --test-years 2023
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_pipeline.nfl_data_fetcher import NFLDataFetcher
from data_pipeline.qb_stats_aggregator import QBStatsAggregator
from features.opponent_adjustments import DefenseRatingSystem, OpponentAdjuster
from features.kalman_filter import QBKalmanFilter
from features.context_adjustments import ContextAdjustmentModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetricPredictivePowerAnalyzer:
    """
    Analyzes predictive power of individual metrics and combinations.
    """

    def __init__(self, min_attempts: int = 100):
        """
        Initialize analyzer.

        Args:
            min_attempts: Minimum attempts to include a QB
        """
        self.min_attempts = min_attempts
        self.results = []

    def prepare_full_dataset(
        self,
        train_years: List[int],
        test_years: List[int]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare comprehensive dataset with all available metrics.

        Args:
            train_years: Years for training
            test_years: Years for testing

        Returns:
            Tuple of (train_stats, test_stats) with all metrics
        """
        logger.info(f"Preparing comprehensive dataset: train={train_years}, test={test_years}")

        # Fetch data
        fetcher = NFLDataFetcher()
        all_years = train_years + test_years
        qb_pbp = fetcher.get_qb_play_by_play(all_years)

        # Aggregate game stats
        aggregator = QBStatsAggregator()
        game_stats = aggregator.aggregate_game_stats(qb_pbp)

        # Calculate defense ratings
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

        # Aggregate by player-season
        player_season_stats = game_stats.groupby(['passer_player_id', 'season']).agg({
            'passer_player_name': 'first',
            'attempts': 'sum',
            # Raw metrics
            'epa_per_play': 'mean',
            'cpoe_mean': 'mean',
            'success_mean': 'mean',
            'completion_pct': 'mean',
            'yards_per_attempt': 'mean',
            'td_rate': 'mean',
            'int_rate': 'mean',
            'sack_rate': 'mean',
            # Opponent-adjusted metrics
            'opp_adj_epa': 'mean',
            'opp_adj_cpoe': 'mean',
            'opp_adj_success_rate': 'mean',
            # Context-adjusted (if available)
        }).reset_index()

        # Add context-adjusted columns if they exist
        if 'epa_per_play_ctx_adj' in game_stats.columns:
            ctx_stats = game_stats.groupby(['passer_player_id', 'season']).agg({
                'epa_per_play_ctx_adj': 'mean',
                'cpoe_mean_ctx_adj': 'mean',
                'success_mean_ctx_adj': 'mean'
            }).reset_index()
            player_season_stats = player_season_stats.merge(ctx_stats, on=['passer_player_id', 'season'], how='left')

        # Merge Kalman estimates
        player_season_stats = player_season_stats.merge(
            kalman_stats,
            on='passer_player_id',
            how='left',
            suffixes=('', '_kalman_full')
        )

        # Split into train/test
        train_stats = player_season_stats[player_season_stats['season'].isin(train_years)].copy()
        test_stats = player_season_stats[player_season_stats['season'].isin(test_years)].copy()

        # Filter by minimum attempts
        train_stats = train_stats[train_stats['attempts'] >= self.min_attempts]
        test_stats = test_stats[test_stats['attempts'] >= self.min_attempts]

        logger.info(f"  Train: {len(train_stats)} QB-seasons")
        logger.info(f"  Test: {len(test_stats)} QB-seasons")
        logger.info(f"  Available columns: {sorted(train_stats.columns.tolist())}")

        return train_stats, test_stats

    def test_metric_predictive_power(
        self,
        train_stats: pd.DataFrame,
        test_stats: pd.DataFrame,
        metric_name: str,
        target_metric: str = 'epa_per_play'
    ) -> Dict[str, float]:
        """
        Test how well a single metric predicts future performance.

        Args:
            train_stats: Training data
            test_stats: Test data
            metric_name: Name of predictor metric
            target_metric: What we're trying to predict

        Returns:
            Dictionary with correlation, RMSE, MAE
        """
        # Check if columns exist
        if metric_name not in train_stats.columns:
            return {
                'metric': metric_name,
                'correlation': np.nan,
                'rmse': np.nan,
                'mae': np.nan,
                'n_qbs': 0
            }

        if target_metric not in test_stats.columns:
            return {
                'metric': metric_name,
                'correlation': np.nan,
                'rmse': np.nan,
                'mae': np.nan,
                'n_qbs': 0
            }

        # Merge train and test on player_id
        train_subset = train_stats[['passer_player_id', metric_name]].copy()
        test_subset = test_stats[['passer_player_id', target_metric]].copy()

        comparison = train_subset.merge(
            test_subset,
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

        # Handle column names after merge
        # If metric_name and target_metric are the same, pandas adds suffixes
        pred_col = f'{metric_name}_train' if f'{metric_name}_train' in comparison.columns else metric_name
        target_col = f'{target_metric}_test' if f'{target_metric}_test' in comparison.columns else target_metric

        pred = comparison[pred_col]
        actual = comparison[target_col]

        # Remove NaN values
        valid_mask = ~(pred.isna() | actual.isna())
        pred = pred[valid_mask]
        actual = actual[valid_mask]

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

    def analyze_all_metrics(
        self,
        train_stats: pd.DataFrame,
        test_stats: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Test predictive power of all available metrics.

        Args:
            train_stats: Training data
            test_stats: Test data

        Returns:
            DataFrame with results for each metric
        """
        logger.info("Testing predictive power of all metrics...")

        # Define metric categories
        raw_metrics = [
            'epa_per_play', 'cpoe_mean', 'success_mean',
            'completion_pct', 'yards_per_attempt', 'td_rate', 'int_rate', 'sack_rate'
        ]

        opponent_adjusted = [
            'opp_adj_epa', 'opp_adj_cpoe', 'opp_adj_success_rate'
        ]

        context_adjusted = [
            'epa_per_play_ctx_adj', 'cpoe_mean_ctx_adj', 'success_mean_ctx_adj'
        ]

        kalman_filtered = [
            'epa_per_play_kalman', 'cpoe_mean_kalman', 'success_mean_kalman',
            'opp_adj_epa_kalman', 'opp_adj_cpoe_kalman', 'opp_adj_success_rate_kalman'
        ]

        all_metrics = raw_metrics + opponent_adjusted + context_adjusted + kalman_filtered

        results = []

        for metric in all_metrics:
            if metric not in train_stats.columns:
                logger.debug(f"  Skipping {metric} (not in dataset)")
                continue

            logger.info(f"  Testing {metric}...")
            result = self.test_metric_predictive_power(train_stats, test_stats, metric)

            # Add category
            if metric in raw_metrics:
                result['category'] = 'Raw'
            elif metric in opponent_adjusted:
                result['category'] = 'Opponent-Adjusted'
            elif metric in context_adjusted:
                result['category'] = 'Context-Adjusted'
            elif metric in kalman_filtered:
                result['category'] = 'Kalman-Filtered'
            else:
                result['category'] = 'Other'

            results.append(result)

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('correlation', ascending=False)

        return results_df

    def compare_metric_versions(
        self,
        train_stats: pd.DataFrame,
        test_stats: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compare different versions of the same base metric.

        For example: EPA raw vs opponent-adjusted vs context-adjusted vs Kalman

        Args:
            train_stats: Training data
            test_stats: Test data

        Returns:
            DataFrame comparing versions of each metric
        """
        logger.info("Comparing different versions of base metrics...")

        base_metrics = ['epa', 'cpoe', 'success']

        comparisons = []

        for base in base_metrics:
            logger.info(f"\n  Comparing versions of {base.upper()}...")

            versions = {
                'Raw': f'{base}_per_play' if base == 'epa' else f'{base}_mean',
                'Opponent-Adj': f'opp_adj_{base}' if base == 'epa' else f'opp_adj_{base}',
                'Context-Adj': f'{base}_per_play_ctx_adj' if base == 'epa' else f'{base}_mean_ctx_adj',
                'Kalman': f'{base}_per_play_kalman' if base == 'epa' else f'{base}_mean_kalman',
                'Opp+Kalman': f'opp_adj_{base}_kalman'
            }

            for version_name, metric_name in versions.items():
                # Handle naming variations
                if base == 'success' and 'mean' not in metric_name and 'kalman' not in metric_name and 'opp_adj' in metric_name:
                    metric_name = 'opp_adj_success_rate'

                if metric_name not in train_stats.columns:
                    continue

                result = self.test_metric_predictive_power(train_stats, test_stats, metric_name)
                result['base_metric'] = base.upper()
                result['version'] = version_name
                comparisons.append(result)

        return pd.DataFrame(comparisons)


def main():
    """Run comprehensive predictive power analysis."""
    parser = argparse.ArgumentParser(
        description='Analyze predictive power of all LEAF components'
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

    logger.info("=" * 80)
    logger.info("LEAF Predictive Power Analysis")
    logger.info("=" * 80)

    analyzer = MetricPredictivePowerAnalyzer(min_attempts=args.min_attempts)

    # Prepare data
    train_stats, test_stats = analyzer.prepare_full_dataset(args.train_years, args.test_years)

    # Analyze all metrics
    logger.info("\n" + "=" * 80)
    logger.info("Testing All Metrics")
    logger.info("=" * 80)

    all_results = analyzer.analyze_all_metrics(train_stats, test_stats)
    all_results.to_csv('results/metric_predictive_power.csv', index=False)

    print(f"\n{'='*80}")
    print("Predictive Power of All Metrics (Correlation with Future EPA)")
    print(f"{'='*80}")
    print(all_results[['metric', 'category', 'correlation', 'rmse', 'n_qbs']].to_string(index=False))

    # Compare versions
    logger.info("\n" + "=" * 80)
    logger.info("Comparing Metric Versions")
    logger.info("=" * 80)

    version_comparison = analyzer.compare_metric_versions(train_stats, test_stats)
    version_comparison.to_csv('results/metric_version_comparison.csv', index=False)

    print(f"\n{'='*80}")
    print("Metric Version Comparison")
    print(f"{'='*80}")

    for base_metric in version_comparison['base_metric'].unique():
        base_data = version_comparison[version_comparison['base_metric'] == base_metric].copy()
        base_data = base_data.sort_values('correlation', ascending=False)

        print(f"\n{base_metric}:")
        print(base_data[['version', 'metric', 'correlation', 'rmse']].to_string(index=False))

    # Top performers
    print(f"\n{'='*80}")
    print("Top 10 Most Predictive Metrics")
    print(f"{'='*80}")
    top10 = all_results.head(10)
    print(top10[['metric', 'category', 'correlation', 'rmse']].to_string(index=False))

    # Bottom performers
    print(f"\n{'='*80}")
    print("Bottom 10 Least Predictive Metrics")
    print(f"{'='*80}")
    bottom10 = all_results.tail(10)
    print(bottom10[['metric', 'category', 'correlation', 'rmse']].to_string(index=False))

    # Category summary
    print(f"\n{'='*80}")
    print("Average Correlation by Category")
    print(f"{'='*80}")
    category_summary = all_results.groupby('category').agg({
        'correlation': ['mean', 'std', 'count']
    }).round(4)
    print(category_summary)

    logger.info("\n" + "=" * 80)
    logger.info("Analysis Complete!")
    logger.info("=" * 80)
    logger.info(f"Results saved to results/metric_predictive_power.csv")
    logger.info(f"Version comparison saved to results/metric_version_comparison.csv")


if __name__ == "__main__":
    exit(main())
