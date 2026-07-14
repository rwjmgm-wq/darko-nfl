"""
Teammate Adjustment Validation - Option C1: Iterative Simultaneous Estimation

Tests whether iterative QB-WR adjustment (like defense ratings) improves prediction.

Key Difference from V2 (residual approach):
  V2: Regressed out receiver quality from QB EPA (one-way adjustment)
  C1: Simultaneously adjust QB and WR ratings (two-way interdependent)

Methodology:
  - Start: QB_0 = Kalman+Opp-Adj EPA, WR_0 = raw EPA per target
  - Iterate:
    * Adjust QB for average WR quality (weighted by targets)
    * Adjust WR for average QB quality (weighted by targets)
  - Converge when delta < threshold
  - Test: Does adjusted QB rating predict better than baseline?

Usage:
    python validate_teammate_adjustments_c1.py --train-years 2020 2021 2022 --test-years 2023
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_pipeline.nfl_data_fetcher import NFLDataFetcher
from data_pipeline.qb_stats_aggregator import QBStatsAggregator
from data_pipeline.receiver_stats_aggregator import ReceiverStatsAggregator
from features.opponent_adjustments import DefenseRatingSystem, OpponentAdjuster
from features.kalman_filter import QBKalmanFilter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IterativeTeammateAdjuster:
    """
    Iteratively adjust QB and WR ratings for teammate quality.

    Similar to DefenseRatingSystem but for QB-WR interactions.
    """

    def __init__(self, max_iterations=10, convergence_threshold=0.001,
                 adjustment_rate=0.5, min_targets_per_game=3):
        """
        Args:
            max_iterations: Maximum iterations
            convergence_threshold: Stop when max rating change < threshold
            adjustment_rate: How much to adjust (0.5 = adjust halfway)
            min_targets_per_game: Minimum targets per GAME to include WR
        """
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.adjustment_rate = adjustment_rate
        self.min_targets_per_game = min_targets_per_game

    def adjust_ratings(self, qb_games, receiver_games):
        """
        Iteratively adjust QB and WR ratings.

        Args:
            qb_games: QB game-level stats with opp_adj_epa_kalman
            receiver_games: Receiver game-level stats with epa_per_target

        Returns:
            qb_games with 'qb_adj_rating' column
            receiver_games with 'wr_adj_rating' column
        """
        logger.info("\nStarting iterative teammate adjustment...")

        # Initialize ratings
        qb_games = qb_games.copy()
        receiver_games = receiver_games.copy()

        # QB initial: Kalman+Opponent-Adjusted EPA
        qb_games['qb_rating'] = qb_games['opp_adj_epa_kalman'].fillna(0)

        # WR initial: EPA per target
        receiver_games['wr_rating'] = receiver_games['epa_per_target'].fillna(0)

        # Filter receivers by minimum targets per game
        receiver_games = receiver_games[receiver_games['targets'] >= self.min_targets_per_game].copy()

        logger.info(f"  QB games: {len(qb_games)}")
        logger.info(f"  WR games: {len(receiver_games)} (min {self.min_targets_per_game} targets/game)")

        # Iterate
        for iteration in range(self.max_iterations):
            logger.info(f"\nIteration {iteration + 1}/{self.max_iterations}")

            # Drop old adjustment columns if they exist
            if 'avg_wr_rating' in qb_games.columns:
                qb_games = qb_games.drop(columns=['avg_wr_rating'])
            if 'avg_qb_rating' in receiver_games.columns:
                receiver_games = receiver_games.drop(columns=['avg_qb_rating'])

            # Calculate average teammate quality
            qb_avg_wr_quality = self._calculate_qb_receiver_quality(qb_games, receiver_games)
            wr_avg_qb_quality = self._calculate_wr_qb_quality(qb_games, receiver_games)

            # Store old ratings to check convergence
            qb_old = qb_games['qb_rating'].copy()
            wr_old = receiver_games['wr_rating'].copy()

            # Adjust QB ratings for WR quality
            qb_games = qb_games.merge(
                qb_avg_wr_quality[['passer_player_id', 'game_id', 'avg_wr_rating']],
                on=['passer_player_id', 'game_id'],
                how='left'
            )
            qb_games['avg_wr_rating'] = qb_games['avg_wr_rating'].fillna(0)

            # QB adjustment: If you had good receivers, your true talent is lower
            qb_games['qb_rating'] = (
                qb_games['opp_adj_epa_kalman'].fillna(0) -
                self.adjustment_rate * qb_games['avg_wr_rating']
            )

            # Adjust WR ratings for QB quality
            receiver_games = receiver_games.merge(
                wr_avg_qb_quality[['receiver_player_id', 'game_id', 'avg_qb_rating']],
                on=['receiver_player_id', 'game_id'],
                how='left'
            )
            receiver_games['avg_qb_rating'] = receiver_games['avg_qb_rating'].fillna(0)

            # WR adjustment: If you had good QBs, your true talent is lower
            receiver_games['wr_rating'] = (
                receiver_games['epa_per_target'].fillna(0) -
                self.adjustment_rate * receiver_games['avg_qb_rating']
            )

            # Check convergence
            qb_delta = (qb_games['qb_rating'] - qb_old).abs().max()
            wr_delta = (receiver_games['wr_rating'] - wr_old).abs().max()
            max_delta = max(qb_delta, wr_delta)

            logger.info(f"  QB delta: {qb_delta:.6f}")
            logger.info(f"  WR delta: {wr_delta:.6f}")
            logger.info(f"  Max delta: {max_delta:.6f}")

            if max_delta < self.convergence_threshold:
                logger.info(f"  Converged after {iteration + 1} iterations!")
                break
        else:
            logger.info(f"  Did not converge after {self.max_iterations} iterations")

        # Rename final ratings
        qb_games['qb_adj_rating'] = qb_games['qb_rating']
        receiver_games['wr_adj_rating'] = receiver_games['wr_rating']

        return qb_games, receiver_games

    def _calculate_qb_receiver_quality(self, qb_games, receiver_games):
        """Calculate average receiver quality for each QB-game."""
        # Merge QB games with receiver games to get targets in that game
        merged = qb_games[['passer_player_id', 'game_id', 'posteam']].merge(
            receiver_games[['game_id', 'posteam', 'receiver_player_id', 'targets', 'wr_rating']],
            on=['game_id', 'posteam'],
            how='left'
        )

        # Calculate weighted average WR rating per QB-game
        merged['weighted_rating'] = merged['targets'] * merged['wr_rating']

        qb_wr_quality = merged.groupby(['passer_player_id', 'game_id']).agg({
            'weighted_rating': 'sum',
            'targets': 'sum'
        }).reset_index()

        qb_wr_quality['avg_wr_rating'] = (
            qb_wr_quality['weighted_rating'] / qb_wr_quality['targets']
        ).fillna(0)

        return qb_wr_quality

    def _calculate_wr_qb_quality(self, qb_games, receiver_games):
        """Calculate average QB quality for each WR-game."""
        # Each receiver game has one QB (passer_player_id from the team)
        # Merge to get QB rating for that game
        merged = receiver_games[['receiver_player_id', 'game_id', 'posteam', 'targets']].merge(
            qb_games[['game_id', 'posteam', 'passer_player_id', 'qb_rating']],
            on=['game_id', 'posteam'],
            how='left'
        )

        # For each WR-game, get the QB rating (should be 1:1 in most cases)
        wr_qb_quality = merged.groupby(['receiver_player_id', 'game_id']).agg({
            'qb_rating': 'mean'  # Average if multiple QBs in same game
        }).reset_index()

        wr_qb_quality = wr_qb_quality.rename(columns={'qb_rating': 'avg_qb_rating'})

        return wr_qb_quality


def main():
    """Run Option C1 validation."""
    parser = argparse.ArgumentParser(
        description='Validate iterative teammate adjustments (Option C1)'
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
    logger.info("Teammate Adjustment Validation - Option C1")
    logger.info("Iterative Simultaneous Estimation")
    logger.info("="*80)
    logger.info(f"\nTrain: {args.train_years}")
    logger.info(f"Test: {args.test_years}")

    # Fetch data
    fetcher = NFLDataFetcher()
    all_years = args.train_years + args.test_years
    pbp = fetcher.fetch_pbp_data(all_years)

    # Aggregate QB stats
    logger.info("\n[Step 1/6] Aggregating QB statistics...")
    qb_agg = QBStatsAggregator()
    qb_games = qb_agg.aggregate_game_stats(pbp)

    # Aggregate receiver stats
    logger.info("\n[Step 2/6] Aggregating receiver statistics...")
    rec_agg = ReceiverStatsAggregator()
    rec_games = rec_agg.aggregate_game_stats(pbp)

    # Calculate defense ratings
    logger.info("\n[Step 3/6] Calculating defense ratings...")
    defense_system = DefenseRatingSystem(decay_rate=0.996, min_attempts=100, iterations=5)
    defense_ratings = defense_system.calculate_defense_ratings(pbp)

    # Apply opponent adjustments
    logger.info("\n[Step 4/6] Applying opponent adjustments...")
    adjuster = OpponentAdjuster(defense_ratings)
    qb_opp_adj = adjuster.adjust_qb_game_stats(qb_games)

    # Apply Kalman filtering
    logger.info("\n[Step 5/6] Applying Kalman filtering...")
    kf = QBKalmanFilter(auto_tune=True, em_iterations=10)

    kalman_metrics = ['epa_per_play', 'opp_adj_epa']
    kalman_prior_means = {
        'epa_per_play': 0.0,
        'opp_adj_epa': 0.0
    }

    kalman_stats = kf.process_all_players(
        game_stats=qb_opp_adj,
        metrics=kalman_metrics,
        prior_means=kalman_prior_means
    )

    # Merge Kalman estimates back
    qb_full = qb_opp_adj.merge(
        kalman_stats[['passer_player_id', 'opp_adj_epa_kalman']],
        on='passer_player_id',
        how='left'
    )

    # Split train/test
    train_qb_games = qb_full[qb_full['season'].isin(args.train_years)].copy()
    test_qb_games = qb_full[qb_full['season'].isin(args.test_years)].copy()

    train_rec_games = rec_games[rec_games['season'].isin(args.train_years)].copy()
    test_rec_games = rec_games[rec_games['season'].isin(args.test_years)].copy()

    logger.info(f"\nTrain QB games: {len(train_qb_games)}")
    logger.info(f"Test QB games: {len(test_qb_games)}")

    # Apply iterative adjustment on TRAINING data only
    logger.info("\n[Step 6/6] Applying iterative teammate adjustment...")
    adjuster = IterativeTeammateAdjuster(
        max_iterations=10,
        convergence_threshold=0.001,
        adjustment_rate=0.5,
        min_targets_per_game=3
    )

    train_qb_adj, train_rec_adj = adjuster.adjust_ratings(train_qb_games, train_rec_games)

    # Aggregate to season level
    def aggregate_season(games, min_att):
        return games.groupby(['passer_player_id', 'season']).agg({
            'passer_player_name': 'first',
            'attempts': 'sum',
            'opp_adj_epa_kalman': 'mean',  # Baseline
            'qb_adj_rating': 'mean'  # Adjusted
        }).reset_index().query(f'attempts >= {min_att}')

    train_seasons = aggregate_season(train_qb_adj, args.min_attempts)

    logger.info(f"\nTrain QB-seasons: {len(train_seasons)}")

    # Get RAW 2023 EPA as target
    test_raw_epa = qb_games[qb_games['season'].isin(args.test_years)].groupby(
        ['passer_player_id', 'season']
    ).agg({
        'epa_per_play': 'mean',
        'attempts': 'sum'
    }).reset_index().query(f'attempts >= {args.min_attempts}')

    # Test predictive power
    def test_metric(train, test_target, metric):
        """Compare train metric against RAW test EPA."""
        comparison = train[['passer_player_id', metric]].merge(
            test_target[['passer_player_id', 'epa_per_play']],
            on='passer_player_id',
            how='inner',
            suffixes=('_train', '_test')
        )

        if len(comparison) < 10:
            return {'correlation': np.nan, 'n_qbs': len(comparison)}

        pred_col = f'{metric}_train' if f'{metric}_train' in comparison.columns else metric
        actual_col = 'epa_per_play_test' if 'epa_per_play_test' in comparison.columns else 'epa_per_play'

        pred = comparison[pred_col]
        actual = comparison[actual_col]

        valid = ~(pred.isna() | actual.isna())
        pred = pred[valid]
        actual = actual[valid]

        if len(pred) < 10:
            return {'correlation': np.nan, 'n_qbs': len(pred)}

        return {
            'correlation': pred.corr(actual),
            'rmse': np.sqrt(((pred - actual) ** 2).mean()),
            'mae': (pred - actual).abs().mean(),
            'n_qbs': len(pred)
        }

    logger.info("\n" + "="*80)
    logger.info("Testing Predictive Power")
    logger.info("="*80)

    # Baseline: Kalman+Opponent-Adjusted EPA
    baseline = test_metric(train_seasons, test_raw_epa, 'opp_adj_epa_kalman')

    # Option C1: Iterative adjustment
    c1_adjusted = test_metric(train_seasons, test_raw_epa, 'qb_adj_rating')

    # Results
    print(f"\n{'='*80}")
    print("Option C1: Iterative Simultaneous Estimation")
    print(f"{'='*80}")

    print(f"\nBaseline (Kalman + Opponent-Adjusted EPA):")
    print(f"  Correlation with future EPA: {baseline['correlation']:.4f}")
    print(f"  RMSE: {baseline.get('rmse', 0):.4f}")
    print(f"  Sample size: {baseline['n_qbs']} QBs")

    print(f"\nOption C1 (Iterative QB-WR Adjustment):")
    print(f"  Correlation with future EPA: {c1_adjusted['correlation']:.4f}")
    print(f"  RMSE: {c1_adjusted.get('rmse', 0):.4f}")
    print(f"  Sample size: {c1_adjusted['n_qbs']} QBs")

    improvement = c1_adjusted['correlation'] - baseline['correlation']
    pct_improvement = (improvement / baseline['correlation']) * 100 if baseline['correlation'] > 0 else 0

    print(f"\nImprovement:")
    print(f"  Absolute: {improvement:+.4f}")
    print(f"  Relative: {pct_improvement:+.2f}%")

    # Decision
    print(f"\n{'='*80}")
    print("OPTION C1 EVALUATION")
    print(f"{'='*80}")

    if improvement > 0.01:
        print("[+] SUCCESS: Iterative adjustment IMPROVES prediction")
        print(f"  Improvement of {improvement:.4f} exceeds 0.01 threshold")
        print("  Recommendation: Use this method for multi-position LEAF")
    elif improvement > 0 and improvement <= 0.01:
        print("[~] MARGINAL: Small improvement")
        print(f"  Improvement of {improvement:.4f} is positive but < 0.01")
        print("  Recommendation: Try Options C2/C3 before deciding")
    else:
        print("[-] FAILED: Iterative adjustment does not improve")
        print(f"  Change of {improvement:.4f} does not meet threshold")
        if improvement < -0.005:
            print(f"  WARNING: Actually HURT prediction by {abs(improvement):.4f}")
        print("\n  Recommendation: Try Options C2 and C3")

    print("="*80)

    # Save results
    results = pd.DataFrame([
        {'method': 'Kalman+Opp-Adj (Baseline)', **baseline},
        {'method': 'C1: Iterative QB-WR', **c1_adjusted}
    ])
    Path('results').mkdir(exist_ok=True)
    results.to_csv('results/teammate_adjustment_c1_results.csv', index=False)
    logger.info("\nSaved results to results/teammate_adjustment_c1_results.csv")

    return 0


if __name__ == "__main__":
    exit(main())
