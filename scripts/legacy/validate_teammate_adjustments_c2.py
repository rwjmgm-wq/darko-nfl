"""
Teammate Adjustment Validation - Option C2: Individual Receiver Quality

Tests whether using individual receiver quality (instead of team aggregate) improves prediction.

Key Difference from V2:
  V2: Used team aggregate receiver corps quality (average across all WRs)
  C2: Uses individual receiver quality weighted by target share

Rationale:
  - A team's WR1 matters more than WR3
  - Target distribution is uneven
  - Individual WR quality might be less confounded

Methodology:
  - Calculate each receiver's EPA per target (season-level)
  - For each QB game, calculate weighted receiver quality (by targets)
  - Adjust QB performance for individual receiver contributions
  - Test: Does this predict better than baseline?

Usage:
    python validate_teammate_adjustments_c2.py --train-years 2020 2021 2022 --test-years 2023
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
from sklearn.linear_model import Ridge

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


def calculate_individual_receiver_quality(qb_games, receiver_games, min_targets=30):
    """
    Calculate QB's receiver quality based on individual WR performance, not team aggregate.

    Args:
        qb_games: QB game-level stats
        receiver_games: Receiver game-level stats
        min_targets: Minimum season targets to include WR in calculation

    Returns:
        qb_games with 'individual_wr_quality' column
    """
    logger.info("\nCalculating individual receiver quality...")

    # Calculate season-level WR quality (EPA per target)
    wr_season_quality = receiver_games.groupby(['receiver_player_id', 'season']).agg({
        'receiver_player_name': 'first',
        'targets': 'sum',
        'epa_per_target': 'mean'
    }).reset_index()

    # Filter by minimum targets
    wr_season_quality = wr_season_quality[wr_season_quality['targets'] >= min_targets].copy()

    logger.info(f"  {len(wr_season_quality)} WR-seasons with {min_targets}+ targets")

    # For each QB-game, calculate weighted average of their receivers' quality
    # Merge QB games with receiver games to get receiver targets
    merged = qb_games[['passer_player_id', 'game_id', 'season', 'posteam']].merge(
        receiver_games[['game_id', 'season', 'posteam', 'receiver_player_id', 'targets']],
        on=['game_id', 'season', 'posteam'],
        how='left'
    )

    # Merge with season-level WR quality
    merged = merged.merge(
        wr_season_quality[['receiver_player_id', 'season', 'epa_per_target']],
        on=['receiver_player_id', 'season'],
        how='left',
        suffixes=('', '_wr')
    )

    # Calculate weighted average WR quality per QB-game
    merged['weighted_epa'] = merged['targets'] * merged['epa_per_target'].fillna(0)

    qb_wr_quality = merged.groupby(['passer_player_id', 'game_id']).agg({
        'weighted_epa': 'sum',
        'targets': 'sum'
    }).reset_index()

    qb_wr_quality['individual_wr_quality'] = (
        qb_wr_quality['weighted_epa'] / qb_wr_quality['targets']
    ).fillna(0)

    # Merge back to QB games
    qb_with_quality = qb_games.merge(
        qb_wr_quality[['passer_player_id', 'game_id', 'individual_wr_quality']],
        on=['passer_player_id', 'game_id'],
        how='left'
    )

    qb_with_quality['individual_wr_quality'] = qb_with_quality['individual_wr_quality'].fillna(0)

    logger.info(f"  Added individual WR quality to {len(qb_with_quality)} QB games")

    return qb_with_quality


def main():
    """Run Option C2 validation."""
    parser = argparse.ArgumentParser(
        description='Validate individual receiver quality adjustments (Option C2)'
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
    logger.info("Teammate Adjustment Validation - Option C2")
    logger.info("Individual Receiver Quality (not team aggregate)")
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
    train_games = qb_full[qb_full['season'].isin(args.train_years)].copy()
    test_games = qb_full[qb_full['season'].isin(args.test_years)].copy()

    train_rec_games = rec_games[rec_games['season'].isin(args.train_years)].copy()
    test_rec_games = rec_games[rec_games['season'].isin(args.test_years)].copy()

    logger.info(f"\nTrain QB games: {len(train_games)}")
    logger.info(f"Test QB games: {len(test_games)}")

    # Calculate individual receiver quality
    logger.info("\n[Step 6/6] Calculating individual receiver quality...")
    train_with_wr = calculate_individual_receiver_quality(train_games, train_rec_games, min_targets=30)
    test_with_wr = calculate_individual_receiver_quality(test_games, test_rec_games, min_targets=30)

    # Adjust QB performance for individual WR quality using Ridge
    logger.info("\nTraining adjustment model...")
    X_train = train_with_wr[['individual_wr_quality']].values
    y_train = train_with_wr['opp_adj_epa_kalman'].fillna(0).values

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    logger.info(f"  Coefficient: {model.coef_[0]:.4f}")

    train_expected = model.predict(X_train)
    X_test = test_with_wr[['individual_wr_quality']].values
    test_expected = model.predict(X_test)

    train_with_wr = train_with_wr.copy()
    test_with_wr = test_with_wr.copy()
    train_with_wr['qb_adj_c2'] = y_train - train_expected
    test_with_wr['qb_adj_c2'] = test_with_wr['opp_adj_epa_kalman'].fillna(0) - test_expected

    # Aggregate to season level
    def aggregate_season(games, min_att):
        return games.groupby(['passer_player_id', 'season']).agg({
            'passer_player_name': 'first',
            'attempts': 'sum',
            'opp_adj_epa_kalman': 'mean',
            'qb_adj_c2': 'mean'
        }).reset_index().query(f'attempts >= {min_att}')

    train_seasons = aggregate_season(train_with_wr, args.min_attempts)

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

    # Option C2: Individual receiver adjustment
    c2_adjusted = test_metric(train_seasons, test_raw_epa, 'qb_adj_c2')

    # Results
    print(f"\n{'='*80}")
    print("Option C2: Individual Receiver Quality")
    print(f"{'='*80}")

    print(f"\nBaseline (Kalman + Opponent-Adjusted EPA):")
    print(f"  Correlation with future EPA: {baseline['correlation']:.4f}")
    print(f"  RMSE: {baseline.get('rmse', 0):.4f}")
    print(f"  Sample size: {baseline['n_qbs']} QBs")

    print(f"\nOption C2 (Individual WR Quality Adjustment):")
    print(f"  Correlation with future EPA: {c2_adjusted['correlation']:.4f}")
    print(f"  RMSE: {c2_adjusted.get('rmse', 0):.4f}")
    print(f"  Sample size: {c2_adjusted['n_qbs']} QBs")

    improvement = c2_adjusted['correlation'] - baseline['correlation']
    pct_improvement = (improvement / baseline['correlation']) * 100 if baseline['correlation'] > 0 else 0

    print(f"\nImprovement:")
    print(f"  Absolute: {improvement:+.4f}")
    print(f"  Relative: {pct_improvement:+.2f}%")

    # Decision
    print(f"\n{'='*80}")
    print("OPTION C2 EVALUATION")
    print(f"{'='*80}")

    if improvement > 0.01:
        print("[+] SUCCESS: Individual WR adjustment IMPROVES prediction")
        print(f"  Improvement of {improvement:.4f} exceeds 0.01 threshold")
        print("  Recommendation: Use this method for multi-position LEAF")
    elif improvement > 0 and improvement <= 0.01:
        print("[~] MARGINAL: Small improvement")
        print(f"  Improvement of {improvement:.4f} is positive but < 0.01")
        print("  Recommendation: Try Option C3 before deciding")
    else:
        print("[-] FAILED: Individual WR adjustment does not improve")
        print(f"  Change of {improvement:.4f} does not meet threshold")
        if improvement < -0.005:
            print(f"  WARNING: Actually HURT prediction by {abs(improvement):.4f}")
        print("\n  Recommendation: Try Option C3 (direct covariate)")

    print("="*80)

    # Save results
    results = pd.DataFrame([
        {'method': 'Kalman+Opp-Adj (Baseline)', **baseline},
        {'method': 'C2: Individual WR Quality', **c2_adjusted}
    ])
    Path('results').mkdir(exist_ok=True)
    results.to_csv('results/teammate_adjustment_c2_results.csv', index=False)
    logger.info("\nSaved results to results/teammate_adjustment_c2_results.csv")

    return 0


if __name__ == "__main__":
    exit(main())
