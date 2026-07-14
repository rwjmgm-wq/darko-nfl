"""
Teammate Adjustment Validation V2 - Test on Strong Baseline

Tests whether adjusting QB performance for receiver quality improves
prediction AFTER applying opponent adjustments and Kalman filtering.

Key Question:
  Does receiver adjustment improve 0.886 (Kalman+Opp-Adj) → 0.89+?

Methodology:
  - Train on 2020-2022, test on 2023
  - Apply opponent adjustments + Kalman filtering first
  - Then test if receiver adjustments add value on top

Usage:
    python validate_teammate_adjustments_v2.py --train-years 2020 2021 2022 --test-years 2023
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


def main():
    """Run enhanced teammate adjustment validation."""
    parser = argparse.ArgumentParser(
        description='Validate teammate adjustments on strong baseline'
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
    logger.info("Teammate Adjustment Validation V2 - Strong Baseline Test")
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

    # Calculate team receiver quality
    team_receiver_quality = rec_agg.calculate_team_receiver_quality(rec_games)

    # Merge QB with receiver context
    qb_with_receivers = qb_games.merge(
        team_receiver_quality[['game_id', 'posteam', 'receiver_corps_epa',
                               'receiver_corps_catch_rate', 'receiver_corps_yards_per_target']],
        on=['game_id', 'posteam'],
        how='left'
    )
    qb_with_receivers['receiver_corps_epa'] = qb_with_receivers['receiver_corps_epa'].fillna(0)
    qb_with_receivers['receiver_corps_catch_rate'] = qb_with_receivers['receiver_corps_catch_rate'].fillna(0.6)
    qb_with_receivers['receiver_corps_yards_per_target'] = qb_with_receivers['receiver_corps_yards_per_target'].fillna(6.5)

    # Calculate defense ratings
    logger.info("\n[Step 3/6] Calculating defense ratings...")
    defense_system = DefenseRatingSystem(decay_rate=0.996, min_attempts=100, iterations=5)
    defense_ratings = defense_system.calculate_defense_ratings(pbp)

    # Apply opponent adjustments
    logger.info("\n[Step 4/6] Applying opponent adjustments...")
    adjuster = OpponentAdjuster(defense_ratings)
    qb_opp_adj = adjuster.adjust_qb_game_stats(qb_with_receivers)

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
    train_games = qb_full[qb_full['season'].isin(args.train_years)]
    test_games = qb_full[qb_full['season'].isin(args.test_years)]

    logger.info(f"\nTrain games: {len(train_games)}")
    logger.info(f"Test games: {len(test_games)}")

    # Adjust opp_adj_epa_kalman for receivers
    logger.info("\n[Step 6/6] Testing receiver adjustment on strong baseline...")

    X_train = train_games[['receiver_corps_epa', 'receiver_corps_catch_rate',
                            'receiver_corps_yards_per_target']].values
    y_train = train_games['opp_adj_epa_kalman'].fillna(0).values

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    logger.info(f"  Receiver adjustment coefficients: {model.coef_}")

    train_expected = model.predict(X_train)
    X_test = test_games[['receiver_corps_epa', 'receiver_corps_catch_rate',
                          'receiver_corps_yards_per_target']].values
    test_expected = model.predict(X_test)

    train_games = train_games.copy()
    test_games = test_games.copy()
    train_games['epa_receiver_adj'] = y_train - train_expected
    test_games['epa_receiver_adj'] = test_games['opp_adj_epa_kalman'].fillna(0) - test_expected

    # Aggregate to season
    def aggregate_season(games, min_att):
        return games.groupby(['passer_player_id', 'season']).agg({
            'passer_player_name': 'first',
            'attempts': 'sum',
            'opp_adj_epa_kalman': 'mean',
            'epa_receiver_adj': 'mean'
        }).reset_index().query(f'attempts >= {min_att}')

    train_seasons = aggregate_season(train_games, args.min_attempts)
    test_seasons = aggregate_season(test_games, args.min_attempts)

    logger.info(f"\nTrain QB-seasons: {len(train_seasons)}")
    logger.info(f"Test QB-seasons: {len(test_seasons)}")

    # Aggregate RAW test EPA (not Kalman filtered) as target
    test_raw_epa = qb_games[qb_games['season'].isin(args.test_years)].groupby(
        ['passer_player_id', 'season']
    ).agg({
        'epa_per_play': 'mean',  # Raw EPA for 2023
        'attempts': 'sum'
    }).reset_index().query(f'attempts >= {args.min_attempts}')

    # Test predictive power
    def test_metric(train, test_target, metric):
        """Compare train metric against RAW test EPA."""
        comparison = train[['passer_player_id', metric]].merge(
            test_target[['passer_player_id', 'epa_per_play']],  # Target is RAW 2023 EPA
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

    # Adjusted: Add receiver adjustment on top
    adjusted = test_metric(train_seasons, test_raw_epa, 'epa_receiver_adj')

    # Results
    print(f"\n{'='*80}")
    print("Predictive Power Comparison - Strong Baseline Test")
    print(f"{'='*80}")

    print(f"\nBaseline (Kalman + Opponent-Adjusted EPA):")
    print(f"  Correlation with future EPA: {baseline['correlation']:.4f}")
    print(f"  RMSE: {baseline.get('rmse', 'N/A'):.4f}" if 'rmse' in baseline else "  RMSE: N/A")
    print(f"  Sample size: {baseline['n_qbs']} QBs")
    print(f"  [This is our strong baseline from validation]")

    print(f"\nReceiver-Adjusted (on top of Kalman+Opp-Adj):")
    print(f"  Correlation with future EPA: {adjusted['correlation']:.4f}")
    print(f"  RMSE: {adjusted.get('rmse', 'N/A'):.4f}" if 'rmse' in adjusted else "  RMSE: N/A")
    print(f"  Sample size: {adjusted['n_qbs']} QBs")

    improvement = adjusted['correlation'] - baseline['correlation']
    pct_improvement = (improvement / baseline['correlation']) * 100 if baseline['correlation'] > 0 else 0

    print(f"\nImprovement:")
    print(f"  Absolute: {improvement:+.4f}")
    print(f"  Relative: {pct_improvement:+.2f}%")

    # Decision
    print(f"\n{'='*80}")
    print("DECISION FRAMEWORK")
    print(f"{'='*80}")

    if improvement > 0.01:  # 1% absolute threshold
        print("[+] PROCEED: Receiver adjustments IMPROVE strong baseline")
        print(f"  Improvement of {improvement:.4f} exceeds 0.01 threshold")
        print("  Recommendation: Build full teammate adjustment infrastructure (Phase 2)")
    elif improvement > 0 and improvement <= 0.01:
        print("[~] MARGINAL: Small positive improvement")
        print(f"  Improvement of {improvement:.4f} is positive but < 0.01")
        print("  Recommendation: Try alternative method (Option C) before deciding")
    else:
        print("[-] STOP: Receiver adjustments DO NOT improve strong baseline")
        print(f"  Change of {improvement:.4f} does not meet threshold")
        if improvement < -0.005:
            print(f"  WARNING: Actually HURT prediction by {abs(improvement):.4f}")
        print("\n  Possible explanations:")
        print("    1. Receiver quality fully captured by Kalman+Opp-Adj EPA")
        print("    2. Good QBs get good receivers (circular dependency)")
        print("    3. Regression technique wrong for this problem")
        print("\n  Next Step: Try Option C (alternative adjustment method)")
        print("    - Use receiver quality as covariate (not residual)")
        print("    - Try iterative solution (like defense ratings)")
        print("    - Test individual receiver quality (not team aggregate)")

    print("="*80)

    # Save results
    results = pd.DataFrame([
        {'method': 'Kalman+Opp-Adj', **baseline},
        {'method': 'Kalman+Opp-Adj+Receiver', **adjusted}
    ])
    results.to_csv('results/teammate_adjustment_v2_results.csv', index=False)
    logger.info("\nSaved results to results/teammate_adjustment_v2_results.csv")

    return 0


if __name__ == "__main__":
    exit(main())
