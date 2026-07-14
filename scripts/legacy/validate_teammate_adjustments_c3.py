"""
Teammate Adjustment Validation - Option C3: Direct Covariate Approach

Tests whether receiver quality works better as a DIRECT PREDICTOR
instead of trying to adjust it out.

Key Difference from V2/C1/C2:
  V2/C1/C2: Try to isolate "true QB talent" by subtracting receiver effects
  C3: Use receiver quality as additional feature in prediction model

Model:
  Future QB EPA ~ Past QB EPA + Past Receiver Quality

Rationale:
  - Maybe receiver quality SHOULD be part of QB prediction
  - Good QBs often have good receivers (by design)
  - Let the model learn optimal weights instead of forcing residuals

Methodology:
  - Train Ridge regression: Next Year EPA ~ This Year EPA + Receiver Quality
  - Test: Does adding receiver quality improve prediction?

Usage:
    python validate_teammate_adjustments_c3.py --train-years 2020 2021 2022 --test-years 2023
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

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
    """Run Option C3 validation."""
    parser = argparse.ArgumentParser(
        description='Validate direct covariate approach (Option C3)'
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
    logger.info("Teammate Adjustment Validation - Option C3")
    logger.info("Direct Covariate Approach (not residual)")
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
    train_games = qb_full[qb_full['season'].isin(args.train_years)].copy()
    test_games = qb_full[qb_full['season'].isin(args.test_years)].copy()

    logger.info(f"\nTrain QB games: {len(train_games)}")
    logger.info(f"Test QB games: {len(test_games)}")

    # Aggregate to season level
    logger.info("\n[Step 6/6] Testing direct covariate approach...")

    def aggregate_season(games, min_att):
        return games.groupby(['passer_player_id', 'season']).agg({
            'passer_player_name': 'first',
            'attempts': 'sum',
            'opp_adj_epa_kalman': 'mean',
            'receiver_corps_epa': 'mean',
            'receiver_corps_catch_rate': 'mean',
            'receiver_corps_yards_per_target': 'mean'
        }).reset_index().query(f'attempts >= {min_att}')

    train_seasons = aggregate_season(train_games, args.min_attempts)
    test_seasons = aggregate_season(test_games, args.min_attempts)

    logger.info(f"\nTrain QB-seasons: {len(train_seasons)}")
    logger.info(f"Test QB-seasons: {len(test_seasons)}")

    # Get RAW 2023 EPA as target
    test_raw_epa = qb_games[qb_games['season'].isin(args.test_years)].groupby(
        ['passer_player_id', 'season']
    ).agg({
        'epa_per_play': 'mean',
        'attempts': 'sum'
    }).reset_index().query(f'attempts >= {args.min_attempts}')

    # Prepare features for prediction
    # Model: Future EPA ~ Past EPA + Receiver Quality
    feature_cols = ['opp_adj_epa_kalman', 'receiver_corps_epa',
                    'receiver_corps_catch_rate', 'receiver_corps_yards_per_target']

    # Merge train with future (next year) EPA
    train_with_future = []
    for _, row in train_seasons.iterrows():
        qb_id = row['passer_player_id']
        season = row['season']
        next_season = season + 1

        # Find next season performance
        next_perf = qb_games[
            (qb_games['passer_player_id'] == qb_id) &
            (qb_games['season'] == next_season)
        ].groupby('passer_player_id').agg({'epa_per_play': 'mean', 'attempts': 'sum'}).reset_index()

        if len(next_perf) > 0 and next_perf['attempts'].iloc[0] >= args.min_attempts:
            train_with_future.append({
                'passer_player_id': qb_id,
                'season': season,
                **{col: row[col] for col in feature_cols},
                'future_epa': next_perf['epa_per_play'].iloc[0]
            })

    train_df = pd.DataFrame(train_with_future)

    if len(train_df) == 0:
        logger.error("No valid training data with future seasons!")
        return 1

    logger.info(f"\nYear-to-year pairs available: {len(train_df)}")

    # Train models
    X_train = train_df[feature_cols].values
    y_train = train_df['future_epa'].values

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Model 1: Baseline (EPA only)
    model_baseline = Ridge(alpha=1.0)
    model_baseline.fit(X_train[:, [0]], y_train)  # Only opp_adj_epa_kalman

    # Model 2: EPA + Receiver Quality
    model_c3 = Ridge(alpha=1.0)
    model_c3.fit(X_train_scaled, y_train)

    logger.info(f"\nModel coefficients (standardized):")
    for i, col in enumerate(feature_cols):
        logger.info(f"  {col}: {model_c3.coef_[i]:.4f}")

    # Test on 2023
    # Merge 2022 train data with 2023 actual
    test_2022 = train_seasons[train_seasons['season'] == max(args.train_years)]
    test_2023_actual = test_raw_epa[test_raw_epa['season'] == min(args.test_years)]

    test_merged = test_2022.merge(
        test_2023_actual[['passer_player_id', 'epa_per_play']],
        on='passer_player_id',
        how='inner'
    )

    if len(test_merged) < 10:
        logger.error(f"Not enough test data: {len(test_merged)} QBs")
        return 1

    logger.info(f"\n2022->2023 test sample: {len(test_merged)} QBs")

    X_test = test_merged[feature_cols].values
    y_test = test_merged['epa_per_play'].values

    X_test_scaled = scaler.transform(X_test)

    # Predictions
    pred_baseline = model_baseline.predict(X_test[:, [0]])
    pred_c3 = model_c3.predict(X_test_scaled)

    # Evaluate
    corr_baseline = np.corrcoef(pred_baseline, y_test)[0, 1]
    rmse_baseline = np.sqrt(((pred_baseline - y_test) ** 2).mean())

    corr_c3 = np.corrcoef(pred_c3, y_test)[0, 1]
    rmse_c3 = np.sqrt(((pred_c3 - y_test) ** 2).mean())

    # Results
    print(f"\n{'='*80}")
    print("Option C3: Direct Covariate Approach")
    print(f"{'='*80}")

    print(f"\nBaseline (EPA only):")
    print(f"  Correlation with future EPA: {corr_baseline:.4f}")
    print(f"  RMSE: {rmse_baseline:.4f}")
    print(f"  Sample size: {len(test_merged)} QBs")

    print(f"\nOption C3 (EPA + Receiver Quality as covariates):")
    print(f"  Correlation with future EPA: {corr_c3:.4f}")
    print(f"  RMSE: {rmse_c3:.4f}")
    print(f"  Sample size: {len(test_merged)} QBs")

    improvement = corr_c3 - corr_baseline
    pct_improvement = (improvement / corr_baseline) * 100 if corr_baseline > 0 else 0

    print(f"\nImprovement:")
    print(f"  Absolute: {improvement:+.4f}")
    print(f"  Relative: {pct_improvement:+.2f}%")

    # Decision
    print(f"\n{'='*80}")
    print("OPTION C3 EVALUATION")
    print(f"{'='*80}")

    if improvement > 0.01:
        print("[+] SUCCESS: Direct covariate approach IMPROVES prediction")
        print(f"  Improvement of {improvement:.4f} exceeds 0.01 threshold")
        print("  Recommendation: Use receiver quality as direct feature, not residual")
    elif improvement > 0 and improvement <= 0.01:
        print("[~] MARGINAL: Small improvement")
        print(f"  Improvement of {improvement:.4f} is positive but < 0.01")
        print("  Recommendation: Marginal benefit, may not be worth complexity")
    else:
        print("[-] FAILED: Direct covariate does not improve")
        print(f"  Change of {improvement:.4f} does not meet threshold")
        if improvement < -0.005:
            print(f"  WARNING: Actually HURT prediction by {abs(improvement):.4f}")
        print("\n  CONCLUSION: Teammate adjustments do NOT improve this baseline")
        print("  Recommendation: Do not pursue multi-position LEAF with current method")

    print("="*80)

    # Save results
    results = pd.DataFrame([
        {'method': 'Baseline (EPA only)', 'correlation': corr_baseline, 'rmse': rmse_baseline, 'n_qbs': len(test_merged)},
        {'method': 'C3: Direct Covariate', 'correlation': corr_c3, 'rmse': rmse_c3, 'n_qbs': len(test_merged)}
    ])
    Path('results').mkdir(exist_ok=True)
    results.to_csv('results/teammate_adjustment_c3_results.csv', index=False)
    logger.info("\nSaved results to results/teammate_adjustment_c3_results.csv")

    return 0


if __name__ == "__main__":
    exit(main())
