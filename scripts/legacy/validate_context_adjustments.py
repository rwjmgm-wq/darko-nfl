"""
Context Adjustment Validation

Tests whether context adjustments (weather, home/away, game script)
improve prediction over the strong Kalman+Opponent-Adjusted baseline.

Key Question:
  Does context-adjusted EPA predict 2023 better than baseline (0.906)?

Methodology:
  - Train on 2020-2022, test on 2023
  - Fit context model on train play-by-play
  - Apply context adjustments to game-level stats
  - Apply opponent adjustments + Kalman filtering
  - Compare correlation with future EPA

Usage:
    python validate_context_adjustments.py --train-years 2020 2021 2022 --test-years 2023
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
from features.opponent_adjustments import DefenseRatingSystem, OpponentAdjuster
from features.kalman_filter import QBKalmanFilter
from features.context_adjustments import ContextAdjustmentModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run context adjustment validation."""
    parser = argparse.ArgumentParser(
        description='Validate context adjustments on strong baseline'
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
    logger.info("Context Adjustment Validation")
    logger.info("="*80)
    logger.info(f"\nTrain: {args.train_years}")
    logger.info(f"Test: {args.test_years}")

    # Fetch data
    fetcher = NFLDataFetcher()
    all_years = args.train_years + args.test_years
    pbp = fetcher.fetch_pbp_data(all_years)

    # Split train/test PBP
    train_pbp = pbp[pbp['season'].isin(args.train_years)].copy()
    test_pbp = pbp[pbp['season'].isin(args.test_years)].copy()

    # Aggregate QB stats (raw, no adjustments yet)
    logger.info("\n[Step 1/7] Aggregating QB statistics...")
    qb_agg = QBStatsAggregator()
    qb_games = qb_agg.aggregate_game_stats(pbp)

    # Fit context model on TRAIN play-by-play only
    logger.info("\n[Step 2/7] Fitting context adjustment model on training data...")
    context_model = ContextAdjustmentModel(alpha=1.0, use_ridge=True)

    try:
        context_model.fit(
            train_pbp,
            metrics=['qb_epa', 'cpoe', 'success']
        )
        logger.info("  Context model fitted successfully!")
    except Exception as e:
        logger.error(f"  Failed to fit context model: {e}")
        logger.info("  Proceeding without context adjustments (baseline only)")
        context_model = None

    # Apply context adjustments to game stats if model exists
    if context_model is not None:
        logger.info("\n[Step 3/7] Applying context adjustments to game stats...")

        try:
            metrics_map = {
                'qb_epa': 'epa_per_play',
                'cpoe': 'cpoe',
                'success': 'success_rate'
            }

            qb_context_adj = context_model.adjust_game_stats(
                game_stats=qb_games,
                pbp_data=pbp,
                metrics_map=metrics_map
            )

            # Create context-adjusted EPA column
            if 'epa_per_play_ctx_adj' in qb_context_adj.columns:
                qb_games = qb_context_adj.copy()
                logger.info("  Context adjustments applied!")
            else:
                logger.warning("  Context adjustment column not found, using raw EPA")
                qb_games['epa_per_play_ctx_adj'] = qb_games['epa_per_play']

        except Exception as e:
            logger.error(f"  Failed to apply context adjustments: {e}")
            logger.info("  Using raw EPA instead")
            qb_games['epa_per_play_ctx_adj'] = qb_games['epa_per_play']
    else:
        qb_games['epa_per_play_ctx_adj'] = qb_games['epa_per_play']

    # Calculate defense ratings
    logger.info("\n[Step 4/7] Calculating defense ratings...")
    defense_system = DefenseRatingSystem(decay_rate=0.996, min_attempts=100, iterations=5)
    defense_ratings = defense_system.calculate_defense_ratings(pbp)

    # Apply opponent adjustments to BOTH raw and context-adjusted EPA
    logger.info("\n[Step 5/7] Applying opponent adjustments...")
    adjuster = OpponentAdjuster(defense_ratings)
    qb_opp_adj = adjuster.adjust_qb_game_stats(qb_games)

    # Also create opponent-adjusted context EPA
    if 'epa_per_play_ctx_adj' in qb_opp_adj.columns:
        # Calculate opponent adjustment for context-adjusted EPA
        qb_opp_adj['opp_adj_context_epa'] = qb_opp_adj['epa_per_play_ctx_adj'] - (
            qb_opp_adj['epa_per_play'] - qb_opp_adj['opp_adj_epa']
        )
    else:
        qb_opp_adj['opp_adj_context_epa'] = qb_opp_adj['opp_adj_epa']

    # Apply Kalman filtering to both metrics
    logger.info("\n[Step 6/7] Applying Kalman filtering...")
    kf = QBKalmanFilter(auto_tune=True, em_iterations=10)

    kalman_metrics = ['epa_per_play', 'opp_adj_epa', 'opp_adj_context_epa']
    kalman_prior_means = {
        'epa_per_play': 0.0,
        'opp_adj_epa': 0.0,
        'opp_adj_context_epa': 0.0
    }

    kalman_stats = kf.process_all_players(
        game_stats=qb_opp_adj,
        metrics=kalman_metrics,
        prior_means=kalman_prior_means
    )

    # Merge Kalman estimates back
    qb_full = qb_opp_adj.merge(
        kalman_stats[['passer_player_id', 'opp_adj_epa_kalman', 'opp_adj_context_epa_kalman']],
        on='passer_player_id',
        how='left'
    )

    # Split train/test
    train_games = qb_full[qb_full['season'].isin(args.train_years)].copy()
    test_games = qb_full[qb_full['season'].isin(args.test_years)].copy()

    logger.info(f"\nTrain games: {len(train_games)}")
    logger.info(f"Test games: {len(test_games)}")

    # Aggregate to season level
    def aggregate_season(games, min_att):
        agg_dict = {
            'passer_player_name': 'first',
            'attempts': 'sum',
            'opp_adj_epa_kalman': 'mean',
        }

        if 'opp_adj_context_epa_kalman' in games.columns:
            agg_dict['opp_adj_context_epa_kalman'] = 'mean'

        result = games.groupby(['passer_player_id', 'season']).agg(agg_dict).reset_index()
        return result.query(f'attempts >= {min_att}')

    train_seasons = aggregate_season(train_games, args.min_attempts)

    logger.info(f"\nTrain QB-seasons: {len(train_seasons)}")

    # Get RAW 2023 EPA as target
    test_raw_epa = qb_games[qb_games['season'].isin(args.test_years)].groupby(
        ['passer_player_id', 'season']
    ).agg({
        'epa_per_play': 'mean',
        'attempts': 'sum'
    }).reset_index().query(f'attempts >= {args.min_attempts}')

    # Test predictive power
    logger.info("\n[Step 7/7] Testing predictive power...")

    def test_metric(train, test_target, metric):
        """Compare train metric against RAW test EPA."""
        if metric not in train.columns:
            logger.warning(f"Metric {metric} not found in train data")
            return {'correlation': np.nan, 'rmse': np.nan, 'n_qbs': 0}

        comparison = train[['passer_player_id', metric]].merge(
            test_target[['passer_player_id', 'epa_per_play']],
            on='passer_player_id',
            how='inner',
            suffixes=('_train', '_test')
        )

        if len(comparison) < 10:
            return {'correlation': np.nan, 'rmse': np.nan, 'n_qbs': len(comparison)}

        pred_col = f'{metric}_train' if f'{metric}_train' in comparison.columns else metric
        actual_col = 'epa_per_play_test' if 'epa_per_play_test' in comparison.columns else 'epa_per_play'

        pred = comparison[pred_col]
        actual = comparison[actual_col]

        valid = ~(pred.isna() | actual.isna())
        pred = pred[valid]
        actual = actual[valid]

        if len(pred) < 10:
            return {'correlation': np.nan, 'rmse': np.nan, 'n_qbs': len(pred)}

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

    # Context-adjusted: Kalman+Opponent+Context EPA
    context_adjusted = test_metric(train_seasons, test_raw_epa, 'opp_adj_context_epa_kalman')

    # Results
    print(f"\n{'='*80}")
    print("Context Adjustment Validation Results")
    print(f"{'='*80}")

    print(f"\nBaseline (Kalman + Opponent-Adjusted EPA):")
    print(f"  Correlation with future EPA: {baseline['correlation']:.4f}")
    print(f"  RMSE: {baseline.get('rmse', 0):.4f}")
    print(f"  Sample size: {baseline['n_qbs']} QBs")
    print(f"  [This is our strong baseline: 0.886]")

    print(f"\nContext-Adjusted (Kalman + Opponent + Context):")
    print(f"  Correlation with future EPA: {context_adjusted['correlation']:.4f}")
    print(f"  RMSE: {context_adjusted.get('rmse', 0):.4f}")
    print(f"  Sample size: {context_adjusted['n_qbs']} QBs")

    improvement = context_adjusted['correlation'] - baseline['correlation']
    pct_improvement = (improvement / baseline['correlation']) * 100 if baseline['correlation'] > 0 else 0

    print(f"\nImprovement:")
    print(f"  Absolute: {improvement:+.4f}")
    print(f"  Relative: {pct_improvement:+.2f}%")

    # Decision
    print(f"\n{'='*80}")
    print("DECISION FRAMEWORK")
    print(f"{'='*80}")

    if improvement > 0.005:
        print("[+] SUCCESS: Context adjustments IMPROVE prediction")
        print(f"  Improvement of {improvement:.4f} exceeds 0.005 threshold")
        print("  Recommendation: Integrate context adjustments into LEAF pipeline")
    elif improvement > 0 and improvement <= 0.005:
        print("[~] MARGINAL: Small positive improvement")
        print(f"  Improvement of {improvement:.4f} is positive but < 0.005")
        print("  Recommendation: Context effects may already be captured")
        print("  Consider: Keep system simpler without explicit context adjustments")
    else:
        print("[-] NO IMPROVEMENT: Context adjustments do not help")
        print(f"  Change of {improvement:.4f} does not meet threshold")
        if improvement < -0.005:
            print(f"  WARNING: Actually HURT prediction by {abs(improvement):.4f}")
        print("\n  Conclusion: Context effects already captured by Kalman+Opponent")
        print("  Recommendation: Skip context adjustments to keep system simpler")

    print("="*80)

    # Save results
    results = pd.DataFrame([
        {'method': 'Kalman+Opp-Adj (Baseline)', **baseline},
        {'method': 'Kalman+Opp+Context', **context_adjusted}
    ])
    Path('results').mkdir(exist_ok=True)
    results.to_csv('results/context_adjustment_validation.csv', index=False)
    logger.info("\nSaved results to results/context_adjustment_validation.csv")

    return 0


if __name__ == "__main__":
    exit(main())
