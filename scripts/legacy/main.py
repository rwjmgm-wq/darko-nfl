"""
LEAF - Layered EPA Adaptive Framework

Main script to run the complete LEAF NFL QB evaluation and projection system.

Usage:
    python main.py --years 2022 2023 2024 --output results/qb_projections.csv
"""

import argparse
import pandas as pd
from pathlib import Path
import logging
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_pipeline.nfl_data_fetcher import NFLDataFetcher
from data_pipeline.qb_stats_aggregator import QBStatsAggregator
from features.time_weighted_metrics import TimeWeightedMetrics
from features.opponent_adjustments import DefenseRatingSystem, OpponentAdjuster
from features.kalman_filter import QBKalmanFilter
from features.context_adjustments import ContextAdjustmentModel
from features.situational_splits import SituationalSplits
from models.age_curves import AgeCurveModel, SimpleAgeCurve
from projections.bayesian_projections import BayesianProjection, DARKOProjectionSystem
from projections.composite_metric import LEAFCompositeMetric
from utils.config_loader import get_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = ['data/raw', 'data/processed', 'data/cache', 'results']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='LEAF QB Projection System'
    )
    parser.add_argument(
        '--years',
        type=int,
        nargs='+',
        default=[2022, 2023, 2024],
        help='Years to include in analysis'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/qb_projections.csv',
        help='Output file for projections'
    )
    parser.add_argument(
        '--min-attempts',
        type=int,
        default=100,
        help='Minimum attempts to include a QB'
    )
    parser.add_argument(
        '--refresh',
        action='store_true',
        help='Force refresh of cached data'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--situational-splits',
        action='store_true',
        help='Generate detailed situational splits analysis'
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("LEAF - QB Projection System")
    logger.info("=" * 60)

    # Setup
    setup_directories()
    config = get_config(args.config)

    # Step 1: Fetch Data
    logger.info("\n[Step 1/12] Fetching NFL data...")
    fetcher = NFLDataFetcher(cache_dir=config.get('data.cache_dir', 'data/cache'))

    try:
        qb_pbp = fetcher.get_qb_play_by_play(args.years, force_refresh=args.refresh)
        rosters = fetcher.fetch_rosters(args.years, force_refresh=args.refresh)
        logger.info(f"  ✓ Loaded {len(qb_pbp):,} QB passing plays")
    except Exception as e:
        logger.error(f"  ✗ Error fetching data: {e}")
        return 1

    # Step 2: Aggregate Game Stats
    logger.info("\n[Step 2/12] Aggregating QB statistics...")
    aggregator = QBStatsAggregator()
    game_stats = aggregator.aggregate_game_stats(qb_pbp)
    logger.info(f"  ✓ Aggregated {len(game_stats)} QB-game combinations")

    # Save game stats
    game_stats_path = Path('data/processed/qb_game_stats.csv')
    game_stats.to_csv(game_stats_path, index=False)
    logger.info(f"  ✓ Saved to {game_stats_path}")

    # Step 3: Calculate Defense Ratings & Opponent Adjustments
    logger.info("\n[Step 3/12] Calculating defense ratings (5 iterations)...")
    defense_system = DefenseRatingSystem(
        decay_rate=config.get('decay.default', 0.996),
        min_attempts=100,
        iterations=5
    )
    defense_ratings = defense_system.calculate_defense_ratings(qb_pbp)

    # Save defense rankings
    defense_rankings = defense_system.get_defense_rankings('epa_allowed', ascending=True)
    defense_path = Path('results/defense_rankings.csv')
    defense_rankings.to_csv(defense_path, index=False)
    logger.info(f"  ✓ Calculated ratings for {len(defense_ratings)} defenses")
    logger.info(f"  ✓ Saved defense rankings to {defense_path}")

    # Show top/bottom defenses
    print(f"\n{'='*60}")
    print("Top 5 Defenses (EPA allowed)")
    print(f"{'='*60}")
    print(defense_rankings[['rank', 'team', 'epa_allowed', 'attempts_faced']].head(5).to_string(index=False))

    print(f"\n{'='*60}")
    print("Bottom 5 Defenses (EPA allowed)")
    print(f"{'='*60}")
    print(defense_rankings[['rank', 'team', 'epa_allowed', 'attempts_faced']].tail(5).to_string(index=False))

    # Step 4: Apply Opponent Adjustments
    logger.info("\n[Step 4/12] Applying opponent adjustments to QB stats...")
    adjuster = OpponentAdjuster(defense_ratings)
    game_stats = adjuster.adjust_qb_game_stats(game_stats)

    # Save opponent-adjusted game stats
    adjusted_game_stats_path = Path('data/processed/qb_game_stats_opp_adj.csv')
    game_stats.to_csv(adjusted_game_stats_path, index=False)

    # Calculate strength of schedule
    sos = adjuster.get_strength_of_schedule(game_stats)
    sos_path = Path('results/strength_of_schedule.csv')
    sos.to_csv(sos_path, index=False)
    logger.info(f"  ✓ Applied opponent adjustments")
    logger.info(f"  ✓ Saved opponent-adjusted stats to {adjusted_game_stats_path}")
    logger.info(f"  ✓ Saved strength of schedule to {sos_path}")

    # Step 5: Apply Context Adjustments (Weather, Game Script, Situation)
    logger.info("\n[Step 5/12] Fitting context adjustment models...")
    context_model = ContextAdjustmentModel(
        alpha=config.get('context.ridge_alpha', 1.0),
        use_ridge=config.get('context.use_ridge', True)
    )

    # Fit on all available data for better context estimates
    context_model.fit(
        qb_pbp,
        metrics=['qb_epa', 'cpoe', 'success']
    )

    # Apply context adjustments to game stats
    metrics_map = {
        'qb_epa': 'epa_per_play',
        'cpoe': 'cpoe_mean',
        'success': 'success_mean'
    }

    game_stats = context_model.adjust_game_stats(game_stats, qb_pbp, metrics_map)

    # Save context-adjusted game stats
    context_adjusted_path = Path('data/processed/qb_game_stats_ctx_adj.csv')
    game_stats.to_csv(context_adjusted_path, index=False)
    logger.info(f"  ✓ Applied context adjustments")
    logger.info(f"  ✓ Saved context-adjusted stats to {context_adjusted_path}")

    # Show top context effects
    print(f"\n{'='*60}")
    print("Top Weather/Context Effects on EPA")
    print(f"{'='*60}")
    print(context_model.get_feature_importance('qb_epa', top_n=5).to_string(index=False))

    # Optional: Situational Splits Analysis
    if args.situational_splits or config.get('analysis.situational_splits', False):
        logger.info("\n[Analysis] Generating situational splits...")

        # Initialize analyzer
        splits_analyzer = SituationalSplits(
            min_attempts_per_situation=config.get('analysis.min_attempts_situation', 10)
        )

        # Calculate league averages
        splits_analyzer.calculate_league_averages(qb_pbp, metrics=['qb_epa', 'cpoe', 'success'])

        # Calculate QB splits
        qb_splits = splits_analyzer.calculate_qb_splits(qb_pbp, metrics=['qb_epa', 'cpoe', 'success'])

        # Save detailed splits
        splits_path = Path('results/qb_situational_splits.csv')
        qb_splits.to_csv(splits_path, index=False)
        logger.info(f"  ✓ Saved situational splits to {splits_path}")

        # Generate summary
        summary = splits_analyzer.get_summary_stats(qb_splits, min_attempts=args.min_attempts)
        summary_path = Path('results/situational_summary.csv')
        summary.to_csv(summary_path, index=False)
        logger.info(f"  ✓ Saved situational summary to {summary_path}")

        # Show example: top QBs by situational consistency
        print(f"\n{'='*60}")
        print("Top 10 QBs by Situations Tracked (Consistency)")
        print(f"{'='*60}")
        print(summary.nlargest(10, 'situations_tracked')[['player_name', 'attempts', 'situations_tracked', 'qb_epa_vs_avg']].to_string(index=False))

    # Step 6: Apply Kalman Filtering
    logger.info("\n[Step 6/12] Applying Kalman filter for adaptive learning...")
    kf = QBKalmanFilter(
        auto_tune=config.get('kalman.auto_tune', True),
        em_iterations=config.get('kalman.em_iterations', 10),
        initial_process_noise=config.get('kalman.process_noise', 0.02),
        initial_observation_noise=config.get('kalman.observation_noise', 0.15)
    )

    # Apply Kalman filter to both raw and opponent-adjusted metrics
    kalman_metrics = ['epa_per_play', 'cpoe_mean', 'success_mean',
                      'opp_adj_epa', 'opp_adj_cpoe', 'opp_adj_success_rate']

    kalman_prior_means = {
        'epa_per_play': config.get('priors.league_avg_epa', 0.0),
        'cpoe_mean': config.get('priors.league_avg_cpoe', 0.0),
        'success_mean': config.get('priors.league_avg_success', 0.5),
        'opp_adj_epa': 0.0,
        'opp_adj_cpoe': 0.0,
        'opp_adj_success_rate': 0.0
    }

    kalman_stats = kf.process_all_players(
        game_stats=game_stats,
        metrics=kalman_metrics,
        prior_means=kalman_prior_means
    )

    # Save Kalman-filtered stats
    kalman_stats_path = Path('data/processed/qb_kalman_filtered.csv')
    kalman_stats.to_csv(kalman_stats_path, index=False)
    logger.info(f"  ✓ Applied Kalman filtering to {len(kalman_stats)} QBs")
    logger.info(f"  ✓ Saved Kalman-filtered stats to {kalman_stats_path}")

    # Step 7: Calculate Time-Weighted Metrics
    logger.info("\n[Step 7/12] Calculating time-weighted metrics...")
    decay_rates = {
        'epa_per_play': config.get('decay.epa_beta', 0.996),
        'cpoe_mean': config.get('decay.cpoe_beta', 0.995),
        'success_mean': config.get('decay.success_beta', 0.996),
        'completion_pct': 0.997,
        'yards_per_attempt': 0.996,
        'opp_adj_epa': 0.996,  # Opponent-adjusted metrics
        'opp_adj_cpoe': 0.995,
        'opp_adj_success_rate': 0.996,
        'default': 0.996
    }
    twm = TimeWeightedMetrics(decay_rates=decay_rates)

    weighted_stats = twm.aggregate_player_metrics(
        game_stats,
        metrics=['epa_per_play', 'cpoe_mean', 'success_mean',
                'completion_pct', 'yards_per_attempt',
                'opp_adj_epa', 'opp_adj_cpoe', 'opp_adj_success_rate']
    )

    # Rename for consistency with later code
    weighted_stats = weighted_stats.rename(columns={
        'cpoe_mean_weighted': 'cpoe_weighted',
        'success_mean_weighted': 'success_rate_weighted'
    })
    logger.info(f"  ✓ Calculated weighted metrics for {len(weighted_stats)} QBs")

    # Step 8: Fit Age Curves
    logger.info("\n[Step 8/12] Fitting age curve models...")

    # Prepare data with ages
    game_stats_with_age = game_stats.merge(
        rosters[['player_id', 'season', 'age']].drop_duplicates(),
        left_on=['passer_player_id', 'season'],
        right_on=['player_id', 'season'],
        how='left'
    )

    # Get season-level stats for age curve fitting
    season_stats = game_stats_with_age.groupby(
        ['passer_player_id', 'passer_player_name', 'season', 'age']
    ).agg({
        'attempts': 'sum',
        'epa_per_play': 'mean',
        'cpoe_mean': 'mean',
        'success_mean': 'mean',
    }).reset_index()

    #  Rename for consistency
    season_stats = season_stats.rename(columns={
        'cpoe_mean': 'cpoe',
        'success_mean': 'success_rate'
    })

    age_model = AgeCurveModel(
        degree=3,
        peak_age=config.get('age_curves.peak_age', 28),
        min_attempts=config.get('age_curves.min_attempts', 100)
    )

    age_model.fit(
        season_stats,
        metrics=['epa_per_play', 'cpoe', 'success_rate'],
        age_col='age'
    )
    logger.info("  ✓ Fitted age curves for key metrics")

    # Step 9: Merge Age Information and Kalman Results
    logger.info("\n[Step 9/12] Merging age information and Kalman results...")
    latest_ages = rosters.groupby('player_id')['season'].max().reset_index()
    latest_ages = latest_ages.merge(
        rosters[['player_id', 'season', 'age']],
        on=['player_id', 'season']
    )

    weighted_stats = weighted_stats.merge(
        latest_ages[['player_id', 'age']],
        left_on='passer_player_id',
        right_on='player_id',
        how='left'
    )

    # Merge Kalman-filtered results with weighted stats
    weighted_stats = weighted_stats.merge(
        kalman_stats[[col for col in kalman_stats.columns if col != 'player_name' and col != 'total_attempts' and col != 'games_played']],
        on='passer_player_id',
        how='left'
    )
    logger.info("  ✓ Added age information and merged Kalman-filtered estimates")

    # Step 10: Generate Bayesian Projections
    logger.info("\n[Step 10/12] Generating Bayesian projections (using Kalman-filtered estimates)...")

    prior_means = {
        'epa_per_play': config.get('priors.league_avg_epa', 0.0),
        'cpoe': config.get('priors.league_avg_cpoe', 0.0),
        'success_rate': config.get('priors.league_avg_success', 0.5),
    }
    prior_weight = config.get('priors.prior_weight', 200)

    bayesian = BayesianProjection(
        prior_means=prior_means,
        prior_weight=prior_weight,
        confidence_level=config.get('projections.confidence_level', 0.95)
    )

    # Apply age adjustments and generate projections
    projections_list = []

    for _, player in weighted_stats.iterrows():
        if player['total_attempts'] < args.min_attempts:
            continue

        # Calculate age adjustments
        age_adjustments = {}
        if pd.notna(player.get('age')):
            for metric in ['epa_per_play', 'cpoe', 'success_rate']:
                age_adjustments[metric] = age_model.get_age_adjustment_factor(
                    int(player['age']), metric
                )

        # Generate projection
        proj = bayesian.project_player(
            player,
            metrics=['epa_per_play', 'cpoe', 'success_rate'],
            age_adjustment=age_adjustments
        )

        # Add player info
        proj['passer_player_id'] = player['passer_player_id']
        proj['player_name'] = player.get('player_name', '')
        proj['age'] = player.get('age', None)
        proj['total_attempts'] = player['total_attempts']

        projections_list.append(proj)

    projections = pd.concat(projections_list, ignore_index=True)
    logger.info(f"  ✓ Generated projections for {len(weighted_stats)} QBs")

    # Step 11: Calculate LEAF Rating
    logger.info("\n[Step 11/12] Calculating LEAF Rating...")

    # Load situational stats if available
    situational_data = None
    if args.situational_splits or config.get('analysis.situational_splits', False):
        try:
            situational_data = pd.read_csv('results/qb_situational_splits.csv')
        except:
            logger.warning("  Situational splits not found, calculating LEAF without situational component")

    # Initialize LEAF calculator
    leaf_calculator = LEAFCompositeMetric(
        component_weights=config.get('composite.weights', None),
        use_kalman_estimates=True,
        use_opponent_adjusted=True
    )

    # Calculate LEAF
    leaf_results = leaf_calculator.calculate_leaf(
        weighted_stats,
        situational_data,
        return_components=True
    )

    # Convert to per-game values
    leaf_results = leaf_calculator.convert_to_per_game(
        leaf_results,
        plays_per_game=config.get('composite.plays_per_game', 35.0)
    )

    # Calculate WAR
    leaf_results = leaf_calculator.calculate_wins_above_replacement(
        leaf_results,
        replacement_level_percentile=config.get('composite.replacement_percentile', 25.0),
        points_per_win=config.get('composite.points_per_win', 25.0)
    )

    # Rank by LEAF
    leaf_rankings = leaf_calculator.rank_players(leaf_results, min_attempts=args.min_attempts)

    # Save LEAF rankings
    leaf_path = Path('results/leaf_ratings.csv')
    leaf_rankings.to_csv(leaf_path, index=False)
    logger.info(f"  ✓ Saved LEAF Ratings to {leaf_path}")

    # Display top QBs by LEAF
    print(f"\n{'='*80}")
    print("LEAF Ratings - Top 10 QBs (Unified Composite Metric)")
    print(f"{'='*80}")
    display_cols = [col for col in ['rank', 'player_name', 'age', 'total_attempts',
                    'leaf', 'leaf_per_game', 'war', 'leaf_tier'] if col in leaf_rankings.columns]
    print(leaf_rankings[display_cols].head(10).to_string(index=False))

    # Step 12: Save Component Rankings
    logger.info("\n[Step 12/12] Saving component metric rankings...")

    # Create rankings for each metric
    for metric in ['epa_per_play', 'cpoe', 'success_rate']:
        metric_proj = projections[projections['metric'] == metric].copy()
        metric_proj = metric_proj.sort_values('projection', ascending=False)
        metric_proj['rank'] = range(1, len(metric_proj) + 1)

        # Save to file
        output_file = Path(args.output).parent / f"{metric}_rankings.csv"
        metric_proj.to_csv(output_file, index=False)
        logger.info(f"  ✓ Saved {metric} rankings to {output_file}")

        # Show top 10
        print(f"\n{'='*60}")
        print(f"Top 10 QBs by {metric.upper()}")
        print(f"{'='*60}")
        print(metric_proj[['rank', 'player_name', 'age', 'total_attempts',
                          'projection', 'lower_ci', 'upper_ci', 'uncertainty']].head(10).to_string(index=False))

    # Save complete projections
    output_path = Path(args.output)
    projections.to_csv(output_path, index=False)
    logger.info(f"\n  ✓ Saved complete projections to {output_path}")

    logger.info("\n" + "=" * 60)
    logger.info("LEAF QB Projections Complete!")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
