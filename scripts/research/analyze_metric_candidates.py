"""
Phase 1: Metric Candidate Selection

Analyzes all available WR metrics to select the best 15-20 candidates
for exhaustive 5-metric composite testing.

Evaluates metrics by:
- Year-over-year predictive power (2023→2024 correlation)
- Data coverage (percentage non-null)
- Inter-metric correlation (identifies redundancy)
- WR-specificity (lower QB dependence)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_pipeline.nfl_data_fetcher import NFLDataFetcher
from data_pipeline.receiver_stats_aggregator import ReceiverStatsAggregator
from features.wr_leaf_pipeline import WRLEAFPipeline

import pandas as pd
import numpy as np
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_individual_correlations(
    ratings_2023: pd.DataFrame,
    ratings_2024: pd.DataFrame,
    metrics: list
) -> pd.DataFrame:
    """
    Calculate year-over-year correlation for each metric.

    Args:
        ratings_2023: 2023 season ratings
        ratings_2024: 2024 season ratings
        metrics: List of metric names to analyze

    Returns:
        DataFrame with correlation statistics for each metric
    """
    results = []

    for metric in metrics:
        # Check if metric exists in both years
        if metric not in ratings_2023.columns or metric not in ratings_2024.columns:
            logger.debug(f"  Metric {metric} not found in data, skipping")
            continue

        # Merge on receiver_player_id
        merged = ratings_2023[['receiver_player_id', metric]].merge(
            ratings_2024[['receiver_player_id', metric]],
            on='receiver_player_id',
            suffixes=('_2023', '_2024')
        )

        # Remove rows with NaN in either year
        merged = merged.dropna()

        if len(merged) < 10:
            logger.debug(f"  Metric {metric} has insufficient data ({len(merged)} players), skipping")
            continue

        # Calculate correlation
        try:
            correlation, p_value = stats.pearsonr(
                merged[f'{metric}_2023'],
                merged[f'{metric}_2024']
            )

            results.append({
                'metric': metric,
                'correlation': correlation,
                'p_value': p_value,
                'n_players': len(merged),
                'mean_2023': merged[f'{metric}_2023'].mean(),
                'std_2023': merged[f'{metric}_2023'].std(),
                'mean_2024': merged[f'{metric}_2024'].mean(),
                'std_2024': merged[f'{metric}_2024'].std(),
                'coverage': len(merged) / max(len(ratings_2023), len(ratings_2024))
            })
        except Exception as e:
            logger.debug(f"  Error calculating correlation for {metric}: {e}")
            continue

    results_df = pd.DataFrame(results).sort_values('correlation', ascending=False)
    return results_df


def calculate_inter_metric_correlations(
    game_stats: pd.DataFrame,
    metrics: list
) -> pd.DataFrame:
    """
    Calculate correlation matrix between all metrics.
    Identifies redundant/highly correlated metrics.

    Args:
        game_stats: Game-level statistics
        metrics: List of metric names

    Returns:
        Correlation matrix DataFrame
    """
    # Filter to metrics that exist
    available_metrics = [m for m in metrics if m in game_stats.columns]

    # Calculate correlation matrix
    corr_matrix = game_stats[available_metrics].corr()

    return corr_matrix


def score_metrics(
    individual_corrs: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    weights: dict = None
) -> pd.DataFrame:
    """
    Score metrics based on multiple criteria.

    Args:
        individual_corrs: Individual year-over-year correlations
        corr_matrix: Inter-metric correlation matrix
        weights: Dict of scoring weights (default: equal)

    Returns:
        DataFrame with composite scores
    """
    if weights is None:
        weights = {
            'predictive_power': 0.50,  # Year-over-year correlation
            'uniqueness': 0.30,        # Low inter-metric correlation
            'significance': 0.10,       # Statistical significance
            'coverage': 0.10           # Data availability
        }

    scored = individual_corrs.copy()

    # 1. Predictive power (already have correlation)
    scored['predictive_power_score'] = scored['correlation'].abs()

    # 2. Uniqueness (average correlation with other metrics)
    uniqueness_scores = []
    for metric in scored['metric']:
        if metric in corr_matrix.columns:
            # Get average absolute correlation with other metrics
            other_corrs = corr_matrix[metric].drop(metric, errors='ignore').abs()
            avg_corr = other_corrs.mean()
            # Lower correlation = more unique = higher score
            uniqueness_scores.append(1 - avg_corr)
        else:
            uniqueness_scores.append(0.5)  # Neutral score

    scored['uniqueness_score'] = uniqueness_scores

    # 3. Statistical significance
    scored['significance_score'] = (scored['p_value'] < 0.05).astype(float)

    # 4. Coverage
    scored['coverage_score'] = scored['coverage']

    # 5. Composite score
    scored['composite_score'] = (
        weights['predictive_power'] * scored['predictive_power_score'] +
        weights['uniqueness'] * scored['uniqueness_score'] +
        weights['significance'] * scored['significance_score'] +
        weights['coverage'] * scored['coverage_score']
    )

    return scored.sort_values('composite_score', ascending=False)


def main():
    """Run Phase 1: Metric candidate selection."""

    logger.info("="*80)
    logger.info("PHASE 1: METRIC CANDIDATE SELECTION")
    logger.info("="*80)

    # ==========================================
    # Load Data
    # ==========================================
    logger.info("\n[1/5] Loading data...")
    fetcher = NFLDataFetcher()
    pbp_2023 = fetcher.fetch_pbp_data([2023])
    pbp_2024 = fetcher.fetch_pbp_data([2024])
    pbp_full = fetcher.fetch_pbp_data([2023, 2024])

    # Fetch roster data for WR-only filtering
    rosters = fetcher.fetch_rosters([2023, 2024])

    aggregator = ReceiverStatsAggregator(min_targets=1, filter_wr_only=True)
    game_stats_2023 = aggregator.aggregate_game_stats(pbp_2023, roster_data=rosters)
    game_stats_2024 = aggregator.aggregate_game_stats(pbp_2024, roster_data=rosters)
    game_stats_full = aggregator.aggregate_game_stats(pbp_full, roster_data=rosters)

    logger.info(f"  2023: {len(game_stats_2023):,} receiver-games")
    logger.info(f"  2024: {len(game_stats_2024):,} receiver-games")

    # ==========================================
    # Process through WR LEAF pipeline
    # ==========================================
    logger.info("\n[2/5] Processing through WR LEAF pipeline...")
    pipeline = WRLEAFPipeline(
        use_context=True,
        use_opponent_adj=True,
        use_kalman=True
    )

    # Fit on full data for context/opponent adjustments
    pipeline.fit_context_model(pbp_full, metrics=['epa', 'air_epa', 'yac_epa'])
    secondary_ratings = pipeline.calculate_secondary_ratings(pbp_full)

    # Get season-level stats (these have all the base metrics)
    season_stats_2023 = aggregator.aggregate_season_stats(game_stats_2023, min_targets=30)
    season_stats_2024 = aggregator.aggregate_season_stats(game_stats_2024, min_targets=30)

    logger.info(f"  2023: {len(season_stats_2023)} WRs (30+ targets)")
    logger.info(f"  2024: {len(season_stats_2024)} WRs (30+ targets)")

    # Also get pipeline outputs for comparison
    processed_2023 = pipeline.process_full_pipeline(
        game_stats=game_stats_2023,
        pbp_data=pbp_2023,
        secondary_ratings=secondary_ratings
    )
    ratings_2023_pipeline = pipeline.get_current_ratings(processed_2023, min_targets=30)

    processed_2024 = pipeline.process_full_pipeline(
        game_stats=game_stats_2024,
        pbp_data=pbp_2024,
        secondary_ratings=secondary_ratings
    )
    ratings_2024_pipeline = pipeline.get_current_ratings(processed_2024, min_targets=30)

    # Merge season stats with pipeline ratings
    ratings_2023 = season_stats_2023.merge(
        ratings_2023_pipeline[['receiver_player_id', 'wr_leaf_rating', 'wr_leaf_uncertainty']],
        on='receiver_player_id',
        how='left'
    )

    ratings_2024 = season_stats_2024.merge(
        ratings_2024_pipeline[['receiver_player_id', 'wr_leaf_rating', 'wr_leaf_uncertainty']],
        on='receiver_player_id',
        how='left'
    )

    # ==========================================
    # Define candidate metrics
    # ==========================================
    logger.info("\n[3/5] Analyzing candidate metrics...")

    # All potential metrics to test (from season_stats)
    candidate_metrics = [
        # Core EPA metrics
        'air_epa_per_target',
        'yac_epa_per_target',
        'epa_per_target',
        'total_air_epa',
        'total_yac_epa',
        'total_epa',

        # Success/efficiency
        'success_rate',
        'catch_rate',
        'yards_per_target',
        'yards_per_reception',
        'racr',

        # Volume metrics
        'targets',
        'receptions',
        'yards',
        'touchdowns',
        'first_downs',

        # Advanced metrics
        'air_yards',
        'yac',
        'targets_per_game',
        'interceptions',

        # Pipeline outputs (from merged data)
        'wr_leaf_rating',
        'wr_leaf_uncertainty',
    ]

    # Filter to only metrics that exist in both years
    available_metrics = [m for m in candidate_metrics
                        if m in ratings_2023.columns and m in ratings_2024.columns]

    logger.info(f"  Found {len(available_metrics)}/{len(candidate_metrics)} metrics available in both years")

    # ==========================================
    # Calculate individual correlations
    # ==========================================
    logger.info("\n[4/5] Calculating year-over-year correlations...")

    individual_corrs = calculate_individual_correlations(
        ratings_2023,
        ratings_2024,
        available_metrics
    )

    logger.info(f"  Successfully analyzed {len(individual_corrs)} metrics")

    # ==========================================
    # Calculate inter-metric correlations
    # ==========================================
    logger.info("\n[5/5] Calculating inter-metric correlations...")

    # Use full game-level data for inter-metric correlations (more data points)
    season_stats_full = aggregator.aggregate_season_stats(game_stats_full, min_targets=30)

    corr_matrix = calculate_inter_metric_correlations(
        season_stats_full,
        individual_corrs['metric'].tolist()
    )

    # ==========================================
    # Score and rank metrics
    # ==========================================
    logger.info("\nScoring metrics...")

    scored_metrics = score_metrics(individual_corrs, corr_matrix)

    # ==========================================
    # Display results
    # ==========================================
    print(f"\n{'='*80}")
    print("TOP 20 METRICS BY COMPOSITE SCORE")
    print(f"{'='*80}")
    print(scored_metrics.head(20)[['metric', 'correlation', 'p_value', 'composite_score',
                                     'uniqueness_score', 'coverage']].to_string(index=False))

    print(f"\n{'='*80}")
    print("TOP 20 METRICS BY RAW PREDICTIVE POWER")
    print(f"{'='*80}")
    print(scored_metrics.sort_values('correlation', ascending=False).head(20)[
        ['metric', 'correlation', 'p_value', 'n_players']
    ].to_string(index=False))

    # ==========================================
    # Save results
    # ==========================================
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full results
    scored_metrics.to_csv(output_dir / 'metric_candidate_analysis.csv', index=False)

    # Save correlation matrix
    corr_matrix.to_csv(output_dir / 'metric_correlation_matrix.csv')

    # Save top 20 candidates for Phase 2
    top_20_metrics = scored_metrics.head(20)['metric'].tolist()
    with open(output_dir / 'top_20_metrics.txt', 'w') as f:
        for metric in top_20_metrics:
            f.write(f"{metric}\n")

    logger.info(f"\n✓ Results saved to {output_dir}")
    logger.info(f"  - metric_candidate_analysis.csv")
    logger.info(f"  - metric_correlation_matrix.csv")
    logger.info(f"  - top_20_metrics.txt")

    # ==========================================
    # Summary statistics
    # ==========================================
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Total metrics analyzed: {len(scored_metrics)}")
    print(f"Statistically significant (p<0.05): {(scored_metrics['p_value'] < 0.05).sum()}")
    print(f"Positive correlation: {(scored_metrics['correlation'] > 0).sum()}")
    print(f"Correlation > 0.10: {(scored_metrics['correlation'] > 0.10).sum()}")
    print(f"Correlation > 0.20: {(scored_metrics['correlation'] > 0.20).sum()}")
    print(f"\nBest single metric:")
    best = scored_metrics.iloc[0]
    print(f"  {best['metric']}: r = {best['correlation']:.4f} (p = {best['p_value']:.4f})")

    print(f"\n{'='*80}")
    print("RECOMMENDED TOP 15 METRICS FOR PHASE 2")
    print(f"{'='*80}")
    top_15 = scored_metrics.head(15)['metric'].tolist()
    for i, metric in enumerate(top_15, 1):
        row = scored_metrics[scored_metrics['metric'] == metric].iloc[0]
        print(f"{i:2d}. {metric:30s} (r={row['correlation']:6.3f}, score={row['composite_score']:.3f})")


if __name__ == "__main__":
    main()
