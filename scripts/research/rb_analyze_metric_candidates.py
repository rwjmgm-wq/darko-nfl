"""
RB Composite Analysis - Phase 1: Metric Candidate Selection

Identifies the top RB metrics with highest predictive power for year-over-year performance.
Tests both rushing and receiving metrics for dual-threat evaluation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_pipeline.nfl_data_fetcher import NFLDataFetcher
from data_pipeline.rb_stats_aggregator import RBStatsAggregator

import pandas as pd
import numpy as np
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_year_over_year_correlation(metric_name, year1_data, year2_data):
    """Calculate correlation for a single metric across two years."""

    # Merge on player ID
    merged = year1_data[['player_id', metric_name]].merge(
        year2_data[['player_id', metric_name]],
        on='player_id',
        suffixes=('_y1', '_y2')
    )

    # Remove NaN/inf values
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna()

    if len(merged) < 10:  # Need minimum sample size
        return None, None, 0

    # Calculate correlation
    r, p = stats.pearsonr(merged[f'{metric_name}_y1'], merged[f'{metric_name}_y2'])

    return r, p, len(merged)

def main():
    """Run Phase 1: Metric candidate selection for RBs."""

    logger.info("="*80)
    logger.info("RB COMPOSITE ANALYSIS - PHASE 1: METRIC CANDIDATES")
    logger.info("="*80)

    # ==================================================
    # Load Data
    # ==================================================
    logger.info("\n[1/3] Loading RB data for 2023-2024...")

    fetcher = NFLDataFetcher()

    # Load 2023 and 2024 data
    pbp_2023 = fetcher.fetch_pbp_data([2023])
    pbp_2024 = fetcher.fetch_pbp_data([2024])

    # Load roster data for position filtering
    rosters_2023 = fetcher.fetch_rosters([2023])
    rosters_2024 = fetcher.fetch_rosters([2024])

    logger.info(f"  2023: {len(pbp_2023):,} plays")
    logger.info(f"  2024: {len(pbp_2024):,} plays")

    # Aggregate to season level
    aggregator = RBStatsAggregator(filter_rb_only=True)

    game_stats_2023 = aggregator.aggregate_game_stats(pbp_2023, rosters_2023)
    season_stats_2023 = aggregator.aggregate_season_stats(game_stats_2023, min_touches=50)

    game_stats_2024 = aggregator.aggregate_game_stats(pbp_2024, rosters_2024)
    season_stats_2024 = aggregator.aggregate_season_stats(game_stats_2024, min_touches=50)

    logger.info(f"  2023: {len(season_stats_2023)} RBs with 50+ touches")
    logger.info(f"  2024: {len(season_stats_2024)} RBs with 50+ touches")

    # ==================================================
    # Define Candidate Metrics
    # ==================================================
    logger.info("\n[2/3] Testing candidate metrics...")

    # All numeric columns that might be predictive
    candidate_metrics = [
        # Rushing efficiency
        'yards_per_carry',
        'rush_epa_per_attempt',
        'rush_success_rate',
        'rush_td_rate',

        # Rushing volume
        'rush_attempts_per_game',
        'rush_yards_per_game',
        'rush_attempts',

        # Receiving efficiency
        'catch_rate',
        'yards_per_target',
        'yards_per_reception',
        'rec_epa_per_target',
        'yac_epa_per_target',
        'rec_success_rate',

        # Receiving volume
        'targets_per_game',
        'rec_yards_per_game',
        'targets',
        'receptions',

        # Combined metrics
        'epa_per_touch',
        'total_epa',
        'touches_per_game',
        'total_touches',
        'yards_per_touch',
        'total_yards_per_game',
        'rush_share',

        # Impact metrics
        'total_first_downs',
        'total_tds',

        # Ball security
        'fumble_rate',
    ]

    # Check which columns actually exist
    available_metrics = [m for m in candidate_metrics if m in season_stats_2023.columns]

    logger.info(f"  Testing {len(available_metrics)} candidate metrics")

    # ==================================================
    # Calculate Correlations
    # ==================================================
    results = []

    for metric in available_metrics:
        r, p, n = calculate_year_over_year_correlation(
            metric, season_stats_2023, season_stats_2024
        )

        if r is not None:
            results.append({
                'metric': metric,
                'correlation': r,
                'p_value': p,
                'sample_size': n,
                'significant': p < 0.05
            })

            logger.info(f"  {metric:30s}: r = {r:6.3f}, p = {p:.4f}, n = {n}")

    # ==================================================
    # Analyze Results
    # ==================================================
    logger.info("\n[3/3] Analyzing results...")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('correlation', ascending=False)

    # Calculate composite score (correlation + significance)
    results_df['composite_score'] = results_df['correlation'] + (results_df['significant'].astype(int) * 0.1)
    results_df = results_df.sort_values('composite_score', ascending=False)

    # Save results
    output_file = Path("data/processed/rb_metric_candidates.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file, index=False)

    logger.info(f"\n  Saved results to: {output_file}")

    # Display top 15
    print("\n" + "="*80)
    print("TOP 15 RB METRIC CANDIDATES")
    print("="*80)
    print("\nBased on 2023->2024 year-over-year correlation:\n")

    top_15 = results_df.head(15)
    print(top_15[['metric', 'correlation', 'composite_score', 'p_value', 'sample_size']].to_string(index=False))

    # Summary stats
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"\nMetrics tested: {len(results_df)}")
    print(f"Significant correlations (p < 0.05): {results_df['significant'].sum()}")
    print(f"\nBest single metric: {results_df.iloc[0]['metric']}")
    print(f"  Correlation: {results_df.iloc[0]['correlation']:.4f}")
    print(f"  Sample size: {int(results_df.iloc[0]['sample_size'])} RBs")
    print(f"\nMedian correlation: {results_df['correlation'].median():.4f}")
    print(f"Mean correlation: {results_df['correlation'].mean():.4f}")

    # Breakdown by metric type
    print("\n" + "="*80)
    print("METRIC TYPE BREAKDOWN")
    print("="*80)

    rushing_metrics = [m for m in results_df['metric'] if 'rush' in m.lower()]
    receiving_metrics = [m for m in results_df['metric'] if any(x in m.lower() for x in ['rec', 'target', 'catch', 'yac'])]
    combined_metrics = [m for m in results_df['metric'] if any(x in m.lower() for x in ['total', 'touch', 'epa_per_touch'])]

    rush_corrs = results_df[results_df['metric'].isin(rushing_metrics)]['correlation']
    rec_corrs = results_df[results_df['metric'].isin(receiving_metrics)]['correlation']
    comb_corrs = results_df[results_df['metric'].isin(combined_metrics)]['correlation']

    print(f"\nRushing metrics ({len(rushing_metrics)}):")
    print(f"  Mean correlation: {rush_corrs.mean():.4f}")
    print(f"  Best: {rush_corrs.max():.4f}")

    print(f"\nReceiving metrics ({len(receiving_metrics)}):")
    print(f"  Mean correlation: {rec_corrs.mean():.4f}")
    print(f"  Best: {rec_corrs.max():.4f}")

    print(f"\nCombined metrics ({len(combined_metrics)}):")
    print(f"  Mean correlation: {comb_corrs.mean():.4f}")
    print(f"  Best: {comb_corrs.max():.4f}")

    logger.info("\n" + "="*80)
    logger.info("PHASE 1 COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nTop 15 metrics identified for Phase 2 exhaustive combination search.")
    logger.info(f"Next step: Run rb_exhaustive_composite_search.py")

if __name__ == "__main__":
    main()
