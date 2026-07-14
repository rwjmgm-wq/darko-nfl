"""
QB Composite Analysis - Phase 2: Exhaustive Combination Search

Tests all possible 5-metric combinations from the top 15 candidates
to find the optimal QB performance composite.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_pipeline.nfl_data_fetcher import NFLDataFetcher
from data_pipeline.qb_stats_aggregator import QBStatsAggregator

import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_composite_correlation(metrics, year1_data, year2_data):
    """
    Calculate correlation for a composite of multiple metrics.

    Uses z-score standardization and equal weighting.
    """

    # Select only the metrics we need
    y1 = year1_data[['passer_player_id'] + list(metrics)].copy()
    y2 = year2_data[['passer_player_id'] + list(metrics)].copy()

    # Standardize each metric (z-score)
    for metric in metrics:
        y1_mean = y1[metric].mean()
        y1_std = y1[metric].std()
        if y1_std > 0:
            y1[f'{metric}_z'] = (y1[metric] - y1_mean) / y1_std
        else:
            y1[f'{metric}_z'] = 0

        y2_mean = y2[metric].mean()
        y2_std = y2[metric].std()
        if y2_std > 0:
            y2[f'{metric}_z'] = (y2[metric] - y2_mean) / y2_std
        else:
            y2[f'{metric}_z'] = 0

    # Calculate composite (equal-weighted average of z-scores)
    z_cols = [f'{m}_z' for m in metrics]
    y1['composite'] = y1[z_cols].mean(axis=1)
    y2['composite'] = y2[z_cols].mean(axis=1)

    # Merge and calculate correlation
    merged = y1[['passer_player_id', 'composite']].merge(
        y2[['passer_player_id', 'composite']],
        on='passer_player_id',
        suffixes=('_y1', '_y2')
    )

    merged = merged.replace([np.inf, -np.inf], np.nan).dropna()

    if len(merged) < 10:
        return None, None, 0

    r, p = stats.pearsonr(merged['composite_y1'], merged['composite_y2'])

    return r, p, len(merged)

def main():
    """Run Phase 2: Exhaustive combination search for QBs."""

    logger.info("="*80)
    logger.info("QB COMPOSITE ANALYSIS - PHASE 2: EXHAUSTIVE SEARCH")
    logger.info("="*80)

    # ==================================================
    # Load Phase 1 Results
    # ==================================================
    logger.info("\n[1/3] Loading Phase 1 metric candidates...")

    candidates_file = Path("data/processed/qb_metric_candidates.csv")
    if not candidates_file.exists():
        raise FileNotFoundError("Phase 1 results not found. Run qb_analyze_metric_candidates.py first.")

    candidates_df = pd.read_csv(candidates_file)
    candidates_df = candidates_df.sort_values('composite_score', ascending=False)

    # Take top 15 metrics
    top_15 = candidates_df.head(15)['metric'].tolist()

    logger.info(f"  Top 15 metrics: {', '.join(top_15[:5])}... (+{len(top_15)-5} more)")

    # ==================================================
    # Load QB Data
    # ==================================================
    logger.info("\n[2/3] Loading QB data for 2023-2024...")

    fetcher = NFLDataFetcher()
    pbp_2023 = fetcher.fetch_pbp_data([2023])
    pbp_2024 = fetcher.fetch_pbp_data([2024])

    pbp_2023_pass = pbp_2023[pbp_2023['play_type'] == 'pass'].copy()
    pbp_2024_pass = pbp_2024[pbp_2024['play_type'] == 'pass'].copy()

    aggregator = QBStatsAggregator()
    game_stats_2023 = aggregator.aggregate_game_stats(pbp_2023_pass)
    season_stats_2023 = aggregator.aggregate_season_stats(game_stats_2023, min_attempts=100)

    game_stats_2024 = aggregator.aggregate_game_stats(pbp_2024_pass)
    season_stats_2024 = aggregator.aggregate_season_stats(game_stats_2024, min_attempts=100)

    logger.info(f"  2023: {len(season_stats_2023)} QBs")
    logger.info(f"  2024: {len(season_stats_2024)} QBs")

    # ==================================================
    # Exhaustive Combination Search
    # ==================================================
    logger.info("\n[3/3] Testing all 5-metric combinations...")

    # Generate all 5-metric combinations
    all_combos = list(combinations(top_15, 5))
    logger.info(f"  Total combinations to test: {len(all_combos):,}")

    results = []

    for i, combo in enumerate(all_combos):
        if (i + 1) % 500 == 0:
            logger.info(f"  Progress: {i+1}/{len(all_combos)} ({(i+1)/len(all_combos)*100:.1f}%)")

        # Check if all metrics exist in both years
        if not all(m in season_stats_2023.columns for m in combo):
            continue
        if not all(m in season_stats_2024.columns for m in combo):
            continue

        r, p, n = calculate_composite_correlation(combo, season_stats_2023, season_stats_2024)

        if r is not None:
            results.append({
                'rank': i + 1,
                'metrics': list(combo),
                'correlation': r,
                'p_value': p,
                'sample_size': n
            })

    logger.info(f"  Completed: {len(results)} valid combinations tested")

    # ==================================================
    # Analyze Results
    # ==================================================
    logger.info("\nAnalyzing results...")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('correlation', ascending=False)
    results_df['rank'] = range(1, len(results_df) + 1)

    # Save results
    output_file = Path("data/processed/qb_exhaustive_combinations.csv")
    results_df.to_csv(output_file, index=False)

    logger.info(f"  Saved results to: {output_file}")

    # Display top 10
    print("\n" + "="*80)
    print("TOP 10 QB COMPOSITE COMBINATIONS")
    print("="*80)
    print("\nBased on 2023->2024 year-over-year correlation (equal weights):\n")

    for idx, row in results_df.head(10).iterrows():
        metrics_str = ' + '.join(row['metrics'])
        print(f"\nRank {row['rank']}: r = {row['correlation']:.4f}")
        print(f"  Metrics: {metrics_str}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"\nCombinations tested: {len(results_df):,}")
    print(f"Best combination: r = {results_df.iloc[0]['correlation']:.4f}")
    print(f"Sample size: {int(results_df.iloc[0]['sample_size'])} QBs")
    print(f"\nTop 3 metrics by frequency:")

    # Count metric frequency in top 10
    all_top_metrics = []
    for metrics_list in results_df.head(10)['metrics']:
        all_top_metrics.extend(metrics_list)

    from collections import Counter
    metric_counts = Counter(all_top_metrics)
    for metric, count in metric_counts.most_common(5):
        print(f"  {metric}: appears in {count}/10 top combinations")

    # Compare to best single metric
    best_single = candidates_df.iloc[0]
    improvement = (results_df.iloc[0]['correlation'] - best_single['correlation']) / best_single['correlation'] * 100

    print(f"\nBest single metric: {best_single['metric']} (r = {best_single['correlation']:.4f})")
    print(f"Best composite: r = {results_df.iloc[0]['correlation']:.4f}")
    print(f"Improvement: {improvement:+.1f}%")

    logger.info("\n" + "="*80)
    logger.info("PHASE 2 COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nTop 10 combinations identified.")
    logger.info(f"Next step: Run qb_optimize_composite_weights.py (Phase 3)")

if __name__ == "__main__":
    main()
