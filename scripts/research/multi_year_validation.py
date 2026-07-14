"""
Phase 4: Multi-Year Validation

Tests the top 10 optimized combinations across multiple year pairs
to ensure stability and generalizability.

Year pairs tested:
- 2020→2021
- 2021→2022
- 2022→2023
- 2023→2024
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_pipeline.nfl_data_fetcher import NFLDataFetcher
from data_pipeline.receiver_stats_aggregator import ReceiverStatsAggregator

import pandas as pd
import numpy as np
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def standardize_metrics(data: pd.DataFrame, metrics: list) -> pd.DataFrame:
    """Standardize metrics to z-scores."""
    standardized = data.copy()
    for metric in metrics:
        if metric in data.columns:
            standardized[metric] = (data[metric] - data[metric].mean()) / data[metric].std()
    return standardized


def validate_combination(
    ratings_year1: pd.DataFrame,
    ratings_year2: pd.DataFrame,
    metrics: list,
    weights: list
) -> dict:
    """
    Validate a metric combination on a year pair.

    Args:
        ratings_year1: First year ratings
        ratings_year2: Second year ratings
        metrics: List of metrics
        weights: Optimized weights

    Returns:
        Dictionary with validation results
    """
    # Standardize metrics
    ratings_year1_std = standardize_metrics(ratings_year1, metrics)
    ratings_year2_std = standardize_metrics(ratings_year2, metrics)

    # Get common players
    common_players = set(ratings_year1['receiver_player_id']) & set(ratings_year2['receiver_player_id'])

    # Filter and drop duplicates
    train_year1 = ratings_year1_std[ratings_year1_std['receiver_player_id'].isin(common_players)]
    train_year2 = ratings_year2_std[ratings_year2_std['receiver_player_id'].isin(common_players)]

    train_year1 = train_year1.drop_duplicates('receiver_player_id', keep='first')
    train_year2 = train_year2.drop_duplicates('receiver_player_id', keep='first')

    # Sort and align
    train_year1 = train_year1.sort_values('receiver_player_id').reset_index(drop=True)
    train_year2 = train_year2.sort_values('receiver_player_id').reset_index(drop=True)

    # Filter to players with complete data
    train_year1_complete = train_year1[['receiver_player_id'] + metrics].dropna()
    train_year2_complete = train_year2[['receiver_player_id'] + metrics].dropna()

    valid_players = set(train_year1_complete['receiver_player_id']) & set(train_year2_complete['receiver_player_id'])

    train_year1_clean = train_year1[train_year1['receiver_player_id'].isin(valid_players)].copy()
    train_year2_clean = train_year2[train_year2['receiver_player_id'].isin(valid_players)].copy()

    train_year1_clean = train_year1_clean.sort_values('receiver_player_id').reset_index(drop=True)
    train_year2_clean = train_year2_clean.sort_values('receiver_player_id').reset_index(drop=True)

    if len(train_year1_clean) < 10:
        return None

    # Create weighted composite
    composite_year1 = (train_year1_clean[metrics].values * weights).sum(axis=1)
    composite_year2 = (train_year2_clean[metrics].values * weights).sum(axis=1)

    # Calculate correlation
    try:
        correlation, p_value = stats.pearsonr(composite_year1, composite_year2)

        # Calculate error metrics
        error = composite_year2 - composite_year1
        mae = np.abs(error).mean()
        rmse = np.sqrt((error ** 2).mean())

        return {
            'correlation': correlation,
            'p_value': p_value,
            'n_players': len(valid_players),
            'mae': mae,
            'rmse': rmse
        }
    except Exception as e:
        return None


def main():
    """Run Phase 4: Multi-year validation."""

    logger.info("="*80)
    logger.info("PHASE 4: MULTI-YEAR VALIDATION")
    logger.info("="*80)

    # ==========================================
    # Load top 10 optimized combinations
    # ==========================================
    logger.info("\n[1/3] Loading top 10 optimized combinations...")

    top_optimized_file = Path('data/processed/top_20_optimized.csv')
    if not top_optimized_file.exists():
        logger.error("top_20_optimized.csv not found. Run optimize_composite_weights.py first.")
        return

    top_optimized = pd.read_csv(top_optimized_file)
    top_10 = top_optimized.head(10)

    logger.info(f"  Testing top {len(top_10)} combinations")
    logger.info(f"  Best: r = {top_10.iloc[0]['correlation']:.4f} (2023→2024)")

    # ==========================================
    # Load data for all year pairs
    # ==========================================
    logger.info("\n[2/3] Loading data for all year pairs...")

    fetcher = NFLDataFetcher()

    # Fetch roster data for WR-only filtering
    all_years = list(range(2020, 2025))
    rosters = fetcher.fetch_rosters(all_years)

    aggregator = ReceiverStatsAggregator(min_targets=1, filter_wr_only=True)

    # Year pairs to test
    year_pairs = [
        (2020, 2021),
        (2021, 2022),
        (2022, 2023),
        (2023, 2024)
    ]

    # Load all years
    ratings_by_year = {}
    for year in range(2020, 2025):
        logger.info(f"  Loading {year} data...")
        pbp = fetcher.fetch_pbp_data([year])
        game_stats = aggregator.aggregate_game_stats(pbp, roster_data=rosters)
        ratings = aggregator.aggregate_season_stats(game_stats, min_targets=30)
        ratings_by_year[year] = ratings
        logger.info(f"    {year}: {len(ratings)} WRs")

    # ==========================================
    # Validate each combination on all year pairs
    # ==========================================
    logger.info("\n[3/3] Validating combinations across year pairs...")

    results = []

    for idx, row in top_10.iterrows():
        # Parse metrics and weights
        metrics = eval(row['metrics']) if isinstance(row['metrics'], str) else row['metrics']

        # Parse weights - handle numpy array string format
        if isinstance(row['weights'], str):
            # Convert numpy array string to list
            weights_str = row['weights'].strip('[]')
            weights = np.fromstring(weights_str, sep=' ')
        else:
            weights = row['weights']

        combo_results = {
            'rank': int(row['rank']),
            'metrics': metrics,
            'weights': weights,
            'method': row['method']
        }

        # Test on each year pair
        for year1, year2 in year_pairs:
            year_pair_key = f"{year1}_{year2}"

            result = validate_combination(
                ratings_by_year[year1],
                ratings_by_year[year2],
                list(metrics),
                weights
            )

            if result is not None:
                combo_results[f'r_{year_pair_key}'] = result['correlation']
                combo_results[f'n_{year_pair_key}'] = result['n_players']
            else:
                combo_results[f'r_{year_pair_key}'] = np.nan
                combo_results[f'n_{year_pair_key}'] = 0

        # Calculate average correlation across all years
        r_values = [combo_results[f'r_{y1}_{y2}'] for y1, y2 in year_pairs if not np.isnan(combo_results.get(f'r_{y1}_{y2}', np.nan))]
        combo_results['r_mean'] = np.mean(r_values) if r_values else np.nan
        combo_results['r_std'] = np.std(r_values) if r_values else np.nan
        combo_results['r_min'] = np.min(r_values) if r_values else np.nan
        combo_results['r_max'] = np.max(r_values) if r_values else np.nan

        results.append(combo_results)

        logger.info(f"  Rank {combo_results['rank']}: mean r = {combo_results['r_mean']:.4f} (std = {combo_results['r_std']:.4f})")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Sort by mean correlation
    results_df = results_df.sort_values('r_mean', ascending=False).reset_index(drop=True)

    # ==========================================
    # Save results
    # ==========================================
    output_dir = Path('data/processed')
    results_df.to_csv(output_dir / 'multi_year_validation.csv', index=False)

    logger.info(f"\n  Saved results to multi_year_validation.csv")

    # ==========================================
    # Display results
    # ==========================================
    print(f"\n{'='*80}")
    print("MULTI-YEAR VALIDATION RESULTS")
    print(f"{'='*80}")

    for _, row in results_df.iterrows():
        metrics_list = eval(row['metrics']) if isinstance(row['metrics'], str) else row['metrics']
        print(f"\nRank {int(row['rank'])}: {row['method']}")
        print(f"  Metrics: {' + '.join(metrics_list)}")
        print(f"  Average: r = {row['r_mean']:.4f} ± {row['r_std']:.4f}")
        print(f"  Range: [{row['r_min']:.4f}, {row['r_max']:.4f}]")
        print(f"  By year pair:")
        for year1, year2 in year_pairs:
            r = row[f'r_{year1}_{year2}']
            n = int(row[f'n_{year1}_{year2}'])
            print(f"    {year1}->{year2}: r = {r:.4f} (n={n})")

    # ==========================================
    # Summary statistics
    # ==========================================
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    best_combo = results_df.iloc[0]
    print(f"Most stable combination:")
    print(f"  Rank: {int(best_combo['rank'])}")
    print(f"  Average r: {best_combo['r_mean']:.4f}")
    print(f"  Std dev: {best_combo['r_std']:.4f}")
    print(f"  Consistency: {(1 - best_combo['r_std']/best_combo['r_mean'])*100:.1f}%")

    # Compare to original 2023->2024 performance
    print(f"\n{'='*80}")
    print("COMPARISON TO ORIGINAL 2023->2024 RESULTS")
    print(f"{'='*80}")
    for i in range(min(3, len(results_df))):
        row = results_df.iloc[i]
        original_r = row['r_2023_2024']
        mean_r = row['r_mean']
        diff = mean_r - original_r
        print(f"Rank {int(row['rank'])}: 2023->2024 r={original_r:.4f}, Multi-year mean={mean_r:.4f} (Delta={diff:+.4f})")


if __name__ == "__main__":
    main()
