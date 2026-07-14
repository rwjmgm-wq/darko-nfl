"""
Phase 3: Weight Optimization for Top Combinations

Takes the top N combinations from Phase 2 and finds optimal weights
using Ridge regression to maximize 2023→2024 correlation.

Methods:
1. Ridge regression (L2 regularization)
2. Grid search (discrete weight space)
3. Comparison to equal weighting
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_pipeline.nfl_data_fetcher import NFLDataFetcher
from data_pipeline.receiver_stats_aggregator import ReceiverStatsAggregator

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
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


def optimize_weights_ridge(
    ratings_2023: pd.DataFrame,
    ratings_2024: pd.DataFrame,
    metrics: list,
    alpha: float = 1.0
) -> dict:
    """
    Optimize weights using Ridge regression.

    Args:
        ratings_2023: 2023 season ratings
        ratings_2024: 2024 season ratings
        metrics: List of metrics
        alpha: Regularization strength

    Returns:
        Dictionary with results
    """
    # Standardize metrics
    ratings_2023_std = standardize_metrics(ratings_2023, metrics)
    ratings_2024_std = standardize_metrics(ratings_2024, metrics)

    # Get common players
    common_players = set(ratings_2023['receiver_player_id']) & set(ratings_2024['receiver_player_id'])
    common_players = list(common_players)

    # Filter to common players
    train_2023 = ratings_2023_std[ratings_2023_std['receiver_player_id'].isin(common_players)]
    train_2024 = ratings_2024_std[ratings_2024_std['receiver_player_id'].isin(common_players)]

    # Drop duplicate player_ids (data quality fix)
    train_2023 = train_2023.drop_duplicates('receiver_player_id', keep='first')
    train_2024 = train_2024.drop_duplicates('receiver_player_id', keep='first')

    # Sort by player_id to ensure alignment
    train_2023 = train_2023.sort_values('receiver_player_id').reset_index(drop=True)
    train_2024 = train_2024.sort_values('receiver_player_id').reset_index(drop=True)

    # Identify players with complete data in BOTH years
    train_2023_complete = train_2023[['receiver_player_id'] + metrics].dropna()
    train_2024_complete = train_2024[['receiver_player_id'] + metrics].dropna()

    # Get players who have complete data in both years
    players_2023 = set(train_2023_complete['receiver_player_id'])
    players_2024 = set(train_2024_complete['receiver_player_id'])
    valid_players = players_2023 & players_2024

    # Filter to these players and ensure alignment
    train_2023_clean = train_2023[train_2023['receiver_player_id'].isin(valid_players)].copy()
    train_2024_clean = train_2024[train_2024['receiver_player_id'].isin(valid_players)].copy()

    # Sort to ensure alignment
    train_2023_clean = train_2023_clean.sort_values('receiver_player_id').reset_index(drop=True)
    train_2024_clean = train_2024_clean.sort_values('receiver_player_id').reset_index(drop=True)

    # Extract X (2023 metrics) and y (2024 composite - using equal weights as target)
    X = train_2023_clean[metrics].values

    # Create target: 2024 equal-weighted composite
    y = train_2024_clean[metrics].mean(axis=1).values

    # Verify lengths match
    assert len(X) == len(y), f"X and y length mismatch: {len(X)} vs {len(y)}"

    # Fit Ridge regression
    model = Ridge(alpha=alpha)
    model.fit(X, y)

    # Get optimized weights
    weights = model.coef_

    # Normalize weights to sum to 1
    weights = weights / weights.sum()

    # Create composite with optimized weights
    composite_2023 = (train_2023_clean[metrics].values * weights).sum(axis=1)
    composite_2024 = (train_2024_clean[metrics].values * weights).sum(axis=1)

    # Calculate correlation
    correlation, p_value = stats.pearsonr(composite_2023, composite_2024)

    # Calculate error metrics
    error = composite_2024 - composite_2023
    mae = np.abs(error).mean()
    rmse = np.sqrt((error ** 2).mean())

    return {
        'metrics': metrics,
        'weights': weights,
        'correlation': correlation,
        'p_value': p_value,
        'n_players': len(valid_players),
        'mae': mae,
        'rmse': rmse,
        'method': f'ridge_alpha_{alpha}'
    }


def evaluate_with_weights(
    ratings_2023: pd.DataFrame,
    ratings_2024: pd.DataFrame,
    metrics: list,
    weights: list
) -> dict:
    """Evaluate a combination with specific weights."""
    # Standardize
    ratings_2023_std = standardize_metrics(ratings_2023, metrics)
    ratings_2024_std = standardize_metrics(ratings_2024, metrics)

    # Get common players
    common_players = set(ratings_2023['receiver_player_id']) & set(ratings_2024['receiver_player_id'])
    common_players = list(common_players)

    train_2023 = ratings_2023_std[ratings_2023_std['receiver_player_id'].isin(common_players)]
    train_2024 = ratings_2024_std[ratings_2024_std['receiver_player_id'].isin(common_players)]

    train_2023 = train_2023.sort_values('receiver_player_id')
    train_2024 = train_2024.sort_values('receiver_player_id')

    # Create weighted composite
    composite_2023 = (train_2023[metrics].values * weights).sum(axis=1)
    composite_2024 = (train_2024[metrics].values * weights).sum(axis=1)

    # Calculate correlation
    correlation, p_value = stats.pearsonr(composite_2023, composite_2024)

    error = composite_2024 - composite_2023
    mae = np.abs(error).mean()
    rmse = np.sqrt((error ** 2).mean())

    return {
        'correlation': correlation,
        'p_value': p_value,
        'n_players': len(common_players),
        'mae': mae,
        'rmse': rmse
    }


def main():
    """Run Phase 3: Weight optimization."""

    logger.info("="*80)
    logger.info("PHASE 3: WEIGHT OPTIMIZATION")
    logger.info("="*80)

    # ==========================================
    # Load top combinations from Phase 2
    # ==========================================
    logger.info("\n[1/4] Loading top combinations from Phase 2...")

    top_100_file = Path('data/processed/top_100_combinations.csv')
    if not top_100_file.exists():
        logger.error("top_100_combinations.csv not found. Run exhaustive_composite_search.py first.")
        return

    top_100 = pd.read_csv(top_100_file)

    # Use top 50 for optimization (manageable computation)
    N_COMBOS = 50
    top_combinations = top_100.head(N_COMBOS)

    logger.info(f"  Optimizing weights for top {N_COMBOS} combinations")
    logger.info(f"  Best equal-weighted: r = {top_combinations.iloc[0]['correlation']:.4f}")

    # ==========================================
    # Load data
    # ==========================================
    logger.info("\n[2/4] Loading data...")

    fetcher = NFLDataFetcher()
    pbp_2023 = fetcher.fetch_pbp_data([2023])
    pbp_2024 = fetcher.fetch_pbp_data([2024])

    # Fetch roster data for WR-only filtering
    rosters = fetcher.fetch_rosters([2023, 2024])

    aggregator = ReceiverStatsAggregator(min_targets=1, filter_wr_only=True)
    game_stats_2023 = aggregator.aggregate_game_stats(pbp_2023, roster_data=rosters)
    game_stats_2024 = aggregator.aggregate_game_stats(pbp_2024, roster_data=rosters)

    ratings_2023 = aggregator.aggregate_season_stats(game_stats_2023, min_targets=30)
    ratings_2024 = aggregator.aggregate_season_stats(game_stats_2024, min_targets=30)

    logger.info(f"  2023: {len(ratings_2023)} WRs")
    logger.info(f"  2024: {len(ratings_2024)} WRs")

    # ==========================================
    # Optimize weights for each combination
    # ==========================================
    logger.info("\n[3/4] Optimizing weights with Ridge regression...")

    optimized_results = []

    for idx, row in tqdm(top_combinations.iterrows(), total=len(top_combinations), desc="Optimizing"):
        # Parse metrics from tuple string
        metrics = eval(row['metrics']) if isinstance(row['metrics'], str) else row['metrics']

        # Try multiple alpha values
        for alpha in [0.1, 1.0, 10.0]:
            result = optimize_weights_ridge(
                ratings_2023,
                ratings_2024,
                list(metrics),
                alpha=alpha
            )
            optimized_results.append(result)

    # Convert to DataFrame
    optimized_df = pd.DataFrame(optimized_results)
    optimized_df = optimized_df.sort_values('correlation', ascending=False)

    # Add rank
    optimized_df['rank'] = range(1, len(optimized_df) + 1)

    # ==========================================
    # Save results
    # ==========================================
    logger.info("\n[4/4] Saving results...")

    output_dir = Path('data/processed')

    # Save optimized results
    optimized_df.to_csv(output_dir / 'optimized_weights_results.csv', index=False)

    # Save top 20
    top_20_optimized = optimized_df.head(20)
    top_20_optimized.to_csv(output_dir / 'top_20_optimized.csv', index=False)

    logger.info(f"  Saved {len(optimized_df):,} results to optimized_weights_results.csv")
    logger.info(f"  Saved top 20 to top_20_optimized.csv")

    # ==========================================
    # Display results
    # ==========================================
    print(f"\n{'='*80}")
    print("TOP 10 OPTIMIZED COMBINATIONS")
    print(f"{'='*80}")

    for _, row in top_20_optimized.head(10).iterrows():
        metrics_list = eval(row['metrics']) if isinstance(row['metrics'], str) else row['metrics']
        weights_list = eval(row['weights']) if isinstance(row['weights'], str) else row['weights']

        print(f"\nRank {int(row['rank'])}: r = {row['correlation']:.4f} ({row['method']})")
        print(f"  Metrics & Weights:")
        for metric, weight in zip(metrics_list, weights_list):
            print(f"    {weight:6.3f} × {metric}")

    # ==========================================
    # Compare to Phase 2 results
    # ==========================================
    best_optimized = optimized_df.iloc[0]
    best_equal_weighted = top_combinations.iloc[0]

    print(f"\n{'='*80}")
    print("COMPARISON: OPTIMIZED VS EQUAL WEIGHTING")
    print(f"{'='*80}")
    print(f"Best equal-weighted (Phase 2):")
    print(f"  Correlation: {best_equal_weighted['correlation']:.4f}")
    print(f"\nBest optimized weights (Phase 3):")
    print(f"  Correlation: {best_optimized['correlation']:.4f}")
    print(f"  Improvement: {(best_optimized['correlation'] - best_equal_weighted['correlation']):.4f} ({(best_optimized['correlation']/best_equal_weighted['correlation'] - 1)*100:+.2f}%)")

    # Compare to best single metric
    single_metric_file = Path('data/processed/metric_candidate_analysis.csv')
    if single_metric_file.exists():
        single_metrics = pd.read_csv(single_metric_file)
        best_single = single_metrics.iloc[0]

        print(f"\n{'='*80}")
        print("COMPARISON TO BEST SINGLE METRIC")
        print(f"{'='*80}")
        print(f"Best single metric: {best_single['metric']}")
        print(f"  Correlation: {best_single['correlation']:.4f}")
        print(f"\nBest optimized composite:")
        print(f"  Correlation: {best_optimized['correlation']:.4f}")

        if best_optimized['correlation'] > best_single['correlation']:
            print(f"  IMPROVEMENT: +{(best_optimized['correlation'] - best_single['correlation']):.4f} ({(best_optimized['correlation']/best_single['correlation'] - 1)*100:+.2f}%)")
        else:
            print(f"  Single metric still better by {(best_single['correlation'] - best_optimized['correlation']):.4f}")


if __name__ == "__main__":
    main()
