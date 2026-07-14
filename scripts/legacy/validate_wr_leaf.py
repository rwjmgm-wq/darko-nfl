"""
WR LEAF Validation Test

Validates the WR LEAF system by:
1. Training on 2023 data
2. Predicting 2024 performance
3. Checking correlation with actual 2024 results

This tests the system's predictive validity.
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


def main():
    """Run WR LEAF validation test."""

    # ==========================================
    # Fetch Data
    # ==========================================
    logger.info("Fetching nflfastR data...")
    fetcher = NFLDataFetcher()
    pbp_2023 = fetcher.fetch_pbp_data([2023])
    pbp_2024 = fetcher.fetch_pbp_data([2024])
    pbp_full = fetcher.fetch_pbp_data([2023, 2024])

    # ==========================================
    # Aggregate Stats
    # ==========================================
    logger.info("Aggregating WR statistics...")
    aggregator = ReceiverStatsAggregator(min_targets=1)

    game_stats_2023 = aggregator.aggregate_game_stats(pbp_2023)
    game_stats_2024 = aggregator.aggregate_game_stats(pbp_2024)

    logger.info(f"  2023: {len(game_stats_2023):,} receiver-games")
    logger.info(f"  2024: {len(game_stats_2024):,} receiver-games")

    # ==========================================
    # Train WR LEAF on 2023 Data
    # ==========================================
    logger.info("\nTraining WR LEAF on 2023 data...")
    pipeline = WRLEAFPipeline(
        use_context=True,
        use_opponent_adj=True,
        use_kalman=True
    )

    # Fit models on 2023 data only
    pipeline.fit_context_model(pbp_2023, metrics=['epa', 'air_epa', 'yac_epa'])
    secondary_ratings_2023 = pipeline.calculate_secondary_ratings(pbp_2023)

    # Process 2023 to get end-of-season ratings
    processed_2023 = pipeline.process_full_pipeline(
        game_stats=game_stats_2023,
        pbp_data=pbp_2023,
        secondary_ratings=secondary_ratings_2023
    )

    # Get 2023 final ratings (min 30 targets)
    ratings_2023 = pipeline.get_current_ratings(processed_2023, min_targets=30)

    logger.info(f"  {len(ratings_2023)} WRs rated in 2023 (30+ targets)")

    # ==========================================
    # Get 2024 Actual Performance
    # ==========================================
    logger.info("\nCalculating 2024 actual performance...")

    # Re-fit context model on full 2023+2024 data for fair 2024 adjustments
    pipeline.fit_context_model(pbp_full, metrics=['epa', 'air_epa', 'yac_epa'])
    secondary_ratings_full = pipeline.calculate_secondary_ratings(pbp_full)

    processed_2024 = pipeline.process_full_pipeline(
        game_stats=game_stats_2024,
        pbp_data=pbp_2024,
        secondary_ratings=secondary_ratings_full
    )

    ratings_2024 = pipeline.get_current_ratings(processed_2024, min_targets=30)

    logger.info(f"  {len(ratings_2024)} WRs rated in 2024 (30+ targets)")

    # ==========================================
    # Validation Analysis
    # ==========================================
    logger.info("\nValidation Analysis:")

    # Merge 2023 predictions with 2024 actuals
    validation = ratings_2023[['receiver_player_id', 'receiver_player_name',
                                'wr_leaf_rating', 'wr_leaf_uncertainty', 'total_targets']].merge(
        ratings_2024[['receiver_player_id', 'wr_leaf_rating', 'total_targets']],
        on='receiver_player_id',
        suffixes=('_2023', '_2024')
    )

    logger.info(f"  {len(validation)} WRs with 30+ targets in both seasons")

    # Calculate correlation
    correlation, p_value = stats.pearsonr(
        validation['wr_leaf_rating_2023'],
        validation['wr_leaf_rating_2024']
    )

    logger.info(f"\n{'='*80}")
    logger.info("WR LEAF Validation Results")
    logger.info(f"{'='*80}")
    logger.info(f"Year-to-year correlation: r = {correlation:.3f} (p = {p_value:.4f})")

    # Calculate prediction error
    validation['prediction_error'] = np.abs(
        validation['wr_leaf_rating_2024'] - validation['wr_leaf_rating_2023']
    )
    mae = validation['prediction_error'].mean()
    rmse = np.sqrt((validation['prediction_error'] ** 2).mean())

    logger.info(f"Mean Absolute Error: {mae:.4f}")
    logger.info(f"Root Mean Squared Error: {rmse:.4f}")

    # Split by rating tiers
    validation['tier_2023'] = pd.cut(
        validation['wr_leaf_rating_2023'],
        bins=[-np.inf, -0.3, 0.0, np.inf],
        labels=['Below Average', 'Average', 'Above Average']
    )

    logger.info(f"\n{'='*80}")
    logger.info("Correlation by 2023 Rating Tier")
    logger.info(f"{'='*80}")
    for tier in ['Below Average', 'Average', 'Above Average']:
        tier_data = validation[validation['tier_2023'] == tier]
        if len(tier_data) >= 5:
            tier_corr, tier_p = stats.pearsonr(
                tier_data['wr_leaf_rating_2023'],
                tier_data['wr_leaf_rating_2024']
            )
            logger.info(f"{tier}: r = {tier_corr:.3f} (n = {len(tier_data)}, p = {tier_p:.4f})")

    # Show top/bottom movers
    validation['change'] = validation['wr_leaf_rating_2024'] - validation['wr_leaf_rating_2023']
    validation = validation.sort_values('change', ascending=False)

    print(f"\n{'='*80}")
    print("Top 10 Improvers (2023 -> 2024)")
    print(f"{'='*80}")
    print(validation.head(10)[['receiver_player_name', 'wr_leaf_rating_2023',
                                'wr_leaf_rating_2024', 'change']].to_string(index=False))

    print(f"\n{'='*80}")
    print("Top 10 Decliners (2023 -> 2024)")
    print(f"{'='*80}")
    print(validation.tail(10)[['receiver_player_name', 'wr_leaf_rating_2023',
                                'wr_leaf_rating_2024', 'change']].to_string(index=False))

    # Save validation results
    output_path = Path('data/processed/wr_leaf_validation.csv')
    validation.to_csv(output_path, index=False)
    logger.info(f"\n✓ Validation results saved to {output_path}")

    # ==========================================
    # Interpretation
    # ==========================================
    print(f"\n{'='*80}")
    print("Validation Interpretation")
    print(f"{'='*80}")

    if correlation > 0.5:
        print(f"[STRONG] Predictive validity (r = {correlation:.3f})")
        print("  WR LEAF ratings are highly predictive of next season performance.")
    elif correlation > 0.3:
        print(f"[MODERATE] Predictive validity (r = {correlation:.3f})")
        print("  WR LEAF shows meaningful predictive power.")
    else:
        print(f"[WEAK] Predictive validity (r = {correlation:.3f})")
        print("  May need further refinement.")
        print("\n  Possible reasons for weak correlation:")
        print("  - WR performance heavily depends on QB/scheme (more than QB depends on WR)")
        print("  - WRs change teams more frequently than QBs")
        print("  - Air EPA may be more volatile than overall QB EPA")
        print("  - Consider: Using this as a talent metric rather than predictive tool")

    if p_value < 0.01:
        print(f"\n[SIGNIFICANT] Statistically significant (p < 0.01)")
    elif p_value < 0.05:
        print(f"\n[SIGNIFICANT] Statistically significant (p < 0.05)")
    else:
        print(f"\n[NOT SIGNIFICANT] Not statistically significant (p = {p_value:.4f})")
        print("  Results may be due to chance. Use with caution.")


if __name__ == "__main__":
    main()
