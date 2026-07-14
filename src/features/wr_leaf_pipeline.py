"""
WR LEAF Pipeline v1.0

Integrated rating system for wide receivers combining:
1. Air EPA & YAC EPA tracking (position-specific metrics)
2. Context adjustments (weather, target depth, game script)
3. Opponent adjustments (secondary quality)
4. Kalman filtering (adaptive learning)
5. Game-by-game ratings (temporal dynamics)

Mirrors QB LEAF architecture but optimized for WR evaluation.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import logging
from pathlib import Path

try:
    from .wr_context_adjustments import WRContextAdjustmentModel
    from .secondary_ratings import SecondaryRatingSystem
    from .kalman_filter import QBKalmanFilter  # Reusable for any metric
    from .game_by_game_ratings import GameByGameRatingSystem
except ImportError:
    # When running as main script
    from wr_context_adjustments import WRContextAdjustmentModel
    from secondary_ratings import SecondaryRatingSystem
    from kalman_filter import QBKalmanFilter
    from game_by_game_ratings import GameByGameRatingSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WRLEAFPipeline:
    """
    Complete WR LEAF rating system with all enhancements.

    Pipeline stages:
    1. Context adjustment: Remove weather/target depth/situation effects
    2. Opponent adjustment: Control for secondary quality
    3. Kalman filtering: Adaptive learning over time
    4. Game-by-game: Full trajectory tracking
    """

    def __init__(
        self,
        context_model: Optional[WRContextAdjustmentModel] = None,
        secondary_system: Optional[SecondaryRatingSystem] = None,
        kalman_filter: Optional[QBKalmanFilter] = None,
        use_context: bool = True,
        use_opponent_adj: bool = True,
        use_kalman: bool = True
    ):
        """
        Initialize WR LEAF pipeline.

        Args:
            context_model: WRContextAdjustmentModel instance (creates default if None)
            secondary_system: SecondaryRatingSystem instance (creates default if None)
            kalman_filter: QBKalmanFilter instance (creates default if None)
            use_context: Whether to apply context adjustments
            use_opponent_adj: Whether to apply opponent adjustments
            use_kalman: Whether to apply Kalman filtering
        """
        self.context_model = context_model or WRContextAdjustmentModel(alpha=1.0, use_ridge=True)
        self.secondary_system = secondary_system or SecondaryRatingSystem(
            decay_rate=0.996, min_targets=50, iterations=5
        )
        self.kalman_filter = kalman_filter or QBKalmanFilter(auto_tune=True, em_iterations=10)
        self.game_by_game_system = GameByGameRatingSystem(kalman_filter=self.kalman_filter)

        self.use_context = use_context
        self.use_opponent_adj = use_opponent_adj
        self.use_kalman = use_kalman

        # Track what adjustments were applied
        self.pipeline_config = {
            'context_adjustments': use_context,
            'opponent_adjustments': use_opponent_adj,
            'kalman_filtering': use_kalman
        }

    def fit_context_model(
        self,
        train_pbp: pd.DataFrame,
        metrics: List[str] = ['epa', 'air_epa', 'yac_epa']
    ):
        """
        Fit WR context adjustment model on training play-by-play data.

        Args:
            train_pbp: Training play-by-play data with receiver targets
            metrics: WR metrics to model context effects for
        """
        if not self.use_context:
            logger.info("Context adjustments disabled, skipping fit")
            return

        logger.info("\nFitting WR context adjustment model...")
        self.context_model.fit(train_pbp, metrics=metrics)
        logger.info("  Context model fitted!")

    def calculate_secondary_ratings(
        self,
        pbp_data: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate secondary ratings from play-by-play data.

        Args:
            pbp_data: Full play-by-play data with receiver targets

        Returns:
            Dictionary with secondary ratings
        """
        if not self.use_opponent_adj:
            logger.info("Opponent adjustments disabled, skipping secondary ratings")
            return {}

        logger.info("\nCalculating secondary ratings...")
        secondary_ratings = self.secondary_system.calculate_secondary_ratings(pbp_data)
        logger.info("  Secondary ratings calculated!")

        return secondary_ratings

    def process_full_pipeline(
        self,
        game_stats: pd.DataFrame,
        pbp_data: pd.DataFrame,
        secondary_ratings: Optional[Dict] = None,
        primary_metric: str = 'air_epa_per_target'
    ) -> pd.DataFrame:
        """
        Run complete WR LEAF pipeline on game-level stats.

        Args:
            game_stats: Game-level WR statistics (from ReceiverStatsAggregator)
            pbp_data: Play-by-play data (for context adjustments)
            secondary_ratings: Pre-calculated secondary ratings (calculates if None)
            primary_metric: Primary metric for LEAF rating (default: air_epa_per_target)

        Returns:
            DataFrame with all adjustments and game-by-game WR LEAF ratings
        """
        logger.info("\n" + "="*80)
        logger.info("Running WR LEAF Pipeline")
        logger.info("="*80)

        result = game_stats.copy()

        # ==========================================
        # Stage 1: Context Adjustments
        # ==========================================
        if self.use_context:
            logger.info("\n[Stage 1/4] Applying context adjustments...")

            try:
                # Map play-by-play to receiver game stats
                receiver_pbp = pbp_data[
                    (pbp_data['pass_attempt'] == 1) &
                    (pbp_data['receiver_player_id'].notna())
                ].copy()

                # Apply context adjustments at play level
                adjusted_pbp = self.context_model.adjust(receiver_pbp, metrics=['air_epa', 'yac_epa'])

                # Aggregate context-adjusted metrics to game level
                game_ctx = adjusted_pbp.groupby(
                    ['game_id', 'receiver_player_id']
                ).agg({
                    'air_epa_ctx_adj': 'mean',
                    'yac_epa_ctx_adj': 'mean',
                    'air_epa_context_impact': 'mean',
                    'yac_epa_context_impact': 'mean'
                }).reset_index()

                # Merge with game stats
                result = result.merge(
                    game_ctx,
                    on=['game_id', 'receiver_player_id'],
                    how='left'
                )

                # Use context-adjusted air EPA as base
                if 'air_epa_ctx_adj' in result.columns:
                    result['base_air_epa'] = result['air_epa_ctx_adj']
                    result['base_yac_epa'] = result['yac_epa_ctx_adj']
                    logger.info("  Context adjustments applied!")
                else:
                    result['base_air_epa'] = result.get('air_epa_per_target', 0)
                    result['base_yac_epa'] = result.get('yac_epa_per_target', 0)
                    logger.warning("  Context adjustment failed, using raw metrics")

            except Exception as e:
                logger.error(f"  Context adjustment error: {e}")
                result['base_air_epa'] = result.get('air_epa_per_target', 0)
                result['base_yac_epa'] = result.get('yac_epa_per_target', 0)
        else:
            result['base_air_epa'] = result.get('air_epa_per_target', 0)
            result['base_yac_epa'] = result.get('yac_epa_per_target', 0)
            logger.info("\n[Stage 1/4] Skipping context adjustments")

        # ==========================================
        # Stage 2: Opponent Adjustments (Secondary Quality)
        # ==========================================
        if self.use_opponent_adj:
            logger.info("\n[Stage 2/4] Applying opponent adjustments...")

            try:
                if secondary_ratings is None:
                    secondary_ratings = self.calculate_secondary_ratings(pbp_data)

                # Map secondary ratings to games
                result['secondary_air_epa_allowed'] = result['defteam'].map(
                    {team: ratings.get('air_epa_allowed', 0.0)
                     for team, ratings in secondary_ratings.items()}
                ).fillna(0.0)

                result['secondary_yac_epa_allowed'] = result['defteam'].map(
                    {team: ratings.get('yac_epa_allowed', 0.0)
                     for team, ratings in secondary_ratings.items()}
                ).fillna(0.0)

                # Apply opponent adjustment
                # Subtract what secondary typically allows (lower is tougher)
                result['opp_adj_base_air_epa'] = (
                    result['base_air_epa'] - result['secondary_air_epa_allowed']
                )
                result['opp_adj_base_yac_epa'] = (
                    result['base_yac_epa'] - result['secondary_yac_epa_allowed']
                )

                logger.info("  Opponent adjustments applied!")

            except Exception as e:
                logger.error(f"  Opponent adjustment error: {e}")
                result['opp_adj_base_air_epa'] = result['base_air_epa']
                result['opp_adj_base_yac_epa'] = result['base_yac_epa']
        else:
            result['opp_adj_base_air_epa'] = result['base_air_epa']
            result['opp_adj_base_yac_epa'] = result['base_yac_epa']
            logger.info("\n[Stage 2/4] Skipping opponent adjustments")

        # ==========================================
        # Stage 3: Kalman Filtering
        # ==========================================
        if self.use_kalman:
            logger.info("\n[Stage 3/4] Applying Kalman filtering...")

            try:
                # Prepare data for Kalman filter
                kalman_input = result[[
                    'receiver_player_id', 'season', 'week',
                    'opp_adj_base_air_epa', 'opp_adj_base_yac_epa', 'targets'
                ]].copy()

                kalman_input['games_played'] = kalman_input.groupby('receiver_player_id').cumcount() + 1

                # Apply Kalman filter to air_epa (primary metric)
                # Process each WR separately
                filtered_data = []
                for wr_id in kalman_input['receiver_player_id'].unique():
                    wr_data = kalman_input[kalman_input['receiver_player_id'] == wr_id].sort_values(['season', 'week']).copy()

                    if len(wr_data) >= 2:  # Need at least 2 games for Kalman
                        # Add game_date column for Kalman filter (use season/week as proxy)
                        # Create a simple datetime from season and week (week * 7 days offset from season start)
                        wr_data['game_date'] = pd.to_datetime(wr_data['season'], format='%Y') + pd.to_timedelta(wr_data['week'] * 7, unit='D')

                        # Filter air_epa
                        air_estimates, air_uncertainties = self.kalman_filter.filter_player_metric(
                            player_id=wr_id,
                            player_games=wr_data,
                            metric='opp_adj_base_air_epa',
                            prior_mean=0.0,
                            game_date_col='game_date',
                            attempts_col='targets'
                        )
                        wr_data['opp_adj_base_air_epa_kalman'] = air_estimates
                        wr_data['opp_adj_base_air_epa_uncertainty'] = air_uncertainties

                        # Filter yac_epa
                        yac_estimates, yac_uncertainties = self.kalman_filter.filter_player_metric(
                            player_id=wr_id,
                            player_games=wr_data,
                            metric='opp_adj_base_yac_epa',
                            prior_mean=0.0,
                            game_date_col='game_date',
                            attempts_col='targets'
                        )
                        wr_data['opp_adj_base_yac_epa_kalman'] = yac_estimates
                        wr_data['opp_adj_base_yac_epa_uncertainty'] = yac_uncertainties

                        filtered_data.append(wr_data)
                    elif len(wr_data) == 1:
                        # Single game - use raw values with default uncertainty
                        wr_data['opp_adj_base_air_epa_kalman'] = wr_data['opp_adj_base_air_epa']
                        wr_data['opp_adj_base_air_epa_uncertainty'] = 0.15
                        wr_data['opp_adj_base_yac_epa_kalman'] = wr_data['opp_adj_base_yac_epa']
                        wr_data['opp_adj_base_yac_epa_uncertainty'] = 0.15
                        filtered_data.append(wr_data)

                filtered = pd.concat(filtered_data, ignore_index=True) if filtered_data else kalman_input

                # Merge filtered results
                result = result.merge(
                    filtered[['receiver_player_id', 'season', 'week',
                             'opp_adj_base_air_epa_kalman', 'opp_adj_base_air_epa_uncertainty',
                             'opp_adj_base_yac_epa_kalman', 'opp_adj_base_yac_epa_uncertainty']],
                    on=['receiver_player_id', 'season', 'week'],
                    how='left'
                )

                # Primary LEAF rating = air_epa (coverage ability)
                result['wr_leaf_rating'] = result['opp_adj_base_air_epa_kalman']
                result['wr_leaf_uncertainty'] = result['opp_adj_base_air_epa_uncertainty']

                # Fill NaN values with non-Kalman fallback
                result['wr_leaf_rating'] = result['wr_leaf_rating'].fillna(result['opp_adj_base_air_epa'])
                result['wr_leaf_uncertainty'] = result['wr_leaf_uncertainty'].fillna(0.15)

                # Final fallback: use league average (0.0) for any remaining NaN
                result['wr_leaf_rating'] = result['wr_leaf_rating'].fillna(0.0)

                logger.info("  Kalman filtering applied!")

            except Exception as e:
                import traceback
                logger.error(f"  Kalman filtering error: {e}")
                logger.error(f"  Full traceback:\n{traceback.format_exc()}")
                result['wr_leaf_rating'] = result['opp_adj_base_air_epa']
                result['wr_leaf_uncertainty'] = 0.15  # Default uncertainty

        else:
            result['wr_leaf_rating'] = result['opp_adj_base_air_epa']
            result['wr_leaf_uncertainty'] = 0.15
            logger.info("\n[Stage 3/4] Skipping Kalman filtering")

        # ==========================================
        # Stage 4: Game-by-Game Ratings
        # ==========================================
        logger.info("\n[Stage 4/4] Generating game-by-game ratings...")

        try:
            # Add game number for each WR
            result['game_number'] = result.groupby('receiver_player_id').cumcount() + 1

            # Ensure required columns exist
            if 'wr_leaf_rating' not in result.columns:
                result['wr_leaf_rating'] = result.get('opp_adj_base_air_epa', 0)
            if 'wr_leaf_uncertainty' not in result.columns:
                result['wr_leaf_uncertainty'] = 0.15

            logger.info("  Game-by-game ratings generated!")

        except Exception as e:
            logger.error(f"  Game-by-game rating error: {e}")

        # ==========================================
        # Pipeline Complete
        # ==========================================
        logger.info("\n" + "="*80)
        logger.info("Pipeline Complete!")
        logger.info("="*80)

        return result

    def get_current_ratings(
        self,
        processed_data: pd.DataFrame,
        min_targets: int = 30
    ) -> pd.DataFrame:
        """
        Get current WR LEAF ratings (most recent for each player).

        Args:
            processed_data: Output from process_full_pipeline()
            min_targets: Minimum targets to include WR

        Returns:
            DataFrame with current ratings
        """
        # Get most recent rating for each WR
        current = processed_data.sort_values(
            ['receiver_player_id', 'season', 'week']
        ).groupby('receiver_player_id').tail(1)

        # Calculate total targets
        total_targets = processed_data.groupby('receiver_player_id')['targets'].sum()
        current = current.merge(
            total_targets.rename('total_targets'),
            left_on='receiver_player_id',
            right_index=True
        )

        # Filter by minimum targets
        current = current[current['total_targets'] >= min_targets]

        # Sort by rating
        current = current.sort_values('wr_leaf_rating', ascending=False)

        return current[[
            'receiver_player_id', 'receiver_player_name',
            'season', 'week', 'wr_leaf_rating', 'wr_leaf_uncertainty',
            'total_targets', 'game_number'
        ]]

    def calculate_composite_rating(
        self,
        season_stats: pd.DataFrame,
        use_optimized_weights: bool = False
    ) -> pd.DataFrame:
        """
        Calculate WR composite rating from season-level statistics.

        Uses the optimal 5-metric combination validated across 2020-2024:
        - targets_per_game
        - receptions
        - total_yac_epa
        - first_downs
        - targets

        This composite focuses on WR-specific skills (YAC ability, volume, impact)
        and minimizes QB-dependent metrics.

        Args:
            season_stats: Season-aggregated statistics (from ReceiverStatsAggregator)
            use_optimized_weights: If True, use Ridge-optimized weights.
                                  If False, use equal weighting (more stable).

        Returns:
            season_stats with added 'wr_composite_rating' column
        """
        required_metrics = ['targets_per_game', 'receptions', 'total_yac_epa',
                           'first_downs', 'targets']

        # Check all required metrics are present
        missing = [m for m in required_metrics if m not in season_stats.columns]
        if missing:
            raise ValueError(f"Missing required metrics: {missing}")

        # Standardize each metric to z-scores
        standardized = season_stats.copy()
        for metric in required_metrics:
            mean = season_stats[metric].mean()
            std = season_stats[metric].std()
            if std > 0:
                standardized[f'{metric}_z'] = (season_stats[metric] - mean) / std
            else:
                standardized[f'{metric}_z'] = 0.0

        # Define weights (from multi-year validation - Rank 10, most stable)
        if use_optimized_weights:
            # Ridge alpha=10.0 optimized weights
            weights = {
                'targets_per_game': 0.14864289,
                'receptions': 0.18068462,
                'total_yac_epa': 0.27755785,
                'first_downs': 0.19066506,
                'targets': 0.20244958
            }
            logger.info("Using optimized weights for composite rating")
        else:
            # Equal weighting (more stable, nearly identical performance)
            weights = {metric: 0.20 for metric in required_metrics}
            logger.info("Using equal weights for composite rating")

        # Calculate weighted composite
        standardized['wr_composite_rating'] = sum(
            standardized[f'{metric}_z'] * weights[metric]
            for metric in required_metrics
        )

        # Add percentile rank
        standardized['wr_composite_percentile'] = (
            standardized['wr_composite_rating'].rank(pct=True) * 100
        )

        # Add to original dataframe
        season_stats = season_stats.copy()
        season_stats['wr_composite_rating'] = standardized['wr_composite_rating']
        season_stats['wr_composite_percentile'] = standardized['wr_composite_percentile']

        logger.info(f"  Calculated composite rating for {len(season_stats)} WRs")
        logger.info(f"  Mean: {season_stats['wr_composite_rating'].mean():.3f}")
        logger.info(f"  Std: {season_stats['wr_composite_rating'].std():.3f}")
        logger.info(f"  Range: [{season_stats['wr_composite_rating'].min():.3f}, "
                   f"{season_stats['wr_composite_rating'].max():.3f}]")

        return season_stats


def main():
    """Example usage of WR LEAF Pipeline."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from data_pipeline.nfl_data_fetcher import NFLDataFetcher
    from data_pipeline.receiver_stats_aggregator import ReceiverStatsAggregator

    # Fetch data
    logger.info("Fetching nflfastR data...")
    fetcher = NFLDataFetcher()
    pbp_2023 = fetcher.fetch_pbp_data([2023])
    pbp_2024 = fetcher.fetch_pbp_data([2024])
    pbp_full = fetcher.fetch_pbp_data([2023, 2024])

    # Fetch roster data for WR-only filtering
    logger.info("Fetching roster data...")
    rosters = fetcher.fetch_rosters([2023, 2024])

    # Aggregate WR stats (WR-only filtering)
    logger.info("Aggregating WR game statistics...")
    aggregator = ReceiverStatsAggregator(min_targets=1, filter_wr_only=True)
    game_stats_2023 = aggregator.aggregate_game_stats(pbp_2023, roster_data=rosters)
    game_stats_2024 = aggregator.aggregate_game_stats(pbp_2024, roster_data=rosters)

    # Get season-level stats for composite rating
    season_stats_2023 = aggregator.aggregate_season_stats(game_stats_2023, min_targets=30)
    season_stats_2024 = aggregator.aggregate_season_stats(game_stats_2024, min_targets=30)

    # Initialize pipeline
    pipeline = WRLEAFPipeline(
        use_context=True,
        use_opponent_adj=True,
        use_kalman=True
    )

    # Fit context model
    pipeline.fit_context_model(pbp_full, metrics=['epa', 'air_epa', 'yac_epa'])

    # Calculate secondary ratings
    secondary_ratings = pipeline.calculate_secondary_ratings(pbp_full)

    # Run full pipeline on 2024 data
    game_stats_full = aggregator.aggregate_game_stats(pbp_full, roster_data=rosters)
    processed = pipeline.process_full_pipeline(
        game_stats=game_stats_full,
        pbp_data=pbp_full,
        secondary_ratings=secondary_ratings
    )

    # Get current LEAF ratings (game-by-game)
    current_leaf = pipeline.get_current_ratings(processed, min_targets=30)

    # Calculate composite ratings (season-level)
    logger.info("\n" + "="*80)
    logger.info("Calculating WR Composite Ratings")
    logger.info("="*80)
    season_stats_2024_with_composite = pipeline.calculate_composite_rating(
        season_stats_2024,
        use_optimized_weights=False  # Use equal weighting for stability
    )

    # Display both ratings
    print(f"\n{'='*80}")
    print("Top 20 WRs - LEAF Ratings (2023-2024, Air EPA focus)")
    print(f"{'='*80}")
    print(current_leaf.head(20)[['receiver_player_name', 'wr_leaf_rating',
                            'wr_leaf_uncertainty', 'total_targets']].to_string(index=False))

    print(f"\n{'='*80}")
    print("Top 20 WRs - Composite Ratings (2024, YAC + Volume focus)")
    print(f"{'='*80}")
    composite_display = season_stats_2024_with_composite.sort_values(
        'wr_composite_rating', ascending=False
    ).head(20)
    print(composite_display[['receiver_player_name', 'wr_composite_rating',
                            'wr_composite_percentile', 'targets', 'total_yac_epa']].to_string(index=False))

    print(f"\n{'='*80}")
    print("WR LEAF Rating Statistics")
    print(f"{'='*80}")
    print(f"Mean: {processed['wr_leaf_rating'].mean():.4f}")
    print(f"Std: {processed['wr_leaf_rating'].std():.4f}")
    print(f"Min: {processed['wr_leaf_rating'].min():.4f}")
    print(f"Max: {processed['wr_leaf_rating'].max():.4f}")


if __name__ == "__main__":
    main()
