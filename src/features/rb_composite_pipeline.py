"""
RB Composite Rating Pipeline

Implements the validated 5-metric RB composite from exhaustive analysis.

Based on multi-year validation (2020-2024):
- Multi-year stability: r = 0.7241 (±0.0556) - BEST of all positions!
- Consistency: 92.3%
- RBs are +16.9% more stable than QBs, +9.5% more stable than WRs

Validated Metrics (equal weighted):
1. targets_per_game (20%) - receiving volume
2. touches_per_game (20%) - total usage
3. total_yards_per_game (20%) - total production
4. rec_yards_per_game (20%) - receiving production
5. rush_share (20%) - usage split (rushing vs receiving)

Captures both rushing and receiving dimensions of RB value.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RBCompositePipeline:
    """
    RB Composite Rating calculation using validated multi-metric combination.

    Uses z-score standardization and equal weighting for optimal stability.
    """

    def __init__(self, use_optimized_weights: bool = False):
        """
        Initialize RB composite pipeline.

        Args:
            use_optimized_weights: If True, use ridge-optimized weights.
                                   If False, use equal weights (recommended).
        """
        self.use_optimized_weights = use_optimized_weights

        # Validated metrics from multi-year analysis
        self.required_metrics = [
            'targets_per_game',
            'touches_per_game',
            'total_yards_per_game',
            'rec_yards_per_game',
            'rush_share'
        ]

        # Equal weights (recommended for stability)
        self.equal_weights = {metric: 0.20 for metric in self.required_metrics}

        # Ridge-optimized weights (if needed in future)
        self.optimized_weights = {
            'targets_per_game': 0.21,
            'touches_per_game': 0.20,
            'total_yards_per_game': 0.20,
            'rec_yards_per_game': 0.21,
            'rush_share': 0.18
        }

    def calculate_composite_rating(
        self,
        season_stats: pd.DataFrame,
        season: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Calculate RB composite rating from season-level statistics.

        Args:
            season_stats: DataFrame with RB season statistics
            season: Optional season year for filtering

        Returns:
            DataFrame with added 'rb_composite_rating' column
        """
        logger.info(f"Calculating RB composite ratings for {len(season_stats)} RB-seasons...")

        # Filter to specific season if requested
        if season is not None:
            season_stats = season_stats[season_stats['season'] == season].copy()
        else:
            season_stats = season_stats.copy()

        # Check for required metrics
        missing = [m for m in self.required_metrics if m not in season_stats.columns]
        if missing:
            raise ValueError(f"Missing required metrics: {missing}")

        # Standardize metrics to z-scores
        standardized = season_stats.copy()

        for metric in self.required_metrics:
            # Remove inf/nan values
            valid_values = season_stats[metric].replace([np.inf, -np.inf], np.nan).dropna()

            if len(valid_values) < 2:
                logger.warning(f"Insufficient data for {metric}, setting z-scores to 0")
                standardized[f'{metric}_z'] = 0
                continue

            mean = valid_values.mean()
            std = valid_values.std()

            if std > 0:
                standardized[f'{metric}_z'] = (season_stats[metric] - mean) / std
            else:
                logger.warning(f"{metric} has zero variance, setting z-scores to 0")
                standardized[f'{metric}_z'] = 0

            # Handle inf/nan after standardization
            standardized[f'{metric}_z'] = standardized[f'{metric}_z'].replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0)

        # Choose weights
        weights = self.optimized_weights if self.use_optimized_weights else self.equal_weights

        # Calculate composite as weighted average of z-scores
        standardized['rb_composite_rating'] = sum(
            standardized[f'{metric}_z'] * weights[metric]
            for metric in self.required_metrics
        )

        # Add percentile ranking
        standardized['rb_composite_percentile'] = (
            standardized['rb_composite_rating'].rank(pct=True) * 100
        )

        # Copy composite back to original dataframe
        season_stats['rb_composite_rating'] = standardized['rb_composite_rating']
        season_stats['rb_composite_percentile'] = standardized['rb_composite_percentile']

        logger.info(f"  ✓ Calculated RB composite ratings")
        logger.info(f"  Range: [{season_stats['rb_composite_rating'].min():.3f}, "
                   f"{season_stats['rb_composite_rating'].max():.3f}]")
        logger.info(f"  Mean: {season_stats['rb_composite_rating'].mean():.3f}")
        logger.info(f"  Std: {season_stats['rb_composite_rating'].std():.3f}")

        return season_stats

    def calculate_multi_season_ratings(
        self,
        season_stats: pd.DataFrame,
        seasons: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Calculate RB composite ratings across multiple seasons.

        Each season is standardized independently to account for era effects.

        Args:
            season_stats: DataFrame with RB season statistics
            seasons: Optional list of seasons to include

        Returns:
            DataFrame with RB composite ratings for all seasons
        """
        if seasons is not None:
            season_stats = season_stats[season_stats['season'].isin(seasons)].copy()

        all_ratings = []

        for season in sorted(season_stats['season'].unique()):
            season_data = season_stats[season_stats['season'] == season].copy()
            rated_season = self.calculate_composite_rating(season_data, season=season)
            all_ratings.append(rated_season)

        result = pd.concat(all_ratings, ignore_index=True)

        logger.info(f"\n✓ Calculated RB composite ratings for {len(result)} RB-seasons "
                   f"across {len(result['season'].unique())} seasons")

        return result

    def get_top_rbs(
        self,
        season_stats: pd.DataFrame,
        n: int = 20,
        min_touches: int = 100
    ) -> pd.DataFrame:
        """
        Get top N RBs by composite rating.

        Args:
            season_stats: DataFrame with RB composite ratings
            n: Number of top RBs to return
            min_touches: Minimum touches threshold

        Returns:
            DataFrame with top N RBs
        """
        # Filter by minimum touches
        qualified = season_stats[season_stats['total_touches'] >= min_touches].copy()

        # Sort by composite rating
        top_rbs = qualified.nlargest(n, 'rb_composite_rating')

        return top_rbs[['player_name', 'season', 'team', 'rb_composite_rating',
                       'rb_composite_percentile', 'targets_per_game', 'touches_per_game',
                       'total_yards_per_game', 'rec_yards_per_game', 'rush_share', 'total_touches']]

    def compare_rbs(
        self,
        season_stats: pd.DataFrame,
        rb_names: List[str]
    ) -> pd.DataFrame:
        """
        Compare specific RBs across seasons.

        Args:
            season_stats: DataFrame with RB composite ratings
            rb_names: List of RB names to compare

        Returns:
            DataFrame with RB comparisons
        """
        # Filter to requested RBs
        comparison = season_stats[
            season_stats['player_name'].isin(rb_names)
        ].copy()

        # Sort by RB and season
        comparison = comparison.sort_values(['player_name', 'season'])

        return comparison[['player_name', 'season', 'team', 'rb_composite_rating',
                          'rb_composite_percentile', 'targets_per_game', 'touches_per_game',
                          'total_yards_per_game', 'rec_yards_per_game', 'rush_share']]

    def get_season_leaders(
        self,
        season_stats: pd.DataFrame,
        season: int,
        n: int = 10,
        min_touches: int = 100
    ) -> pd.DataFrame:
        """
        Get top RBs for a specific season.

        Args:
            season_stats: DataFrame with RB composite ratings
            season: Season year
            n: Number of top RBs to return
            min_touches: Minimum touches threshold

        Returns:
            DataFrame with season leaders
        """
        season_data = season_stats[season_stats['season'] == season].copy()
        return self.get_top_rbs(season_data, n=n, min_touches=min_touches)

    def classify_rb_role(self, season_stats: pd.DataFrame) -> pd.DataFrame:
        """
        Classify RBs into role archetypes based on usage.

        Args:
            season_stats: DataFrame with RB stats

        Returns:
            DataFrame with 'rb_role' column added
        """
        def get_role(rush_share_pct):
            if pd.isna(rush_share_pct):
                return 'unknown'
            elif rush_share_pct >= 85:
                return 'pure_rusher'
            elif rush_share_pct <= 50:
                return 'pass_catching_specialist'
            else:
                return 'three_down_back'

        season_stats['rb_role'] = season_stats['rush_share'].apply(get_role)

        return season_stats


def main():
    """Example usage of RB composite pipeline."""
    from ..data_pipeline.nfl_data_fetcher import NFLDataFetcher
    from ..data_pipeline.rb_stats_aggregator import RBStatsAggregator

    logger.info("="*80)
    logger.info("RB COMPOSITE RATING PIPELINE - EXAMPLE")
    logger.info("="*80)

    # Load data
    fetcher = NFLDataFetcher()
    aggregator = RBStatsAggregator(filter_rb_only=True)

    pbp_2023 = fetcher.fetch_pbp_data([2023])
    pbp_2024 = fetcher.fetch_pbp_data([2024])

    rosters_2023 = fetcher.fetch_rosters([2023])
    rosters_2024 = fetcher.fetch_rosters([2024])

    game_stats_2023 = aggregator.aggregate_game_stats(pbp_2023, rosters_2023)
    game_stats_2024 = aggregator.aggregate_game_stats(pbp_2024, rosters_2024)

    season_stats_2023 = aggregator.aggregate_season_stats(game_stats_2023, min_touches=50)
    season_stats_2024 = aggregator.aggregate_season_stats(game_stats_2024, min_touches=50)

    all_seasons = pd.concat([season_stats_2023, season_stats_2024], ignore_index=True)

    # Calculate composite ratings
    pipeline = RBCompositePipeline(use_optimized_weights=False)
    rated_rbs = pipeline.calculate_multi_season_ratings(all_seasons, seasons=[2023, 2024])

    # Classify RB roles
    rated_rbs = pipeline.classify_rb_role(rated_rbs)

    # Show 2024 leaders
    print("\n" + "="*80)
    print("2024 RB COMPOSITE LEADERS")
    print("="*80)
    leaders_2024 = pipeline.get_season_leaders(rated_rbs, season=2024, n=10)
    print(leaders_2024.to_string(index=False))

    # Show specific RB comparison
    print("\n" + "="*80)
    print("RB TRAJECTORY: Selected RBs")
    print("="*80)
    comparison = pipeline.compare_rbs(rated_rbs, ['C.McCaffrey', 'S.Barkley', 'D.Henry'])
    print(comparison.to_string(index=False))

    # Show role distribution
    print("\n" + "="*80)
    print("RB ROLE DISTRIBUTION (2024)")
    print("="*80)
    role_counts = rated_rbs[rated_rbs['season'] == 2024]['rb_role'].value_counts()
    print(role_counts)


if __name__ == "__main__":
    main()
