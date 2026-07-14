"""
QB Composite Rating Pipeline

Implements the validated 5-metric QB composite from exhaustive analysis.

Based on multi-year validation (2020-2024):
- Multi-year stability: r = 0.6194 (±0.1457)
- Consistency: 76.5%

Validated Metrics (equal weighted):
1. yards_per_game (20%)
2. completion_pct (20%)
3. yards_per_attempt (20%)
4. attempts_per_game (20%)
5. sack_rate (20% - inverted)

This is complementary to the existing QB LEAF Kalman filter system.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QBCompositePipeline:
    """
    QB Composite Rating calculation using validated multi-metric combination.

    Uses z-score standardization and equal weighting for optimal stability.
    """

    def __init__(self, use_optimized_weights: bool = False):
        """
        Initialize QB composite pipeline.

        Args:
            use_optimized_weights: If True, use ridge-optimized weights.
                                   If False, use equal weights (recommended).
        """
        self.use_optimized_weights = use_optimized_weights

        # Validated metrics from multi-year analysis
        self.required_metrics = [
            'yards_per_game',
            'completion_pct',
            'yards_per_attempt',
            'attempts_per_game',
            'sack_rate'
        ]

        # Equal weights (recommended for stability)
        self.equal_weights = {metric: 0.20 for metric in self.required_metrics}

        # Ridge-optimized weights (if needed in future)
        self.optimized_weights = {
            'yards_per_game': 0.21,
            'completion_pct': 0.19,
            'yards_per_attempt': 0.22,
            'attempts_per_game': 0.20,
            'sack_rate': 0.18
        }

    def calculate_composite_rating(
        self,
        season_stats: pd.DataFrame,
        season: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Calculate QB composite rating from season-level statistics.

        Args:
            season_stats: DataFrame with QB season statistics
            season: Optional season year for filtering

        Returns:
            DataFrame with added 'qb_composite_rating' column
        """
        logger.info(f"Calculating QB composite ratings for {len(season_stats)} QB-seasons...")

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

        # Invert sack_rate (lower is better)
        standardized['sack_rate_z'] = -standardized['sack_rate_z']

        # Choose weights
        weights = self.optimized_weights if self.use_optimized_weights else self.equal_weights

        # Calculate composite as weighted average of z-scores
        standardized['qb_composite_rating'] = sum(
            standardized[f'{metric}_z'] * weights[metric]
            for metric in self.required_metrics
        )

        # Add percentile ranking
        standardized['qb_composite_percentile'] = (
            standardized['qb_composite_rating'].rank(pct=True) * 100
        )

        # Copy composite back to original dataframe
        season_stats['qb_composite_rating'] = standardized['qb_composite_rating']
        season_stats['qb_composite_percentile'] = standardized['qb_composite_percentile']

        logger.info(f"  ✓ Calculated QB composite ratings")
        logger.info(f"  Range: [{season_stats['qb_composite_rating'].min():.3f}, "
                   f"{season_stats['qb_composite_rating'].max():.3f}]")
        logger.info(f"  Mean: {season_stats['qb_composite_rating'].mean():.3f}")
        logger.info(f"  Std: {season_stats['qb_composite_rating'].std():.3f}")

        return season_stats

    def calculate_multi_season_ratings(
        self,
        season_stats: pd.DataFrame,
        seasons: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Calculate QB composite ratings across multiple seasons.

        Each season is standardized independently to account for era effects.

        Args:
            season_stats: DataFrame with QB season statistics
            seasons: Optional list of seasons to include

        Returns:
            DataFrame with QB composite ratings for all seasons
        """
        if seasons is not None:
            season_stats = season_stats[season_stats['season'].isin(seasons)].copy()

        all_ratings = []

        for season in sorted(season_stats['season'].unique()):
            season_data = season_stats[season_stats['season'] == season].copy()
            rated_season = self.calculate_composite_rating(season_data, season=season)
            all_ratings.append(rated_season)

        result = pd.concat(all_ratings, ignore_index=True)

        logger.info(f"\n✓ Calculated QB composite ratings for {len(result)} QB-seasons "
                   f"across {len(result['season'].unique())} seasons")

        return result

    def get_top_qbs(
        self,
        season_stats: pd.DataFrame,
        n: int = 20,
        min_attempts: int = 100
    ) -> pd.DataFrame:
        """
        Get top N QBs by composite rating.

        Args:
            season_stats: DataFrame with QB composite ratings
            n: Number of top QBs to return
            min_attempts: Minimum attempts threshold

        Returns:
            DataFrame with top N QBs
        """
        # Filter by minimum attempts
        qualified = season_stats[season_stats['attempts'] >= min_attempts].copy()

        # Sort by composite rating
        top_qbs = qualified.nlargest(n, 'qb_composite_rating')

        return top_qbs[['passer_player_name', 'season', 'team', 'qb_composite_rating',
                       'qb_composite_percentile', 'yards_per_game', 'completion_pct',
                       'yards_per_attempt', 'attempts_per_game', 'sack_rate', 'attempts']]

    def compare_qbs(
        self,
        season_stats: pd.DataFrame,
        qb_names: List[str]
    ) -> pd.DataFrame:
        """
        Compare specific QBs across seasons.

        Args:
            season_stats: DataFrame with QB composite ratings
            qb_names: List of QB names to compare

        Returns:
            DataFrame with QB comparisons
        """
        # Filter to requested QBs
        comparison = season_stats[
            season_stats['passer_player_name'].isin(qb_names)
        ].copy()

        # Sort by QB and season
        comparison = comparison.sort_values(['passer_player_name', 'season'])

        return comparison[['passer_player_name', 'season', 'team', 'qb_composite_rating',
                          'qb_composite_percentile', 'yards_per_game', 'completion_pct',
                          'yards_per_attempt', 'attempts_per_game', 'sack_rate']]

    def get_season_leaders(
        self,
        season_stats: pd.DataFrame,
        season: int,
        n: int = 10,
        min_attempts: int = 100
    ) -> pd.DataFrame:
        """
        Get top QBs for a specific season.

        Args:
            season_stats: DataFrame with QB composite ratings
            season: Season year
            n: Number of top QBs to return
            min_attempts: Minimum attempts threshold

        Returns:
            DataFrame with season leaders
        """
        season_data = season_stats[season_stats['season'] == season].copy()
        return self.get_top_qbs(season_data, n=n, min_attempts=min_attempts)


def main():
    """Example usage of QB composite pipeline."""
    from ..data_pipeline.nfl_data_fetcher import NFLDataFetcher
    from ..data_pipeline.qb_stats_aggregator import QBStatsAggregator

    logger.info("="*80)
    logger.info("QB COMPOSITE RATING PIPELINE - EXAMPLE")
    logger.info("="*80)

    # Load data
    fetcher = NFLDataFetcher()
    aggregator = QBStatsAggregator()

    pbp_2023 = fetcher.fetch_pbp_data([2023])
    pbp_2024 = fetcher.fetch_pbp_data([2024])

    pbp_pass_2023 = pbp_2023[pbp_2023['play_type'] == 'pass'].copy()
    pbp_pass_2024 = pbp_2024[pbp_2024['play_type'] == 'pass'].copy()

    game_stats_2023 = aggregator.aggregate_game_stats(pbp_pass_2023)
    game_stats_2024 = aggregator.aggregate_game_stats(pbp_pass_2024)

    season_stats_2023 = aggregator.aggregate_season_stats(game_stats_2023, min_attempts=100)
    season_stats_2024 = aggregator.aggregate_season_stats(game_stats_2024, min_attempts=100)

    all_seasons = pd.concat([season_stats_2023, season_stats_2024], ignore_index=True)

    # Calculate composite ratings
    pipeline = QBCompositePipeline(use_optimized_weights=False)
    rated_qbs = pipeline.calculate_multi_season_ratings(all_seasons, seasons=[2023, 2024])

    # Show 2024 leaders
    print("\n" + "="*80)
    print("2024 QB COMPOSITE LEADERS")
    print("="*80)
    leaders_2024 = pipeline.get_season_leaders(rated_qbs, season=2024, n=10)
    print(leaders_2024.to_string(index=False))

    # Show specific QB comparison
    print("\n" + "="*80)
    print("QB TRAJECTORY: Selected QBs")
    print("="*80)
    comparison = pipeline.compare_qbs(rated_qbs, ['P.Mahomes', 'J.Allen', 'L.Jackson'])
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
