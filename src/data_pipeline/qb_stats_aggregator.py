"""
QB Statistics Aggregator

Processes play-by-play data to calculate QB-level statistics including:
- EPA (Expected Points Added)
- CPOE (Completion Percentage Over Expected)
- Success Rate
- Air Yards
- Traditional box score stats
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QBStatsAggregator:
    """Aggregates QB statistics from play-by-play data."""

    def __init__(self):
        """Initialize the aggregator."""
        pass

    def aggregate_game_stats(self, pbp_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate QB statistics at the game level.

        Args:
            pbp_df: Play-by-play DataFrame (filtered for passing plays)

        Returns:
            DataFrame with one row per QB per game
        """
        # Ensure we have required columns
        required_cols = ['game_id', 'game_date', 'passer_player_id', 'passer_player_name']
        missing = [col for col in required_cols if col not in pbp_df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Group by game and QB
        groupby_cols = ['game_id', 'game_date', 'season', 'week',
                       'passer_player_id', 'passer_player_name', 'posteam', 'defteam']

        # Filter for valid columns
        groupby_cols = [col for col in groupby_cols if col in pbp_df.columns]

        # Calculate aggregated stats
        game_stats = pbp_df.groupby(groupby_cols).agg({
            # EPA-based metrics
            'qb_epa': ['mean', 'sum', 'count'],
            'epa': ['mean', 'sum'],  # Raw EPA as backup

            # CPOE and success
            'cpoe': 'mean',
            'success': 'mean',

            # Traditional stats
            'complete_pass': 'sum',
            'incomplete_pass': 'sum',
            'interception': 'sum',
            'pass_touchdown': 'sum',
            'sack': 'sum',
            'yards_gained': 'sum',

            # Air yards and depth
            'air_yards': ['mean', 'sum'],
            'yards_after_catch': 'sum',

            # Advanced metrics
            'qb_hit': 'sum',
            'qb_scramble': 'sum',
            'pass_attempt': 'sum',

        }).reset_index()

        # Flatten column names
        game_stats.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                             for col in game_stats.columns.values]

        # Calculate derived metrics
        game_stats = self._calculate_derived_metrics(game_stats)

        # Sort by date
        game_stats = game_stats.sort_values(['passer_player_id', 'game_date'])

        logger.info(f"Aggregated stats for {len(game_stats)} QB-game combinations")

        return game_stats

    def _calculate_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate derived metrics from aggregated stats.

        Args:
            df: DataFrame with aggregated game stats

        Returns:
            DataFrame with additional derived metrics
        """
        # Attempts = completions + incompletions + sacks + interceptions
        if 'pass_attempt_sum' in df.columns:
            df['attempts'] = df['pass_attempt_sum']
        else:
            df['attempts'] = (
                df['complete_pass_sum'] +
                df['incomplete_pass_sum'] +
                df.get('sack_sum', 0)
            )

        df['completions'] = df['complete_pass_sum']

        # Completion percentage
        df['completion_pct'] = np.where(
            df['attempts'] > 0,
            df['completions'] / df['attempts'],
            np.nan
        )

        # Yards per attempt
        df['yards_per_attempt'] = np.where(
            df['attempts'] > 0,
            df['yards_gained_sum'] / df['attempts'],
            np.nan
        )

        # Air yards per attempt
        df['air_yards_per_attempt'] = np.where(
            df['attempts'] > 0,
            df['air_yards_sum'] / df['attempts'],
            np.nan
        )

        # EPA per play (use qb_epa if available, otherwise epa)
        if 'qb_epa_mean' in df.columns:
            df['epa_per_play'] = df['qb_epa_mean']
        elif 'epa_mean' in df.columns:
            df['epa_per_play'] = df['epa_mean']

        # Success rate
        if 'success_mean' not in df.columns:
            df['success_rate'] = np.nan
        else:
            df['success_rate'] = df['success_mean']

        # TD rate
        df['td_rate'] = np.where(
            df['attempts'] > 0,
            df['pass_touchdown_sum'] / df['attempts'],
            np.nan
        )

        # INT rate
        df['int_rate'] = np.where(
            df['attempts'] > 0,
            df['interception_sum'] / df['attempts'],
            np.nan
        )

        # Sack rate
        if 'sack_sum' in df.columns:
            df['sack_rate'] = np.where(
                df['attempts'] > 0,
                df['sack_sum'] / df['attempts'],
                np.nan
            )

        return df

    def aggregate_cumulative_stats(
        self,
        game_stats: pd.DataFrame,
        window: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Calculate cumulative or rolling statistics for each QB.

        Args:
            game_stats: DataFrame with game-level stats
            window: If specified, use rolling window instead of cumulative

        Returns:
            DataFrame with cumulative/rolling stats
        """
        cumulative_stats = []

        for player_id in game_stats['passer_player_id'].unique():
            player_games = game_stats[
                game_stats['passer_player_id'] == player_id
            ].sort_values('game_date').copy()

            # Calculate cumulative stats
            if window is None:
                # Cumulative sum
                player_games['cumulative_attempts'] = player_games['attempts'].cumsum()
                player_games['cumulative_completions'] = player_games['completions'].cumsum()
                player_games['cumulative_yards'] = player_games['yards_gained_sum'].cumsum()
                player_games['cumulative_tds'] = player_games['pass_touchdown_sum'].cumsum()
                player_games['cumulative_ints'] = player_games['interception_sum'].cumsum()

                # Cumulative EPA
                if 'qb_epa_sum' in player_games.columns:
                    player_games['cumulative_epa'] = player_games['qb_epa_sum'].cumsum()
            else:
                # Rolling window
                player_games['rolling_attempts'] = player_games['attempts'].rolling(window).sum()
                player_games['rolling_epa_per_play'] = player_games['epa_per_play'].rolling(window).mean()
                player_games['rolling_cpoe'] = player_games['cpoe'].rolling(window).mean()

            cumulative_stats.append(player_games)

        result = pd.concat(cumulative_stats, ignore_index=True)

        return result

    def aggregate_season_stats(
        self,
        game_stats: pd.DataFrame,
        min_attempts: int = 100
    ) -> pd.DataFrame:
        """
        Aggregate QB stats to season level for exhaustive analysis.

        Args:
            game_stats: Game-level statistics
            min_attempts: Minimum attempts to include QB

        Returns:
            DataFrame with season-level statistics
        """
        logger.info("Aggregating QB season-level statistics...")

        # Build aggregation dict only for columns that exist
        agg_dict = {}

        # Core metrics (should always exist)
        if 'attempts' in game_stats.columns:
            agg_dict['attempts'] = 'sum'
        if 'completions' in game_stats.columns:
            agg_dict['completions'] = 'sum'
        if 'yards_gained_sum' in game_stats.columns:
            agg_dict['yards_gained_sum'] = 'sum'
        if 'pass_touchdown_sum' in game_stats.columns:
            agg_dict['pass_touchdown_sum'] = 'sum'
        if 'interception_sum' in game_stats.columns:
            agg_dict['interception_sum'] = 'sum'

        # Advanced metrics (may not exist)
        if 'epa_per_play' in game_stats.columns:
            agg_dict['epa_per_play'] = 'mean'
        if 'cpoe' in game_stats.columns:
            agg_dict['cpoe'] = 'mean'
        if 'success_rate' in game_stats.columns:
            agg_dict['success_rate'] = 'mean'
        if 'air_yards_mean' in game_stats.columns:
            agg_dict['air_yards_mean'] = 'mean'
        if 'air_yards_sum' in game_stats.columns:
            agg_dict['air_yards_sum'] = 'sum'
        if 'yards_after_catch_sum' in game_stats.columns:
            agg_dict['yards_after_catch_sum'] = 'sum'
        if 'qb_scramble_sum' in game_stats.columns:
            agg_dict['qb_scramble_sum'] = 'sum'
        if 'sack_sum' in game_stats.columns:
            agg_dict['sack_sum'] = 'sum'

        season_stats = game_stats.groupby(
            ['season', 'passer_player_id', 'passer_player_name']
        ).agg(agg_dict).reset_index()

        # Add EPA sums if available
        if 'qb_epa_sum' in game_stats.columns:
            epa_sums = game_stats.groupby(['season', 'passer_player_id']).agg({
                'qb_epa_sum': 'sum'
            }).reset_index()
            season_stats = season_stats.merge(epa_sums, on=['season', 'passer_player_id'], how='left')
            season_stats.rename(columns={'qb_epa_sum': 'total_epa'}, inplace=True)
        elif 'epa_sum' in game_stats.columns:
            epa_sums = game_stats.groupby(['season', 'passer_player_id']).agg({
                'epa_sum': 'sum'
            }).reset_index()
            season_stats = season_stats.merge(epa_sums, on=['season', 'passer_player_id'], how='left')
            season_stats.rename(columns={'epa_sum': 'total_epa'}, inplace=True)

        # Rename columns for clarity (only if they exist)
        rename_map = {}
        if 'yards_gained_sum' in season_stats.columns:
            rename_map['yards_gained_sum'] = 'yards'
        if 'pass_touchdown_sum' in season_stats.columns:
            rename_map['pass_touchdown_sum'] = 'touchdowns'
        if 'interception_sum' in season_stats.columns:
            rename_map['interception_sum'] = 'interceptions'
        if 'air_yards_mean' in season_stats.columns:
            rename_map['air_yards_mean'] = 'air_yards'
        if 'air_yards_sum' in season_stats.columns:
            rename_map['air_yards_sum'] = 'total_air_yards'
        if 'yards_after_catch_sum' in season_stats.columns:
            rename_map['yards_after_catch_sum'] = 'total_yac'
        if 'qb_scramble_sum' in season_stats.columns:
            rename_map['qb_scramble_sum'] = 'scrambles'
        if 'sack_sum' in season_stats.columns:
            rename_map['sack_sum'] = 'sacks'

        if rename_map:
            season_stats.rename(columns=rename_map, inplace=True)

        # Filter by minimum attempts
        season_stats = season_stats[season_stats['attempts'] >= min_attempts]

        # Games played
        games_played = game_stats.groupby(
            ['season', 'passer_player_id']
        ).size().reset_index(name='games')

        season_stats = season_stats.merge(games_played, on=['season', 'passer_player_id'])

        # Calculate derived metrics
        season_stats['completion_pct'] = (season_stats['completions'] / season_stats['attempts'] * 100)
        season_stats['yards_per_attempt'] = season_stats['yards'] / season_stats['attempts']
        season_stats['td_rate'] = (season_stats['touchdowns'] / season_stats['attempts'] * 100)
        season_stats['int_rate'] = (season_stats['interceptions'] / season_stats['attempts'] * 100)
        season_stats['sack_rate'] = (season_stats['sacks'] / season_stats['attempts'] * 100)
        season_stats['attempts_per_game'] = season_stats['attempts'] / season_stats['games']
        season_stats['yards_per_game'] = season_stats['yards'] / season_stats['games']

        # Add team (most recent team in season)
        if 'posteam' in game_stats.columns:
            most_recent_team = (
                game_stats.sort_values(['season', 'passer_player_id', 'week'])
                .groupby(['season', 'passer_player_id'])
                ['posteam']
                .last()
                .reset_index()
                .rename(columns={'posteam': 'team'})
            )
            season_stats = season_stats.merge(
                most_recent_team,
                on=['season', 'passer_player_id'],
                how='left'
            )

        logger.info(f"  ✓ Aggregated {len(season_stats)} QB-seasons")
        logger.info(f"  Minimum attempts: {min_attempts}")

        return season_stats

    def get_latest_stats(
        self,
        game_stats: pd.DataFrame,
        min_attempts: int = 50
    ) -> pd.DataFrame:
        """
        Get the most recent season stats for each QB.

        Args:
            game_stats: DataFrame with game-level stats
            min_attempts: Minimum attempts to include QB

        Returns:
            DataFrame with one row per QB (latest season)
        """
        # Use aggregate_season_stats and filter to latest season
        season_stats = self.aggregate_season_stats(game_stats, min_attempts)

        # Get latest season for each QB
        latest_season = season_stats.groupby('passer_player_id')['season'].max()
        latest_stats = season_stats.merge(
            latest_season.reset_index(),
            on=['passer_player_id', 'season']
        )

        return latest_stats


def main():
    """Example usage of QBStatsAggregator."""
    from nfl_data_fetcher import NFLDataFetcher

    # Fetch data
    fetcher = NFLDataFetcher()
    qb_pbp = fetcher.get_qb_play_by_play([2023, 2024])

    # Aggregate stats
    aggregator = QBStatsAggregator()
    game_stats = aggregator.aggregate_game_stats(qb_pbp)

    print(f"\nAggregated stats for {len(game_stats)} QB-game combinations")
    print(f"\nColumns: {game_stats.columns.tolist()}")

    # Show top QBs by EPA
    latest = aggregator.get_latest_stats(game_stats, min_attempts=100)
    latest_sorted = latest.sort_values('epa_per_play', ascending=False)

    print("\nTop 10 QBs by EPA/play (2024 season):")
    print(latest_sorted[['passer_player_name', 'attempts', 'epa_per_play',
                         'cpoe', 'completion_pct', 'yards_per_attempt']].head(10))


if __name__ == "__main__":
    main()
