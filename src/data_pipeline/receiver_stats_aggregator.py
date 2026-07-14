"""
Receiver Statistics Aggregator

Aggregates receiver performance from play-by-play data at game and season level.

Key metrics:
- EPA per target
- Air EPA per target (EPA at catch point - independent of YAC)
- YAC EPA per target (EPA from yards after catch - pure WR skill)
- Catch rate
- Yards per target
- YAC (yards after catch)
- RACR (Receiver Air Conversion Ratio)

Enhanced for WR LEAF v1.0 compatibility.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReceiverStatsAggregator:
    """
    Aggregates receiver statistics from play-by-play data.

    Mirrors QBStatsAggregator but for pass catchers.
    """

    def __init__(self, min_targets: int = 1, filter_wr_only: bool = True):
        """
        Initialize aggregator.

        Args:
            min_targets: Minimum targets to include in game-level stats
            filter_wr_only: If True, filter to WR position only (excludes RBs, TEs)
        """
        self.min_targets = min_targets
        self.filter_wr_only = filter_wr_only

    def aggregate_game_stats(
        self,
        pbp_data: pd.DataFrame,
        roster_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Aggregate receiver stats at game level.

        Args:
            pbp_data: Play-by-play DataFrame from nflfastR
            roster_data: Optional roster DataFrame with position info
                        (required if filter_wr_only=True)

        Returns:
            DataFrame with game-level receiver statistics (WRs only if filter enabled)
        """
        logger.info("Aggregating receiver game-level statistics...")

        # Filter to pass attempts with receiver info
        receiver_plays = pbp_data[
            (pbp_data['pass_attempt'] == 1) &
            (pbp_data['receiver_player_id'].notna())
        ].copy()

        if len(receiver_plays) == 0:
            logger.warning("No receiver plays found in data")
            return pd.DataFrame()

        # Filter to WRs only if enabled
        if self.filter_wr_only:
            if roster_data is None:
                raise ValueError("roster_data required when filter_wr_only=True")

            # Create simplified roster lookup (most common position per player per season)
            roster_lookup = (
                roster_data
                .groupby(['season', 'player_id'])
                ['position']
                .agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])
                .reset_index()
            )

            # Merge position info
            receiver_plays = receiver_plays.merge(
                roster_lookup,
                left_on=['season', 'receiver_player_id'],
                right_on=['season', 'player_id'],
                how='left'
            )

            # Count positions before filtering
            total_receivers = receiver_plays['receiver_player_id'].nunique()
            position_counts = receiver_plays.groupby('position')['receiver_player_id'].nunique()

            logger.info(f"  Position distribution ({total_receivers} unique receivers):")
            for pos in ['WR', 'TE', 'RB', 'FB', 'QB']:
                if pos in position_counts.index:
                    count = position_counts[pos]
                    pct = 100 * count / total_receivers
                    logger.info(f"    {pos}: {count} receivers ({pct:.1f}%)")

            # Filter to WRs only
            receiver_plays = receiver_plays[receiver_plays['position'] == 'WR'].copy()

            wr_count = receiver_plays['receiver_player_id'].nunique()
            logger.info(f"  Filtered to {wr_count} WRs only")

            if len(receiver_plays) == 0:
                logger.warning("No WR plays found after filtering")
                return pd.DataFrame()

        # Group by game and receiver
        game_stats = receiver_plays.groupby(
            ['game_id', 'season', 'week', 'receiver_player_id', 'receiver_player_name',
             'posteam', 'defteam']
        ).apply(self._calculate_receiver_game_stats, include_groups=False).reset_index()

        # Filter by minimum targets
        game_stats = game_stats[game_stats['targets'] >= self.min_targets]

        logger.info(f"  ✓ Aggregated {len(game_stats):,} receiver-games")
        logger.info(f"  Unique receivers: {game_stats['receiver_player_id'].nunique()}")

        return game_stats

    def _calculate_receiver_game_stats(self, game_plays: pd.DataFrame) -> pd.Series:
        """
        Calculate statistics for a single receiver in a single game.

        Args:
            game_plays: All plays for this receiver in this game

        Returns:
            Series with game statistics
        """
        stats = {}

        # Basic counts
        stats['targets'] = len(game_plays)
        stats['receptions'] = game_plays['complete_pass'].sum()
        stats['touchdowns'] = game_plays['touchdown'].sum()

        # Catch rate
        stats['catch_rate'] = stats['receptions'] / stats['targets'] if stats['targets'] > 0 else 0

        # EPA metrics
        stats['epa_per_target'] = game_plays['epa'].mean()
        stats['total_epa'] = game_plays['epa'].sum()
        stats['success_rate'] = game_plays['success'].mean()

        # WR LEAF metrics: air_epa and yac_epa
        # air_epa: EPA at catch point (independent of YAC, minimizes QB influence)
        # yac_epa: EPA from yards after catch (pure WR skill post-catch)
        if 'air_epa' in game_plays.columns:
            stats['air_epa_per_target'] = game_plays['air_epa'].mean()
            stats['total_air_epa'] = game_plays['air_epa'].sum()
        else:
            stats['air_epa_per_target'] = np.nan
            stats['total_air_epa'] = np.nan

        if 'yac_epa' in game_plays.columns:
            stats['yac_epa_per_target'] = game_plays['yac_epa'].mean()
            stats['total_yac_epa'] = game_plays['yac_epa'].sum()
        else:
            stats['yac_epa_per_target'] = np.nan
            stats['total_yac_epa'] = np.nan

        # Yards metrics (only on completions)
        completions = game_plays[game_plays['complete_pass'] == 1]
        if len(completions) > 0:
            stats['yards'] = completions['yards_gained'].sum()
            stats['yards_per_reception'] = completions['yards_gained'].mean()
            stats['air_yards'] = completions['air_yards'].mean()
            stats['yac'] = completions['yards_after_catch'].mean()

            # RACR (Receiver Air Conversion Ratio): YAC / air_yards
            if completions['air_yards'].sum() > 0:
                stats['racr'] = completions['yards_after_catch'].sum() / completions['air_yards'].sum()
            else:
                stats['racr'] = np.nan
        else:
            stats['yards'] = 0
            stats['yards_per_reception'] = 0
            stats['air_yards'] = 0
            stats['yac'] = 0
            stats['racr'] = np.nan

        # Yards per target (all targets, not just catches)
        stats['yards_per_target'] = stats['yards'] / stats['targets'] if stats['targets'] > 0 else 0

        # Advanced metrics
        stats['first_downs'] = game_plays['first_down'].sum()
        stats['interceptions'] = game_plays['interception'].sum()  # INTs count against WR

        return pd.Series(stats)

    def aggregate_season_stats(
        self,
        game_stats: pd.DataFrame,
        min_targets: int = 30
    ) -> pd.DataFrame:
        """
        Aggregate to season level.

        Args:
            game_stats: Game-level statistics
            min_targets: Minimum targets to include receiver

        Returns:
            DataFrame with season-level statistics
        """
        logger.info("Aggregating receiver season-level statistics...")

        season_stats = game_stats.groupby(
            ['season', 'receiver_player_id', 'receiver_player_name']
        ).agg({
            'targets': 'sum',
            'receptions': 'sum',
            'yards': 'sum',
            'touchdowns': 'sum',
            'total_epa': 'sum',
            'epa_per_target': 'mean',  # Weighted by games
            'success_rate': 'mean',
            'catch_rate': 'mean',
            'yards_per_reception': 'mean',
            'yards_per_target': 'mean',
            'air_yards': 'mean',
            'yac': 'mean',
            'racr': 'mean',
            'first_downs': 'sum',
            'interceptions': 'sum',
            # WR LEAF metrics
            'total_air_epa': 'sum',
            'air_epa_per_target': 'mean',
            'total_yac_epa': 'sum',
            'yac_epa_per_target': 'mean'
        }).reset_index()

        # Filter by minimum targets
        season_stats = season_stats[season_stats['targets'] >= min_targets]

        # Games played
        games_played = game_stats.groupby(
            ['season', 'receiver_player_id']
        ).size().reset_index(name='games')

        season_stats = season_stats.merge(games_played, on=['season', 'receiver_player_id'])

        # Add team (most recent team played for in season)
        if 'posteam' in game_stats.columns:
            # Get most recent team by taking the team from the latest week
            most_recent_team = (
                game_stats.sort_values(['season', 'receiver_player_id', 'week'])
                .groupby(['season', 'receiver_player_id'])
                ['posteam']
                .last()
                .reset_index()
                .rename(columns={'posteam': 'team'})
            )
            season_stats = season_stats.merge(
                most_recent_team,
                on=['season', 'receiver_player_id'],
                how='left'
            )

        # Targets per game
        season_stats['targets_per_game'] = season_stats['targets'] / season_stats['games']

        logger.info(f"  ✓ Aggregated {len(season_stats)} receiver-seasons")
        logger.info(f"  Minimum targets: {min_targets}")

        return season_stats

    def calculate_team_receiver_quality(
        self,
        game_stats: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate aggregate receiver quality for each team-game.

        This gives us the "QB's supporting cast quality" metric.

        Args:
            game_stats: Game-level receiver statistics

        Returns:
            DataFrame with team-game-level receiver quality
        """
        logger.info("Calculating team-game receiver quality...")

        # Aggregate by team-game
        agg_dict = {
            'targets': 'sum',
            'receptions': 'sum',
            'epa_per_target': 'mean',  # Average EPA per target for all receivers
            'catch_rate': 'mean',
            'yards_per_target': 'mean',
            'success_rate': 'mean'
        }

        # Add WR LEAF metrics if available
        if 'air_epa_per_target' in game_stats.columns:
            agg_dict['air_epa_per_target'] = 'mean'
        if 'yac_epa_per_target' in game_stats.columns:
            agg_dict['yac_epa_per_target'] = 'mean'

        team_quality = game_stats.groupby(
            ['game_id', 'season', 'week', 'posteam']
        ).agg(agg_dict).reset_index()

        # Rename for clarity
        rename_dict = {
            'epa_per_target': 'receiver_corps_epa',
            'catch_rate': 'receiver_corps_catch_rate',
            'yards_per_target': 'receiver_corps_yards_per_target',
            'success_rate': 'receiver_corps_success_rate'
        }

        if 'air_epa_per_target' in team_quality.columns:
            rename_dict['air_epa_per_target'] = 'receiver_corps_air_epa'
        if 'yac_epa_per_target' in team_quality.columns:
            rename_dict['yac_epa_per_target'] = 'receiver_corps_yac_epa'

        team_quality = team_quality.rename(columns=rename_dict)

        logger.info(f"  ✓ Calculated receiver quality for {len(team_quality)} team-games")

        return team_quality

    def calculate_target_share(
        self,
        game_stats: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate each receiver's share of team targets.

        Args:
            game_stats: Game-level receiver statistics

        Returns:
            game_stats with target_share column added
        """
        # Calculate team totals
        team_totals = game_stats.groupby(
            ['game_id', 'posteam']
        )['targets'].sum().reset_index(name='team_targets')

        # Merge and calculate share
        stats_with_share = game_stats.merge(
            team_totals,
            on=['game_id', 'posteam'],
            how='left'
        )

        stats_with_share['target_share'] = (
            stats_with_share['targets'] / stats_with_share['team_targets']
        )

        return stats_with_share


def main():
    """Example usage of ReceiverStatsAggregator."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from data_pipeline.nfl_data_fetcher import NFLDataFetcher

    # Fetch data
    logger.info("Fetching nflfastR data...")
    fetcher = NFLDataFetcher()
    pbp = fetcher.fetch_pbp_data([2023, 2024])

    # Fetch roster data for position filtering
    logger.info("Fetching roster data...")
    rosters = fetcher.fetch_rosters([2023, 2024])

    # Aggregate receiver stats (WRs only by default)
    aggregator = ReceiverStatsAggregator(min_targets=1, filter_wr_only=True)
    game_stats = aggregator.aggregate_game_stats(pbp, roster_data=rosters)

    # Add target share
    game_stats = aggregator.calculate_target_share(game_stats)

    # Ensure output directory exists
    from pathlib import Path
    output_dir = Path('..') / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save game-level stats
    game_path = output_dir / 'receiver_game_stats.csv'
    game_stats.to_csv(game_path, index=False)
    logger.info(f"Saved receiver game stats to {game_path}")

    # Aggregate to season level
    season_stats = aggregator.aggregate_season_stats(game_stats, min_targets=30)
    season_path = output_dir / 'receiver_season_stats.csv'
    season_stats.to_csv(season_path, index=False)
    logger.info(f"Saved receiver season stats to {season_path}")

    # Calculate team receiver quality
    team_quality = aggregator.calculate_team_receiver_quality(game_stats)
    team_path = output_dir / 'team_receiver_quality.csv'
    team_quality.to_csv(team_path, index=False)
    logger.info(f"Saved team receiver quality to {team_path}")

    # Display top receivers
    print(f"\n{'='*80}")
    print("Top 10 Receivers by EPA per Target (2023-2024, min 30 targets)")
    print(f"{'='*80}")
    top_receivers = season_stats.nlargest(10, 'epa_per_target')

    # Display columns based on what's available
    display_cols = ['receiver_player_name', 'season', 'targets', 'epa_per_target',
                    'catch_rate', 'yards_per_target']
    if 'air_epa_per_target' in season_stats.columns:
        display_cols.insert(4, 'air_epa_per_target')
    if 'yac_epa_per_target' in season_stats.columns:
        display_cols.insert(5 if 'air_epa_per_target' in display_cols else 4, 'yac_epa_per_target')

    print(top_receivers[display_cols].to_string(index=False))

    # Display summary statistics for WR LEAF metrics
    if 'air_epa_per_target' in season_stats.columns and 'yac_epa_per_target' in season_stats.columns:
        print(f"\n{'='*80}")
        print("WR LEAF Metrics Summary")
        print(f"{'='*80}")
        print(f"Air EPA per Target - Mean: {season_stats['air_epa_per_target'].mean():.4f}, "
              f"Std: {season_stats['air_epa_per_target'].std():.4f}")
        print(f"YAC EPA per Target - Mean: {season_stats['yac_epa_per_target'].mean():.4f}, "
              f"Std: {season_stats['yac_epa_per_target'].std():.4f}")
        print(f"Total EPA per Target - Mean: {season_stats['epa_per_target'].mean():.4f}, "
              f"Std: {season_stats['epa_per_target'].std():.4f}")


if __name__ == "__main__":
    main()
