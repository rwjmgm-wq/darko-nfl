"""
RB Statistics Aggregator

Processes play-by-play data to calculate RB-level statistics including:
- Rushing: EPA, success rate, yards, TDs, fumbles
- Receiving: targets, receptions, receiving EPA, YAC
- Combined metrics for dual-threat evaluation
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RBStatsAggregator:
    """Aggregates RB statistics from play-by-play data."""

    def __init__(
        self,
        min_rush_attempts: int = 1,
        min_targets: int = 1,
        filter_rb_only: bool = True
    ):
        """
        Initialize the RB aggregator.

        Args:
            min_rush_attempts: Minimum rush attempts per game
            min_targets: Minimum targets per game for receiving stats
            filter_rb_only: If True, filter to RB position only
        """
        self.min_rush_attempts = min_rush_attempts
        self.min_targets = min_targets
        self.filter_rb_only = filter_rb_only

    def aggregate_game_stats(
        self,
        pbp_df: pd.DataFrame,
        roster_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Aggregate RB statistics at the game level (rushing + receiving).

        Args:
            pbp_df: Play-by-play DataFrame
            roster_data: Optional roster DataFrame for position filtering

        Returns:
            DataFrame with one row per RB per game
        """
        logger.info("Aggregating RB game-level statistics...")

        # Aggregate rushing and receiving separately
        rush_stats = self._aggregate_rushing_stats(pbp_df, roster_data)
        rec_stats = self._aggregate_receiving_stats(pbp_df, roster_data)

        # Merge rushing and receiving stats
        game_stats = self._merge_rush_and_rec_stats(rush_stats, rec_stats)

        # Calculate combined metrics
        game_stats = self._calculate_combined_metrics(game_stats)

        logger.info(f"  ✓ Aggregated stats for {len(game_stats)} RB-game combinations")

        return game_stats

    def _aggregate_rushing_stats(
        self,
        pbp_df: pd.DataFrame,
        roster_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Aggregate rushing statistics for RBs."""

        # Filter to rush plays only
        rush_plays = pbp_df[pbp_df['rush_attempt'] == 1].copy()

        if len(rush_plays) == 0:
            logger.warning("No rush plays found in data")
            return pd.DataFrame()

        # Filter to RB position if roster data provided
        if roster_data is not None and self.filter_rb_only:
            # Create roster lookup: season + player_id -> position
            roster_lookup = roster_data.groupby(['season', 'player_id'])['position'].first().reset_index()

            # Merge with rush plays
            rush_plays = rush_plays.merge(
                roster_lookup,
                left_on=['season', 'rusher_player_id'],
                right_on=['season', 'player_id'],
                how='left'
            )

            # Log position distribution before filtering
            if 'position' in rush_plays.columns:
                position_counts = rush_plays['position'].value_counts()
                total_rushes = len(rush_plays)
                logger.info(f"  Rushing position distribution:")
                for pos, count in position_counts.head(10).items():
                    pct = count / total_rushes * 100
                    logger.info(f"    {pos}: {count:,} rushes ({pct:.1f}%)")

                # Filter to RB only
                rush_plays = rush_plays[rush_plays['position'] == 'RB'].copy()
                logger.info(f"  Filtered to {len(rush_plays):,} RB rushing plays")

        # Group by game and rusher
        groupby_cols = ['game_id', 'game_date', 'season', 'week',
                       'rusher_player_id', 'rusher_player_name', 'posteam', 'defteam']

        # Filter to valid columns
        groupby_cols = [col for col in groupby_cols if col in rush_plays.columns]

        # Aggregate rushing stats
        rush_stats = rush_plays.groupby(groupby_cols).agg({
            'rush_attempt': 'sum',              # attempts
            'rushing_yards': 'sum',             # yards
            'epa': ['mean', 'sum'],             # rush_epa
            'success': 'mean',                  # rush_success_rate
            'rush_touchdown': 'sum',            # rush_tds
            'fumble_lost': 'sum',               # fumbles_lost
            'first_down_rush': 'sum',           # first_downs
            'tackled_for_loss': 'sum',          # tfl
        }).reset_index()

        # Flatten column names
        rush_stats.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                             for col in rush_stats.columns.values]

        # Rename for clarity
        rename_map = {
            'rush_attempt_sum': 'rush_attempts',
            'rushing_yards_sum': 'rush_yards',
            'epa_mean': 'rush_epa_per_attempt',
            'epa_sum': 'total_rush_epa',
            'success_mean': 'rush_success_rate',
            'rush_touchdown_sum': 'rush_tds',
            'fumble_lost_sum': 'rush_fumbles_lost',
            'first_down_rush_sum': 'rush_first_downs',
            'tackled_for_loss_sum': 'rush_tfl'
        }

        rush_stats.rename(columns=rename_map, inplace=True)

        # Calculate derived rushing metrics
        rush_stats['yards_per_carry'] = np.where(
            rush_stats['rush_attempts'] > 0,
            rush_stats['rush_yards'] / rush_stats['rush_attempts'],
            np.nan
        )

        rush_stats['rush_td_rate'] = np.where(
            rush_stats['rush_attempts'] > 0,
            rush_stats['rush_tds'] / rush_stats['rush_attempts'],
            np.nan
        )

        return rush_stats

    def _aggregate_receiving_stats(
        self,
        pbp_df: pd.DataFrame,
        roster_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Aggregate receiving statistics for RBs."""

        # Filter to pass plays with a receiver
        rec_plays = pbp_df[
            (pbp_df['pass_attempt'] == 1) &
            (pbp_df['receiver_player_id'].notna())
        ].copy()

        if len(rec_plays) == 0:
            logger.warning("No receiving plays found in data")
            return pd.DataFrame()

        # Filter to RB position if roster data provided
        if roster_data is not None and self.filter_rb_only:
            # Create roster lookup
            roster_lookup = roster_data.groupby(['season', 'player_id'])['position'].first().reset_index()

            # Merge with rec plays
            rec_plays = rec_plays.merge(
                roster_lookup,
                left_on=['season', 'receiver_player_id'],
                right_on=['season', 'player_id'],
                how='left'
            )

            # Log position distribution
            if 'position' in rec_plays.columns:
                position_counts = rec_plays['position'].value_counts()
                total_targets = len(rec_plays)
                logger.info(f"  Receiving position distribution:")
                for pos, count in position_counts.head(10).items():
                    pct = count / total_targets * 100
                    logger.info(f"    {pos}: {count:,} targets ({pct:.1f}%)")

                # Filter to RB only
                rec_plays = rec_plays[rec_plays['position'] == 'RB'].copy()
                logger.info(f"  Filtered to {len(rec_plays):,} RB receiving plays")

        # Group by game and receiver
        groupby_cols = ['game_id', 'game_date', 'season', 'week',
                       'receiver_player_id', 'receiver_player_name', 'posteam', 'defteam']

        groupby_cols = [col for col in groupby_cols if col in rec_plays.columns]

        # Aggregate receiving stats
        rec_stats = rec_plays.groupby(groupby_cols).agg({
            'pass_attempt': 'sum',              # targets
            'complete_pass': 'sum',             # receptions
            'yards_gained': 'sum',              # receiving_yards
            'epa': ['mean', 'sum'],             # rec_epa
            'air_epa': 'mean',                  # air_epa
            'yac_epa': 'mean',                  # yac_epa
            'success': 'mean',                  # rec_success_rate
            'touchdown': 'sum',                 # rec_tds
            'yards_after_catch': ['mean', 'sum'],  # yac
            'first_down': 'sum',                # first_downs
        }).reset_index()

        # Flatten column names
        rec_stats.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                            for col in rec_stats.columns.values]

        # Rename for clarity
        rename_map = {
            'pass_attempt_sum': 'targets',
            'complete_pass_sum': 'receptions',
            'yards_gained_sum': 'rec_yards',
            'epa_mean': 'rec_epa_per_target',
            'epa_sum': 'total_rec_epa',
            'air_epa_mean': 'air_epa_per_target',
            'yac_epa_mean': 'yac_epa_per_target',
            'success_mean': 'rec_success_rate',
            'touchdown_sum': 'rec_tds',
            'yards_after_catch_mean': 'avg_yac',
            'yards_after_catch_sum': 'total_yac',
            'first_down_sum': 'rec_first_downs'
        }

        rec_stats.rename(columns=rename_map, inplace=True)

        # Standardize player ID column name
        if 'receiver_player_id' in rec_stats.columns:
            rec_stats.rename(columns={
                'receiver_player_id': 'player_id',
                'receiver_player_name': 'player_name'
            }, inplace=True)

        # Calculate derived receiving metrics
        rec_stats['catch_rate'] = np.where(
            rec_stats['targets'] > 0,
            rec_stats['receptions'] / rec_stats['targets'],
            np.nan
        )

        rec_stats['yards_per_target'] = np.where(
            rec_stats['targets'] > 0,
            rec_stats['rec_yards'] / rec_stats['targets'],
            np.nan
        )

        rec_stats['yards_per_reception'] = np.where(
            rec_stats['receptions'] > 0,
            rec_stats['rec_yards'] / rec_stats['receptions'],
            np.nan
        )

        return rec_stats

    def _merge_rush_and_rec_stats(
        self,
        rush_stats: pd.DataFrame,
        rec_stats: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge rushing and receiving stats into unified game stats."""

        if len(rush_stats) == 0 and len(rec_stats) == 0:
            logger.warning("No RB stats to merge")
            return pd.DataFrame()

        # Standardize rush stats player ID column
        if 'rusher_player_id' in rush_stats.columns:
            rush_stats.rename(columns={
                'rusher_player_id': 'player_id',
                'rusher_player_name': 'player_name'
            }, inplace=True)

        # Merge on game and player
        merge_cols = ['game_id', 'season', 'week', 'player_id', 'player_name', 'posteam', 'defteam']
        merge_cols = [col for col in merge_cols if col in rush_stats.columns and col in rec_stats.columns]

        if len(rush_stats) > 0 and len(rec_stats) > 0:
            game_stats = rush_stats.merge(
                rec_stats,
                on=merge_cols,
                how='outer',
                suffixes=('', '_rec')
            )
        elif len(rush_stats) > 0:
            game_stats = rush_stats.copy()
        else:
            game_stats = rec_stats.copy()

        # Fill NaN values for players with only rushing or only receiving
        rush_cols = ['rush_attempts', 'rush_yards', 'rush_epa_per_attempt', 'total_rush_epa',
                    'rush_success_rate', 'rush_tds', 'rush_fumbles_lost', 'rush_first_downs',
                    'rush_tfl', 'yards_per_carry', 'rush_td_rate']

        rec_cols = ['targets', 'receptions', 'rec_yards', 'rec_epa_per_target', 'total_rec_epa',
                   'air_epa_per_target', 'yac_epa_per_target', 'rec_success_rate', 'rec_tds',
                   'avg_yac', 'total_yac', 'rec_first_downs', 'catch_rate', 'yards_per_target',
                   'yards_per_reception']

        for col in rush_cols:
            if col in game_stats.columns:
                game_stats[col] = game_stats[col].fillna(0)

        for col in rec_cols:
            if col in game_stats.columns:
                game_stats[col] = game_stats[col].fillna(0)

        # Handle game_date columns if duplicated
        if 'game_date' in game_stats.columns and 'game_date_rec' in game_stats.columns:
            game_stats['game_date'] = game_stats['game_date'].fillna(game_stats['game_date_rec'])
            game_stats.drop(columns=['game_date_rec'], inplace=True)

        return game_stats

    def _calculate_combined_metrics(self, game_stats: pd.DataFrame) -> pd.DataFrame:
        """Calculate combined rushing + receiving metrics."""

        # Total touches
        rush_attempts = game_stats.get('rush_attempts', 0)
        targets = game_stats.get('targets', 0)
        game_stats['total_touches'] = rush_attempts + targets

        # Total EPA
        total_rush_epa = game_stats.get('total_rush_epa', 0)
        total_rec_epa = game_stats.get('total_rec_epa', 0)
        game_stats['total_epa'] = total_rush_epa + total_rec_epa

        # EPA per touch
        game_stats['epa_per_touch'] = np.where(
            game_stats['total_touches'] > 0,
            game_stats['total_epa'] / game_stats['total_touches'],
            np.nan
        )

        # Rush share (% of touches that are rushes)
        game_stats['rush_share'] = np.where(
            game_stats['total_touches'] > 0,
            rush_attempts / game_stats['total_touches'],
            np.nan
        )

        # Total yards
        rush_yards = game_stats.get('rush_yards', 0)
        rec_yards = game_stats.get('rec_yards', 0)
        game_stats['total_yards'] = rush_yards + rec_yards

        # Total TDs
        rush_tds = game_stats.get('rush_tds', 0)
        rec_tds = game_stats.get('rec_tds', 0)
        game_stats['total_tds'] = rush_tds + rec_tds

        # Total first downs
        rush_fds = game_stats.get('rush_first_downs', 0)
        rec_fds = game_stats.get('rec_first_downs', 0)
        game_stats['total_first_downs'] = rush_fds + rec_fds

        # Fumble rate (per touch)
        fumbles = game_stats.get('rush_fumbles_lost', 0)
        game_stats['fumble_rate'] = np.where(
            game_stats['total_touches'] > 0,
            fumbles / game_stats['total_touches'],
            np.nan
        )

        # Sort by player and date
        game_stats = game_stats.sort_values(['player_id', 'game_date'])

        return game_stats

    def aggregate_season_stats(
        self,
        game_stats: pd.DataFrame,
        min_touches: int = 100
    ) -> pd.DataFrame:
        """
        Aggregate RB stats to season level.

        Args:
            game_stats: Game-level statistics
            min_touches: Minimum total touches (rushes + targets) to include

        Returns:
            DataFrame with season-level statistics
        """
        logger.info("Aggregating RB season-level statistics...")

        # Build aggregation dictionary
        agg_dict = {}

        # Rushing metrics
        if 'rush_attempts' in game_stats.columns:
            agg_dict['rush_attempts'] = 'sum'
        if 'rush_yards' in game_stats.columns:
            agg_dict['rush_yards'] = 'sum'
        if 'rush_epa_per_attempt' in game_stats.columns:
            agg_dict['rush_epa_per_attempt'] = 'mean'
        if 'total_rush_epa' in game_stats.columns:
            agg_dict['total_rush_epa'] = 'sum'
        if 'rush_success_rate' in game_stats.columns:
            agg_dict['rush_success_rate'] = 'mean'
        if 'rush_tds' in game_stats.columns:
            agg_dict['rush_tds'] = 'sum'
        if 'rush_fumbles_lost' in game_stats.columns:
            agg_dict['rush_fumbles_lost'] = 'sum'
        if 'rush_first_downs' in game_stats.columns:
            agg_dict['rush_first_downs'] = 'sum'
        if 'rush_tfl' in game_stats.columns:
            agg_dict['rush_tfl'] = 'sum'

        # Receiving metrics
        if 'targets' in game_stats.columns:
            agg_dict['targets'] = 'sum'
        if 'receptions' in game_stats.columns:
            agg_dict['receptions'] = 'sum'
        if 'rec_yards' in game_stats.columns:
            agg_dict['rec_yards'] = 'sum'
        if 'rec_epa_per_target' in game_stats.columns:
            agg_dict['rec_epa_per_target'] = 'mean'
        if 'total_rec_epa' in game_stats.columns:
            agg_dict['total_rec_epa'] = 'sum'
        if 'yac_epa_per_target' in game_stats.columns:
            agg_dict['yac_epa_per_target'] = 'mean'
        if 'rec_success_rate' in game_stats.columns:
            agg_dict['rec_success_rate'] = 'mean'
        if 'rec_tds' in game_stats.columns:
            agg_dict['rec_tds'] = 'sum'
        if 'total_yac' in game_stats.columns:
            agg_dict['total_yac'] = 'sum'
        if 'rec_first_downs' in game_stats.columns:
            agg_dict['rec_first_downs'] = 'sum'

        # Combined metrics
        if 'total_touches' in game_stats.columns:
            agg_dict['total_touches'] = 'sum'
        if 'total_epa' in game_stats.columns:
            agg_dict['total_epa'] = 'sum'
        if 'total_yards' in game_stats.columns:
            agg_dict['total_yards'] = 'sum'
        if 'total_tds' in game_stats.columns:
            agg_dict['total_tds'] = 'sum'
        if 'total_first_downs' in game_stats.columns:
            agg_dict['total_first_downs'] = 'sum'

        season_stats = game_stats.groupby(
            ['season', 'player_id', 'player_name']
        ).agg(agg_dict).reset_index()

        # Filter by minimum touches
        if 'total_touches' in season_stats.columns:
            season_stats = season_stats[season_stats['total_touches'] >= min_touches]

        # Games played
        games_played = game_stats.groupby(
            ['season', 'player_id']
        ).size().reset_index(name='games')

        season_stats = season_stats.merge(games_played, on=['season', 'player_id'])

        # Calculate derived season metrics
        if 'rush_attempts' in season_stats.columns and 'games' in season_stats.columns:
            season_stats['rush_attempts_per_game'] = season_stats['rush_attempts'] / season_stats['games']
            season_stats['yards_per_carry'] = season_stats['rush_yards'] / season_stats['rush_attempts']
            season_stats['rush_yards_per_game'] = season_stats['rush_yards'] / season_stats['games']
            season_stats['rush_td_rate'] = season_stats['rush_tds'] / season_stats['rush_attempts'] * 100

        if 'targets' in season_stats.columns and 'games' in season_stats.columns:
            season_stats['targets_per_game'] = season_stats['targets'] / season_stats['games']
            season_stats['catch_rate'] = season_stats['receptions'] / season_stats['targets'] * 100
            season_stats['yards_per_target'] = season_stats['rec_yards'] / season_stats['targets']
            season_stats['yards_per_reception'] = season_stats['rec_yards'] / season_stats['receptions']
            season_stats['rec_yards_per_game'] = season_stats['rec_yards'] / season_stats['games']

        if 'total_touches' in season_stats.columns:
            season_stats['touches_per_game'] = season_stats['total_touches'] / season_stats['games']
            season_stats['epa_per_touch'] = season_stats['total_epa'] / season_stats['total_touches']
            season_stats['yards_per_touch'] = season_stats['total_yards'] / season_stats['total_touches']
            season_stats['total_yards_per_game'] = season_stats['total_yards'] / season_stats['games']

            # Rush share
            season_stats['rush_share'] = season_stats['rush_attempts'] / season_stats['total_touches'] * 100

            # Fumble rate
            season_stats['fumble_rate'] = season_stats['rush_fumbles_lost'] / season_stats['total_touches'] * 100

        # Add team (most recent team in season)
        if 'posteam' in game_stats.columns:
            most_recent_team = (
                game_stats.sort_values(['season', 'player_id', 'week'])
                .groupby(['season', 'player_id'])
                ['posteam']
                .last()
                .reset_index()
                .rename(columns={'posteam': 'team'})
            )
            season_stats = season_stats.merge(
                most_recent_team,
                on=['season', 'player_id'],
                how='left'
            )

        logger.info(f"  ✓ Aggregated {len(season_stats)} RB-seasons")
        logger.info(f"  Minimum touches: {min_touches}")

        return season_stats


def main():
    """Example usage of RBStatsAggregator."""
    from nfl_data_fetcher import NFLDataFetcher

    # Fetch data
    fetcher = NFLDataFetcher()
    pbp_2024 = fetcher.fetch_pbp_data([2024])

    # Load roster data
    rosters_2024 = fetcher.fetch_roster_data([2024])

    # Aggregate stats
    aggregator = RBStatsAggregator()
    game_stats = aggregator.aggregate_game_stats(pbp_2024, rosters_2024)

    print(f"\nAggregated stats for {len(game_stats)} RB-game combinations")
    print(f"\nColumns: {game_stats.columns.tolist()}")

    # Show season stats
    season_stats = aggregator.aggregate_season_stats(game_stats, min_touches=50)

    print(f"\n{len(season_stats)} RBs with 50+ touches in 2024")
    print("\nTop 10 RBs by total EPA:")
    print(season_stats.nlargest(10, 'total_epa')[
        ['player_name', 'team', 'total_touches', 'total_epa', 'epa_per_touch',
         'rush_attempts', 'targets', 'rush_share']
    ])


if __name__ == "__main__":
    main()
