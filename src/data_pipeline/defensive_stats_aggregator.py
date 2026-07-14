"""
Defensive Statistics Aggregator

Aggregates EDGE rusher (DE/OLB) performance from play-by-play data at game and season level.

Key metrics:
- Sacks (full + half sacks)
- QB hits (pressures that contact QB)
- Tackles for loss (run disruption)
- Forced fumbles (playmaking)
- Tackles (solo + assist)
- EPA against (offensive EPA on plays where defender was involved)
- Success rate against

Note: This tracks PASS RUSH and RUN DEFENSE only. Coverage stats are not measured
due to limitations in publicly available data.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DefensiveStatsAggregator:
    """
    Aggregates EDGE rusher statistics from play-by-play data.

    Tracks sacks, QB hits, TFLs, forced fumbles, and tackles for
    defensive edge players (DE, OLB positions).
    """

    def __init__(self, min_plays: int = 1, edge_positions: List[str] = None):
        """
        Initialize aggregator.

        Args:
            min_plays: Minimum defensive plays to include in game-level stats
            edge_positions: List of positions to include (default: ['DL', 'LB'])
                           Note: nflfastR roster uses simplified positions:
                           - 'DL' = Defensive Line (includes DEs, DTs)
                           - 'LB' = Linebackers (includes OLB, ILB, MLB)
                           EDGE rushers are primarily DL, with some LBs in 3-4 schemes
        """
        self.min_plays = min_plays
        self.edge_positions = edge_positions or ['DL', 'LB']

    def aggregate_game_stats(
        self,
        pbp_data: pd.DataFrame,
        roster_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Aggregate EDGE rusher stats at game level.

        Args:
            pbp_data: Play-by-play DataFrame from nflfastR
            roster_data: Roster DataFrame with position info

        Returns:
            DataFrame with game-level EDGE rusher statistics
        """
        logger.info("Aggregating EDGE rusher game-level statistics...")

        # Create simplified roster lookup (most common position per player per season)
        roster_lookup = (
            roster_data
            .groupby(['season', 'player_id'])
            ['position']
            .agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])
            .reset_index()
        )

        # Get player name lookup
        name_lookup = (
            roster_data[['season', 'player_id', 'player_name']]
            .drop_duplicates()
        )

        # Extract defensive events across all plays
        logger.info("  Extracting defensive events from play-by-play data...")
        defensive_events = self._extract_all_defensive_events(pbp_data)

        if len(defensive_events) == 0:
            logger.warning("No defensive events found in data")
            return pd.DataFrame()

        logger.info(f"  Found {len(defensive_events):,} defensive events")

        # Merge position info
        defensive_events = defensive_events.merge(
            roster_lookup,
            on=['season', 'player_id'],
            how='left'
        )

        # Filter to EDGE positions only
        defensive_events = defensive_events[defensive_events['position'].isin(self.edge_positions)]

        logger.info(f"  Filtered to {len(defensive_events):,} EDGE rusher events")

        if len(defensive_events) == 0:
            logger.warning(f"No events found for positions: {self.edge_positions}")
            return pd.DataFrame()

        # Group by player-game and aggregate
        game_stats = defensive_events.groupby(
            ['game_id', 'season', 'week', 'player_id', 'position']
        ).apply(self._calculate_defender_game_stats, include_groups=False).reset_index()

        # Merge player names
        game_stats = game_stats.merge(
            name_lookup,
            on=['season', 'player_id'],
            how='left'
        )

        # Filter by minimum plays
        game_stats = game_stats[game_stats['total_plays'] >= self.min_plays]

        logger.info(f"  ✓ Aggregated {len(game_stats):,} EDGE rusher-games")
        logger.info(f"  Unique EDGE rushers: {game_stats['player_id'].nunique()}")
        logger.info(f"  Positions included: {', '.join(self.edge_positions)}")

        # Count positions
        position_counts = game_stats.groupby('position')['player_id'].nunique()
        for pos in position_counts.index:
            count = position_counts[pos]
            logger.info(f"    {pos}: {count} players")

        return game_stats

    def _extract_all_defensive_events(self, pbp_data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all defensive events from play-by-play data.

        Returns long DataFrame with one row per defensive event (sack, hit, TFL, tackle, etc.)

        Args:
            pbp_data: Play-by-play DataFrame

        Returns:
            DataFrame with columns: game_id, season, week, player_id, event_type,
                                   sack_value, epa, is_pass, is_run
        """
        events = []

        # Full sacks
        sacks = pbp_data[pbp_data['sack'] == 1].copy()
        for _, play in sacks.iterrows():
            if pd.notna(play.get('sack_player_id')):
                events.append({
                    'game_id': play['game_id'],
                    'season': play['season'],
                    'week': play['week'],
                    'player_id': play['sack_player_id'],
                    'event_type': 'sack',
                    'sack_value': 1.0,
                    'qb_hit_value': 0,
                    'tfl_value': 0,
                    'ff_value': 0,
                    'solo_tackle_value': 0,
                    'assist_tackle_value': 0,
                    'epa': -play.get('epa', 0),  # Negative EPA is good for defense
                    'is_pass': 1,
                    'is_run': 0
                })

        # Half sacks
        for _, play in sacks.iterrows():
            for half_col in ['half_sack_1_player_id', 'half_sack_2_player_id']:
                if pd.notna(play.get(half_col)):
                    events.append({
                        'game_id': play['game_id'],
                        'season': play['season'],
                        'week': play['week'],
                        'player_id': play[half_col],
                        'event_type': 'half_sack',
                        'sack_value': 0.5,
                        'qb_hit_value': 0,
                        'tfl_value': 0,
                        'ff_value': 0,
                        'solo_tackle_value': 0,
                        'assist_tackle_value': 0,
                        'epa': -play.get('epa', 0),
                        'is_pass': 1,
                        'is_run': 0
                    })

        # QB hits
        qb_hits = pbp_data[pbp_data['qb_hit'] == 1].copy()
        for _, play in qb_hits.iterrows():
            for hit_col in ['qb_hit_1_player_id', 'qb_hit_2_player_id']:
                if pd.notna(play.get(hit_col)):
                    events.append({
                        'game_id': play['game_id'],
                        'season': play['season'],
                        'week': play['week'],
                        'player_id': play[hit_col],
                        'event_type': 'qb_hit',
                        'sack_value': 0,
                        'qb_hit_value': 1,
                        'tfl_value': 0,
                        'ff_value': 0,
                        'solo_tackle_value': 0,
                        'assist_tackle_value': 0,
                        'epa': -play.get('epa', 0),
                        'is_pass': 1,
                        'is_run': 0
                    })

        # Tackles for loss
        tfls = pbp_data[pbp_data['tackled_for_loss'] == 1].copy()
        for _, play in tfls.iterrows():
            for tfl_col in ['tackle_for_loss_1_player_id', 'tackle_for_loss_2_player_id']:
                if pd.notna(play.get(tfl_col)):
                    events.append({
                        'game_id': play['game_id'],
                        'season': play['season'],
                        'week': play['week'],
                        'player_id': play[tfl_col],
                        'event_type': 'tfl',
                        'sack_value': 0,
                        'qb_hit_value': 0,
                        'tfl_value': 1,
                        'ff_value': 0,
                        'solo_tackle_value': 0,
                        'assist_tackle_value': 0,
                        'epa': -play.get('epa', 0),
                        'is_pass': 0,
                        'is_run': 1
                    })

        # Forced fumbles
        ffs = pbp_data[pbp_data['fumble_forced'] == 1].copy()
        for _, play in ffs.iterrows():
            for ff_col in ['forced_fumble_player_1_player_id', 'forced_fumble_player_2_player_id']:
                if pd.notna(play.get(ff_col)):
                    events.append({
                        'game_id': play['game_id'],
                        'season': play['season'],
                        'week': play['week'],
                        'player_id': play[ff_col],
                        'event_type': 'forced_fumble',
                        'sack_value': 0,
                        'qb_hit_value': 0,
                        'tfl_value': 0,
                        'ff_value': 1,
                        'solo_tackle_value': 0,
                        'assist_tackle_value': 0,
                        'epa': -play.get('epa', 0),
                        'is_pass': int(play.get('pass_attempt', 0) == 1),
                        'is_run': int(play.get('rush_attempt', 0) == 1)
                    })

        # Solo tackles
        solo_tackles = pbp_data[pbp_data['solo_tackle'] == 1].copy()
        for _, play in solo_tackles.iterrows():
            for tackle_col in ['solo_tackle_1_player_id', 'solo_tackle_2_player_id']:
                if pd.notna(play.get(tackle_col)):
                    events.append({
                        'game_id': play['game_id'],
                        'season': play['season'],
                        'week': play['week'],
                        'player_id': play[tackle_col],
                        'event_type': 'solo_tackle',
                        'sack_value': 0,
                        'qb_hit_value': 0,
                        'tfl_value': 0,
                        'ff_value': 0,
                        'solo_tackle_value': 1,
                        'assist_tackle_value': 0,
                        'epa': -play.get('epa', 0),
                        'is_pass': int(play.get('pass_attempt', 0) == 1),
                        'is_run': int(play.get('rush_attempt', 0) == 1)
                    })

        # Assist tackles
        assist_tackles = pbp_data[pbp_data['assist_tackle'] == 1].copy()
        for _, play in assist_tackles.iterrows():
            for i in range(1, 5):
                assist_col = f'assist_tackle_{i}_player_id'
                if pd.notna(play.get(assist_col)):
                    events.append({
                        'game_id': play['game_id'],
                        'season': play['season'],
                        'week': play['week'],
                        'player_id': play[assist_col],
                        'event_type': 'assist_tackle',
                        'sack_value': 0,
                        'qb_hit_value': 0,
                        'tfl_value': 0,
                        'ff_value': 0,
                        'solo_tackle_value': 0,
                        'assist_tackle_value': 1,
                        'epa': -play.get('epa', 0),
                        'is_pass': int(play.get('pass_attempt', 0) == 1),
                        'is_run': int(play.get('rush_attempt', 0) == 1)
                    })

        return pd.DataFrame(events)

    def _calculate_defender_game_stats(self, events: pd.DataFrame) -> pd.Series:
        """
        Calculate statistics for a single defender in a single game.

        Args:
            events: All defensive events for this player in this game

        Returns:
            Series with game statistics
        """
        stats = {}

        # Count events
        stats['sacks'] = events['sack_value'].sum()
        stats['qb_hits'] = events['qb_hit_value'].sum()
        stats['tfls'] = events['tfl_value'].sum()
        stats['forced_fumbles'] = events['ff_value'].sum()
        stats['solo_tackles'] = events['solo_tackle_value'].sum()
        stats['assist_tackles'] = events['assist_tackle_value'].sum()
        stats['total_tackles'] = stats['solo_tackles'] + stats['assist_tackles']

        # Play counts
        stats['total_plays'] = len(events)
        stats['pass_rush_plays'] = events[events['is_pass'] == 1].shape[0]
        stats['run_defense_plays'] = events[events['is_run'] == 1].shape[0]

        # Derived metrics
        stats['pressures'] = stats['sacks'] + stats['qb_hits']
        stats['impact_plays'] = stats['sacks'] + stats['tfls'] + stats['forced_fumbles']

        # EPA metrics (already negated, so higher is better for defense)
        stats['epa_against'] = events['epa'].mean() if len(events) > 0 else 0.0

        pass_events = events[events['is_pass'] == 1]
        stats['epa_against_pass'] = pass_events['epa'].mean() if len(pass_events) > 0 else 0.0

        run_events = events[events['is_run'] == 1]
        stats['epa_against_run'] = run_events['epa'].mean() if len(run_events) > 0 else 0.0

        return pd.Series(stats)

    def aggregate_season_stats(
        self,
        game_stats: pd.DataFrame,
        min_plays: int = 50
    ) -> pd.DataFrame:
        """
        Aggregate to season level.

        Args:
            game_stats: Game-level statistics
            min_plays: Minimum defensive plays to include player

        Returns:
            DataFrame with season-level statistics
        """
        logger.info("Aggregating EDGE rusher season-level statistics...")

        season_stats = game_stats.groupby(
            ['season', 'player_id', 'player_name', 'position']
        ).agg({
            'sacks': 'sum',
            'qb_hits': 'sum',
            'tfls': 'sum',
            'forced_fumbles': 'sum',
            'solo_tackles': 'sum',
            'assist_tackles': 'sum',
            'total_tackles': 'sum',
            'pressures': 'sum',
            'impact_plays': 'sum',
            'total_plays': 'sum',
            'pass_rush_plays': 'sum',
            'run_defense_plays': 'sum',
            'epa_against': 'mean',
            'epa_against_pass': 'mean',
            'epa_against_run': 'mean',
        }).reset_index()

        # Filter by minimum plays
        season_stats = season_stats[season_stats['total_plays'] >= min_plays]

        # Games played
        games_played = game_stats.groupby(
            ['season', 'player_id']
        ).size().reset_index(name='games')

        season_stats = season_stats.merge(games_played, on=['season', 'player_id'])

        # Calculate per-game rates
        season_stats['sacks_per_game'] = season_stats['sacks'] / season_stats['games']
        season_stats['qb_hits_per_game'] = season_stats['qb_hits'] / season_stats['games']
        season_stats['tfls_per_game'] = season_stats['tfls'] / season_stats['games']
        season_stats['pressures_per_game'] = season_stats['pressures'] / season_stats['games']
        season_stats['impact_plays_per_game'] = season_stats['impact_plays'] / season_stats['games']
        season_stats['tackles_per_game'] = season_stats['total_tackles'] / season_stats['games']

        logger.info(f"  ✓ Aggregated {len(season_stats)} EDGE rusher-seasons")
        logger.info(f"  Minimum plays: {min_plays}")

        return season_stats


def main():
    """Example usage of DefensiveStatsAggregator."""
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

    # Aggregate EDGE stats (DL + LB includes all edge rushers)
    aggregator = DefensiveStatsAggregator(min_plays=1, edge_positions=['DL', 'LB'])
    game_stats = aggregator.aggregate_game_stats(pbp, roster_data=rosters)

    # Ensure output directory exists
    output_dir = Path('..') / '..' / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save game-level stats
    game_path = output_dir / 'edge_game_stats.csv'
    game_stats.to_csv(game_path, index=False)
    logger.info(f"Saved EDGE game stats to {game_path}")

    # Aggregate to season level
    season_stats = aggregator.aggregate_season_stats(game_stats, min_plays=50)
    season_path = output_dir / 'edge_season_stats.csv'
    season_stats.to_csv(season_path, index=False)
    logger.info(f"Saved EDGE season stats to {season_path}")

    # Display top EDGE rushers
    print(f"\n{'='*80}")
    print("Top 10 EDGE Rushers by Sacks (2023-2024, min 50 plays)")
    print(f"{'='*80}")
    top_edge = season_stats.nlargest(10, 'sacks')

    display_cols = ['player_name', 'position', 'season', 'games', 'sacks',
                    'qb_hits', 'tfls', 'pressures', 'epa_against']
    print(top_edge[display_cols].to_string(index=False))

    print(f"\n{'='*80}")
    print("Top 10 EDGE Rushers by Pressures per Game")
    print(f"{'='*80}")
    top_pressure = season_stats.nlargest(10, 'pressures_per_game')
    display_cols2 = ['player_name', 'position', 'season', 'games', 'pressures_per_game',
                     'sacks_per_game', 'impact_plays_per_game']
    print(top_pressure[display_cols2].to_string(index=False))


if __name__ == "__main__":
    main()
