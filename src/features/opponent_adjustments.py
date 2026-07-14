"""
Opponent Defense Adjustments

Adjusts QB statistics based on opponent defense quality using an iterative approach.
Accounts for strength of schedule and provides opponent-adjusted metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DefenseRatingSystem:
    """
    Calculates defensive quality ratings for all NFL teams.

    Uses iterative approach to account for:
    - Strength of QBs faced
    - Time-weighting (recent games matter more)
    - Multiple defensive metrics
    """

    def __init__(
        self,
        decay_rate: float = 0.996,
        min_attempts: int = 100,
        iterations: int = 5
    ):
        """
        Initialize defense rating system.

        Args:
            decay_rate: Exponential decay for time weighting (beta value)
            min_attempts: Minimum pass attempts to include a defense
            iterations: Number of iterations for convergence
        """
        self.decay_rate = decay_rate
        self.min_attempts = min_attempts
        self.iterations = iterations
        self.defense_ratings = {}
        self.convergence_history = []

    def calculate_defense_ratings(
        self,
        pbp_data: pd.DataFrame,
        qb_quality: Optional[Dict[str, float]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate defensive ratings with iterative adjustment for QB quality.

        Args:
            pbp_data: Play-by-play data with QB passing plays
            qb_quality: Optional dict of {qb_id: quality_rating} for weighting

        Returns:
            Dictionary of {team: {metric: rating}} for all defenses
        """
        # Filter to passing plays only
        pass_plays = pbp_data[pbp_data['play_type'] == 'pass'].copy()

        # Initialize QB quality if not provided (use league average)
        if qb_quality is None:
            qb_quality = {qb: 0.0 for qb in pass_plays['passer_player_id'].unique()}

        # Iterate to convergence
        for iteration in range(self.iterations):
            logger.info(f"Defense rating iteration {iteration + 1}/{self.iterations}")

            # Calculate raw defensive metrics
            defense_ratings = self._calculate_raw_defense_metrics(
                pass_plays,
                qb_quality
            )

            # Update QB quality based on defenses faced
            if iteration < self.iterations - 1:  # Don't update on last iteration
                qb_quality = self._update_qb_quality(pass_plays, defense_ratings)

            # Track convergence
            self.convergence_history.append(defense_ratings.copy())

        self.defense_ratings = defense_ratings
        logger.info(f"Calculated ratings for {len(defense_ratings)} defenses")

        return defense_ratings

    def _calculate_raw_defense_metrics(
        self,
        pass_plays: pd.DataFrame,
        qb_quality: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate defensive metrics adjusted for QB quality faced.

        Args:
            pass_plays: Filtered passing plays
            qb_quality: Quality ratings for each QB

        Returns:
            Dictionary of defensive ratings
        """
        # Add QB quality to plays
        pass_plays['qb_quality'] = pass_plays['passer_player_id'].map(qb_quality).fillna(0)

        # Add time weights
        if 'game_date' in pass_plays.columns:
            reference_date = pd.to_datetime(pass_plays['game_date']).max()
            days_elapsed = (reference_date - pd.to_datetime(pass_plays['game_date'])).dt.days
            pass_plays['time_weight'] = self.decay_rate ** days_elapsed
        else:
            pass_plays['time_weight'] = 1.0

        # Group by defense (defteam = team on defense)
        defense_stats = {}

        for team in pass_plays['defteam'].unique():
            team_plays = pass_plays[pass_plays['defteam'] == team].copy()

            # Skip if insufficient data
            if len(team_plays) < self.min_attempts:
                continue

            # Calculate weighted metrics accounting for QB quality
            metrics = self._calculate_team_defense_metrics(team_plays)

            defense_stats[team] = metrics

        return defense_stats

    def _calculate_team_defense_metrics(
        self,
        team_plays: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate all defensive metrics for a single team.

        Args:
            team_plays: Plays against this defense

        Returns:
            Dictionary of defensive metrics
        """
        weights = team_plays['time_weight']
        qb_quality = team_plays['qb_quality']

        metrics = {}

        # EPA allowed (lower is better for defense)
        if 'qb_epa' in team_plays.columns and team_plays['qb_epa'].notna().any():
            # Adjust for QB quality: if facing good QBs, allowing EPA is more forgivable
            epa_values = team_plays['qb_epa'].fillna(0)
            weighted_epa = (epa_values * weights).sum() / weights.sum()

            # Adjust for average QB quality faced
            avg_qb_quality = (qb_quality * weights).sum() / weights.sum()
            metrics['epa_allowed'] = weighted_epa - avg_qb_quality
        else:
            metrics['epa_allowed'] = 0.0

        # CPOE allowed (lower is better for defense)
        if 'cpoe' in team_plays.columns and team_plays['cpoe'].notna().any():
            cpoe_values = team_plays['cpoe'].fillna(0)
            metrics['cpoe_allowed'] = (cpoe_values * weights).sum() / weights.sum()
        else:
            metrics['cpoe_allowed'] = 0.0

        # Success rate allowed (lower is better)
        if 'success' in team_plays.columns and team_plays['success'].notna().any():
            success_values = team_plays['success'].fillna(0)
            metrics['success_rate_allowed'] = (success_values * weights).sum() / weights.sum()
        else:
            metrics['success_rate_allowed'] = 0.5

        # TD rate allowed (lower is better)
        if 'pass_touchdown' in team_plays.columns:
            td_values = team_plays['pass_touchdown'].fillna(0)
            metrics['td_rate_allowed'] = (td_values * weights).sum() / weights.sum()
        else:
            metrics['td_rate_allowed'] = 0.045

        # INT rate generated (higher is better for defense)
        if 'interception' in team_plays.columns:
            int_values = team_plays['interception'].fillna(0)
            metrics['int_rate_generated'] = (int_values * weights).sum() / weights.sum()
        else:
            metrics['int_rate_generated'] = 0.025

        # Sack rate generated (higher is better for defense)
        if 'sack' in team_plays.columns:
            sack_values = team_plays['sack'].fillna(0)
            metrics['sack_rate_generated'] = (sack_values * weights).sum() / weights.sum()
        else:
            metrics['sack_rate_generated'] = 0.06

        # Pressure rate (if available)
        if 'was_pressure' in team_plays.columns:
            pressure_values = pd.to_numeric(team_plays['was_pressure'], errors='coerce').fillna(0)
            metrics['pressure_rate'] = (pressure_values * weights).sum() / weights.sum()
        else:
            metrics['pressure_rate'] = 0.25

        # Completion percentage allowed
        if 'complete_pass' in team_plays.columns:
            comp_values = team_plays['complete_pass'].fillna(0)
            metrics['completion_pct_allowed'] = (comp_values * weights).sum() / weights.sum()
        else:
            metrics['completion_pct_allowed'] = 0.64

        # Attempts faced
        metrics['attempts_faced'] = len(team_plays)

        return metrics

    def _update_qb_quality(
        self,
        pass_plays: pd.DataFrame,
        defense_ratings: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Update QB quality ratings based on defenses faced.

        Args:
            pass_plays: All passing plays
            defense_ratings: Current defensive ratings

        Returns:
            Updated QB quality dictionary
        """
        qb_quality = {}

        for qb_id in pass_plays['passer_player_id'].unique():
            qb_plays = pass_plays[pass_plays['passer_player_id'] == qb_id]

            if len(qb_plays) < 30:  # Skip QBs with very few attempts
                qb_quality[qb_id] = 0.0
                continue

            # Calculate QB performance vs expected based on defenses faced
            expected_epa = 0.0
            actual_epa = 0.0
            total_weight = 0.0

            for _, play in qb_plays.iterrows():
                defense = play['defteam']
                if defense not in defense_ratings:
                    continue

                weight = play.get('time_weight', 1.0)

                # Expected EPA = what defense typically allows
                expected_epa += defense_ratings[defense]['epa_allowed'] * weight

                # Actual EPA
                if pd.notna(play.get('qb_epa')):
                    actual_epa += play['qb_epa'] * weight

                total_weight += weight

            if total_weight > 0:
                # QB quality = how much better than expected vs defenses faced
                qb_quality[qb_id] = (actual_epa - expected_epa) / total_weight
            else:
                qb_quality[qb_id] = 0.0

        return qb_quality

    def get_defense_rankings(
        self,
        metric: str = 'epa_allowed',
        ascending: bool = True
    ) -> pd.DataFrame:
        """
        Get ranked list of defenses by a specific metric.

        Args:
            metric: Metric to rank by
            ascending: If True, lower is better (for EPA, CPOE, etc.)

        Returns:
            DataFrame with ranked defenses
        """
        if not self.defense_ratings:
            raise ValueError("Must call calculate_defense_ratings first")

        rankings = []
        for team, metrics in self.defense_ratings.items():
            if metric in metrics:
                rankings.append({
                    'team': team,
                    metric: metrics[metric],
                    'attempts_faced': metrics.get('attempts_faced', 0)
                })

        df = pd.DataFrame(rankings)
        df = df.sort_values(metric, ascending=ascending)
        df['rank'] = range(1, len(df) + 1)

        return df


class OpponentAdjuster:
    """
    Adjusts QB statistics based on opponent defense quality.

    Uses "performance over expected" approach:
    - Calculate what average QB would do vs this defense
    - QB's adjusted stat = actual - expected
    """

    def __init__(self, defense_ratings: Dict[str, Dict[str, float]]):
        """
        Initialize opponent adjuster.

        Args:
            defense_ratings: Defense ratings from DefenseRatingSystem
        """
        self.defense_ratings = defense_ratings

        # Calculate league averages for "expected" baseline
        self.league_averages = self._calculate_league_averages()

    def _calculate_league_averages(self) -> Dict[str, float]:
        """Calculate league average metrics across all defenses."""
        all_metrics = {}

        for team, metrics in self.defense_ratings.items():
            for metric, value in metrics.items():
                if metric != 'attempts_faced':
                    if metric not in all_metrics:
                        all_metrics[metric] = []
                    all_metrics[metric].append(value)

        league_avgs = {
            metric: np.mean(values)
            for metric, values in all_metrics.items()
        }

        return league_avgs

    def adjust_qb_game_stats(
        self,
        game_stats: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add opponent-adjusted columns to game-level QB stats.

        Args:
            game_stats: DataFrame with game-level QB statistics

        Returns:
            DataFrame with additional opponent-adjusted columns
        """
        adjusted = game_stats.copy()

        # Initialize adjusted columns
        adjusted['opp_adj_epa'] = np.nan
        adjusted['opp_adj_cpoe'] = np.nan
        adjusted['opp_adj_success_rate'] = np.nan
        adjusted['defense_quality'] = np.nan

        for idx, row in adjusted.iterrows():
            defense = row.get('defteam')

            if defense not in self.defense_ratings:
                # No rating available, use unadjusted
                adjusted.loc[idx, 'opp_adj_epa'] = row.get('epa_per_play', 0)
                adjusted.loc[idx, 'opp_adj_cpoe'] = row.get('cpoe_mean', 0)
                adjusted.loc[idx, 'opp_adj_success_rate'] = row.get('success_mean', 0.5)
                continue

            defense_metrics = self.defense_ratings[defense]

            # EPA adjustment
            expected_epa = defense_metrics['epa_allowed']
            actual_epa = row.get('epa_per_play', 0)
            adjusted.loc[idx, 'opp_adj_epa'] = actual_epa - expected_epa

            # CPOE adjustment
            expected_cpoe = defense_metrics['cpoe_allowed']
            actual_cpoe = row.get('cpoe_mean', 0)
            adjusted.loc[idx, 'opp_adj_cpoe'] = actual_cpoe - expected_cpoe

            # Success rate adjustment
            expected_success = defense_metrics['success_rate_allowed']
            actual_success = row.get('success_mean', 0.5)
            adjusted.loc[idx, 'opp_adj_success_rate'] = actual_success - expected_success

            # Store defense quality (negative EPA allowed = good defense)
            adjusted.loc[idx, 'defense_quality'] = -defense_metrics['epa_allowed']

        return adjusted

    def get_strength_of_schedule(
        self,
        game_stats: pd.DataFrame,
        player_col: str = 'passer_player_id'
    ) -> pd.DataFrame:
        """
        Calculate strength of schedule for each QB.

        Args:
            game_stats: Game-level stats with defteam column
            player_col: Column identifying players

        Returns:
            DataFrame with QB and their schedule strength
        """
        sos_data = []

        for player_id in game_stats[player_col].unique():
            player_games = game_stats[game_stats[player_col] == player_id]

            # Calculate average defense quality faced
            defense_qualities = []
            for defense in player_games['defteam']:
                if defense in self.defense_ratings:
                    # Negative EPA allowed = good defense
                    defense_qualities.append(-self.defense_ratings[defense]['epa_allowed'])

            if defense_qualities:
                sos_data.append({
                    player_col: player_id,
                    'player_name': player_games['passer_player_name'].iloc[0] if 'passer_player_name' in player_games.columns else '',
                    'avg_defense_quality': np.mean(defense_qualities),
                    'toughest_defense': np.max(defense_qualities),
                    'easiest_defense': np.min(defense_qualities),
                    'games': len(player_games)
                })

        sos_df = pd.DataFrame(sos_data)
        if len(sos_df) > 0:
            sos_df = sos_df.sort_values('avg_defense_quality', ascending=False)

        return sos_df


def main():
    """Example usage of opponent adjustment system."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from data_pipeline.nfl_data_fetcher import NFLDataFetcher
    from data_pipeline.qb_stats_aggregator import QBStatsAggregator

    # Fetch data
    fetcher = NFLDataFetcher()
    pbp = fetcher.get_qb_play_by_play([2024])

    # Calculate defense ratings
    defense_system = DefenseRatingSystem(iterations=5, min_attempts=100)
    defense_ratings = defense_system.calculate_defense_ratings(pbp)

    # Show top/bottom defenses
    print("\nTop 10 Defenses (by EPA allowed):")
    print(defense_system.get_defense_rankings('epa_allowed', ascending=True).head(10))

    print("\nWorst 10 Defenses (by EPA allowed):")
    print(defense_system.get_defense_rankings('epa_allowed', ascending=False).head(10))

    # Adjust QB stats
    aggregator = QBStatsAggregator()
    game_stats = aggregator.aggregate_game_stats(pbp)

    adjuster = OpponentAdjuster(defense_ratings)
    adjusted_stats = adjuster.adjust_qb_game_stats(game_stats)

    # Show strength of schedule
    print("\nStrength of Schedule (toughest defenses faced):")
    sos = adjuster.get_strength_of_schedule(game_stats)
    print(sos.head(10))


if __name__ == "__main__":
    main()
