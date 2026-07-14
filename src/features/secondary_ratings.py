"""
Secondary Defense Ratings

Rates NFL secondaries (pass coverage units) based on performance against receivers.
Used for WR LEAF opponent adjustments.

Key differences from overall defense ratings:
- Focuses on pass defense only (not overall defense)
- Rates against WR performance (air_epa, yac_epa, catch rate)
- Accounts for WR quality faced (iterative approach)
- Time-weighted for recent performance

Designed for WR LEAF v1.0 system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecondaryRatingSystem:
    """
    Calculates pass coverage quality ratings for all NFL secondaries.

    Uses iterative approach to account for:
    - Strength of WRs faced
    - Time-weighting (recent games matter more)
    - Multiple pass defense metrics (EPA allowed, catch rate allowed)
    """

    def __init__(
        self,
        decay_rate: float = 0.996,
        min_targets: int = 50,
        iterations: int = 5
    ):
        """
        Initialize secondary rating system.

        Args:
            decay_rate: Exponential decay for time weighting (beta value)
            min_targets: Minimum targets to include a secondary
            iterations: Number of iterations for convergence
        """
        self.decay_rate = decay_rate
        self.min_targets = min_targets
        self.iterations = iterations
        self.secondary_ratings = {}
        self.convergence_history = []

    def calculate_secondary_ratings(
        self,
        pbp_data: pd.DataFrame,
        wr_quality: Optional[Dict[str, float]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate secondary ratings with iterative adjustment for WR quality.

        Args:
            pbp_data: Play-by-play data with receiver targets
            wr_quality: Optional dict of {receiver_id: quality_rating} for weighting

        Returns:
            Dictionary of {team: {metric: rating}} for all secondaries
        """
        # Filter to pass attempts with receiver info
        receiver_plays = pbp_data[
            (pbp_data['pass_attempt'] == 1) &
            (pbp_data['receiver_player_id'].notna())
        ].copy()

        logger.info(f"Calculating secondary ratings from {len(receiver_plays):,} receiver targets")

        # Initialize WR quality if not provided (use league average)
        if wr_quality is None:
            wr_quality = {wr: 0.0 for wr in receiver_plays['receiver_player_id'].unique()}

        # Iterate to convergence
        for iteration in range(self.iterations):
            logger.info(f"Secondary rating iteration {iteration + 1}/{self.iterations}")

            # Calculate raw secondary metrics
            secondary_ratings = self._calculate_raw_secondary_metrics(
                receiver_plays,
                wr_quality
            )

            # Update WR quality based on secondaries faced
            if iteration < self.iterations - 1:  # Don't update on last iteration
                wr_quality = self._update_wr_quality(receiver_plays, secondary_ratings)

            # Track convergence
            self.convergence_history.append(secondary_ratings.copy())

        self.secondary_ratings = secondary_ratings
        logger.info(f"Calculated ratings for {len(secondary_ratings)} secondaries")

        return secondary_ratings

    def _calculate_raw_secondary_metrics(
        self,
        receiver_plays: pd.DataFrame,
        wr_quality: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate secondary metrics adjusted for WR quality faced.

        Args:
            receiver_plays: Filtered receiver target plays
            wr_quality: Quality ratings for each WR

        Returns:
            Dictionary of secondary ratings
        """
        # Add WR quality to plays
        receiver_plays['wr_quality'] = receiver_plays['receiver_player_id'].map(wr_quality).fillna(0)

        # Add time weights (more recent games weighted higher)
        if 'game_date' in receiver_plays.columns:
            reference_date = pd.to_datetime(receiver_plays['game_date']).max()
            days_elapsed = (reference_date - pd.to_datetime(receiver_plays['game_date'])).dt.days
            receiver_plays['time_weight'] = self.decay_rate ** days_elapsed
        else:
            receiver_plays['time_weight'] = 1.0

        # Group by defense (defteam = team on defense)
        secondary_stats = {}

        for team in receiver_plays['defteam'].unique():
            team_plays = receiver_plays[receiver_plays['defteam'] == team].copy()

            # Skip if insufficient data
            if len(team_plays) < self.min_targets:
                continue

            # Calculate weighted metrics accounting for WR quality
            metrics = self._calculate_team_secondary_metrics(team_plays)

            secondary_stats[team] = metrics

        return secondary_stats

    def _calculate_team_secondary_metrics(
        self,
        team_plays: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate all pass coverage metrics for a single team's secondary.

        Args:
            team_plays: Receiver targets against this secondary

        Returns:
            Dictionary of secondary metrics
        """
        weights = team_plays['time_weight']
        wr_quality = team_plays['wr_quality']

        metrics = {}

        # Total targets faced
        metrics['targets'] = len(team_plays)

        # ==========================================
        # EPA Metrics (lower is better for defense)
        # ==========================================

        # Total EPA allowed (raw)
        if 'epa' in team_plays.columns:
            valid_epa = team_plays['epa'].notna()
            if valid_epa.sum() > 0:
                metrics['epa_allowed'] = np.average(
                    team_plays.loc[valid_epa, 'epa'],
                    weights=weights[valid_epa]
                )
            else:
                metrics['epa_allowed'] = 0.0

        # Air EPA allowed (at catch point - coverage quality)
        if 'air_epa' in team_plays.columns:
            valid_air = team_plays['air_epa'].notna()
            if valid_air.sum() > 0:
                metrics['air_epa_allowed'] = np.average(
                    team_plays.loc[valid_air, 'air_epa'],
                    weights=weights[valid_air]
                )
            else:
                metrics['air_epa_allowed'] = 0.0

        # YAC EPA allowed (tackling ability)
        if 'yac_epa' in team_plays.columns:
            valid_yac = team_plays['yac_epa'].notna()
            if valid_yac.sum() > 0:
                metrics['yac_epa_allowed'] = np.average(
                    team_plays.loc[valid_yac, 'yac_epa'],
                    weights=weights[valid_yac]
                )
            else:
                metrics['yac_epa_allowed'] = 0.0

        # ==========================================
        # Completion/Coverage Metrics
        # ==========================================

        # Catch rate allowed (lower is better)
        if 'complete_pass' in team_plays.columns:
            metrics['catch_rate_allowed'] = np.average(
                team_plays['complete_pass'],
                weights=weights
            )

        # Success rate allowed (lower is better)
        if 'success' in team_plays.columns:
            metrics['success_rate_allowed'] = np.average(
                team_plays['success'],
                weights=weights
            )

        # ==========================================
        # Big Play Prevention
        # ==========================================

        # Yards per target allowed
        if 'yards_gained' in team_plays.columns:
            metrics['yards_per_target_allowed'] = np.average(
                team_plays['yards_gained'].fillna(0),
                weights=weights
            )

        # Deep ball defense (targets > 15 air yards)
        if 'air_yards' in team_plays.columns:
            deep_targets = team_plays[team_plays['air_yards'] > 15]
            if len(deep_targets) > 0:
                metrics['deep_epa_allowed'] = np.average(
                    deep_targets['epa'],
                    weights=deep_targets['time_weight']
                )
                metrics['deep_catch_rate_allowed'] = np.average(
                    deep_targets['complete_pass'],
                    weights=deep_targets['time_weight']
                )
            else:
                metrics['deep_epa_allowed'] = 0.0
                metrics['deep_catch_rate_allowed'] = 0.0

        # ==========================================
        # Quality-Adjusted Rating (overall)
        # ==========================================

        # Adjust for WR quality faced
        # Negative values = good defense (prevented expected EPA)
        if 'epa' in team_plays.columns:
            expected_epa = wr_quality.mean()  # Average WR quality faced
            actual_epa = metrics.get('epa_allowed', 0.0)
            metrics['quality_adj_rating'] = -(actual_epa - expected_epa)  # Negative = good

        return metrics

    def _update_wr_quality(
        self,
        receiver_plays: pd.DataFrame,
        secondary_ratings: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Update WR quality estimates based on secondaries faced.

        Args:
            receiver_plays: Receiver target plays
            secondary_ratings: Current secondary ratings

        Returns:
            Updated WR quality dictionary
        """
        wr_quality = {}

        # Create a mapping of defense quality (use air_epa_allowed as proxy)
        defense_quality = {
            team: ratings.get('air_epa_allowed', 0.0)
            for team, ratings in secondary_ratings.items()
        }

        # Update each WR's quality estimate
        for wr_id in receiver_plays['receiver_player_id'].unique():
            wr_plays = receiver_plays[receiver_plays['receiver_player_id'] == wr_id]

            # Get WR's raw EPA
            if 'air_epa' in wr_plays.columns:
                raw_epa = wr_plays['air_epa'].mean()

                # Adjust for difficulty of secondaries faced
                defenses_faced = wr_plays['defteam'].map(defense_quality).fillna(0)
                avg_defense_quality = defenses_faced.mean()

                # Quality = raw performance - defense difficulty
                wr_quality[wr_id] = raw_epa - avg_defense_quality
            else:
                wr_quality[wr_id] = 0.0

        return wr_quality

    def get_team_rating(
        self,
        team: str,
        metric: str = 'air_epa_allowed'
    ) -> float:
        """
        Get specific metric for a team's secondary.

        Args:
            team: Team abbreviation
            metric: Metric name

        Returns:
            Metric value (0.0 if not found)
        """
        if team not in self.secondary_ratings:
            return 0.0

        return self.secondary_ratings[team].get(metric, 0.0)

    def get_rankings(
        self,
        metric: str = 'air_epa_allowed',
        ascending: bool = True
    ) -> pd.DataFrame:
        """
        Get ranked list of secondaries by specified metric.

        Args:
            metric: Metric to rank by
            ascending: True for best to worst (lower EPA is better)

        Returns:
            DataFrame with team rankings
        """
        if not self.secondary_ratings:
            return pd.DataFrame()

        rankings = []
        for team, metrics in self.secondary_ratings.items():
            if metric in metrics:
                rankings.append({
                    'team': team,
                    'rank': 0,  # Will be calculated
                    metric: metrics[metric],
                    'targets': metrics.get('targets', 0)
                })

        df = pd.DataFrame(rankings).sort_values(metric, ascending=ascending)
        df['rank'] = range(1, len(df) + 1)

        return df[['rank', 'team', metric, 'targets']]

    def apply_adjustments(
        self,
        pbp_data: pd.DataFrame,
        metrics: List[str] = ['epa', 'air_epa', 'yac_epa']
    ) -> pd.DataFrame:
        """
        Apply secondary adjustments to receiver plays.

        Args:
            pbp_data: Play-by-play data with receiver targets
            metrics: List of metrics to adjust

        Returns:
            DataFrame with opponent-adjusted metrics
        """
        adjusted = pbp_data.copy()

        # Create defense quality mapping
        defense_quality = {
            team: ratings.get('air_epa_allowed', 0.0)
            for team, ratings in self.secondary_ratings.items()
        }

        # Map defense quality to each play
        adjusted['secondary_quality'] = adjusted['defteam'].map(defense_quality).fillna(0.0)

        # Apply adjustments
        for metric in metrics:
            if metric not in adjusted.columns:
                continue

            # Opponent adjustment: metric - defense_quality
            # If facing tough secondary (negative air_epa_allowed), boost WR metric
            adj_col = f"{metric}_opp_adj"
            adjusted[adj_col] = adjusted[metric] - adjusted['secondary_quality']

        return adjusted


def main():
    """Example usage of SecondaryRatingSystem."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from data_pipeline.nfl_data_fetcher import NFLDataFetcher

    # Fetch data
    logger.info("Fetching nflfastR data...")
    fetcher = NFLDataFetcher()
    pbp = fetcher.fetch_pbp_data([2023, 2024])

    # Calculate secondary ratings
    system = SecondaryRatingSystem(decay_rate=0.996, min_targets=50, iterations=5)
    ratings = system.calculate_secondary_ratings(pbp)

    # Display top/bottom secondaries
    print(f"\n{'='*80}")
    print("Top 10 Secondaries (Best Pass Coverage)")
    print(f"{'='*80}")
    print("(Lower air_epa_allowed = better coverage)")
    rankings = system.get_rankings('air_epa_allowed', ascending=True)
    print(rankings.head(10).to_string(index=False))

    print(f"\n{'='*80}")
    print("Bottom 10 Secondaries (Worst Pass Coverage)")
    print(f"{'='*80}")
    rankings_bottom = system.get_rankings('air_epa_allowed', ascending=False)
    print(rankings_bottom.head(10).to_string(index=False))

    # Display catch rate allowed
    print(f"\n{'='*80}")
    print("Catch Rate Allowed Rankings")
    print(f"{'='*80}")
    catch_rate_rankings = system.get_rankings('catch_rate_allowed', ascending=True)
    print(catch_rate_rankings.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
