"""
Injury Tracking System

Tracks player injury status and adjusts LEAF ratings for injury-related uncertainty.

Data Sources:
- ESPN API (recommended)
- Pro Football Reference (web scraping)
- Manual entry (CSV upload)

Adjustments:
- Injury status → Uncertainty multiplier
- Games missed → Recovery model
- First game back → Reduced rating
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import logging
from pathlib import Path
import requests
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InjuryStatus:
    """Injury status levels."""
    HEALTHY = "Healthy"
    QUESTIONABLE = "Questionable"
    DOUBTFUL = "Doubtful"
    OUT = "Out"
    IR = "IR"  # Injured Reserve
    PUP = "PUP"  # Physically Unable to Perform


class InjuryDataFetcher:
    """
    Fetch injury data from various sources.

    Primary: ESPN API
    Fallback: Manual CSV
    """

    def __init__(self, data_source: str = "espn", api_key: Optional[str] = None):
        """
        Initialize injury data fetcher.

        Args:
            data_source: Data source ('espn', 'manual', or 'pfr')
            api_key: API key if required
        """
        self.data_source = data_source
        self.api_key = api_key
        self.base_url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"

    def fetch_weekly_injuries(
        self,
        week: int,
        season: int
    ) -> pd.DataFrame:
        """
        Fetch injury reports for a specific week.

        Args:
            week: NFL week number (1-18)
            season: Season year

        Returns:
            DataFrame with injury information
        """
        if self.data_source == "espn":
            return self._fetch_from_espn(week, season)
        elif self.data_source == "manual":
            return self._load_from_csv(week, season)
        else:
            logger.error(f"Unknown data source: {self.data_source}")
            return pd.DataFrame()

    def _fetch_from_espn(self, week: int, season: int) -> pd.DataFrame:
        """
        Fetch injuries from ESPN API.

        ESPN API endpoint:
        https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team}/injuries

        Note: ESPN API is public but may have rate limits.
        """
        logger.info(f"Fetching injury data for {season} Week {week} from ESPN...")

        # Get all NFL teams
        teams = [
            'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE',
            'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC',
            'LV', 'LAC', 'LAR', 'MIA', 'MIN', 'NE', 'NO', 'NYG',
            'NYJ', 'PHI', 'PIT', 'SF', 'SEA', 'TB', 'TEN', 'WAS'
        ]

        all_injuries = []

        for team in teams:
            try:
                # ESPN API endpoint
                url = f"{self.base_url}/teams/{team}/injuries"
                response = requests.get(url, timeout=10)

                if response.status_code == 200:
                    data = response.json()

                    # Parse injuries from response
                    if 'injuries' in data:
                        for injury in data['injuries']:
                            player_info = injury.get('athlete', {})

                            all_injuries.append({
                                'player_name': player_info.get('displayName', ''),
                                'player_id': player_info.get('id', ''),
                                'team': team,
                                'status': injury.get('status', 'Unknown'),
                                'injury_type': injury.get('type', 'Unknown'),
                                'date': injury.get('date', ''),
                                'week': week,
                                'season': season
                            })

                else:
                    logger.warning(f"Failed to fetch {team} injuries: {response.status_code}")

            except Exception as e:
                logger.error(f"Error fetching {team} injuries: {e}")
                continue

        injuries_df = pd.DataFrame(all_injuries)

        if len(injuries_df) > 0:
            logger.info(f"  Fetched {len(injuries_df)} injury reports")
        else:
            logger.warning(f"  No injury data found for Week {week}")

        return injuries_df

    def _load_from_csv(self, week: int, season: int) -> pd.DataFrame:
        """
        Load injury data from CSV file.

        Expected CSV format:
        player_id,player_name,team,status,injury_type,week,season
        """
        csv_path = Path(f'data/injuries/injuries_{season}_week_{week}.csv')

        if not csv_path.exists():
            logger.warning(f"Injury CSV not found: {csv_path}")
            return pd.DataFrame()

        injuries = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(injuries)} injury reports from {csv_path}")

        return injuries

    def fetch_season_injuries(self, season: int) -> pd.DataFrame:
        """
        Fetch all injury data for a season.

        Args:
            season: Season year

        Returns:
            DataFrame with full season injury history
        """
        logger.info(f"Fetching full season injury data for {season}...")

        all_weeks = []

        for week in range(1, 19):  # 18 weeks
            week_injuries = self.fetch_weekly_injuries(week, season)
            if len(week_injuries) > 0:
                all_weeks.append(week_injuries)

        if len(all_weeks) == 0:
            return pd.DataFrame()

        season_injuries = pd.concat(all_weeks, ignore_index=True)
        logger.info(f"  Total injury reports: {len(season_injuries)}")

        return season_injuries


class InjuryAdjuster:
    """
    Adjust LEAF ratings for injury status and recovery.
    """

    def __init__(
        self,
        questionable_multiplier: float = 1.10,
        doubtful_multiplier: float = 1.25,
        recovery_games: int = 3
    ):
        """
        Initialize injury adjuster.

        Args:
            questionable_multiplier: Uncertainty multiplier for Questionable status
            doubtful_multiplier: Uncertainty multiplier for Doubtful status
            recovery_games: Number of games for full recovery
        """
        self.questionable_multiplier = questionable_multiplier
        self.doubtful_multiplier = doubtful_multiplier
        self.recovery_games = recovery_games

    def adjust_for_injury_status(
        self,
        rating: float,
        uncertainty: float,
        status: str
    ) -> Dict[str, float]:
        """
        Adjust rating and uncertainty for injury status.

        Args:
            rating: Current LEAF rating
            uncertainty: Current uncertainty
            status: Injury status

        Returns:
            Dictionary with adjusted rating and uncertainty
        """
        adjusted_rating = rating
        adjusted_uncertainty = uncertainty

        if status == InjuryStatus.OUT or status == InjuryStatus.IR or status == InjuryStatus.PUP:
            # Player not playing - rating is 0
            adjusted_rating = 0.0
            adjusted_uncertainty = 0.0

        elif status == InjuryStatus.DOUBTFUL:
            # Significant uncertainty increase
            adjusted_uncertainty *= self.doubtful_multiplier

        elif status == InjuryStatus.QUESTIONABLE:
            # Moderate uncertainty increase
            adjusted_uncertainty *= self.questionable_multiplier

        return {
            'adjusted_rating': adjusted_rating,
            'adjusted_uncertainty': adjusted_uncertainty,
            'injury_adjustment': adjusted_rating - rating
        }

    def model_injury_recovery(
        self,
        pre_injury_rating: float,
        games_since_return: int
    ) -> float:
        """
        Model gradual return to full performance after injury.

        Args:
            pre_injury_rating: Rating before injury
            games_since_return: Number of games since returning

        Returns:
            Adjusted rating accounting for recovery
        """
        if games_since_return >= self.recovery_games:
            # Full recovery
            return pre_injury_rating

        # Gradual recovery curve
        recovery_factors = {
            1: 0.85,  # First game back: 85% of pre-injury
            2: 0.92,  # Second game: 92%
            3: 1.00   # Third game+: Full recovery
        }

        factor = recovery_factors.get(games_since_return, 1.00)
        return pre_injury_rating * factor

    def identify_returning_players(
        self,
        injury_history: pd.DataFrame,
        current_week: int,
        season: int
    ) -> pd.DataFrame:
        """
        Identify players returning from injury.

        Args:
            injury_history: Full injury history
            current_week: Current week
            season: Current season

        Returns:
            DataFrame with returning players and games since return
        """
        # Find players who were Out/IR and are now playing
        recent_weeks = injury_history[
            (injury_history['season'] == season) &
            (injury_history['week'] <= current_week) &
            (injury_history['week'] >= current_week - 4)
        ]

        # Identify status changes
        returning_players = []

        for player_id in recent_weeks['player_id'].unique():
            player_history = recent_weeks[
                recent_weeks['player_id'] == player_id
            ].sort_values('week')

            # Check if they were out and are now back
            statuses = player_history['status'].tolist()

            if any(s in [InjuryStatus.OUT, InjuryStatus.IR] for s in statuses):
                # They were out at some point
                last_out_week = player_history[
                    player_history['status'].isin([InjuryStatus.OUT, InjuryStatus.IR])
                ]['week'].max()

                games_since_return = current_week - last_out_week

                if games_since_return > 0 and games_since_return <= self.recovery_games:
                    returning_players.append({
                        'player_id': player_id,
                        'player_name': player_history['player_name'].iloc[0],
                        'last_out_week': last_out_week,
                        'games_since_return': games_since_return,
                        'current_week': current_week
                    })

        return pd.DataFrame(returning_players)

    def apply_injury_adjustments(
        self,
        ratings: pd.DataFrame,
        injury_data: pd.DataFrame,
        week: int,
        season: int
    ) -> pd.DataFrame:
        """
        Apply injury adjustments to LEAF ratings.

        Args:
            ratings: DataFrame with LEAF ratings
            injury_data: Injury data for the week
            week: Current week
            season: Current season

        Returns:
            DataFrame with injury-adjusted ratings
        """
        logger.info(f"\nApplying injury adjustments for {season} Week {week}...")

        adjusted_ratings = ratings.copy()

        # Check if injury data is empty or missing required columns
        if len(injury_data) == 0 or 'season' not in injury_data.columns or 'week' not in injury_data.columns:
            logger.info("  No injury data available")
            return adjusted_ratings

        # Merge with injury data
        week_injuries = injury_data[
            (injury_data['season'] == season) &
            (injury_data['week'] == week)
        ]

        if len(week_injuries) == 0:
            logger.info("  No injury data available for this week")
            return adjusted_ratings

        # Match players by ID (or name if ID not available)
        adjusted_ratings = adjusted_ratings.merge(
            week_injuries[['player_id', 'status', 'injury_type']],
            left_on='passer_player_id',
            right_on='player_id',
            how='left'
        )

        # Apply adjustments
        adjustments = []

        for idx, row in adjusted_ratings.iterrows():
            if pd.notna(row.get('status')):
                adj = self.adjust_for_injury_status(
                    rating=row.get('leaf_rating', 0.0),
                    uncertainty=row.get('leaf_uncertainty', 0.1),
                    status=row['status']
                )

                adjusted_ratings.loc[idx, 'injury_adjusted_rating'] = adj['adjusted_rating']
                adjusted_ratings.loc[idx, 'injury_adjusted_uncertainty'] = adj['adjusted_uncertainty']

                if adj['injury_adjustment'] != 0:
                    adjustments.append({
                        'player': row.get('passer_player_name', ''),
                        'status': row['status'],
                        'adjustment': adj['injury_adjustment']
                    })

        logger.info(f"  Applied {len(adjustments)} injury adjustments")

        if len(adjustments) > 0:
            adj_df = pd.DataFrame(adjustments)
            logger.info(f"  Adjustments summary:")
            for _, adj in adj_df.head(5).iterrows():
                logger.info(f"    {adj['player']}: {adj['status']} ({adj['adjustment']:+.3f})")

        return adjusted_ratings


class InjuryTrackingSystem:
    """
    Complete injury tracking and adjustment system.
    """

    def __init__(
        self,
        data_source: str = "espn",
        api_key: Optional[str] = None
    ):
        """
        Initialize injury tracking system.

        Args:
            data_source: Data source for injuries
            api_key: API key if required
        """
        self.fetcher = InjuryDataFetcher(data_source=data_source, api_key=api_key)
        self.adjuster = InjuryAdjuster()

        # Cache injury data
        self.injury_cache = {}

    def fetch_and_cache_injuries(
        self,
        weeks: List[int],
        season: int
    ):
        """
        Fetch and cache injury data for multiple weeks.

        Args:
            weeks: List of week numbers
            season: Season year
        """
        logger.info(f"Fetching injury data for {len(weeks)} weeks...")

        for week in weeks:
            if (season, week) not in self.injury_cache:
                injuries = self.fetcher.fetch_weekly_injuries(week, season)
                self.injury_cache[(season, week)] = injuries

        logger.info(f"  Cached injury data for {len(self.injury_cache)} week-seasons")

    def get_cached_injuries(
        self,
        week: int,
        season: int
    ) -> pd.DataFrame:
        """Get cached injury data."""
        return self.injury_cache.get((season, week), pd.DataFrame())

    def adjust_ratings_with_injuries(
        self,
        ratings: pd.DataFrame,
        week: int,
        season: int
    ) -> pd.DataFrame:
        """
        Apply injury adjustments to ratings.

        Args:
            ratings: LEAF ratings
            week: Week number
            season: Season

        Returns:
            Injury-adjusted ratings
        """
        # Get injury data
        injuries = self.get_cached_injuries(week, season)

        if len(injuries) == 0:
            # Try to fetch
            injuries = self.fetcher.fetch_weekly_injuries(week, season)
            self.injury_cache[(season, week)] = injuries

        # Apply adjustments
        return self.adjuster.apply_injury_adjustments(
            ratings, injuries, week, season
        )


def main():
    """Example usage of injury tracking system."""

    # Initialize system
    injury_system = InjuryTrackingSystem(data_source="espn")

    # Fetch current week injuries
    current_week = 10
    current_season = 2024

    logger.info("="*80)
    logger.info("Injury Tracking System - Example Usage")
    logger.info("="*80)

    # Fetch injuries
    injuries = injury_system.fetcher.fetch_weekly_injuries(current_week, current_season)

    if len(injuries) > 0:
        print(f"\n{len(injuries)} injury reports for Week {current_week}:")
        print(injuries[['player_name', 'team', 'status', 'injury_type']].head(10).to_string(index=False))

        # Example: Adjust a mock rating
        print("\n" + "="*80)
        print("Example Injury Adjustments")
        print("="*80)

        adjuster = InjuryAdjuster()

        # Mock QB rating
        mock_rating = 0.15
        mock_uncertainty = 0.05

        for status in [InjuryStatus.HEALTHY, InjuryStatus.QUESTIONABLE,
                       InjuryStatus.DOUBTFUL, InjuryStatus.OUT]:
            adj = adjuster.adjust_for_injury_status(mock_rating, mock_uncertainty, status)

            print(f"\n{status}:")
            print(f"  Original: Rating={mock_rating:.3f}, Uncertainty=±{mock_uncertainty:.3f}")
            print(f"  Adjusted: Rating={adj['adjusted_rating']:.3f}, Uncertainty=±{adj['adjusted_uncertainty']:.3f}")

    else:
        print(f"\nNo injury data available for Week {current_week}")
        print("Note: ESPN API may require authentication or have rate limits")
        print("\nAlternative: Use manual CSV format:")
        print("  data/injuries/injuries_2024_week_10.csv")


if __name__ == "__main__":
    main()
