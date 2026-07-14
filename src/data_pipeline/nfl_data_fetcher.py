"""
NFL Data Fetcher using nfl_data_py (nflfastR wrapper)

This module handles fetching and caching play-by-play data from nflfastR.
"""

import pandas as pd
import nfl_data_py as nfl
from pathlib import Path
from typing import List, Optional
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NFLDataFetcher:
    """Fetches and caches NFL play-by-play data using nfl_data_py."""

    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize the data fetcher.

        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_pbp_data(
        self,
        years: List[int],
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Fetch play-by-play data for specified years.

        Args:
            years: List of years to fetch
            force_refresh: If True, re-download data even if cached

        Returns:
            DataFrame with play-by-play data
        """
        cache_file = self.cache_dir / f"pbp_{'_'.join(map(str, years))}.parquet"

        # Check cache first
        if cache_file.exists() and not force_refresh:
            logger.info(f"Loading cached play-by-play data from {cache_file}")
            return pd.read_parquet(cache_file)

        # Download data
        logger.info(f"Downloading play-by-play data for years: {years}")
        try:
            pbp_df = nfl.import_pbp_data(years)

            # Cache the data
            pbp_df.to_parquet(cache_file, index=False)
            logger.info(f"Cached play-by-play data to {cache_file}")

            return pbp_df
        except Exception as e:
            logger.error(f"Error fetching play-by-play data: {e}")
            raise

    def fetch_weekly_data(
        self,
        years: List[int],
        stat_type: str = 'pass',
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Fetch weekly player statistics.

        Args:
            years: List of years to fetch
            stat_type: Type of stats ('pass', 'rush', 'rec')
            force_refresh: If True, re-download data even if cached

        Returns:
            DataFrame with weekly statistics
        """
        cache_file = self.cache_dir / f"weekly_{stat_type}_{'_'.join(map(str, years))}.parquet"

        # Check cache first
        if cache_file.exists() and not force_refresh:
            logger.info(f"Loading cached weekly data from {cache_file}")
            return pd.read_parquet(cache_file)

        # Download data
        logger.info(f"Downloading weekly {stat_type} data for years: {years}")
        try:
            weekly_df = nfl.import_weekly_data(years)

            # Cache the data
            weekly_df.to_parquet(cache_file, index=False)
            logger.info(f"Cached weekly data to {cache_file}")

            return weekly_df
        except Exception as e:
            logger.error(f"Error fetching weekly data: {e}")
            raise

    def fetch_seasonal_data(
        self,
        years: List[int],
        stat_type: str = 'pass',
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Fetch seasonal player statistics.

        Args:
            years: List of years to fetch
            stat_type: Type of stats ('pass', 'rush', 'rec')
            force_refresh: If True, re-download data even if cached

        Returns:
            DataFrame with seasonal statistics
        """
        cache_file = self.cache_dir / f"seasonal_{stat_type}_{'_'.join(map(str, years))}.parquet"

        # Check cache first
        if cache_file.exists() and not force_refresh:
            logger.info(f"Loading cached seasonal data from {cache_file}")
            return pd.read_parquet(cache_file)

        # Download data
        logger.info(f"Downloading seasonal {stat_type} data for years: {years}")
        try:
            seasonal_df = nfl.import_seasonal_data(years)

            # Cache the data
            seasonal_df.to_parquet(cache_file, index=False)
            logger.info(f"Cached seasonal data to {cache_file}")

            return seasonal_df
        except Exception as e:
            logger.error(f"Error fetching seasonal data: {e}")
            raise

    def fetch_rosters(
        self,
        years: List[int],
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Fetch weekly player roster data including position.

        Args:
            years: List of years to fetch
            force_refresh: If True, re-download data even if cached

        Returns:
            DataFrame with weekly roster data (includes position, team, etc.)
        """
        cache_file = self.cache_dir / f"weekly_rosters_{'_'.join(map(str, years))}.parquet"

        # Check cache first
        if cache_file.exists() and not force_refresh:
            logger.info(f"Loading cached weekly roster data from {cache_file}")
            return pd.read_parquet(cache_file)

        # Download data
        logger.info(f"Downloading weekly roster data for years: {years}")
        try:
            roster_df = nfl.import_weekly_rosters(years)

            # Keep only essential columns for position lookup
            roster_df = roster_df[['season', 'player_id', 'player_name', 'position', 'team', 'week']].copy()

            # Cache the data
            roster_df.to_parquet(cache_file, index=False)
            logger.info(f"Cached weekly roster data to {cache_file}")

            return roster_df
        except Exception as e:
            logger.error(f"Error fetching roster data: {e}")
            raise

    def get_qb_play_by_play(
        self,
        years: List[int],
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Get play-by-play data filtered for QB passing plays.

        Args:
            years: List of years to fetch
            force_refresh: If True, re-download data even if cached

        Returns:
            DataFrame with QB passing plays only
        """
        pbp_df = self.fetch_pbp_data(years, force_refresh)

        # Filter for passing plays with a passer identified
        qb_plays = pbp_df[
            (pbp_df['play_type'] == 'pass') &
            (pbp_df['passer_player_id'].notna())
        ].copy()

        logger.info(f"Filtered to {len(qb_plays):,} QB passing plays")

        return qb_plays

    def clear_cache(self):
        """Delete all cached files."""
        for cache_file in self.cache_dir.glob("*.parquet"):
            cache_file.unlink()
            logger.info(f"Deleted cache file: {cache_file}")


def main():
    """Example usage of NFLDataFetcher."""
    fetcher = NFLDataFetcher()

    # Fetch recent years
    years = [2022, 2023, 2024]

    # Get QB play-by-play data
    qb_pbp = fetcher.get_qb_play_by_play(years)
    print(f"\nFetched {len(qb_pbp):,} QB passing plays")
    print(f"Columns available: {qb_pbp.columns.tolist()[:10]}...")

    # Show a few key columns
    key_cols = ['game_date', 'passer_player_name', 'posteam', 'defteam',
                'down', 'ydstogo', 'qb_epa', 'cpoe', 'air_yards', 'complete_pass']
    print("\nSample data:")
    print(qb_pbp[key_cols].head(10))

    # Get roster data for ages
    rosters = fetcher.fetch_rosters(years)
    print(f"\nFetched roster data for {len(rosters):,} players")


if __name__ == "__main__":
    main()
