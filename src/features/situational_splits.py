"""
Situational Splits Analysis for QB Performance

Analyzes QB performance across different game situations:
- Down and distance
- Field position (red zone, own territory)
- Game script (leading, trailing, close game)
- Time situations (2-minute drill, 4th quarter)
- Play type (play action, pressure, clean pocket)
- Target depth (short, medium, deep)

Provides detailed performance profiles for each QB.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SituationalSplits:
    """
    Calculates and analyzes QB performance in different game situations.

    Compares QB performance to league averages in each situation to identify
    strengths and weaknesses.
    """

    def __init__(self, min_attempts_per_situation: int = 10):
        """
        Initialize situational splits analyzer.

        Args:
            min_attempts_per_situation: Minimum attempts to include a split
        """
        self.min_attempts = min_attempts_per_situation
        self.league_averages = {}
        self.qb_splits = {}

    def _define_situations(self, pbp_data: pd.DataFrame) -> pd.DataFrame:
        """
        Add situation categories to play-by-play data.

        Args:
            pbp_data: Play-by-play DataFrame

        Returns:
            DataFrame with situation columns added
        """
        df = pbp_data.copy()

        # Down-based situations
        if 'down' in df.columns:
            df['situation_down'] = df['down'].fillna(0).astype(int).astype(str) + '_down'

            # 3rd/4th down by distance
            if 'ydstogo' in df.columns:
                df.loc[(df['down'] == 3) & (df['ydstogo'] <= 3), 'situation_down'] = '3rd_short'
                df.loc[(df['down'] == 3) & (df['ydstogo'] > 3) & (df['ydstogo'] <= 7), 'situation_down'] = '3rd_medium'
                df.loc[(df['down'] == 3) & (df['ydstogo'] > 7), 'situation_down'] = '3rd_long'
                df.loc[(df['down'] == 4), 'situation_down'] = '4th_down'
        else:
            df['situation_down'] = 'unknown'

        # Field position situations
        if 'yardline_100' in df.columns:
            df['situation_field'] = 'midfield'
            df.loc[df['yardline_100'] <= 20, 'situation_field'] = 'red_zone'
            df.loc[df['yardline_100'] <= 10, 'situation_field'] = 'inside_10'
            df.loc[df['yardline_100'] <= 5, 'situation_field'] = 'goal_line'
            df.loc[df['yardline_100'] >= 80, 'situation_field'] = 'own_territory'
        else:
            df['situation_field'] = 'unknown'

        # Game script situations
        if 'score_differential' in df.columns:
            df['situation_script'] = 'close_game'
            df.loc[df['score_differential'] > 14, 'situation_script'] = 'leading_big'
            df.loc[(df['score_differential'] > 7) & (df['score_differential'] <= 14), 'situation_script'] = 'leading_small'
            df.loc[df['score_differential'] < -14, 'situation_script'] = 'trailing_big'
            df.loc[(df['score_differential'] < -7) & (df['score_differential'] >= -14), 'situation_script'] = 'trailing_small'
        else:
            df['situation_script'] = 'unknown'

        # Time-based situations
        if 'half_seconds_remaining' in df.columns and 'qtr' in df.columns:
            df['situation_time'] = 'normal'
            df.loc[(df['half_seconds_remaining'] <= 120) & (df['qtr'].isin([2, 4])), 'situation_time'] = 'two_minute'
            df.loc[df['qtr'] == 4, 'situation_time'] = '4th_quarter'
            df.loc[df['qtr'] == 5, 'situation_time'] = 'overtime'
        else:
            df['situation_time'] = 'unknown'

        # Play type situations
        if 'pass_length' in df.columns:
            df['situation_depth'] = df['pass_length'].fillna('unknown')
        else:
            df['situation_depth'] = 'unknown'

        # Air yards depth
        if 'air_yards' in df.columns:
            df['situation_air_depth'] = 'short'
            df.loc[df['air_yards'] >= 10, 'situation_air_depth'] = 'medium'
            df.loc[df['air_yards'] >= 20, 'situation_air_depth'] = 'deep'
            df.loc[df['air_yards'] < 0, 'situation_air_depth'] = 'behind_los'
        else:
            df['situation_air_depth'] = 'unknown'

        # Pressure situations
        if 'was_pressure' in df.columns:
            df['situation_pressure'] = 'clean_pocket'
            df.loc[pd.to_numeric(df['was_pressure'], errors='coerce') == 1, 'situation_pressure'] = 'under_pressure'
        else:
            df['situation_pressure'] = 'unknown'

        # Play action
        if 'pass_oe' in df.columns or 'qb_dropback' in df.columns:
            # Play action = pass on run expected
            df['situation_play_action'] = 'standard_dropback'
            if 'pass_oe' in df.columns:
                df.loc[pd.to_numeric(df['pass_oe'], errors='coerce') == 1, 'situation_play_action'] = 'play_action'
        else:
            df['situation_play_action'] = 'unknown'

        return df

    def calculate_league_averages(
        self,
        pbp_data: pd.DataFrame,
        metrics: List[str] = ['qb_epa', 'cpoe', 'success']
    ):
        """
        Calculate league average performance in each situation.

        Args:
            pbp_data: Play-by-play data
            metrics: Metrics to calculate averages for
        """
        logger.info("Calculating league averages for situational splits...")

        # Filter to passing plays
        pass_plays = pbp_data[pbp_data['play_type'] == 'pass'].copy()

        # Add situation categories
        pass_plays = self._define_situations(pass_plays)

        # Calculate averages for each situation type
        situation_types = [
            'situation_down', 'situation_field', 'situation_script',
            'situation_time', 'situation_depth', 'situation_air_depth',
            'situation_pressure', 'situation_play_action'
        ]

        for sit_type in situation_types:
            if sit_type not in pass_plays.columns:
                continue

            situation_avgs = {}

            for situation in pass_plays[sit_type].unique():
                if situation == 'unknown':
                    continue

                sit_plays = pass_plays[pass_plays[sit_type] == situation]

                if len(sit_plays) < self.min_attempts:
                    continue

                sit_stats = {
                    'attempts': len(sit_plays)
                }

                for metric in metrics:
                    if metric in sit_plays.columns:
                        values = sit_plays[metric].dropna()
                        if len(values) > 0:
                            sit_stats[f'{metric}_avg'] = values.mean()
                            sit_stats[f'{metric}_std'] = values.std()

                situation_avgs[situation] = sit_stats

            self.league_averages[sit_type] = situation_avgs

        logger.info(f"  ✓ Calculated league averages for {len(self.league_averages)} situation types")

    def calculate_qb_splits(
        self,
        pbp_data: pd.DataFrame,
        metrics: List[str] = ['qb_epa', 'cpoe', 'success']
    ) -> pd.DataFrame:
        """
        Calculate each QB's performance in different situations.

        Args:
            pbp_data: Play-by-play data
            metrics: Metrics to calculate

        Returns:
            DataFrame with QB situational performance
        """
        logger.info("Calculating QB situational splits...")

        # Filter to passing plays
        pass_plays = pbp_data[pbp_data['play_type'] == 'pass'].copy()

        # Add situation categories
        pass_plays = self._define_situations(pass_plays)

        # Calculate splits for each QB
        results = []

        situation_types = [
            'situation_down', 'situation_field', 'situation_script',
            'situation_time', 'situation_depth', 'situation_air_depth',
            'situation_pressure', 'situation_play_action'
        ]

        for qb_id in pass_plays['passer_player_id'].unique():
            qb_plays = pass_plays[pass_plays['passer_player_id'] == qb_id]
            qb_name = qb_plays['passer_player_name'].iloc[0] if 'passer_player_name' in qb_plays.columns else qb_id

            for sit_type in situation_types:
                if sit_type not in qb_plays.columns:
                    continue

                for situation in qb_plays[sit_type].unique():
                    if situation == 'unknown':
                        continue

                    sit_plays = qb_plays[qb_plays[sit_type] == situation]

                    if len(sit_plays) < self.min_attempts:
                        continue

                    row = {
                        'passer_player_id': qb_id,
                        'player_name': qb_name,
                        'situation_type': sit_type.replace('situation_', ''),
                        'situation': situation,
                        'attempts': len(sit_plays)
                    }

                    # Calculate metrics
                    for metric in metrics:
                        if metric in sit_plays.columns:
                            values = sit_plays[metric].dropna()
                            if len(values) > 0:
                                row[f'{metric}'] = values.mean()

                                # Calculate vs league average
                                if sit_type in self.league_averages and situation in self.league_averages[sit_type]:
                                    league_avg = self.league_averages[sit_type][situation].get(f'{metric}_avg', 0)
                                    row[f'{metric}_vs_avg'] = values.mean() - league_avg

                    results.append(row)

        splits_df = pd.DataFrame(results)
        logger.info(f"  ✓ Calculated splits for {len(pass_plays['passer_player_id'].unique())} QBs")

        return splits_df

    def get_qb_strengths_weaknesses(
        self,
        splits_df: pd.DataFrame,
        qb_id: str,
        metric: str = 'qb_epa',
        top_n: int = 5
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Identify QB's top strengths and weaknesses.

        Args:
            splits_df: Situational splits DataFrame
            qb_id: QB player ID
            metric: Metric to analyze
            top_n: Number of top/bottom situations to return

        Returns:
            Tuple of (strengths_df, weaknesses_df)
        """
        qb_splits = splits_df[splits_df['passer_player_id'] == qb_id].copy()

        vs_avg_col = f'{metric}_vs_avg'
        if vs_avg_col not in qb_splits.columns:
            return pd.DataFrame(), pd.DataFrame()

        # Sort by performance vs average
        qb_splits = qb_splits.sort_values(vs_avg_col, ascending=False)

        strengths = qb_splits.head(top_n)[['situation_type', 'situation', 'attempts', metric, vs_avg_col]]
        weaknesses = qb_splits.tail(top_n)[['situation_type', 'situation', 'attempts', metric, vs_avg_col]]

        return strengths, weaknesses

    def create_situation_profile(
        self,
        splits_df: pd.DataFrame,
        qb_id: str,
        metrics: List[str] = ['qb_epa', 'cpoe', 'success']
    ) -> Dict[str, pd.DataFrame]:
        """
        Create comprehensive situational profile for a QB.

        Args:
            splits_df: Situational splits DataFrame
            qb_id: QB player ID
            metrics: Metrics to include

        Returns:
            Dictionary of {situation_type: performance_df}
        """
        qb_splits = splits_df[splits_df['passer_player_id'] == qb_id].copy()

        profile = {}
        for sit_type in qb_splits['situation_type'].unique():
            sit_data = qb_splits[qb_splits['situation_type'] == sit_type].copy()

            # Select relevant columns
            cols = ['situation', 'attempts']
            for metric in metrics:
                if metric in sit_data.columns:
                    cols.append(metric)
                vs_avg_col = f'{metric}_vs_avg'
                if vs_avg_col in sit_data.columns:
                    cols.append(vs_avg_col)

            profile[sit_type] = sit_data[cols].sort_values('attempts', ascending=False)

        return profile

    def get_summary_stats(
        self,
        splits_df: pd.DataFrame,
        min_attempts: int = 50
    ) -> pd.DataFrame:
        """
        Get summary statistics for each QB across all situations.

        Args:
            splits_df: Situational splits DataFrame
            min_attempts: Minimum total attempts to include QB

        Returns:
            DataFrame with summary stats
        """
        summary = splits_df.groupby('passer_player_id').agg({
            'player_name': 'first',
            'attempts': 'sum',
            'qb_epa_vs_avg': lambda x: x.mean() if 'qb_epa_vs_avg' in splits_df.columns else np.nan,
            'situation': 'count'  # Number of different situations with enough data
        }).reset_index()

        summary = summary.rename(columns={'situation': 'situations_tracked'})
        summary = summary[summary['attempts'] >= min_attempts]
        summary = summary.sort_values('attempts', ascending=False)

        return summary


def main():
    """Example usage of situational splits analysis."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from data_pipeline.nfl_data_fetcher import NFLDataFetcher

    # Fetch data
    fetcher = NFLDataFetcher()
    pbp = fetcher.get_qb_play_by_play([2023, 2024])

    # Initialize analyzer
    analyzer = SituationalSplits(min_attempts_per_situation=10)

    # Calculate league averages
    analyzer.calculate_league_averages(pbp, metrics=['qb_epa', 'cpoe', 'success'])

    # Calculate QB splits
    splits = analyzer.calculate_qb_splits(pbp, metrics=['qb_epa', 'cpoe', 'success'])

    # Save results
    splits.to_csv('results/qb_situational_splits.csv', index=False)
    print(f"\n{'='*60}")
    print(f"Saved situational splits for {len(splits['passer_player_id'].unique())} QBs")
    print(f"{'='*60}")

    # Example: Show Patrick Mahomes' strengths and weaknesses
    mahomes_id = splits[splits['player_name'].str.contains('Mahomes', na=False)]['passer_player_id'].iloc[0] if len(splits[splits['player_name'].str.contains('Mahomes', na=False)]) > 0 else None

    if mahomes_id:
        strengths, weaknesses = analyzer.get_qb_strengths_weaknesses(
            splits, mahomes_id, metric='qb_epa', top_n=5
        )

        print(f"\nP.Mahomes - Top 5 Situations (EPA):")
        print(strengths.to_string(index=False))

        print(f"\nP.Mahomes - Worst 5 Situations (EPA):")
        print(weaknesses.to_string(index=False))

    # Summary stats
    summary = analyzer.get_summary_stats(splits, min_attempts=100)
    print(f"\n{'='*60}")
    print("QB Situational Coverage Summary (min 100 attempts)")
    print(f"{'='*60}")
    print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
