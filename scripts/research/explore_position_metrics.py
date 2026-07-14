"""
Position Metrics Explorer for Multi-Position LEAF

Comprehensive exploration of nflfastR play-by-play data to determine
what metrics are available for each position.

This script identifies:
1. Available EPA attribution by position
2. Sample sizes (plays per game, players per season)
3. Data quality and completeness
4. Derivable vs directly available metrics

Usage:
    python explore_position_metrics.py --years 2022 2023 2024
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_pipeline.nfl_data_fetcher import NFLDataFetcher

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PositionMetricsExplorer:
    """
    Explores available metrics for all positions in nflfastR data.
    """

    def __init__(self):
        self.position_groups = {
            'QB': ['QB'],
            'RB': ['RB', 'FB'],
            'WR': ['WR'],
            'TE': ['TE'],
            'OL': ['T', 'G', 'C'],
            'DL': ['DE', 'DT', 'NT'],
            'EDGE': ['OLB'],  # Edge rushers
            'LB': ['ILB', 'MLB', 'LB'],
            'DB': ['CB', 'S', 'SS', 'FS', 'DB']
        }

    def explore_qb_metrics(self, pbp: pd.DataFrame) -> pd.DataFrame:
        """
        Explore QB metrics (already well-established).
        """
        logger.info("\n=== QB Metrics ===")

        qb_plays = pbp[pbp['passer_player_id'].notna()].copy()

        metrics = {
            'position': 'QB',
            'total_plays': len(qb_plays),
            'unique_players': qb_plays['passer_player_id'].nunique(),
            'seasons': qb_plays['season'].nunique(),
            'plays_per_player': len(qb_plays) / qb_plays['passer_player_id'].nunique(),
        }

        # Available metrics
        available_cols = [
            'qb_epa', 'cpoe', 'success', 'air_yards', 'yards_after_catch',
            'complete_pass', 'incomplete_pass', 'interception', 'qb_hit',
            'qb_scramble', 'sack'
        ]

        for col in available_cols:
            if col in qb_plays.columns:
                metrics[f'has_{col}'] = (qb_plays[col].notna().sum() / len(qb_plays))

        # Sample QB
        sample_qb = qb_plays.groupby('passer_player_id').size().idxmax()
        sample_name = qb_plays[qb_plays['passer_player_id'] == sample_qb]['passer_player_name'].iloc[0]
        metrics['sample_player'] = sample_name
        metrics['sample_plays'] = qb_plays[qb_plays['passer_player_id'] == sample_qb].shape[0]

        logger.info(f"  Total QB pass plays: {metrics['total_plays']:,}")
        logger.info(f"  Unique QBs: {metrics['unique_players']}")
        logger.info(f"  Avg plays per QB: {metrics['plays_per_player']:.1f}")

        return pd.DataFrame([metrics])

    def explore_receiver_metrics(self, pbp: pd.DataFrame) -> pd.DataFrame:
        """
        Explore receiver metrics (WR/TE combined - position split requires roster matching).
        """
        logger.info("\n=== Receiver Metrics (All Pass Catchers) ===")

        # Get pass plays with receiver info
        receiver_plays = pbp[
            (pbp['pass_attempt'] == 1) &
            (pbp['receiver_player_id'].notna())
        ].copy()

        metrics = {
            'position': 'WR/TE',
            'total_plays': len(receiver_plays),
            'unique_players': receiver_plays['receiver_player_id'].nunique(),
            'plays_per_player': len(receiver_plays) / receiver_plays['receiver_player_id'].nunique() if len(receiver_plays) > 0 else 0,
        }

        # Available metrics
        metrics['has_epa'] = (receiver_plays['epa'].notna().sum() / len(receiver_plays)) if len(receiver_plays) > 0 else 0
        metrics['has_air_yards'] = (receiver_plays['air_yards'].notna().sum() / len(receiver_plays)) if len(receiver_plays) > 0 else 0
        metrics['has_yac'] = (receiver_plays['yards_after_catch'].notna().sum() / len(receiver_plays)) if len(receiver_plays) > 0 else 0
        metrics['has_complete_pass'] = 'complete_pass' in receiver_plays.columns
        metrics['has_incomplete_pass'] = 'incomplete_pass' in receiver_plays.columns

        # Derivable metrics
        metrics['can_calculate_catch_rate'] = metrics['total_plays'] > 0
        metrics['can_calculate_target_share'] = True  # Team-level calculation
        metrics['can_calculate_racr'] = metrics['has_yac'] and metrics['has_air_yards']  # Receiver Air Conversion Ratio

        # Sample receiver
        sample_rec = receiver_plays.groupby('receiver_player_id').size().idxmax()
        sample_name = receiver_plays[receiver_plays['receiver_player_id'] == sample_rec]['receiver_player_name'].iloc[0]
        metrics['sample_player'] = sample_name
        metrics['sample_plays'] = receiver_plays[receiver_plays['receiver_player_id'] == sample_rec].shape[0]

        logger.info(f"  Total targets: {metrics['total_plays']:,}")
        logger.info(f"  Unique receivers: {metrics['unique_players']}")
        logger.info(f"  Avg targets per receiver: {metrics['plays_per_player']:.1f}")
        logger.info(f"  Sample player: {sample_name} ({metrics['sample_plays']} targets)")

        return pd.DataFrame([metrics])

    def explore_rb_metrics(self, pbp: pd.DataFrame) -> pd.DataFrame:
        """
        Explore RB metrics (rushing, receiving, blocking).
        """
        logger.info("\n=== RB Metrics ===")

        # Rushing plays
        rush_plays = pbp[
            (pbp['rush_attempt'] == 1) &
            (pbp['rusher_player_id'].notna())
        ].copy()

        # RB receiving
        rb_targets = pbp[
            (pbp['pass_attempt'] == 1) &
            (pbp['receiver_player_id'].notna())
        ].copy()

        metrics = {
            'position': 'RB',
            'total_rush_plays': len(rush_plays),
            'total_target_plays': len(rb_targets),
            'unique_rushers': rush_plays['rusher_player_id'].nunique(),
            'unique_receivers': rb_targets['receiver_player_id'].nunique(),
            'rush_plays_per_player': len(rush_plays) / rush_plays['rusher_player_id'].nunique() if len(rush_plays) > 0 else 0,
        }

        # Available metrics
        metrics['has_rush_epa'] = (rush_plays['epa'].notna().sum() / len(rush_plays)) if len(rush_plays) > 0 else 0
        metrics['has_yards_gained'] = (rush_plays['yards_gained'].notna().sum() / len(rush_plays)) if len(rush_plays) > 0 else 0

        # Derivable metrics
        metrics['can_estimate_blocking'] = 'pass_attempt' in pbp.columns and 'sack' in pbp.columns

        logger.info(f"  Total rush attempts: {metrics['total_rush_plays']:,}")
        logger.info(f"  Unique RBs (rushing): {metrics['unique_rushers']}")
        logger.info(f"  Avg rush attempts per RB: {metrics['rush_plays_per_player']:.1f}")
        logger.info(f"  RB targets: {metrics['total_target_plays']:,}")

        return pd.DataFrame([metrics])

    def explore_ol_metrics(self, pbp: pd.DataFrame) -> pd.DataFrame:
        """
        Explore OL metrics (must be inferred from QB/RB performance).
        """
        logger.info("\n=== OL Metrics ===")

        # OL doesn't have direct EPA attribution in nflfastR
        # Must be inferred from:
        # 1. Pass protection: sack rate, QB hit rate, pressure rate (if NGS available)
        # 2. Run blocking: yards before contact, stuff rate

        metrics = {
            'position': 'OL',
            'direct_attribution': False,
            'inference_method': 'team_level_reverse_engineering'
        }

        # Check what's available for inference
        pass_plays = pbp[pbp['pass_attempt'] == 1].copy()
        rush_plays = pbp[pbp['rush_attempt'] == 1].copy()

        metrics['can_calculate_sack_rate'] = 'sack' in pbp.columns
        metrics['can_calculate_pressure_rate'] = 'qb_hit' in pbp.columns
        metrics['can_calculate_stuff_rate'] = 'yards_gained' in pbp.columns

        # Sample size for team-level
        metrics['teams_per_season'] = pbp['posteam'].nunique()
        metrics['games_per_team'] = len(pbp) / (pbp['posteam'].nunique() * pbp['season'].nunique())

        logger.info(f"  Direct OL attribution: {metrics['direct_attribution']}")
        logger.info(f"  Must infer from team-level QB/RB performance")
        logger.info(f"  Can calculate sack rate: {metrics['can_calculate_sack_rate']}")
        logger.info(f"  Can calculate pressure rate: {metrics['can_calculate_pressure_rate']}")

        return pd.DataFrame([metrics])

    def explore_defensive_metrics(self, pbp: pd.DataFrame) -> pd.DataFrame:
        """
        Explore defensive position metrics (coverage, pass rush, run defense).
        """
        logger.info("\n=== Defensive Position Metrics ===")

        results = []

        # Check available defensive player tracking
        def_cols = [col for col in pbp.columns if 'defense' in col.lower() or 'tackle' in col.lower()]

        logger.info(f"  Available defensive columns: {def_cols}")

        for pos_group in ['DL', 'EDGE', 'LB', 'DB']:
            metrics = {
                'position': pos_group,
                'direct_attribution': False,  # Limited in standard nflfastR
                'inference_method': 'team_defense_ratings'
            }

            # What can we infer?
            if pos_group in ['DL', 'EDGE']:
                metrics['can_estimate_pressure'] = 'qb_hit' in pbp.columns or 'sack' in pbp.columns
                metrics['can_estimate_run_stop'] = 'yards_gained' in pbp.columns

            elif pos_group == 'LB':
                metrics['can_estimate_coverage'] = 'epa' in pbp.columns  # Via pass defense
                metrics['can_estimate_run_stop'] = 'yards_gained' in pbp.columns

            elif pos_group == 'DB':
                metrics['can_estimate_coverage'] = 'epa' in pbp.columns
                metrics['has_interceptions'] = 'interception' in pbp.columns
                metrics['has_pass_defense'] = 'pass_defense_1_player_name' in pbp.columns

            results.append(metrics)

            logger.info(f"  {pos_group}: Limited direct attribution, must use team-level defense ratings")

        return pd.DataFrame(results)

    def generate_summary_report(self, all_metrics: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Generate comprehensive summary of position metric availability.
        """
        logger.info("\n" + "="*80)
        logger.info("POSITION METRICS SUMMARY")
        logger.info("="*80)

        combined = pd.concat(all_metrics, ignore_index=True)

        # Categorize by data availability
        combined['data_quality'] = combined.apply(self._assess_data_quality, axis=1)

        logger.info("\nData Quality by Position:")
        for _, row in combined.iterrows():
            logger.info(f"  {row['position']:8s}: {row.get('data_quality', 'Unknown')}")

        return combined

    def _assess_data_quality(self, row: pd.Series) -> str:
        """Assess data quality for position."""
        if row['position'] in ['QB']:
            return 'EXCELLENT - Direct EPA attribution, large samples'
        elif row['position'] in ['WR', 'TE', 'RB']:
            return 'GOOD - Direct EPA attribution, moderate samples'
        elif row['position'] == 'OL':
            return 'FAIR - Inference required, team-level only'
        elif row['position'] in ['DL', 'EDGE', 'LB', 'DB']:
            return 'FAIR - Limited direct data, team defense ratings available'
        else:
            return 'UNKNOWN'

    def identify_next_steps(self, summary: pd.DataFrame):
        """
        Recommend implementation order based on data availability.
        """
        logger.info("\n" + "="*80)
        logger.info("RECOMMENDED IMPLEMENTATION ORDER")
        logger.info("="*80)

        logger.info("\n1. PHASE 3 - QB + Pass Catchers (WR/TE)")
        logger.info("   Rationale: Excellent data quality, direct EPA attribution")
        logger.info("   Risk: LOW")

        logger.info("\n2. PHASE 4a - Add RB")
        logger.info("   Rationale: Good data for rushing and receiving")
        logger.info("   Risk: LOW-MEDIUM")

        logger.info("\n3. PHASE 4b - Add OL (team-level)")
        logger.info("   Rationale: Requires reverse engineering from QB/RB")
        logger.info("   Risk: MEDIUM-HIGH")

        logger.info("\n4. PHASE 5 - Add Defense")
        logger.info("   Rationale: Use existing defense rating system")
        logger.info("   Risk: MEDIUM (extend existing system)")

        logger.info("\n5. FUTURE - Individual defensive players")
        logger.info("   Rationale: Limited data, may require PFF or NGS")
        logger.info("   Risk: HIGH")


def main():
    """Run position metrics exploration."""
    parser = argparse.ArgumentParser(
        description='Explore available metrics for all positions'
    )
    parser.add_argument(
        '--years',
        type=int,
        nargs='+',
        default=[2022, 2023, 2024],
        help='Years to analyze'
    )
    args = parser.parse_args()

    logger.info("="*80)
    logger.info("Position Metrics Explorer")
    logger.info("="*80)

    # Fetch data
    logger.info(f"\nFetching nflfastR data for {args.years}...")
    fetcher = NFLDataFetcher()
    pbp = fetcher.fetch_pbp_data(args.years)
    rosters = fetcher.fetch_rosters(args.years)

    logger.info(f"  Loaded {len(pbp):,} plays")
    logger.info(f"  Columns available: {len(pbp.columns)}")

    # Explore each position
    explorer = PositionMetricsExplorer()

    all_metrics = []

    # QB
    qb_metrics = explorer.explore_qb_metrics(pbp)
    all_metrics.append(qb_metrics)

    # Receivers
    receiver_metrics = explorer.explore_receiver_metrics(pbp)
    all_metrics.append(receiver_metrics)

    # RB
    rb_metrics = explorer.explore_rb_metrics(pbp)
    all_metrics.append(rb_metrics)

    # OL
    ol_metrics = explorer.explore_ol_metrics(pbp)
    all_metrics.append(ol_metrics)

    # Defense
    def_metrics = explorer.explore_defensive_metrics(pbp)
    all_metrics.append(def_metrics)

    # Generate summary
    summary = explorer.generate_summary_report(all_metrics)

    # Save results
    Path('docs').mkdir(exist_ok=True)
    summary.to_csv('docs/position_metrics_availability.csv', index=False)
    logger.info(f"\n  Saved summary to docs/position_metrics_availability.csv")

    # Export detailed column list
    col_info = pd.DataFrame({
        'column': pbp.columns,
        'dtype': pbp.dtypes.values,
        'non_null_pct': (pbp.notna().sum() / len(pbp)).values
    })
    col_info = col_info.sort_values('non_null_pct', ascending=False)
    col_info.to_csv('docs/nflfastr_columns.csv', index=False)
    logger.info(f"  Saved column inventory to docs/nflfastr_columns.csv")

    # Recommendations
    explorer.identify_next_steps(summary)

    logger.info("\n" + "="*80)
    logger.info("Exploration Complete!")
    logger.info("="*80)
    logger.info("\nNext Steps:")
    logger.info("1. Review docs/position_metrics_availability.csv")
    logger.info("2. Review docs/nflfastr_columns.csv for detailed column info")
    logger.info("3. Proceed with Phase 2: Build interaction matrix infrastructure")

    return 0


if __name__ == "__main__":
    exit(main())
