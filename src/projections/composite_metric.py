"""
LEAF: Layered EPA Adaptive Framework

A comprehensive QB evaluation system combining multiple performance components
into a single unified metric. Named after Ryan Leaf, the quintessential QB bust,
as a playful nod to DARKO (named after Darko Miličić).

LEAF Components:
- EPA (Expected Points Added) - Primary value metric
- CPOE (Completion % Over Expected) - Accuracy/decision-making
- Success Rate - Consistency
- Opponent-adjusted versions - Context-neutral performance
- Situational performance - Clutch/important situations

Methodology:
- Layered adjustments (opponent, context, situation)
- Kalman filtering for adaptive learning
- Bayesian regression for uncertainty
- Age curve adjustments

Produces LEAF Rating: a single per-play value metric representing QB impact.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LEAFCompositeMetric:
    """
    Creates unified LEAF Rating from multiple component metrics.

    Combines offensive components into a single per-play value metric
    representing total QB contribution.
    """

    def __init__(
        self,
        component_weights: Optional[Dict[str, float]] = None,
        use_kalman_estimates: bool = True,
        use_opponent_adjusted: bool = True
    ):
        """
        Initialize composite metric calculator.

        Args:
            component_weights: Weights for each component (auto-calculated if None)
            use_kalman_estimates: Use Kalman-filtered values
            use_opponent_adjusted: Use opponent-adjusted metrics
        """
        self.component_weights = component_weights or self._default_weights()
        self.use_kalman = use_kalman_estimates
        self.use_opponent_adjusted = use_opponent_adjusted

        self.scaler = StandardScaler()
        self.component_importance = {}
        self.league_average_leaf = 0.0
        self.league_std_leaf = 1.0

    def _default_weights(self) -> Dict[str, float]:
        """
        Default component weights.

        NOTE (July 2026): the r=0.906 these weights were selected on came from
        a leaky validation (validate_weights.py - the Kalman predictor had
        already seen the test games). Honest walk-forward next-season
        correlation is ~0.47 (docs/LEAF_V3_RESULTS.md). The weights themselves
        are kept: the honest v3 fusion fit independently agrees that EPA and
        success rate carry the load and CPOE adds little.

        Weights should sum to 1.0 for interpretability.
        """
        return {
            'epa': 0.65,           # Dominant predictor (r=0.886 Kalman+Opp-Adj)
            'cpoe': 0.15,          # Weaker than expected (r=0.574 Kalman)
            'success_rate': 0.20,  # Nearly matches EPA (r=0.879 Kalman+Opp-Adj)
            'situational': 0.00    # Too noisy, hurts prediction
        }

    def _prepare_components(
        self,
        player_stats: pd.DataFrame,
        situational_stats: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Prepare and standardize component metrics.

        Args:
            player_stats: Player statistics with all components
            situational_stats: Optional situational performance data

        Returns:
            DataFrame with standardized components
        """
        components = player_stats.copy()

        # Select appropriate metric versions
        if self.use_opponent_adjusted:
            epa_col = 'opp_adj_epa_kalman' if self.use_kalman else 'opp_adj_epa'
            cpoe_col = 'opp_adj_cpoe_kalman' if self.use_kalman else 'opp_adj_cpoe'
            success_col = 'opp_adj_success_rate_kalman' if self.use_kalman else 'opp_adj_success_rate'
        else:
            epa_col = 'epa_per_play_kalman' if self.use_kalman else 'epa_per_play_weighted'
            cpoe_col = 'cpoe_mean_kalman' if self.use_kalman else 'cpoe_weighted'
            success_col = 'success_mean_kalman' if self.use_kalman else 'success_rate_weighted'

        # Extract components (handle missing columns gracefully)
        components['epa_component'] = components.get(epa_col, components.get('epa_per_play_weighted', 0))
        components['cpoe_component'] = components.get(cpoe_col, components.get('cpoe_weighted', 0))
        components['success_component'] = components.get(success_col, components.get('success_rate_weighted', 0.5))

        # Situational component (if available)
        if situational_stats is not None:
            # Use high-leverage situations: 3rd down, red zone, close games, 2-minute
            high_leverage = situational_stats[
                situational_stats['situation'].isin([
                    '3rd_short', '3rd_medium', '3rd_long',
                    'red_zone', 'goal_line',
                    'close_game', 'two_minute', '4th_quarter'
                ])
            ].groupby('passer_player_id')['qb_epa_vs_avg'].mean().reset_index()
            high_leverage.columns = ['passer_player_id', 'situational_component']

            components = components.merge(high_leverage, on='passer_player_id', how='left')
            components['situational_component'] = components['situational_component'].fillna(0)
        else:
            components['situational_component'] = 0

        # Standardize components (z-scores)
        component_cols = ['epa_component', 'cpoe_component', 'success_component', 'situational_component']

        for col in component_cols:
            if col in components.columns:
                mean_val = components[col].mean()
                std_val = components[col].std()
                if std_val > 0:
                    components[f'{col}_z'] = (components[col] - mean_val) / std_val
                else:
                    components[f'{col}_z'] = 0

        return components

    def calculate_leaf(
        self,
        player_stats: pd.DataFrame,
        situational_stats: Optional[pd.DataFrame] = None,
        return_components: bool = False
    ) -> pd.DataFrame:
        """
        Calculate DARKO NFL Rating (LEAF-NFL) for all players.

        Args:
            player_stats: Player statistics DataFrame
            situational_stats: Optional situational performance
            return_components: If True, return component contributions

        Returns:
            DataFrame with LEAF and optionally components
        """
        logger.info("Calculating DARKO NFL Rating (LEAF)...")

        # Prepare components
        components = self._prepare_components(player_stats, situational_stats)

        # Calculate weighted composite
        components['leaf_z'] = (
            self.component_weights['epa'] * components['epa_component_z'] +
            self.component_weights['cpoe'] * components['cpoe_component_z'] +
            self.component_weights['success_rate'] * components['success_component_z'] +
            self.component_weights['situational'] * components['situational_component_z']
        )

        # Convert z-score back to interpretable scale (EPA-like)
        # LEAF in EPA per play units
        epa_mean = components['epa_component'].mean()
        epa_std = components['epa_component'].std()

        components['leaf'] = epa_mean + (components['leaf_z'] * epa_std)

        # Store league averages
        self.league_average_leaf = components['leaf'].mean()
        self.league_std_leaf = components['leaf'].std()

        # Calculate percentile ranks
        components['leaf_percentile'] = components['leaf'].rank(pct=True) * 100

        # Classify tiers
        components['leaf_tier'] = pd.cut(
            components['leaf_percentile'],
            bins=[0, 25, 50, 75, 90, 100],
            labels=['Below Average', 'Average', 'Above Average', 'Good', 'Elite']
        )

        # Calculate uncertainty-adjusted LEAF
        if 'epa_per_play_uncertainty' in components.columns:
            # Weight by inverse uncertainty
            uncertainty = components['epa_per_play_uncertainty'].fillna(1.0)
            confidence_weight = 1.0 / (1.0 + uncertainty)

            # Regress uncertain estimates toward mean
            components['leaf_adjusted'] = (
                components['leaf'] * confidence_weight +
                self.league_average_leaf * (1 - confidence_weight)
            )
        else:
            components['leaf_adjusted'] = components['leaf']

        # Select output columns
        output_cols = [
            'passer_player_id', 'player_name', 'total_attempts', 'age',
            'leaf', 'leaf_adjusted', 'leaf_percentile', 'leaf_tier'
        ]

        if return_components:
            output_cols.extend([
                'epa_component', 'epa_component_z',
                'cpoe_component', 'cpoe_component_z',
                'success_component', 'success_component_z',
                'situational_component', 'situational_component_z'
            ])

        # Filter to columns that exist
        output_cols = [col for col in output_cols if col in components.columns]

        result = components[output_cols].copy()

        logger.info(f"  ✓ Calculated LEAF for {len(result)} players")
        logger.info(f"  League average LEAF: {self.league_average_leaf:.3f}")
        logger.info(f"  League std LEAF: {self.league_std_leaf:.3f}")

        return result

    def rank_players(
        self,
        leaf_results: pd.DataFrame,
        min_attempts: int = 100,
        sort_by: str = 'leaf_adjusted'
    ) -> pd.DataFrame:
        """
        Rank players by LEAF Rating.

        Args:
            leaf_results: LEAF calculation results
            min_attempts: Minimum attempts to include
            sort_by: Column to sort by ('leaf' or 'leaf_adjusted')

        Returns:
            Ranked DataFrame
        """
        # Filter by minimum attempts
        qualified = leaf_results[leaf_results['total_attempts'] >= min_attempts].copy()

        # Sort and rank
        qualified = qualified.sort_values(sort_by, ascending=False)
        qualified['rank'] = range(1, len(qualified) + 1)

        return qualified

    def get_component_importance(
        self,
        player_stats: pd.DataFrame,
        situational_stats: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Calculate actual importance of each component via correlation with future performance.

        This can be used to validate/update weights.

        Args:
            player_stats: Multi-year player statistics
            situational_stats: Situational performance

        Returns:
            Dictionary of component correlations with future EPA
        """
        # This would require multi-year data and holdout validation
        # For now, return configured weights
        return self.component_weights

    def convert_to_per_game(
        self,
        leaf_results: pd.DataFrame,
        plays_per_game: float = 35.0
    ) -> pd.DataFrame:
        """
        Convert per-play LEAF to per-game value.

        Args:
            leaf_results: LEAF results
            plays_per_game: Average QB plays per game

        Returns:
            DataFrame with per-game values added
        """
        results = leaf_results.copy()
        results['leaf_per_game'] = results['leaf'] * plays_per_game
        results['leaf_adjusted_per_game'] = results['leaf_adjusted'] * plays_per_game

        return results

    def calculate_wins_above_replacement(
        self,
        leaf_results: pd.DataFrame,
        replacement_level_percentile: float = 25.0,
        points_per_win: float = 25.0
    ) -> pd.DataFrame:
        """
        Calculate Wins Above Replacement (WAR) from LEAF.

        Args:
            leaf_results: LEAF results
            replacement_level_percentile: Percentile for replacement level
            points_per_win: Points needed for one additional win

        Returns:
            DataFrame with WAR added
        """
        results = leaf_results.copy()

        # Define replacement level
        replacement_leaf = np.percentile(results['leaf'], replacement_level_percentile)

        # Calculate value over replacement (per play)
        results['leaf_over_replacement'] = results['leaf'] - replacement_leaf

        # Convert to WAR (assuming 17 games, ~35 plays/game)
        plays_per_season = 17 * 35
        results['war'] = (results['leaf_over_replacement'] * plays_per_season) / points_per_win

        return results


def main():
    """Example usage of LEAF composite metric."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    # Load processed data (would come from main pipeline)
    try:
        player_stats = pd.read_csv('data/processed/qb_kalman_filtered.csv')

        # Try to load situational data
        try:
            situational_stats = pd.read_csv('results/qb_situational_splits.csv')
        except:
            situational_stats = None

        # Initialize composite metric
        leaf_calculator = LEAFCompositeMetric(
            use_kalman_estimates=True,
            use_opponent_adjusted=True
        )

        # Calculate LEAF
        leaf_results = leaf_calculator.calculate_leaf(
            player_stats,
            situational_stats,
            return_components=True
        )

        # Convert to per-game
        leaf_results = leaf_calculator.convert_to_per_game(leaf_results)

        # Calculate WAR
        leaf_results = leaf_calculator.calculate_wins_above_replacement(leaf_results)

        # Rank players
        rankings = leaf_calculator.rank_players(leaf_results, min_attempts=100)

        # Save results
        rankings.to_csv('results/leaf_ratings.csv', index=False)

        # Display top QBs
        print(f"\n{'='*80}")
        print("LEAF Ratings - Top 10 QBs")
        print(f"{'='*80}")
        display_cols = ['rank', 'player_name', 'age', 'total_attempts',
                       'leaf', 'leaf_per_game', 'war', 'leaf_tier']
        print(rankings[display_cols].head(10).to_string(index=False))

        print(f"\n{'='*80}")
        print("Component Breakdown - Top 5")
        print(f"{'='*80}")
        component_cols = ['rank', 'player_name', 'epa_component', 'cpoe_component',
                         'success_component', 'situational_component']
        print(rankings[[col for col in component_cols if col in rankings.columns]].head(5).to_string(index=False))

    except FileNotFoundError as e:
        print(f"Error: Required data files not found. Run main.py first.")
        print(f"Details: {e}")


if __name__ == "__main__":
    main()
