"""
DEPRECATED - do not use. Superseded by v2_evaluate.py.

This script exhaustively searches up to 10,000 feature combinations on ~50 QBs
and reports the best IN-SAMPLE R^2. That number (r=0.64, R^2=41%) is a
selection artifact, not predictive power: evaluated honestly (leave-one-
draft-class-out CV), college stats do not predict rookie-season performance
at all (CV R^2 <= 0). See results/v2_evaluation_report.md.

---
Feature Selection Analysis for College-to-NFL QB Projection

Tests which combination of stats best predicts:
1. NFL rookie LEAF
2. Career longevity (85+ starts)

Uses opponent strength adjustment on all metrics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent


def load_play_by_play_data():
    """Load play-by-play data with SP+ ratings"""
    root = get_project_root()
    path = root / 'data' / 'processed' / 'play_by_play' / 'all_qb_plays_with_sp_plus.csv'
    df = pd.read_csv(path)
    print(f"[OK] Loaded {len(df):,} plays for {df['player_name'].nunique()} QBs")
    return df


def load_qb_metadata():
    """Load QB metadata with NFL ratings"""
    root = get_project_root()
    path = root / 'data' / 'processed' / 'merged_with_sp_plus.csv'
    df = pd.read_csv(path)
    return df


def calculate_qb_stats(df_plays, player_name):
    """
    Calculate comprehensive QB stats from play-by-play data

    Returns dict with raw and opponent-adjusted metrics
    """
    qb_plays = df_plays[df_plays['player_name'] == player_name].copy()

    if len(qb_plays) == 0:
        return None

    stats = {}

    # Basic counting
    total_plays = len(qb_plays)
    stats['attempts'] = total_plays

    # Completion data (assuming all pass plays, completions have positive yards or TDs)
    # We don't have explicit completion flag, so use EPA as proxy
    completions = (qb_plays['ppa'] > -1.5).sum()  # Rough heuristic
    stats['completion_rate'] = completions / total_plays if total_plays > 0 else 0

    # EPA-based metrics
    stats['epa_per_play'] = qb_plays['ppa'].mean()
    stats['epa_std'] = qb_plays['ppa'].std()
    stats['success_rate'] = (qb_plays['ppa'] > 0).mean()
    stats['big_play_rate'] = (qb_plays['ppa'] > 1.5).mean()
    stats['explosive_rate'] = (qb_plays['ppa'] > 2.5).mean()

    # Negative plays
    stats['turnover_worthy_rate'] = (qb_plays['ppa'] < -2.0).mean()
    stats['sack_rate'] = (qb_plays['ppa'] < -1.5).mean()  # Approximation

    # Context-weighted EPA (high-leverage situations)
    high_leverage = qb_plays[
        ((qb_plays['down'] >= 3) & (qb_plays['distance'] >= 7)) |
        (qb_plays['down'] == 4)
    ]
    stats['high_leverage_epa'] = high_leverage['ppa'].mean() if len(high_leverage) > 0 else 0

    # Red zone performance (approximation using yards_to_goal)
    red_zone = qb_plays[qb_plays['yards_to_goal'] <= 20]
    stats['red_zone_epa'] = red_zone['ppa'].mean() if len(red_zone) > 0 else 0

    # Situational
    third_down = qb_plays[qb_plays['down'] == 3]
    stats['third_down_epa'] = third_down['ppa'].mean() if len(third_down) > 0 else 0

    # Opponent-adjusted metrics
    plays_with_opp = qb_plays[qb_plays['opponent_defense_sp'].notna()].copy()

    if len(plays_with_opp) > 0:
        # Calculate opponent-adjusted EPA
        # Stronger opponent defense (higher SP+) = harder = adjust up
        plays_with_opp['opp_adj_epa'] = plays_with_opp.apply(
            lambda row: row['ppa'] * (1 + 0.30 * row['opponent_defense_sp'] / 10)
            if pd.notna(row['opponent_defense_sp']) else row['ppa'],
            axis=1
        )

        stats['opp_adj_epa'] = plays_with_opp['opp_adj_epa'].mean()
        stats['opp_adj_success_rate'] = (plays_with_opp['opp_adj_epa'] > 0).mean()
        stats['opp_adj_big_play_rate'] = (plays_with_opp['opp_adj_epa'] > 1.5).mean()

        # Average opponent strength faced
        stats['avg_opp_defense_sp'] = plays_with_opp['opponent_defense_sp'].mean()
        stats['opp_sp_coverage'] = len(plays_with_opp) / total_plays
    else:
        stats['opp_adj_epa'] = stats['epa_per_play']
        stats['opp_adj_success_rate'] = stats['success_rate']
        stats['opp_adj_big_play_rate'] = stats['big_play_rate']
        stats['avg_opp_defense_sp'] = 0
        stats['opp_sp_coverage'] = 0

    # Team strength adjustment
    plays_with_team = qb_plays[qb_plays['team_offense_sp'].notna()].copy()
    if len(plays_with_team) > 0:
        stats['avg_team_offense_sp'] = plays_with_team['team_offense_sp'].mean()

        # Penalize QBs on elite offenses
        plays_with_team['team_adj_epa'] = plays_with_team.apply(
            lambda row: row['ppa'] * (1 - 0.15 * row['team_offense_sp'] / 10)
            if pd.notna(row['team_offense_sp']) else row['ppa'],
            axis=1
        )
        stats['team_adj_epa'] = plays_with_team['team_adj_epa'].mean()
    else:
        stats['avg_team_offense_sp'] = 0
        stats['team_adj_epa'] = stats['epa_per_play']

    # Full adjustment (both opponent and team)
    if len(plays_with_opp) > 0 and len(plays_with_team) > 0:
        combined = qb_plays[
            qb_plays['opponent_defense_sp'].notna() &
            qb_plays['team_offense_sp'].notna()
        ].copy()

        if len(combined) > 0:
            combined['full_adj_epa'] = combined.apply(
                lambda row: row['ppa'] * (
                    1 + 0.30 * row['opponent_defense_sp'] / 10 -
                    0.15 * row['team_offense_sp'] / 10
                ),
                axis=1
            )
            stats['full_adj_epa'] = combined['full_adj_epa'].mean()
        else:
            stats['full_adj_epa'] = stats['epa_per_play']
    else:
        stats['full_adj_epa'] = stats['epa_per_play']

    return stats


def load_career_starts():
    """Load career starts data for longevity analysis"""
    # We'll need to get this from the NFL data
    # For now, use a placeholder - would need to be extracted from nfl_data_py
    root = get_project_root()
    # This would be extracted similar to how we got rookie LEAF
    return {}


def find_best_feature_combination(features_df, target_col, n_features=5):
    """
    Find best N-feature combination using exhaustive search

    Returns: (best_features, correlation, r_squared)
    """
    feature_cols = [col for col in features_df.columns
                   if col not in ['player_name', target_col] and
                   features_df[col].notna().all()]

    # Remove features with no variance
    feature_cols = [col for col in feature_cols if features_df[col].std() > 0]

    best_r2 = -999
    best_features = None
    best_corr = 0

    print(f"\n  Testing combinations of {n_features} features from {len(feature_cols)} candidates...")
    print(f"  Total combinations to test: {len(list(combinations(feature_cols, n_features)))}")

    # Limit search if too many combinations
    max_combinations = 10000
    all_combinations = list(combinations(feature_cols, n_features))

    if len(all_combinations) > max_combinations:
        print(f"  [NOTE] Limiting to {max_combinations} random combinations")
        import random
        random.seed(42)
        all_combinations = random.sample(all_combinations, max_combinations)

    for i, feature_combo in enumerate(all_combinations):
        if i % 1000 == 0 and i > 0:
            print(f"    Tested {i}/{len(all_combinations)} combinations...")

        # Check for missing values
        X = features_df[list(feature_combo)]
        y = features_df[target_col]

        valid_mask = X.notna().all(axis=1) & y.notna()
        if valid_mask.sum() < 10:  # Need at least 10 samples
            continue

        X_valid = X[valid_mask]
        y_valid = y[valid_mask]

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_valid)

        # Fit linear model
        model = LinearRegression()
        model.fit(X_scaled, y_valid)

        # Calculate R²
        r2 = model.score(X_scaled, y_valid)

        # Calculate correlation
        y_pred = model.predict(X_scaled)
        corr = np.corrcoef(y_valid, y_pred)[0, 1]

        if r2 > best_r2:
            best_r2 = r2
            best_features = feature_combo
            best_corr = corr

    return best_features, best_corr, best_r2


def main():
    """Main execution"""
    print("="*80)
    print("FEATURE SELECTION ANALYSIS")
    print("="*80)
    print("\nFinding optimal stat combinations for predicting NFL success")

    # Load data
    print("\n[1] Loading data...")
    df_plays = load_play_by_play_data()
    df_metadata = load_qb_metadata()

    # Calculate stats for each QB
    print("\n[2] Calculating comprehensive stats for each QB...")
    all_qb_stats = []

    for player_name in sorted(df_plays['player_name'].unique()):
        print(f"  Processing: {player_name}")
        stats = calculate_qb_stats(df_plays, player_name)
        if stats:
            stats['player_name'] = player_name
            all_qb_stats.append(stats)

    df_stats = pd.DataFrame(all_qb_stats)
    print(f"\n[OK] Calculated {len(df_stats.columns)-1} stats for {len(df_stats)} QBs")

    # Merge with NFL outcomes
    print("\n[3] Merging with NFL outcome data...")
    df_merged = df_stats.merge(
        df_metadata[['player_name', 'rookie_leaf_mean', 'draft_pick']],
        on='player_name',
        how='inner'
    )

    print(f"[OK] {len(df_merged)} QBs with complete data")

    # Feature selection for NFL rookie LEAF
    print("\n" + "="*80)
    print("ANALYSIS 1: PREDICTING NFL ROOKIE LEAF")
    print("="*80)

    print("\n[4] Finding best 5-feature combination for rookie LEAF...")
    best_features_leaf, corr_leaf, r2_leaf = find_best_feature_combination(
        df_merged, 'rookie_leaf_mean', n_features=5
    )

    print(f"\n[RESULTS] Best 5-feature model:")
    print(f"  Features: {list(best_features_leaf)}")
    print(f"  Correlation: r = {corr_leaf:+.3f}")
    print(f"  R-squared: {r2_leaf:.3f} ({r2_leaf*100:.1f}%)")
    print(f"  vs. Draft Pick baseline: r = -0.314 (R² = 9.9%)")

    if r2_leaf > 0.099:
        improvement = (r2_leaf - 0.099) / 0.099 * 100
        print(f"  Improvement: {improvement:+.1f}% better than draft capital")
    else:
        decline = (0.099 - r2_leaf) / 0.099 * 100
        print(f"  Performance: {decline:.1f}% worse than draft capital")

    # Show individual feature correlations
    print(f"\n  Individual feature correlations with rookie LEAF:")
    for feature in best_features_leaf:
        valid_mask = df_merged[feature].notna() & df_merged['rookie_leaf_mean'].notna()
        if valid_mask.sum() > 0:
            corr = np.corrcoef(
                df_merged[valid_mask][feature],
                df_merged[valid_mask]['rookie_leaf_mean']
            )[0, 1]
            print(f"    {feature:30s}: r = {corr:+.3f}")

    # Try other combinations
    print("\n[5] Testing other promising combinations...")

    # Try 3-feature models (faster)
    print("\n  Best 3-feature model:")
    best_3, corr_3, r2_3 = find_best_feature_combination(
        df_merged, 'rookie_leaf_mean', n_features=3
    )
    print(f"    Features: {list(best_3)}")
    print(f"    r = {corr_3:+.3f}, R² = {r2_3:.3f} ({r2_3*100:.1f}%)")

    # Try 7-feature models
    print("\n  Best 7-feature model:")
    best_7, corr_7, r2_7 = find_best_feature_combination(
        df_merged, 'rookie_leaf_mean', n_features=7
    )
    print(f"    Features: {list(best_7)}")
    print(f"    r = {corr_7:+.3f}, R² = {r2_7:.3f} ({r2_7*100:.1f}%)")

    # Save results
    print("\n[6] Saving results...")

    # Save comprehensive stats
    output_path = get_project_root() / 'data' / 'processed' / 'qb_comprehensive_stats.csv'
    df_merged.to_csv(output_path, index=False)
    print(f"  [OK] Saved comprehensive stats: {output_path}")

    # Save best model results
    results = {
        'best_5_features': list(best_features_leaf),
        'correlation': corr_leaf,
        'r_squared': r2_leaf,
        'best_3_features': list(best_3),
        'corr_3': corr_3,
        'r2_3': r2_3,
        'best_7_features': list(best_7),
        'corr_7': corr_7,
        'r2_7': r2_7
    }

    results_path = get_project_root() / 'data' / 'processed' / 'feature_selection_results.json'
    import json
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  [OK] Saved model results: {results_path}")

    # Summary
    print("\n" + "="*80)
    print("FEATURE SELECTION COMPLETE")
    print("="*80)
    print(f"\nBest predictive model:")
    print(f"  5 features: {list(best_features_leaf)}")
    print(f"  Explains {r2_leaf*100:.1f}% of variance in NFL rookie LEAF")
    print(f"  Correlation: r = {corr_leaf:+.3f}")

    if r2_leaf < 0.099:
        print(f"\n[NOTE] Still underperforms draft capital (R² = 9.9%)")
        print(f"       College stats explain limited variance in NFL success")
    else:
        print(f"\n[SUCCESS] Outperforms draft capital baseline!")


if __name__ == "__main__":
    main()
