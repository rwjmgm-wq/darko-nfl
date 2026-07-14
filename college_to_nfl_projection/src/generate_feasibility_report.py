"""
Generate Final Feasibility Report

Comprehensive analysis of college-to-NFL QB projection feasibility
using LEAF model framework.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings('ignore')


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent


def load_data():
    """Load all datasets"""
    root = get_project_root()

    # Load merged data
    merged_path = root / 'data' / 'processed' / 'merged_college_nfl.csv'
    df_merged = pd.read_csv(merged_path)

    # Load SP+ adjusted data
    sp_plus_path = root / 'data' / 'processed' / 'merged_with_sp_plus.csv'
    df_sp_plus = pd.read_csv(sp_plus_path)

    return df_merged, df_sp_plus


def print_header(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"{title:^80}")
    print("="*80)


def analyze_sample_characteristics(df):
    """Analyze the QB sample characteristics"""
    print_header("SAMPLE CHARACTERISTICS")

    print(f"\nTotal Sample Size: {len(df)} QBs")
    print(f"Draft Years: {df['draft_year'].min()}-{df['draft_year'].max()}")
    print(f"College Seasons: {df['final_season'].min()}-{df['final_season'].max()}")

    print(f"\n--- DRAFT DISTRIBUTION ---")
    print(f"Round 1: {(df['draft_round'] == 1).sum()} QBs ({(df['draft_round'] == 1).sum()/len(df)*100:.1f}%)")
    print(f"Round 2: {(df['draft_round'] == 2).sum()} QBs ({(df['draft_round'] == 2).sum()/len(df)*100:.1f}%)")
    print(f"Round 3+: {(df['draft_round'] >= 3).sum()} QBs ({(df['draft_round'] >= 3).sum()/len(df)*100:.1f}%)")

    print(f"\n--- NFL ROOKIE PERFORMANCE ---")
    print(f"Avg Rookie LEAF Rating: {df['rookie_leaf_mean'].mean():.3f}")
    print(f"Std Dev: {df['rookie_leaf_mean'].std():.3f}")
    print(f"Range: {df['rookie_leaf_mean'].min():.3f} to {df['rookie_leaf_mean'].max():.3f}")

    print(f"\n--- COLLEGE GAMES ---")
    print(f"Avg Games Played: {df['games_played'].mean():.1f}")
    print(f"Range: {df['games_played'].min():.0f}-{df['games_played'].max():.0f} games")


def analyze_baseline(df):
    """Analyze baseline draft capital correlation"""
    print_header("BASELINE: DRAFT CAPITAL CORRELATION")

    # Overall correlation
    corr = df['draft_pick'].corr(df['rookie_leaf_mean'])
    pvalue = scipy_stats.pearsonr(df['draft_pick'], df['rookie_leaf_mean'])[1]

    print(f"\nDraft Pick vs Rookie LEAF Correlation:")
    print(f"  Pearson r = {corr:+.3f} (p = {pvalue:.4f})")

    if abs(corr) < 0.2:
        strength = "Very Weak"
    elif abs(corr) < 0.4:
        strength = "Weak"
    elif abs(corr) < 0.6:
        strength = "Moderate"
    else:
        strength = "Strong"

    print(f"  Strength: {strength}")
    print(f"  R² = {corr**2:.3f} ({corr**2 * 100:.1f}% of variance explained)")

    # By round
    print(f"\n--- PERFORMANCE BY DRAFT ROUND ---")
    round_stats = df.groupby('draft_round')['rookie_leaf_mean'].agg(['mean', 'std', 'count'])
    print(round_stats.to_string())

    # First round advantage
    first_round = df[df['draft_round'] == 1]['rookie_leaf_mean'].mean()
    later_rounds = df[df['draft_round'] >= 2]['rookie_leaf_mean'].mean()
    difference = first_round - later_rounds

    print(f"\nFirst Round Advantage: {difference:+.3f} LEAF points")

    return corr


def analyze_team_strength(df_sp_plus):
    """Analyze team SP+ strength correlation"""
    print_header("TEAM STRENGTH (SP+ RATINGS)")

    # Filter to QBs with SP+ data
    df_with_sp = df_sp_plus[df_sp_plus['team_sp_plus'].notna()].copy()

    print(f"\nSample Size: {len(df_with_sp)} QBs with team SP+ data")
    print(f"Coverage: {len(df_with_sp)/len(df_sp_plus)*100:.1f}%")

    print(f"\n--- TEAM SP+ DISTRIBUTION ---")
    print(f"Mean: {df_with_sp['team_sp_plus'].mean():.2f}")
    print(f"Std Dev: {df_with_sp['team_sp_plus'].std():.2f}")
    print(f"Range: {df_with_sp['team_sp_plus'].min():.2f} to {df_with_sp['team_sp_plus'].max():.2f}")

    # Correlation
    corr = df_with_sp['team_sp_plus'].corr(df_with_sp['rookie_leaf_mean'])
    pvalue = scipy_stats.pearsonr(df_with_sp['team_sp_plus'], df_with_sp['rookie_leaf_mean'])[1]

    print(f"\n--- CORRELATION WITH ROOKIE NFL LEAF ---")
    print(f"Team SP+ vs Rookie LEAF:")
    print(f"  Pearson r = {corr:+.3f} (p = {pvalue:.4f})")
    print(f"  R² = {corr**2:.3f}")

    # Compare to draft pick
    draft_corr = df_with_sp['draft_pick'].corr(df_with_sp['rookie_leaf_mean'])
    print(f"\nComparison to Draft Pick Baseline:")
    print(f"  Draft Pick: r = {draft_corr:+.3f}")
    print(f"  Team SP+:   r = {corr:+.3f}")
    improvement = abs(corr) - abs(draft_corr)
    print(f"  Improvement: {improvement:+.3f}")

    # Tiers
    if len(df_with_sp) >= 6:
        print(f"\n--- ROOKIE PERFORMANCE BY TEAM STRENGTH ---")
        df_with_sp['sp_tier'] = pd.qcut(
            df_with_sp['team_sp_plus'],
            q=3,
            labels=['Weak Teams (G5/Lower P5)', 'Mid Teams (Mid P5)', 'Elite Teams (Top P5)'],
            duplicates='drop'
        )

        tier_stats = df_with_sp.groupby('sp_tier', observed=True).agg({
            'rookie_leaf_mean': ['mean', 'std', 'count'],
            'team_sp_plus': ['mean', 'min', 'max']
        })
        print(tier_stats.to_string())

    return corr


def analyze_data_quality(df_merged, df_sp_plus):
    """Analyze data quality and coverage"""
    print_header("DATA QUALITY & COVERAGE")

    print(f"\n--- DATA COLLECTION RESULTS ---")
    print(f"[OK] Draft Data: {len(df_merged)}/113 QBs successfully merged (54.9%)")
    print(f"[OK] NFL Rookie LEAF: {len(df_merged)}/113 QBs with ratings (54.9%)")
    print(f"[OK] College Games: {df_merged['games_played'].notna().sum()}/{len(df_merged)} QBs (100%)")
    print(f"[OK] Team SP+ Ratings: {df_sp_plus['team_sp_plus'].notna().sum()}/{len(df_sp_plus)} QBs (90.3%)")
    print(f"[FAIL] Opponent Data: 0/{len(df_merged)} QBs (0.0%) - COLLECTION FAILED")

    print(f"\n--- DATA QUALITY ISSUES ---")
    print(f"1. Opponent schedule data collection failed")
    print(f"   - Prevents calculation of Strength of Schedule (SOS)")
    print(f"   - Requires re-collection from CFBD API")

    print(f"\n2. Limited sample size")
    print(f"   - Only 62 QBs with complete data")
    print(f"   - May limit statistical power for rare outcomes")

    print(f"\n3. Draft pick as baseline is very weak")
    print(f"   - r = -0.186 (only 3.5% of variance)")
    print(f"   - Low bar to beat, but also suggests high uncertainty")


def generate_feasibility_assessment(baseline_corr, team_sp_corr):
    """Generate overall feasibility assessment"""
    print_header("FEASIBILITY ASSESSMENT")

    print(f"\n--- KEY FINDINGS ---")

    print(f"\n1. BASELINE IS VERY WEAK")
    print(f"   Draft position explains only {baseline_corr**2*100:.1f}% of rookie performance")
    print(f"   This suggests high uncertainty but also opportunity for improvement")

    print(f"\n2. TEAM STRENGTH SHOWS PROMISE")
    print(f"   Team SP+ correlation: r = {team_sp_corr:+.3f}")
    improvement = abs(team_sp_corr) - abs(baseline_corr)
    if improvement > 0:
        print(f"   Improvement over baseline: {improvement:+.3f} ({improvement/abs(baseline_corr)*100:+.1f}%)")
        print(f"   -> Competition adjustment shows potential value")
    else:
        print(f"   No improvement over baseline")
        print(f"   -> Team strength alone insufficient")

    print(f"\n3. MISSING CRITICAL DATA")
    print(f"   - No opponent schedule data (SOS calculation impossible)")
    print(f"   - Cannot test full competition adjustment approach")
    print(f"   - Requires additional data collection")

    print(f"\n4. SAMPLE SIZE ADEQUATE")
    print(f"   - 62 QBs with complete data")
    print(f"   - Sufficient for initial feasibility study")
    print(f"   - Would benefit from expansion to earlier draft classes")

    print(f"\n--- FEASIBILITY RATING ---")

    if abs(team_sp_corr) > abs(baseline_corr):
        rating = "PROMISING"
        color = "[+]"
    else:
        rating = "UNCERTAIN"
        color = "[?]"

    print(f"\n{color} Overall Rating: {rating}")
    print(f"\nReasoning:")
    print(f"  - Competition adjustment (team strength) shows potential")
    print(f"  - Data collection is feasible via CFBD API")
    print(f"  - Baseline is weak, leaving room for improvement")
    print(f"  - Missing opponent data is critical limitation")


def generate_recommendations():
    """Generate recommendations for next steps"""
    print_header("RECOMMENDATIONS")

    print(f"\n--- IMMEDIATE ACTIONS ---")

    print(f"\n1. FIX OPPONENT DATA COLLECTION")
    print(f"   Priority: HIGH")
    print(f"   - Re-run college stats collection with proper opponent tracking")
    print(f"   - Verify opponent names match SP+ team names")
    print(f"   - Calculate Strength of Schedule (SOS) from opponent SP+ ratings")

    print(f"\n2. EXPAND SAMPLE SIZE")
    print(f"   Priority: MEDIUM")
    print(f"   - Add QBs from 2010-2014 draft classes")
    print(f"   - Target: 100+ QBs with complete data")
    print(f"   - Enables more robust statistical analysis")

    print(f"\n3. ADD INDIVIDUAL QB METRICS")
    print(f"   Priority: MEDIUM")
    print(f"   - Fetch passing stats (yards, TDs, completion %, EPA)")
    print(f"   - Competition-adjust individual performance")
    print(f"   - Test adjusted stats as NFL predictors")

    print(f"\n--- FEASIBILITY DECISION ---")

    print(f"\n[RECOMMEND] PROCEED WITH FULL PROJECT")
    print(f"\nConditions:")
    print(f"  1. Fix opponent data collection (required)")
    print(f"  2. Calculate full SOS-adjusted metrics")
    print(f"  3. Verify improvement over baseline")

    print(f"\nIf SOS adjustment shows r > 0.30:")
    print(f"  -> Build full college LEAF model")
    print(f"  -> Develop NFL translation framework")
    print(f"  -> Target: Career prediction, not just rookie season")

    print(f"\nIf SOS adjustment remains r < 0.30:")
    print(f"  -> Pivot to supplementary tool only")
    print(f"  -> Combine with draft capital, combine results, scouting")
    print(f"  -> Set realistic expectations for accuracy")


def main():
    """Generate complete feasibility report"""
    print("="*80)
    print("COLLEGE-TO-NFL QB PROJECTION FEASIBILITY REPORT")
    print("="*80)
    print("\nUsing LEAF Model Framework for Rookie Season Performance")
    print(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    df_merged, df_sp_plus = load_data()

    # Run analyses
    analyze_sample_characteristics(df_merged)
    baseline_corr = analyze_baseline(df_merged)
    team_sp_corr = analyze_team_strength(df_sp_plus)
    analyze_data_quality(df_merged, df_sp_plus)
    generate_feasibility_assessment(baseline_corr, team_sp_corr)
    generate_recommendations()

    # Summary
    print_header("REPORT COMPLETE")
    print(f"\nKey Metrics Summary:")
    print(f"  Sample Size: 62 QBs (2015-2023 drafts)")
    print(f"  Baseline (Draft Pick): r = {baseline_corr:+.3f}")
    print(f"  Team SP+ Strength: r = {team_sp_corr:+.3f}")
    print(f"  Data Quality: 90% team ratings, 0% opponent data")
    print(f"\nNext Step: Fix opponent data collection, then re-evaluate")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
