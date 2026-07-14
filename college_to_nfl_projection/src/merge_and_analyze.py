"""
Merge College and NFL Data and Run Preliminary Analysis

This script combines college stats with NFL rookie ratings and performs initial correlation analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent


def load_data():
    """Load college and NFL datasets"""
    root = get_project_root()

    # Load college stats
    college_path = root / 'data' / 'processed' / 'college_qb_stats.csv'
    college_df = pd.read_csv(college_path)
    print(f"[OK] Loaded {len(college_df)} college QB records")

    # Load NFL rookie ratings
    nfl_path = root / 'data' / 'processed' / 'nfl_rookie_ratings.csv'
    nfl_df = pd.read_csv(nfl_path)
    print(f"[OK] Loaded {len(nfl_df)} NFL rookie rating records")

    return college_df, nfl_df


def merge_datasets(college_df, nfl_df):
    """
    Merge college and NFL datasets by player name and draft year

    Args:
        college_df: College stats dataframe
        nfl_df: NFL rookie ratings dataframe

    Returns:
        Merged dataframe
    """
    print("\n" + "="*80)
    print("MERGING DATASETS")
    print("="*80)

    # Merge on player_name and draft_year
    merged = pd.merge(
        college_df,
        nfl_df,
        on=['player_name', 'draft_year'],
        how='inner',
        suffixes=('_college', '_nfl')
    )

    print(f"\n[OK] Merged dataset: {len(merged)} QBs with both college and NFL data")

    # Filter to QBs who actually played in NFL (games > 0)
    merged_played = merged[merged['rookie_games'] > 0].copy()
    print(f"     QBs who played in rookie season: {len(merged_played)}")

    # Show summary
    print(f"\n  Draft years: {merged_played['draft_year'].min()}-{merged_played['draft_year'].max()}")
    print(f"  Avg college games: {merged_played['games_played'].mean():.1f}")
    print(f"  Avg rookie games: {merged_played['rookie_games'].mean():.1f}")

    return merged_played


def analyze_draft_capital(df):
    """Analyze relationship between draft position and rookie performance"""
    print("\n" + "="*80)
    print("DRAFT CAPITAL ANALYSIS")
    print("="*80)

    # Calculate correlation between draft pick and rookie LEAF rating
    corr = df['draft_pick'].corr(df['rookie_leaf_mean'])
    print(f"\nDraft Pick vs Rookie LEAF Correlation: r = {corr:.3f}")

    # By round
    print("\nAverage Rookie LEAF Rating by Round:")
    round_avg = df.groupby('draft_round')['rookie_leaf_mean'].agg(['mean', 'count', 'std'])
    print(round_avg.to_string())

    # Top vs bottom
    first_rounders = df[df['draft_round'] == 1]['rookie_leaf_mean'].mean()
    later_rounds = df[df['draft_round'] >= 2]['rookie_leaf_mean'].mean()

    print(f"\nFirst Round Average: {first_rounders:.3f}")
    print(f"Later Rounds Average: {later_rounds:.3f}")
    print(f"Difference: {first_rounders - later_rounds:.3f}")

    return corr


def calculate_sos_proxy(df):
    """
    Calculate a simple strength of schedule proxy based on opponent names

    For now, uses game count as a basic measure. In future, can integrate
    with actual CFB power rankings.
    """
    print("\n" + "="*80)
    print("STRENGTH OF SCHEDULE PROXY")
    print("="*80)

    # For now, use games_played as a proxy for experience/competition level
    # More games typically means higher level (P5 schools play more games)
    df['college_games_played'] = df['games_played']

    print(f"\nCollege games distribution:")
    print(df['college_games_played'].describe())

    # Categorize by games played (rough proxy for competition level)
    df['competition_tier'] = pd.cut(
        df['college_games_played'],
        bins=[0, 10, 12, 15],
        labels=['Low', 'Medium', 'High']
    )

    print(f"\nRookie LEAF by Competition Tier (games played):")
    tier_avg = df.groupby('competition_tier')['rookie_leaf_mean'].agg(['mean', 'count'])
    print(tier_avg.to_string())

    return df


def preliminary_analysis(df):
    """Run preliminary correlation analysis"""
    print("\n" + "="*80)
    print("PRELIMINARY ANALYSIS")
    print("="*80)

    print(f"\nSample Size: {len(df)} QBs")
    print(f"Draft Years: {df['draft_year'].min()}-{df['draft_year'].max()}")

    # Summary statistics
    print(f"\nRookie LEAF Rating Distribution:")
    print(df['rookie_leaf_mean'].describe())

    # Correlations
    print(f"\n--- CORRELATIONS WITH ROOKIE LEAF RATING ---")

    variables = {
        'Draft Pick': 'draft_pick',
        'Draft Round': 'draft_round',
        'College Games': 'games_played',
        'Rookie Games': 'rookie_games',
    }

    for name, col in variables.items():
        if col in df.columns:
            corr = df[col].corr(df['rookie_leaf_mean'])
            print(f"{name:20s}: r = {corr:+.3f}")

    # Best and worst performers
    print(f"\n--- TOP 10 ROOKIE SEASONS ---")
    top_10 = df.nlargest(10, 'rookie_leaf_mean')[
        ['player_name', 'draft_year', 'draft_round', 'draft_pick',
         'games_played', 'rookie_leaf_mean', 'rookie_games']
    ]
    print(top_10.to_string(index=False))

    print(f"\n--- BOTTOM 10 ROOKIE SEASONS ---")
    bottom_10 = df.nsmallest(10, 'rookie_leaf_mean')[
        ['player_name', 'draft_year', 'draft_round', 'draft_pick',
         'games_played', 'rookie_leaf_mean', 'rookie_games']
    ]
    print(bottom_10.to_string(index=False))

    return df


def save_merged_data(df):
    """Save the merged dataset"""
    root = get_project_root()
    output_path = root / 'data' / 'processed' / 'merged_college_nfl.csv'

    df.to_csv(output_path, index=False)
    print(f"\n[OK] Saved merged dataset to: {output_path}")
    print(f"     Total records: {len(df)}")


def main():
    """Main execution"""
    print("="*80)
    print("COLLEGE-TO-NFL QB PROJECTION - FEASIBILITY ANALYSIS")
    print("="*80)

    # Load data
    print("\n[1] Loading datasets...")
    college_df, nfl_df = load_data()

    # Merge
    print("\n[2] Merging college and NFL data...")
    merged_df = merge_datasets(college_df, nfl_df)

    # Draft capital analysis
    print("\n[3] Analyzing draft capital as baseline...")
    draft_corr = analyze_draft_capital(merged_df)

    # SOS proxy
    print("\n[4] Creating strength of schedule proxy...")
    merged_df = calculate_sos_proxy(merged_df)

    # Preliminary analysis
    print("\n[5] Running preliminary analysis...")
    merged_df = preliminary_analysis(merged_df)

    # Save
    print("\n[6] Saving merged dataset...")
    save_merged_data(merged_df)

    # Summary
    print("\n" + "="*80)
    print("FEASIBILITY ANALYSIS COMPLETE")
    print("="*80)
    print(f"\n✓ Successfully merged and analyzed {len(merged_df)} QBs")
    print(f"✓ Draft capital baseline: r = {draft_corr:.3f}")
    print(f"✓ Ready for competition adjustment analysis")

    print(f"\nNext Steps:")
    print(f"  1. Integrate CFB power rankings for opponent strength")
    print(f"  2. Calculate competition-adjusted college metrics")
    print(f"  3. Compare adjusted vs raw correlations with rookie NFL LEAF")
    print(f"  4. Generate final feasibility report")


if __name__ == "__main__":
    main()
