"""
Calculate College LEAF Ratings (Simplified Version)

Uses cached team SP+ data, calculates without opponent adjustments due to API rate limits.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent


def load_play_by_play_data():
    """Load the play-by-play data"""
    root = get_project_root()
    path = root / 'data' / 'processed' / 'play_by_play' / 'all_qb_plays_full.csv'
    df = pd.read_csv(path)
    print(f"[OK] Loaded {len(df)} plays for {df['player_name'].nunique()} QBs")
    return df


def load_qb_metadata():
    """Load QB metadata with NFL ratings and SP+ data"""
    root = get_project_root()
    path = root / 'data' / 'processed' / 'merged_with_sp_plus.csv'
    df = pd.read_csv(path)
    print(f"[OK] Loaded metadata for {len(df)} QBs")
    return df


def calculate_context_weight(down, distance, score_diff, period, clock_minutes):
    """
    Calculate context weight based on situation (adapted from NFL LEAF)

    Returns multiplier for EPA based on game context
    """
    weight = 1.0

    # Down and distance (high leverage situations)
    if down == 3:
        if distance >= 7:
            weight *= 1.15  # 3rd and long - high leverage
        elif distance <= 3:
            weight *= 1.10  # 3rd and short - high leverage
    elif down == 4:
        weight *= 1.25  # 4th down - very high leverage

    # Score differential (competitive games matter more)
    abs_score_diff = abs(score_diff) if pd.notna(score_diff) else 0
    if abs_score_diff <= 7:
        weight *= 1.10  # One-score game
    elif abs_score_diff > 21:
        weight *= 0.85  # Blowout - garbage time discount

    # Time remaining (late game matters more)
    if period == 4 and clock_minutes <= 5:
        if abs_score_diff <= 7:
            weight *= 1.20  # Crunch time in close game

    return weight


def calculate_college_leaf(df_plays, df_metadata):
    """
    Calculate College LEAF ratings for all QBs (simplified version)

    Returns DataFrame with QB ratings
    """
    print("\n" + "="*80)
    print("CALCULATING COLLEGE LEAF RATINGS (SIMPLIFIED)")
    print("="*80)
    print("\nNote: Using context adjustments only (no opponent SP+ due to API limits)")

    qb_ratings = []

    # Process each QB
    for player_name in df_plays['player_name'].unique():
        print(f"\n  [{player_name}]")

        # Get QB plays
        qb_plays = df_plays[df_plays['player_name'] == player_name].copy()

        # Get QB metadata
        qb_meta = df_metadata[df_metadata['player_name'] == player_name]
        if len(qb_meta) == 0:
            print(f"    [SKIP] No metadata found")
            continue

        qb_meta = qb_meta.iloc[0]
        season = qb_meta['final_season']
        college = qb_meta.get('college_college', '')
        team_sp_plus = qb_meta.get('team_sp_plus', 0)

        print(f"    Season: {season}, Team: {college}")
        if pd.notna(team_sp_plus):
            print(f"    Team SP+: {team_sp_plus:+.1f}")
        print(f"    Total plays: {len(qb_plays)}")

        # Calculate adjusted EPA for each play
        adjusted_epa_values = []

        for _, play in qb_plays.iterrows():
            # Base EPA
            epa = play['ppa']
            if pd.isna(epa):
                continue

            # Context weight
            score_diff = play['offense_score'] - play['defense_score'] if pd.notna(play['offense_score']) else 0
            context_weight = calculate_context_weight(
                play['down'],
                play['distance'],
                score_diff,
                play['period'],
                play['clock_minutes']
            )

            # Adjusted EPA (context only, no opponent adjustment)
            adjusted_epa = epa * context_weight
            adjusted_epa_values.append(adjusted_epa)

        if not adjusted_epa_values:
            print(f"    [SKIP] No valid plays with EPA")
            continue

        # Aggregate to season rating
        raw_epa_mean = qb_plays['ppa'].mean()
        adjusted_epa_mean = np.mean(adjusted_epa_values)

        # Normalize to LEAF scale (similar to NFL LEAF range of -2 to +2)
        college_leaf_raw = raw_epa_mean * 2.0  # Simple scaling
        college_leaf_adjusted = adjusted_epa_mean * 2.0

        print(f"    Raw EPA/play: {raw_epa_mean:+.3f}")
        print(f"    Context-Adjusted EPA/play: {adjusted_epa_mean:+.3f}")
        print(f"    College LEAF (raw): {college_leaf_raw:+.3f}")
        print(f"    College LEAF (context-adjusted): {college_leaf_adjusted:+.3f}")

        qb_ratings.append({
            'player_name': player_name,
            'season': season,
            'college': college,
            'plays': len(adjusted_epa_values),
            'raw_epa_per_play': raw_epa_mean,
            'context_adjusted_epa_per_play': adjusted_epa_mean,
            'college_leaf_raw': college_leaf_raw,
            'college_leaf_adjusted': college_leaf_adjusted,
            'team_sp_plus': team_sp_plus,
        })

    df_ratings = pd.DataFrame(qb_ratings)

    print(f"\n[OK] Calculated College LEAF for {len(df_ratings)} QBs")

    return df_ratings


def merge_with_nfl_data(df_college_leaf, df_metadata):
    """Merge College LEAF with NFL rookie ratings"""
    df_merged = pd.merge(
        df_college_leaf,
        df_metadata[['player_name', 'draft_year', 'draft_round', 'draft_pick',
                     'rookie_leaf_mean', 'rookie_games']],
        on='player_name',
        how='inner'
    )

    print(f"\n[OK] Merged {len(df_merged)} QBs with NFL data")
    return df_merged


def analyze_correlations(df):
    """Analyze correlations between College LEAF and NFL performance"""
    print("\n" + "="*80)
    print("COLLEGE LEAF -> NFL ROOKIE LEAF ANALYSIS")
    print("="*80)

    print(f"\nSample Size: {len(df)} QBs")

    # Correlations
    print(f"\n--- CORRELATIONS WITH NFL ROOKIE LEAF ---")

    correlations = {}

    # Raw EPA
    corr_raw = df['college_leaf_raw'].corr(df['rookie_leaf_mean'])
    pval_raw = scipy_stats.pearsonr(df['college_leaf_raw'], df['rookie_leaf_mean'])[1]
    correlations['raw'] = corr_raw
    print(f"  College LEAF (raw):              r = {corr_raw:+.3f} (p = {pval_raw:.4f})")

    # Context-adjusted EPA
    corr_adj = df['college_leaf_adjusted'].corr(df['rookie_leaf_mean'])
    pval_adj = scipy_stats.pearsonr(df['college_leaf_adjusted'], df['rookie_leaf_mean'])[1]
    correlations['adjusted'] = corr_adj
    print(f"  College LEAF (context-adjusted): r = {corr_adj:+.3f} (p = {pval_adj:.4f})")

    # Draft pick (baseline)
    corr_draft = df['draft_pick'].corr(df['rookie_leaf_mean'])
    correlations['draft_pick'] = corr_draft
    print(f"  Draft Pick (baseline):           r = {corr_draft:+.3f}")

    # Improvement over baseline
    print(f"\n--- IMPROVEMENT OVER BASELINE ---")
    raw_improvement = abs(corr_raw) - abs(corr_draft)
    adj_improvement = abs(corr_adj) - abs(corr_draft)

    print(f"  Raw LEAF improvement:        {raw_improvement:+.3f} ({raw_improvement/abs(corr_draft)*100:+.1f}%)")
    print(f"  Adjusted LEAF improvement:   {adj_improvement:+.3f} ({adj_improvement/abs(corr_draft)*100:+.1f}%)")

    # R-squared comparison
    print(f"\n--- VARIANCE EXPLAINED (R-squared) ---")
    print(f"  Draft Pick:                  {corr_draft**2:.3f} ({corr_draft**2*100:.1f}%)")
    print(f"  College LEAF (raw):          {corr_raw**2:.3f} ({corr_raw**2*100:.1f}%)")
    print(f"  College LEAF (adjusted):     {corr_adj**2:.3f} ({corr_adj**2*100:.1f}%)")

    # Top performers
    print(f"\n--- TOP 5 COLLEGE LEAF PERFORMERS ---")
    top_5 = df.nlargest(5, 'college_leaf_adjusted')[
        ['player_name', 'college_leaf_adjusted', 'rookie_leaf_mean', 'draft_pick']
    ]
    print(top_5.to_string(index=False))

    return correlations


def save_results(df):
    """Save College LEAF results"""
    root = get_project_root()
    output_path = root / 'data' / 'processed' / 'college_leaf_ratings_simple.csv'

    df.to_csv(output_path, index=False)
    print(f"\n[OK] Saved College LEAF ratings to: {output_path}")


def main():
    """Main execution"""
    print("="*80)
    print("COLLEGE LEAF CALCULATOR (SIMPLIFIED VERSION)")
    print("="*80)
    print("\nAdapting NFL LEAF methodology to college football")
    print("Note: Context adjustments only (API rate limited)")

    # Load data
    print("\n[1] Loading play-by-play data...")
    df_plays = load_play_by_play_data()

    print("\n[2] Loading QB metadata...")
    df_metadata = load_qb_metadata()

    # Calculate College LEAF
    print("\n[3] Calculating College LEAF ratings...")
    df_ratings = calculate_college_leaf(df_plays, df_metadata)

    # Merge with NFL data
    print("\n[4] Merging with NFL rookie data...")
    df_final = merge_with_nfl_data(df_ratings, df_metadata)

    # Analyze
    print("\n[5] Analyzing correlations...")
    correlations = analyze_correlations(df_final)

    # Save
    print("\n[6] Saving results...")
    save_results(df_final)

    # Summary
    print("\n" + "="*80)
    print("COLLEGE LEAF CALCULATION COMPLETE")
    print("="*80)

    print(f"\nKEY RESULTS:")
    print(f"  Sample Size:                     {len(df_final)} QBs")
    print(f"  Baseline (Draft Pick):           r = {correlations['draft_pick']:+.3f}")
    print(f"  College LEAF (raw):              r = {correlations['raw']:+.3f}")
    print(f"  College LEAF (context-adjusted): r = {correlations['adjusted']:+.3f}")

    if abs(correlations['adjusted']) > abs(correlations['draft_pick']):
        improvement = abs(correlations['adjusted']) - abs(correlations['draft_pick'])
        print(f"\n[SUCCESS] College LEAF beats baseline by {improvement:+.3f}!")
    else:
        print(f"\n[UNCERTAIN] College LEAF does not beat baseline")

    print(f"\nNote: Full competition adjustment requires opponent SP+ data")
    print(f"      Will add when API rate limit resets")


if __name__ == "__main__":
    main()
