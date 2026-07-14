"""
DEPRECATED - do not use. Superseded by v2_build_adjusted_stats.py.

Two known defects:
1. The competition multiplier ships with "REVERSED LOGIC TEST" constants
   (alpha=-0.30, beta=+0.30) that reward playing weak defenses - the opposite
   of the documented methodology. feature_selection_analysis.py uses yet
   another sign convention; the two never agreed.
2. Multiplying signed EPA by a multiplier is wrong regardless of sign: it
   penalizes negative plays against good opponents. The v2 adjustment is
   additive (EPA relative to opponent-expected EPA) with a slope estimated
   from within-QB schedule variation.

---
Calculate College LEAF Ratings (Full Implementation with Competition Adjustment)

Uses play-level EPA with context and available competition adjustments.
Falls back gracefully when opponent SP+ data is not available.
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
    # Try loading enhanced file with SP+ data first
    enhanced_path = root / 'data' / 'processed' / 'play_by_play' / 'all_qb_plays_with_sp_plus.csv'
    base_path = root / 'data' / 'processed' / 'play_by_play' / 'all_qb_plays_full.csv'

    if enhanced_path.exists():
        path = enhanced_path
        print(f"[OK] Loading enhanced data with cached SP+ ratings")
    else:
        path = base_path
        print(f"[OK] Loading base data (no cached SP+)")

    df = pd.read_csv(path)
    print(f"[OK] Loaded {len(df):,} plays for {df['player_name'].nunique()} QBs")
    return df


def load_qb_metadata():
    """Load QB metadata with NFL ratings and team SP+ data"""
    root = get_project_root()
    path = root / 'data' / 'processed' / 'merged_with_sp_plus.csv'
    df = pd.read_csv(path)
    print(f"[OK] Loaded metadata for {len(df)} QBs")
    return df


def load_sp_plus_data():
    """
    Load all available SP+ data from previous collection

    Returns dict mapping (team, season) -> SP+ ratings
    """
    root = get_project_root()
    metadata_path = root / 'data' / 'processed' / 'merged_with_sp_plus.csv'
    df_meta = pd.read_csv(metadata_path)

    # Build lookup dict for team SP+ ratings
    sp_plus_dict = {}

    for _, row in df_meta.iterrows():
        if pd.notna(row.get('team_sp_plus')):
            team = row.get('college_college', '')
            season = row.get('final_season')

            if team and season:
                # We have overall SP+ but not the component breakdowns
                # Estimate offense/defense as +/- 50% of total
                overall_sp = row['team_sp_plus']

                sp_plus_dict[(team, season)] = {
                    'overall': overall_sp,
                    'offense': overall_sp * 0.6,  # Rough estimate
                    'defense': overall_sp * 0.4,  # Rough estimate
                }

    return sp_plus_dict


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


def calculate_competition_multiplier(opponent_sp_plus, team_sp_plus_offense, alpha=-0.30, beta=0.30):
    """
    Calculate competition multiplier based on opponent and team strength

    REVERSED LOGIC TEST:
    - Weaker opponent defense = HIGHER multiplier (domination matters)
    - Stronger team offense = HIGHER multiplier (elite supporting cast matters)

    Returns multiplier (0.5 to 1.5)
    """
    # If we don't have opponent data, return neutral multiplier
    if pd.isna(opponent_sp_plus):
        opponent_sp_plus = 0  # Assume average opponent

    if pd.isna(team_sp_plus_offense):
        team_sp_plus_offense = 0  # Assume average team

    # REVERSED: Reward domination, penalize struggling vs weak competition
    # α < 0: Playing vs weaker defense = higher multiplier
    # β > 0: Playing with better offense = higher multiplier
    opponent_factor = alpha * (opponent_sp_plus / 10)
    team_factor = beta * (team_sp_plus_offense / 10)

    multiplier = 1.0 + opponent_factor + team_factor

    # Wider clip range to allow more adjustment
    return np.clip(multiplier, 0.5, 1.5)


def estimate_opponent_strength(opponent_name, season, sp_plus_dict):
    """
    Estimate opponent defensive strength

    Try to find opponent in SP+ dict, otherwise return 0 (average)
    """
    # Try exact match
    if (opponent_name, season) in sp_plus_dict:
        return sp_plus_dict[(opponent_name, season)].get('defense', 0)

    # Try case-insensitive partial match
    for (team, szn), ratings in sp_plus_dict.items():
        if szn == season and opponent_name and team:
            if opponent_name.lower() in team.lower() or team.lower() in opponent_name.lower():
                return ratings.get('defense', 0)

    # Default to average
    return 0


def calculate_college_leaf(df_plays, df_metadata):
    """
    Calculate College LEAF ratings for all QBs with full competition adjustment

    Returns DataFrame with QB ratings
    """
    print("\n" + "="*80)
    print("CALCULATING COLLEGE LEAF RATINGS (FULL IMPLEMENTATION)")
    print("="*80)
    print("\nUsing: Context adjustments + Competition adjustments (where available)")

    # Load SP+ data
    print("\nLoading SP+ data...")
    sp_plus_dict = load_sp_plus_data()
    print(f"[OK] Loaded SP+ for {len(sp_plus_dict)} (team, season) pairs")

    qb_ratings = []

    # Process each QB
    for player_name in sorted(df_plays['player_name'].unique()):
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
        team_sp_plus_overall = qb_meta.get('team_sp_plus', 0)

        # Estimate team offense SP+ (60% of overall)
        team_sp_plus_offense = team_sp_plus_overall * 0.6 if pd.notna(team_sp_plus_overall) else 0

        print(f"    Season: {season}, Team: {college}")
        if pd.notna(team_sp_plus_overall):
            print(f"    Team SP+: {team_sp_plus_overall:+.1f} (Est. Offense: {team_sp_plus_offense:+.1f})")
        print(f"    Total plays: {len(qb_plays)}")

        # Calculate adjusted EPA for each play
        raw_epa_values = []
        context_adjusted_epa_values = []
        competition_adjusted_epa_values = []
        full_adjusted_epa_values = []

        plays_with_comp_adj = 0

        for _, play in qb_plays.iterrows():
            # Base EPA
            epa = play['ppa']
            if pd.isna(epa):
                continue

            raw_epa_values.append(epa)

            # Context weight
            score_diff = 0
            if pd.notna(play['offense_score']) and pd.notna(play['defense_score']):
                score_diff = play['offense_score'] - play['defense_score']

            context_weight = calculate_context_weight(
                play['down'],
                play['distance'],
                score_diff,
                play['period'],
                play['clock_minutes']
            )

            context_adjusted_epa = epa * context_weight
            context_adjusted_epa_values.append(context_adjusted_epa)

            # Competition multiplier (use pre-joined opponent SP+ data if available)
            opponent_defense_sp = play.get('opponent_defense_sp', np.nan)
            play_team_offense_sp = play.get('team_offense_sp', np.nan)

            # Fall back to estimated team offense if play-level data not available
            if pd.isna(play_team_offense_sp):
                play_team_offense_sp = team_sp_plus_offense

            comp_multiplier = calculate_competition_multiplier(
                opponent_defense_sp,
                play_team_offense_sp
            )

            # Track if we have real opponent data (not missing)
            if pd.notna(opponent_defense_sp):
                plays_with_comp_adj += 1

            competition_adjusted_epa = epa * comp_multiplier
            competition_adjusted_epa_values.append(competition_adjusted_epa)

            # Full adjustment (context + competition)
            full_adjusted_epa = epa * context_weight * comp_multiplier
            full_adjusted_epa_values.append(full_adjusted_epa)

        if not raw_epa_values:
            print(f"    [SKIP] No valid plays with EPA")
            continue

        # Aggregate to season rating
        raw_epa_mean = np.mean(raw_epa_values)
        context_adj_epa_mean = np.mean(context_adjusted_epa_values)
        comp_adj_epa_mean = np.mean(competition_adjusted_epa_values)
        full_adj_epa_mean = np.mean(full_adjusted_epa_values)

        # Normalize to LEAF scale (similar to NFL LEAF range of -2 to +2)
        college_leaf_raw = raw_epa_mean * 2.0
        college_leaf_context = context_adj_epa_mean * 2.0
        college_leaf_competition = comp_adj_epa_mean * 2.0
        college_leaf_full = full_adj_epa_mean * 2.0

        comp_coverage = plays_with_comp_adj / len(raw_epa_values) * 100 if raw_epa_values else 0

        print(f"    Competition adjustment coverage: {comp_coverage:.1f}%")
        print(f"    Raw EPA/play: {raw_epa_mean:+.3f}")
        print(f"    College LEAF (raw): {college_leaf_raw:+.3f}")
        print(f"    College LEAF (context): {college_leaf_context:+.3f}")
        print(f"    College LEAF (competition): {college_leaf_competition:+.3f}")
        print(f"    College LEAF (FULL): {college_leaf_full:+.3f}")

        qb_ratings.append({
            'player_name': player_name,
            'season': season,
            'college': college,
            'plays': len(raw_epa_values),
            'comp_adj_coverage': comp_coverage,
            'raw_epa_per_play': raw_epa_mean,
            'context_adj_epa_per_play': context_adj_epa_mean,
            'comp_adj_epa_per_play': comp_adj_epa_mean,
            'full_adj_epa_per_play': full_adj_epa_mean,
            'college_leaf_raw': college_leaf_raw,
            'college_leaf_context': college_leaf_context,
            'college_leaf_competition': college_leaf_competition,
            'college_leaf_full': college_leaf_full,
            'team_sp_plus': team_sp_plus_overall,
        })

    df_ratings = pd.DataFrame(qb_ratings)

    print(f"\n[OK] Calculated College LEAF for {len(df_ratings)} QBs")

    # Summary stats
    avg_coverage = df_ratings['comp_adj_coverage'].mean()
    print(f"\nAverage competition adjustment coverage: {avg_coverage:.1f}%")

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
    print("COLLEGE LEAF -> NFL ROOKIE LEAF ANALYSIS (FULL MODEL)")
    print("="*80)

    print(f"\nSample Size: {len(df)} QBs")

    # Correlations
    print(f"\n--- CORRELATIONS WITH NFL ROOKIE LEAF ---")

    correlations = {}

    # Raw EPA
    corr_raw = df['college_leaf_raw'].corr(df['rookie_leaf_mean'])
    pval_raw = scipy_stats.pearsonr(df['college_leaf_raw'], df['rookie_leaf_mean'])[1]
    correlations['raw'] = corr_raw
    print(f"  College LEAF (raw):              r = {corr_raw:+.3f} (p = {pval_raw:.4f}, R² = {corr_raw**2:.3f})")

    # Context-adjusted
    corr_context = df['college_leaf_context'].corr(df['rookie_leaf_mean'])
    pval_context = scipy_stats.pearsonr(df['college_leaf_context'], df['rookie_leaf_mean'])[1]
    correlations['context'] = corr_context
    print(f"  College LEAF (context):          r = {corr_context:+.3f} (p = {pval_context:.4f}, R² = {corr_context**2:.3f})")

    # Competition-adjusted
    corr_comp = df['college_leaf_competition'].corr(df['rookie_leaf_mean'])
    pval_comp = scipy_stats.pearsonr(df['college_leaf_competition'], df['rookie_leaf_mean'])[1]
    correlations['competition'] = corr_comp
    print(f"  College LEAF (competition):      r = {corr_comp:+.3f} (p = {pval_comp:.4f}, R² = {corr_comp**2:.3f})")

    # Full adjustment
    corr_full = df['college_leaf_full'].corr(df['rookie_leaf_mean'])
    pval_full = scipy_stats.pearsonr(df['college_leaf_full'], df['rookie_leaf_mean'])[1]
    correlations['full'] = corr_full
    print(f"  College LEAF (FULL):             r = {corr_full:+.3f} (p = {pval_full:.4f}, R² = {corr_full**2:.3f})")

    # Draft pick (baseline)
    corr_draft = df['draft_pick'].corr(df['rookie_leaf_mean'])
    correlations['draft_pick'] = corr_draft
    print(f"  Draft Pick (baseline):           r = {corr_draft:+.3f} (R² = {corr_draft**2:.3f})")

    # Improvement over baseline
    print(f"\n--- IMPROVEMENT OVER BASELINE ---")

    for name, corr in [('Raw', corr_raw), ('Context', corr_context),
                       ('Competition', corr_comp), ('FULL', corr_full)]:
        improvement = abs(corr) - abs(corr_draft)
        pct_improvement = (improvement / abs(corr_draft)) * 100 if corr_draft != 0 else 0
        status = "[OK]" if improvement > 0 else "[--]"
        print(f"  {name:15s}: {improvement:+.3f} ({pct_improvement:+.1f}%) {status}")

    # R-squared comparison
    print(f"\n--- VARIANCE EXPLAINED (R-squared) ---")
    print(f"  Draft Pick:                  {corr_draft**2:.3f} ({corr_draft**2*100:.1f}%)")
    print(f"  College LEAF (raw):          {corr_raw**2:.3f} ({corr_raw**2*100:.1f}%)")
    print(f"  College LEAF (context):      {corr_context**2:.3f} ({corr_context**2*100:.1f}%)")
    print(f"  College LEAF (competition):  {corr_comp**2:.3f} ({corr_comp**2*100:.1f}%)")
    print(f"  College LEAF (FULL):         {corr_full**2:.3f} ({corr_full**2*100:.1f}%)")

    # Top performers
    print(f"\n--- TOP 10 COLLEGE LEAF (FULL) PERFORMERS ---")
    top_10 = df.nlargest(10, 'college_leaf_full')[
        ['player_name', 'college_leaf_full', 'rookie_leaf_mean', 'draft_pick']
    ]
    print(top_10.to_string(index=False))

    return correlations


def save_results(df):
    """Save College LEAF results"""
    root = get_project_root()
    output_path = root / 'data' / 'processed' / 'college_leaf_ratings_full.csv'

    df.to_csv(output_path, index=False)
    print(f"\n[OK] Saved College LEAF ratings to: {output_path}")


def main():
    """Main execution"""
    print("="*80)
    print("COLLEGE LEAF CALCULATOR (FULL IMPLEMENTATION)")
    print("="*80)
    print("\nAdapting NFL LEAF methodology to college football")
    print("With context + competition adjustments")

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

    best_model = max(correlations.items(), key=lambda x: abs(x[1]) if x[0] != 'draft_pick' else -999)
    baseline_r = correlations['draft_pick']

    print(f"\nKEY RESULTS:")
    print(f"  Sample Size:                     {len(df_final)} QBs")
    print(f"  Baseline (Draft Pick):           r = {baseline_r:+.3f} (R² = {baseline_r**2:.3f})")
    print(f"  Best Model ({best_model[0]}):     r = {best_model[1]:+.3f} (R² = {best_model[1]**2:.3f})")

    improvement = abs(best_model[1]) - abs(baseline_r)
    pct_improvement = (improvement / abs(baseline_r)) * 100 if baseline_r != 0 else 0

    if improvement > 0:
        print(f"\n[SUCCESS] Best model beats baseline by {improvement:+.3f} ({pct_improvement:+.1f}%)")
    else:
        print(f"\n[NOTE] Models do not beat baseline (limited opponent data)")

    print(f"\nNote: Competition adjustment limited by available opponent SP+ data")
    print(f"      Full implementation would require opponent ratings for all games")


if __name__ == "__main__":
    main()
