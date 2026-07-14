"""
DEPRECATED - do not use. Superseded by v2_build_projections.py.

Problems: trained a 5-class multinomial (~40 parameters) on ~90 QBs with no
out-of-sample check (honest CV shows its probabilities are WORSE than quoting
base rates); consumed the no-op "SP+ adjusted" stats from
apply_sp_plus_to_careers.py; ignored right-censoring (recent draft classes
counted as busts); and generated career trajectories from hardcoded numbers.

---
Build Projection Database for All QBs

Generates projections for all 120 QBs in the dataset and saves them
to a single JSON file for the web application.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent


def create_mutually_exclusive_outcomes(df):
    """Create mutually exclusive career outcome categories

    Hierarchy (from best to worst):
    1. Elite: Top-tier QB with sustained excellence AND strong rookie performance
    2. Solid Starter: Quality starter with longevity (85+ starts OR sustained 30+ game streak)
    3. Journeyman: Moderate playing time but limited success (16-84 starts, never elite)
    4. Backup: Long career but minimal starting (reached 85 starts as backup)
    5. Bust: Failed to establish career (≤16 starts OR never exceeded 40th percentile)

    Note: Elite is intentionally very restrictive (only ~5% of QBs) to reflect reality
    """
    outcomes = []

    for _, row in df.iterrows():
        is_elite = row.get('is_elite', 0)
        is_sustained = row.get('is_sustained_starter', 0)
        reached_85 = row.get('reached_85_starts', 0)
        is_bust = row.get('is_bust', 0)
        total_starts = row.get('total_starts', 0)
        rookie_leaf = row.get('rookie_leaf', np.nan)

        if pd.isna(is_elite) or pd.isna(is_sustained) or pd.isna(is_bust):
            outcomes.append(None)
            continue

        # Elite: VERY restrictive - achieved elite peak, long career, AND strong rookie showing
        # This captures only the truly elite QBs (Mahomes, Wilson, Ryan, etc.)
        if is_elite == 1 and total_starts >= 85:
            if not pd.isna(rookie_leaf) and rookie_leaf > 0.05:
                outcomes.append('Elite')
            elif is_sustained == 1:
                outcomes.append('Solid Starter')
            else:
                outcomes.append('Journeyman')
        # Solid Starter: Long career or sustained starting streak
        elif is_sustained == 1 or reached_85 == 1:
            outcomes.append('Solid Starter')
        # Journeyman: Moderate playing time (16-84 starts)
        elif total_starts >= 16:
            outcomes.append('Journeyman')
        # Bust: Failed to establish career
        elif is_bust == 1 or total_starts < 16:
            outcomes.append('Bust')
        else:
            outcomes.append('Bust')

    return outcomes


def train_models():
    """Train all models once"""
    root = get_project_root()

    stats_path = root / 'data' / 'processed' / 'aggregated_stats' / 'stats_career_average_sp_adjusted.csv'
    df_college = pd.read_csv(stats_path)

    outcomes_path = root / 'data' / 'processed' / 'nfl_outcomes_comprehensive.csv'
    df_nfl = pd.read_csv(outcomes_path)

    df = pd.merge(df_college, df_nfl, on=['player_name', 'draft_year'], how='inner')
    df['career_outcome'] = create_mutually_exclusive_outcomes(df)

    features = ['epa_per_play_adj', 'attempts', 'big_play_rate', 'high_leverage_epa', 'success_rate_raw']

    # Train multiclass model
    mask = df[features + ['career_outcome']].notna().all(axis=1)
    X = df[mask][features]
    y = df[mask]['career_outcome']

    scaler_outcome = StandardScaler()
    X_scaled = scaler_outcome.fit_transform(X)

    model_outcome = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs')
    model_outcome.fit(X_scaled, y)

    # Train rookie LEAF model
    mask = df[features + ['rookie_leaf']].notna().all(axis=1)
    X = df[mask][features]
    y = df[mask]['rookie_leaf']

    scaler_rookie = StandardScaler()
    X_scaled = scaler_rookie.fit_transform(X)

    model_rookie = LinearRegression()
    model_rookie.fit(X_scaled, y)

    return {
        'outcome_model': model_outcome,
        'outcome_scaler': scaler_outcome,
        'outcome_classes': model_outcome.classes_,
        'rookie_model': model_rookie,
        'rookie_scaler': scaler_rookie,
        'features': features
    }


def predict_for_player(player_stats, models):
    """Generate prediction for a single player"""
    features = models['features']

    # Check for missing features
    for f in features:
        if pd.isna(player_stats[f]):
            return None, None

    # Extract features
    X = np.array([[player_stats[f] for f in features]])

    # Predict outcome probabilities
    X_scaled = models['outcome_scaler'].transform(X)
    proba = models['outcome_model'].predict_proba(X_scaled)[0]

    outcome_probs = {}
    for i, outcome in enumerate(models['outcome_classes']):
        outcome_probs[outcome] = round(float(proba[i]), 3)

    # Predict rookie LEAF
    X_scaled_rookie = models['rookie_scaler'].transform(X)
    rookie_leaf = models['rookie_model'].predict(X_scaled_rookie)[0]

    return outcome_probs, float(rookie_leaf)


def generate_trajectory(outcome_probs, rookie_leaf, scenario='realistic'):
    """Generate career trajectory"""
    career_lengths = {
        'Elite': (10, 14, 8),
        'Solid Starter': (7, 10, 4),
        'Backup': (5, 8, 3),
        'Journeyman': (4, 6, 2),
        'Bust': (2, 3, 1)
    }

    peak_leafs = {
        'Elite': (0.25, 0.35, 0.15),
        'Solid Starter': (0.10, 0.20, 0.00),
        'Backup': (-0.05, 0.05, -0.15),
        'Journeyman': (-0.15, -0.05, -0.25),
        'Bust': (-0.25, -0.15, -0.35)
    }

    idx = 0 if scenario == 'realistic' else (1 if scenario == 'optimistic' else 2)

    career_length = 0
    peak_leaf = 0

    for outcome, prob in outcome_probs.items():
        career_length += career_lengths.get(outcome, (3, 5, 2))[idx] * prob
        peak_leaf += peak_leafs.get(outcome, (0, 0.1, -0.1))[idx] * prob

    career_length = int(round(career_length))
    career_length = max(1, min(career_length, 14))

    trajectory = []
    for year in range(1, career_length + 1):
        year_pct = year / career_length

        if year_pct < 0.3:
            progress = year_pct / 0.3
            leaf = rookie_leaf + (peak_leaf - rookie_leaf) * progress * 0.4
        elif year_pct < 0.6:
            progress = (year_pct - 0.3) / 0.3
            leaf = rookie_leaf + (peak_leaf - rookie_leaf) * (0.4 + progress * 0.6)
        else:
            progress = (year_pct - 0.6) / 0.4
            leaf = peak_leaf - (peak_leaf - rookie_leaf) * progress * 0.7

        trajectory.append({
            'year': year,
            'leaf': round(leaf, 3)
        })

    return trajectory


def main():
    """Build projection database for all QBs"""
    print("=" * 80)
    print("BUILDING PROJECTION DATABASE FOR ALL QBs")
    print("=" * 80)

    root = get_project_root()

    # Load college stats
    print("\n[1] Loading college stats...")
    stats_path = root / 'data' / 'processed' / 'aggregated_stats' / 'stats_career_average_sp_adjusted.csv'
    df_college = pd.read_csv(stats_path)
    print(f"    [OK] Loaded {len(df_college)} QBs")

    # Load NFL outcomes (for actual results)
    outcomes_path = root / 'data' / 'processed' / 'nfl_outcomes_comprehensive.csv'
    df_nfl = pd.read_csv(outcomes_path)

    # Merge to get actual outcomes
    df_merged = pd.merge(df_college, df_nfl, on=['player_name', 'draft_year'], how='left')
    df_merged['actual_outcome'] = create_mutually_exclusive_outcomes(df_merged)

    # Train models
    print("\n[2] Training prediction models...")
    models = train_models()
    print("    [OK] Models trained")

    # Generate projections for all QBs
    print("\n[3] Generating projections for all QBs...")
    projections = []

    for idx, row in df_merged.iterrows():
        player_name = row['player_name']
        draft_year = int(row['draft_year'])

        # Get prediction
        outcome_probs, rookie_leaf = predict_for_player(row, models)

        # Skip if missing data
        if outcome_probs is None:
            continue

        # Generate trajectories
        trajectories = {
            'pessimistic': generate_trajectory(outcome_probs, rookie_leaf, 'pessimistic'),
            'realistic': generate_trajectory(outcome_probs, rookie_leaf, 'realistic'),
            'optimistic': generate_trajectory(outcome_probs, rookie_leaf, 'optimistic')
        }

        # Get actual NFL data if available
        actual_rookie_leaf = row.get('rookie_leaf')
        actual_outcome = row.get('actual_outcome')
        total_starts = row.get('total_starts')

        # For recent draft classes (2023+), mark as TBD unless they've already achieved Elite/Solid Starter
        if draft_year >= 2023:
            if pd.isna(actual_outcome) or actual_outcome not in ['Elite', 'Solid Starter']:
                actual_outcome = 'TBD'

        projection = {
            'player_name': player_name,
            'draft_year': draft_year,
            'college': row.get('college', 'Unknown'),
            'college_epa': round(float(row['epa_per_play_adj']), 3) if not pd.isna(row.get('epa_per_play_adj')) else None,
            'college_attempts': int(row['attempts']) if not pd.isna(row['attempts']) else None,
            'projected_outcome_probs': outcome_probs,
            'projected_rookie_leaf': round(rookie_leaf, 3),
            'trajectories': trajectories,
            'actual_outcome': actual_outcome if actual_outcome == 'TBD' or not pd.isna(actual_outcome) else None,
            'actual_rookie_leaf': round(float(actual_rookie_leaf), 3) if not pd.isna(actual_rookie_leaf) else None,
            'actual_total_starts': int(total_starts) if not pd.isna(total_starts) else None
        }

        projections.append(projection)

        if (idx + 1) % 10 == 0:
            print(f"    Processed {idx + 1}/{len(df_merged)} QBs...")

    # Save to JSON
    print("\n[4] Saving projection database...")
    output_dir = root / 'data' / 'projections'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / 'all_projections.json'
    with open(output_path, 'w') as f:
        json.dump(projections, f, indent=2)

    print(f"    [OK] Saved {len(projections)} projections to: {output_path}")

    # Print summary stats
    print("\n" + "=" * 80)
    print("PROJECTION DATABASE SUMMARY")
    print("=" * 80)

    print(f"\nTotal QBs: {len(projections)}")
    print(f"Draft years: {df_merged['draft_year'].min()}-{df_merged['draft_year'].max()}")

    with_actuals = sum(1 for p in projections if p['actual_outcome'] is not None)
    print(f"QBs with actual NFL outcomes: {with_actuals}")

    print("\nMost likely projected outcome distribution:")
    outcome_counts = {}
    for p in projections:
        most_likely = max(p['projected_outcome_probs'], key=p['projected_outcome_probs'].get)
        outcome_counts[most_likely] = outcome_counts.get(most_likely, 0) + 1

    for outcome, count in sorted(outcome_counts.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(projections) * 100
        print(f"  {outcome:15s}: {count:3d} ({pct:5.1f}%)")

    print("\n[OK] Database ready for web application!")


if __name__ == "__main__":
    main()
