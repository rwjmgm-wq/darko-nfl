"""
Generate NFL Projections for Active CFB QBs

Processes active QB career data, calculates stats, and generates NFL projections.
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


def calculate_career_stats(career_plays):
    """Calculate aggregated career statistics from play-by-play data"""
    if len(career_plays) == 0:
        return None

    stats = {}

    # Basic counts
    stats['attempts'] = len(career_plays)
    stats['completions'] = career_plays['completion'].sum()
    stats['yards'] = career_plays['yards_gained'].sum()
    stats['tds'] = career_plays['touchdown'].sum()
    stats['ints'] = career_plays['interception'].sum()

    # Advanced stats
    stats['epa_per_play'] = career_plays['epa'].mean()
    stats['success_rate'] = career_plays['success'].mean()

    # Big plays (15+ yards)
    big_plays = career_plays[career_plays['yards_gained'] >= 15]
    stats['big_play_rate'] = len(big_plays) / len(career_plays) if len(career_plays) > 0 else 0

    # High leverage EPA (3rd/4th down)
    high_leverage = career_plays[career_plays['down'].isin([3, 4])]
    stats['high_leverage_epa'] = high_leverage['epa'].mean() if len(high_leverage) > 0 else 0

    return stats


def train_models(df_train):
    """Train projection models from historical data"""

    features = ['epa_per_play', 'attempts', 'big_play_rate', 'high_leverage_epa', 'success_rate']

    # Filter to complete cases
    mask = df_train[features + ['career_outcome']].notna().all(axis=1)
    X = df_train[mask][features]
    y = df_train[mask]['career_outcome']

    scaler_outcome = StandardScaler()
    X_scaled = scaler_outcome.fit_transform(X)

    model_outcome = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs')
    model_outcome.fit(X_scaled, y)

    # Train rookie LEAF model
    mask = df_train[features + ['rookie_leaf']].notna().all(axis=1)
    X = df_train[mask][features]
    y = df_train[mask]['rookie_leaf']

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


def create_mutually_exclusive_outcomes(df):
    """Create mutually exclusive career outcome categories for training data"""
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

        # Elite: VERY restrictive
        if is_elite == 1 and total_starts >= 85:
            if not pd.isna(rookie_leaf) and rookie_leaf > 0.05:
                outcomes.append('Elite')
            elif is_sustained == 1:
                outcomes.append('Solid Starter')
            else:
                outcomes.append('Journeyman')
        # Solid Starter
        elif is_sustained == 1 or reached_85 == 1:
            outcomes.append('Solid Starter')
        # Journeyman
        elif total_starts >= 16:
            outcomes.append('Journeyman')
        # Bust
        elif is_bust == 1 or total_starts < 16:
            outcomes.append('Bust')
        else:
            outcomes.append('Bust')

    return outcomes


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
    """Generate projections for all active CFB QBs"""
    print("=" * 80)
    print("PROJECTING ACTIVE CFB QBs TO NFL")
    print("=" * 80)

    root = get_project_root()

    # Load training data
    print("\n[1] Loading training data...")
    stats_path = root / 'data' / 'processed' / 'aggregated_stats' / 'stats_career_average.csv'
    df_college = pd.read_csv(stats_path)

    outcomes_path = root / 'data' / 'processed' / 'nfl_outcomes_comprehensive.csv'
    df_nfl = pd.read_csv(outcomes_path)

    df_train = pd.merge(df_college, df_nfl, on=['player_name', 'draft_year'], how='inner')
    df_train['career_outcome'] = create_mutually_exclusive_outcomes(df_train)

    print(f"    [OK] Loaded {len(df_train)} historical QBs for training")

    # Train models
    print("\n[2] Training projection models...")
    models = train_models(df_train)
    print("    [OK] Models trained")

    # Load active QB career data
    print("\n[3] Loading active QB careers...")
    careers_dir = root / 'data' / 'processed' / 'active_qb_careers'

    if not careers_dir.exists():
        print("    [ERROR] No active QB career data found")
        print("    Run collect_active_qb_careers.py first")
        return

    career_files = list(careers_dir.glob('*.csv'))
    print(f"    [OK] Found {len(career_files)} active QBs")

    # Generate projections
    print("\n[4] Generating NFL projections...")
    projections = []

    for idx, career_file in enumerate(career_files, 1):
        player_name = career_file.stem.replace('_', ' ')
        print(f"    [{idx}/{len(career_files)}] {player_name}...", end=' ')

        try:
            df_career = pd.read_csv(career_file)

            # Calculate stats
            stats = calculate_career_stats(df_career)

            if stats is None:
                print("[No stats]")
                continue

            # Get prediction
            outcome_probs, rookie_leaf = predict_for_player(stats, models)

            if outcome_probs is None:
                print("[Missing features]")
                continue

            # Generate trajectories
            trajectories = {
                'pessimistic': generate_trajectory(outcome_probs, rookie_leaf, 'pessimistic'),
                'realistic': generate_trajectory(outcome_probs, rookie_leaf, 'realistic'),
                'optimistic': generate_trajectory(outcome_probs, rookie_leaf, 'optimistic')
            }

            projection = {
                'player_name': player_name,
                'draft_year': 'Active 2024',
                'college': df_career['college'].iloc[0],
                'college_epa': round(stats['epa_per_play'], 3),
                'college_attempts': int(stats['attempts']),
                'projected_outcome_probs': outcome_probs,
                'projected_rookie_leaf': round(rookie_leaf, 3),
                'trajectories': trajectories,
                'actual_outcome': None,
                'actual_rookie_leaf': None,
                'actual_total_starts': None
            }

            projections.append(projection)
            print("[OK]")

        except Exception as e:
            print(f"[Error: {str(e)[:40]}]")
            continue

    # Save projections
    print(f"\n[5] Saving {len(projections)} projections...")
    output_dir = root / 'data' / 'projections'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / 'active_cfb_projections.json'
    with open(output_path, 'w') as f:
        json.dump(projections, f, indent=2)

    print(f"    [OK] Saved to: {output_path}")

    # Display top prospects
    print("\n" + "=" * 80)
    print("TOP 10 ACTIVE CFB QB PROSPECTS (by Elite probability)")
    print("=" * 80)

    projections_sorted = sorted(projections, key=lambda x: x['projected_outcome_probs'].get('Elite', 0), reverse=True)

    for idx, p in enumerate(projections_sorted[:10], 1):
        probs = p['projected_outcome_probs']
        print(f"\n{idx}. {p['player_name']} ({p['college']})")
        print(f"   College EPA: {p['college_epa']:+.3f} ({p['college_attempts']} attempts)")
        print(f"   Elite: {probs.get('Elite', 0)*100:.1f}% | Solid Starter: {probs.get('Solid Starter', 0)*100:.1f}% | Bust: {probs.get('Bust', 0)*100:.1f}%")
        print(f"   Projected Rookie LEAF: {p['projected_rookie_leaf']:+.3f}")


if __name__ == "__main__":
    main()
