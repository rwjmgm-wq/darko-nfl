"""
College QB Career Arc Predictor

Generates probabilistic career LEAF projections based on college performance.
Outputs year-by-year projections with uncertainty similar to NFL LEAF visualizer.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
import json


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent


def train_outcome_models():
    """
    Train models for each NFL outcome using best features
    Returns dict of trained models
    """
    root = get_project_root()

    # Load college stats (using career_average aggregation - best for most outcomes)
    stats_path = root / 'data' / 'processed' / 'aggregated_stats' / 'stats_career_average.csv'
    df_college = pd.read_csv(stats_path)

    # Load NFL outcomes
    outcomes_path = root / 'data' / 'processed' / 'nfl_outcomes_comprehensive.csv'
    df_nfl = pd.read_csv(outcomes_path)

    # Merge
    df = pd.merge(df_college, df_nfl, on=['player_name', 'draft_year'], how='inner')

    # Define features (5-feature ensemble - best performer)
    features = ['epa_per_play', 'attempts', 'big_play_rate', 'high_leverage_epa', 'success_rate']

    models = {}
    scalers = {}

    # Train each outcome model
    print("Training outcome models...")

    # 1. 85+ Starts (career length)
    mask = df[features + ['reached_85_starts']].notna().all(axis=1)
    X = df[mask][features]
    y = df[mask]['reached_85_starts']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_scaled, y)

    models['career_length'] = model
    scalers['career_length'] = scaler
    print(f"  [OK] Career length model trained (n={len(X)})")

    # 2. Elite Peak
    mask = df[features + ['is_elite']].notna().all(axis=1)
    X = df[mask][features]
    y = df[mask]['is_elite']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_scaled, y)

    models['elite_peak'] = model
    scalers['elite_peak'] = scaler
    print(f"  [OK] Elite peak model trained (n={len(X)})")

    # 3. Sustained Starter
    mask = df[features + ['is_sustained_starter']].notna().all(axis=1)
    X = df[mask][features]
    y = df[mask]['is_sustained_starter']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_scaled, y)

    models['sustained_starter'] = model
    scalers['sustained_starter'] = scaler
    print(f"  [OK] Sustained starter model trained (n={len(X)})")

    # 4. Bust
    mask = df[features + ['is_bust']].notna().all(axis=1)
    X = df[mask][features]
    y = df[mask]['is_bust']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_scaled, y)

    models['bust'] = model
    scalers['bust'] = scaler
    print(f"  [OK] Bust model trained (n={len(X)})")

    # 5. Rookie LEAF (regression)
    mask = df[features + ['rookie_leaf']].notna().all(axis=1)
    X = df[mask][features]
    y = df[mask]['rookie_leaf']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    models['rookie_leaf'] = model
    scalers['rookie_leaf'] = scaler
    print(f"  [OK] Rookie LEAF model trained (n={len(X)})")

    return models, scalers, features


def predict_outcome_probabilities(college_stats, models, scalers, features):
    """
    Predict probability of each outcome given college stats

    Returns dict with probabilities
    """
    # Extract feature values
    X = np.array([[college_stats[f] for f in features]])

    predictions = {}

    # Career length probability
    X_scaled = scalers['career_length'].transform(X)
    proba = models['career_length'].predict_proba(X_scaled)[0]
    predictions['long_career_prob'] = proba[1]  # Probability of 85+ starts

    # Elite peak probability
    X_scaled = scalers['elite_peak'].transform(X)
    proba = models['elite_peak'].predict_proba(X_scaled)[0]
    predictions['elite_prob'] = proba[1]

    # Sustained starter probability
    X_scaled = scalers['sustained_starter'].transform(X)
    proba = models['sustained_starter'].predict_proba(X_scaled)[0]
    predictions['sustained_prob'] = proba[1]

    # Bust probability
    X_scaled = scalers['bust'].transform(X)
    proba = models['bust'].predict_proba(X_scaled)[0]
    predictions['bust_prob'] = proba[1]

    # Rookie LEAF prediction
    X_scaled = scalers['rookie_leaf'].transform(X)
    predictions['rookie_leaf_est'] = models['rookie_leaf'].predict(X_scaled)[0]

    return predictions


def generate_career_trajectory(predictions, scenario='realistic', career_years=10):
    """
    Generate year-by-year LEAF trajectory based on outcome probabilities

    scenario: 'pessimistic' (10th percentile), 'realistic' (50th), 'optimistic' (90th)

    Returns list of (year, LEAF_estimate, confidence_lower, confidence_upper)
    """
    trajectory = []

    # Determine career archetype based on probabilities
    if scenario == 'pessimistic':
        # Use lower bounds
        elite_factor = max(0, predictions['elite_prob'] - 0.3)
        bust_factor = min(1, predictions['bust_prob'] + 0.3)
        career_length = 3 if predictions['long_career_prob'] < 0.5 else 6
        rookie_leaf = predictions['rookie_leaf_est'] - 0.15

    elif scenario == 'optimistic':
        # Use upper bounds
        elite_factor = min(1, predictions['elite_prob'] + 0.3)
        bust_factor = max(0, predictions['bust_prob'] - 0.3)
        career_length = 10 if predictions['long_career_prob'] > 0.3 else 8
        rookie_leaf = predictions['rookie_leaf_est'] + 0.15

    else:  # realistic
        elite_factor = predictions['elite_prob']
        bust_factor = predictions['bust_prob']
        career_length = 8 if predictions['long_career_prob'] > 0.5 else 4
        rookie_leaf = predictions['rookie_leaf_est']

    # Generate year-by-year trajectory
    for year in range(1, min(career_years + 1, career_length + 1)):

        if year == 1:
            # Rookie year - use rookie LEAF prediction with high uncertainty
            leaf = rookie_leaf
            uncertainty = 0.20  # High uncertainty for rookies

        elif year <= 3:
            # Early career - establishing baseline
            # Typical pattern: slight improvement or decline from rookie year
            base_leaf = rookie_leaf + (0.05 * (year - 1))  # Slight growth

            # Apply bust factor (high bust risk means lower trajectory)
            leaf = base_leaf - (bust_factor * 0.15)
            uncertainty = 0.15

        elif year <= 6:
            # Prime years - potential for peak performance
            # Elite QBs peak here, others plateau or decline

            if elite_factor > 0.5:
                # Elite trajectory - peak in years 4-6
                peak_bonus = elite_factor * 0.25 * np.sin((year - 2) * np.pi / 6)
            else:
                peak_bonus = 0

            base_leaf = rookie_leaf + 0.10 + peak_bonus
            leaf = base_leaf - (bust_factor * 0.10)
            uncertainty = 0.12

        else:
            # Late career - gradual decline
            # Sustained starters maintain performance, others decline faster

            sustained_factor = predictions['sustained_prob']
            decline_rate = 0.02 if sustained_factor > 0.6 else 0.05

            years_past_prime = year - 6
            decline = decline_rate * years_past_prime

            base_leaf = rookie_leaf + 0.10 - decline
            leaf = max(-0.2, base_leaf - (bust_factor * 0.10))
            uncertainty = 0.15

        # Apply realistic bounds (NFL LEAF typically ranges from -0.3 to +0.4)
        leaf = np.clip(leaf, -0.25, 0.35)

        # Calculate confidence intervals
        conf_lower = leaf - uncertainty
        conf_upper = leaf + uncertainty

        trajectory.append({
            'year': year,
            'leaf_estimate': round(leaf, 3),
            'conf_lower': round(conf_lower, 3),
            'conf_upper': round(conf_upper, 3)
        })

    return trajectory


def find_historical_comps(college_stats, top_n=5):
    """
    Find historical QBs with similar college profiles

    Returns list of similar QBs with their actual NFL outcomes
    """
    root = get_project_root()

    # Load college stats
    stats_path = root / 'data' / 'processed' / 'aggregated_stats' / 'stats_career_average.csv'
    df_college = pd.read_csv(stats_path)

    # Load NFL outcomes
    outcomes_path = root / 'data' / 'processed' / 'nfl_outcomes_comprehensive.csv'
    df_nfl = pd.read_csv(outcomes_path)

    df = pd.merge(df_college, df_nfl, on=['player_name', 'draft_year'], how='inner')

    # Calculate similarity scores
    features = ['epa_per_play', 'attempts', 'big_play_rate', 'success_rate']

    # Normalize features
    df_features = df[features].fillna(df[features].mean())
    target_features = np.array([college_stats.get(f, 0) for f in features])

    # Calculate Euclidean distance
    distances = np.sqrt(((df_features - target_features) ** 2).sum(axis=1))
    df['similarity'] = 1 / (1 + distances)  # Convert distance to similarity

    # Get top N similar QBs
    similar_qbs = df.nlargest(top_n, 'similarity')

    comps = []
    for _, row in similar_qbs.iterrows():
        # Determine outcome category
        if row.get('elite_peak', 0) == 1:
            outcome = 'Elite career'
        elif row.get('sustained_starter', 0) == 1:
            outcome = 'Quality starter'
        elif row.get('starts_85_plus', 0) == 1:
            outcome = 'Long career'
        elif row.get('bust', 0) == 1:
            outcome = 'Bust'
        else:
            outcome = 'Backup career'

        comps.append({
            'name': row['player_name'],
            'draft_year': int(row['draft_year']),
            'similarity': round(row['similarity'] * 100, 1),
            'outcome': outcome,
            'rookie_leaf': round(row.get('rookie_leaf_mean', 0), 3) if pd.notna(row.get('rookie_leaf_mean')) else None
        })

    return comps


def predict_career_arc(player_name, draft_year):
    """
    Main function: Predict career arc for a college QB

    Returns complete projection data structure
    """
    print("="*80)
    print(f"CAREER ARC PROJECTION: {player_name} ({draft_year})")
    print("="*80)

    root = get_project_root()

    # Load college stats
    print("\n[1] Loading college stats...")
    stats_path = root / 'data' / 'processed' / 'aggregated_stats' / 'stats_career_average.csv'
    df_college = pd.read_csv(stats_path)

    qb_stats = df_college[(df_college['player_name'] == player_name) &
                          (df_college['draft_year'] == draft_year)]

    if len(qb_stats) == 0:
        print(f"    [ERROR] {player_name} not found in dataset")
        return None

    college_stats = qb_stats.iloc[0].to_dict()
    print(f"    [OK] College stats loaded for {player_name}")
    print(f"         College: {college_stats.get('college', 'Unknown')}")
    print(f"         EPA/play: {college_stats.get('epa_per_play', 0):.3f}")
    print(f"         Attempts: {college_stats.get('attempts', 0):.0f}")

    # Train models
    print("\n[2] Training prediction models...")
    models, scalers, features = train_outcome_models()

    # Generate predictions
    print("\n[3] Generating outcome probabilities...")
    predictions = predict_outcome_probabilities(college_stats, models, scalers, features)

    print(f"\n    Outcome Probabilities:")
    print(f"      Long career (85+ starts): {predictions['long_career_prob']:.1%}")
    print(f"      Elite peak:               {predictions['elite_prob']:.1%}")
    print(f"      Sustained starter:        {predictions['sustained_prob']:.1%}")
    print(f"      Bust risk:                {predictions['bust_prob']:.1%}")
    print(f"      Rookie LEAF estimate:     {predictions['rookie_leaf_est']:+.3f}")

    # Generate career trajectories
    print("\n[4] Generating career trajectories...")
    trajectories = {
        'pessimistic': generate_career_trajectory(predictions, 'pessimistic'),
        'realistic': generate_career_trajectory(predictions, 'realistic'),
        'optimistic': generate_career_trajectory(predictions, 'optimistic')
    }

    print(f"    [OK] Generated 3 trajectory scenarios")

    # Find historical comps
    print("\n[5] Finding historical comparables...")
    comps = find_historical_comps(college_stats, top_n=5)

    print(f"    [OK] Most similar historical QBs:")
    for comp in comps:
        print(f"         {comp['name']:25s} ({comp['draft_year']}) - {comp['similarity']:.0f}% match - {comp['outcome']}")

    # Compile final projection
    projection = {
        'player_name': player_name,
        'draft_year': draft_year,
        'college': college_stats.get('college', 'Unknown'),
        'probabilities': {
            'long_career': round(predictions['long_career_prob'], 3),
            'elite_peak': round(predictions['elite_prob'], 3),
            'sustained_starter': round(predictions['sustained_prob'], 3),
            'bust': round(predictions['bust_prob'], 3)
        },
        'rookie_leaf_estimate': round(predictions['rookie_leaf_est'], 3),
        'trajectories': trajectories,
        'historical_comps': comps,
        'confidence_level': 'MODERATE',
        'model_notes': 'Based on career average college stats. Long-term outcomes more reliable than year-by-year predictions.'
    }

    return projection


def save_projection(projection, output_format='json'):
    """Save projection to file"""
    root = get_project_root()
    output_dir = root / 'data' / 'projections'
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{projection['player_name'].replace(' ', '_')}_{projection['draft_year']}_projection.{output_format}"
    output_path = output_dir / filename

    if output_format == 'json':
        with open(output_path, 'w') as f:
            json.dump(projection, f, indent=2)

    print(f"\n[6] Saving projection...")
    print(f"    [OK] Saved to: {output_path}")

    return output_path


def visualize_projection(projection):
    """Print text visualization of projection"""
    print("\n" + "="*80)
    print("CAREER PROJECTION SUMMARY")
    print("="*80)

    print(f"\nPlayer: {projection['player_name']} ({projection['draft_year']})")
    print(f"College: {projection['college']}")

    print(f"\nCareer Outlook:")
    probs = projection['probabilities']
    print(f"  Long Career (85+ starts):  {probs['long_career']*100:5.1f}%  {'#' * int(probs['long_career']*50)}")
    print(f"  Elite Peak Performance:    {probs['elite_peak']*100:5.1f}%  {'#' * int(probs['elite_peak']*50)}")
    print(f"  Sustained Starter:         {probs['sustained_starter']*100:5.1f}%  {'#' * int(probs['sustained_starter']*50)}")
    print(f"  Bust Risk:                 {probs['bust']*100:5.1f}%  {'#' * int(probs['bust']*50)}")

    print(f"\nProjected Career Trajectories:")
    print(f"{'Year':<6} {'Pessimistic':<15} {'Realistic':<15} {'Optimistic':<15}")
    print("-" * 60)

    # Get max length
    max_years = max(len(projection['trajectories'][s]) for s in ['pessimistic', 'realistic', 'optimistic'])

    for year in range(max_years):
        year_num = year + 1
        pess = projection['trajectories']['pessimistic'][year] if year < len(projection['trajectories']['pessimistic']) else None
        real = projection['trajectories']['realistic'][year] if year < len(projection['trajectories']['realistic']) else None
        opt = projection['trajectories']['optimistic'][year] if year < len(projection['trajectories']['optimistic']) else None

        pess_str = f"{pess['leaf_estimate']:+.3f}" if pess else "---"
        real_str = f"{real['leaf_estimate']:+.3f}" if real else "---"
        opt_str = f"{opt['leaf_estimate']:+.3f}" if opt else "---"

        print(f"{year_num:<6} {pess_str:<15} {real_str:<15} {opt_str:<15}")

    print(f"\nHistorical Comparisons:")
    for i, comp in enumerate(projection['historical_comps'][:5], 1):
        print(f"  {i}. {comp['name']:25s} ({comp['similarity']}% match) - {comp['outcome']}")


def main():
    """Example usage"""
    import sys

    if len(sys.argv) < 3:
        print("Usage: python predict_career_arc.py 'Player Name' draft_year")
        print("\nExample: python predict_career_arc.py 'Patrick Mahomes' 2017")
        print("\nOr run interactively...")

        # Interactive mode
        player_name = input("\nEnter player name: ")
        draft_year = int(input("Enter draft year: "))
    else:
        player_name = sys.argv[1]
        draft_year = int(sys.argv[2])

    # Generate projection
    projection = predict_career_arc(player_name, draft_year)

    if projection:
        # Visualize
        visualize_projection(projection)

        # Save
        save_projection(projection, output_format='json')

        print("\n" + "="*80)
        print("Use this JSON file with your NFL LEAF visualizer!")
        print("="*80)


if __name__ == "__main__":
    main()
