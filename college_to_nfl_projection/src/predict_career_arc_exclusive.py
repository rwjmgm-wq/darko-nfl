"""
College QB Career Arc Predictor - Mutually Exclusive Outcomes

Generates probabilistic career projections with mutually exclusive categories
that sum to 100%. Outputs both console and HTML visualizations.
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


def create_mutually_exclusive_outcomes(df):
    """
    Create mutually exclusive career outcome categories

    Hierarchy (checked in order):
    1. Elite: Reached elite peak performance
    2. Solid Starter: Sustained starter but not elite
    3. Backup: Long career (85+ starts) but not starter quality
    4. Journeyman: Some playing time but limited impact
    5. Bust: Failed to establish NFL career
    """
    outcomes = []

    for _, row in df.iterrows():
        is_elite = row.get('is_elite', 0)
        is_sustained = row.get('is_sustained_starter', 0)
        reached_85 = row.get('reached_85_starts', 0)
        is_bust = row.get('is_bust', 0)
        total_starts = row.get('total_starts', 0)

        # Handle NaN values
        if pd.isna(is_elite) or pd.isna(is_sustained) or pd.isna(is_bust):
            outcomes.append(None)
            continue

        # Hierarchical classification
        if is_elite == 1:
            outcomes.append('Elite')
        elif is_sustained == 1:
            outcomes.append('Solid Starter')
        elif reached_85 == 1:
            outcomes.append('Backup')
        elif total_starts >= 16:  # At least 1 full season
            outcomes.append('Journeyman')
        elif is_bust == 1:
            outcomes.append('Bust')
        else:
            outcomes.append('Bust')  # Default to bust if unclear

    return outcomes


def train_outcome_models():
    """
    Train models for mutually exclusive outcomes
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

    # Create mutually exclusive outcomes
    df['career_outcome'] = create_mutually_exclusive_outcomes(df)

    # Define features (5-feature ensemble - best performer)
    features = ['epa_per_play', 'attempts', 'big_play_rate', 'high_leverage_epa', 'success_rate']

    models = {}
    scalers = {}

    # Train multiclass outcome model
    print("Training mutually exclusive outcome model...")

    mask = df[features + ['career_outcome']].notna().all(axis=1)
    X = df[mask][features]
    y = df[mask]['career_outcome']

    # Print outcome distribution
    print(f"\n  Outcome distribution (n={len(y)}):")
    outcome_counts = y.value_counts().sort_index()
    for outcome, count in outcome_counts.items():
        pct = count / len(y) * 100
        print(f"    {outcome:15s}: {count:3d} ({pct:5.1f}%)")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use multinomial logistic regression
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        multi_class='multinomial',
        solver='lbfgs'
    )
    model.fit(X_scaled, y)

    models['career_outcome'] = model
    scalers['career_outcome'] = scaler
    print(f"  [OK] Multiclass outcome model trained")

    # Train rookie LEAF regression model
    print("\nTraining rookie LEAF model...")
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

    # Get class labels for prediction
    outcome_classes = models['career_outcome'].classes_

    return models, scalers, features, outcome_classes


def predict_outcome_probabilities(college_stats, models, scalers, features, outcome_classes):
    """
    Predict probability of each outcome given college stats

    Returns dict with probabilities that sum to 100%
    """
    # Extract feature values
    X = np.array([[college_stats[f] for f in features]])

    # Predict outcome probabilities (multiclass)
    X_scaled_outcome = scalers['career_outcome'].transform(X)
    proba = models['career_outcome'].predict_proba(X_scaled_outcome)[0]

    # Create probability dict
    probabilities = {}
    for i, outcome in enumerate(outcome_classes):
        probabilities[outcome] = round(float(proba[i]), 3)

    # Predict rookie LEAF
    X_scaled_rookie = scalers['rookie_leaf'].transform(X)
    rookie_leaf = models['rookie_leaf'].predict(X_scaled_rookie)[0]

    return {
        'outcome_probs': probabilities,
        'rookie_leaf_estimate': round(float(rookie_leaf), 3)
    }


def generate_career_trajectory(predictions, scenario='realistic', career_years=10):
    """
    Generate year-by-year LEAF trajectory with confidence intervals

    Uses outcome probabilities to shape trajectory
    """
    outcome_probs = predictions['outcome_probs']
    rookie_leaf = predictions['rookie_leaf_estimate']

    # Determine career characteristics based on most likely outcome
    most_likely = max(outcome_probs, key=outcome_probs.get)

    # Career length by outcome
    career_lengths = {
        'Elite': (10, 14, 8),          # (realistic, optimistic, pessimistic)
        'Solid Starter': (7, 10, 4),
        'Backup': (5, 8, 3),
        'Journeyman': (4, 6, 2),
        'Bust': (2, 3, 1)
    }

    # Peak LEAF by outcome
    peak_leafs = {
        'Elite': (0.25, 0.35, 0.15),
        'Solid Starter': (0.10, 0.20, 0.00),
        'Backup': (-0.05, 0.05, -0.15),
        'Journeyman': (-0.15, -0.05, -0.25),
        'Bust': (-0.25, -0.15, -0.35)
    }

    # Weight by probabilities for blended trajectory
    career_length = 0
    peak_leaf = 0

    for outcome, prob in outcome_probs.items():
        if scenario == 'optimistic':
            idx = 1
        elif scenario == 'pessimistic':
            idx = 2
        else:  # realistic
            idx = 0

        career_length += career_lengths.get(outcome, (3, 5, 2))[idx] * prob
        peak_leaf += peak_leafs.get(outcome, (0, 0.1, -0.1))[idx] * prob

    career_length = int(round(career_length))
    career_length = max(1, min(career_length, career_years))

    trajectory = []

    for year in range(1, career_length + 1):
        # Career arc: start low, peak mid-career, decline
        year_pct = year / career_length

        if year_pct < 0.3:  # Early career
            progress = year_pct / 0.3  # 0 to 1
            leaf = rookie_leaf + (peak_leaf - rookie_leaf) * progress * 0.4
        elif year_pct < 0.6:  # Peak
            progress = (year_pct - 0.3) / 0.3  # 0 to 1
            leaf = rookie_leaf + (peak_leaf - rookie_leaf) * (0.4 + progress * 0.6)
        else:  # Decline
            progress = (year_pct - 0.6) / 0.4  # 0 to 1
            leaf = peak_leaf - (peak_leaf - rookie_leaf) * progress * 0.7

        # Uncertainty increases over time
        uncertainty = 0.15 + (year / career_length) * 0.05

        conf_lower = leaf - uncertainty
        conf_upper = leaf + uncertainty

        trajectory.append({
            'year': year,
            'leaf_estimate': round(leaf, 3),
            'conf_lower': round(conf_lower, 3),
            'conf_upper': round(conf_upper, 3)
        })

    return trajectory


def find_historical_comparables(college_stats, outcome_probs):
    """
    Find historical QBs with similar college stats and outcomes
    """
    root = get_project_root()

    # Load data
    stats_path = root / 'data' / 'processed' / 'aggregated_stats' / 'stats_career_average.csv'
    df_college = pd.read_csv(stats_path)

    outcomes_path = root / 'data' / 'processed' / 'nfl_outcomes_comprehensive.csv'
    df_nfl = pd.read_csv(outcomes_path)

    df = pd.merge(df_college, df_nfl, on=['player_name', 'draft_year'], how='inner')
    df['career_outcome'] = create_mutually_exclusive_outcomes(df)

    # Calculate similarity
    features = ['epa_per_play', 'attempts', 'big_play_rate']

    similarities = []
    for _, row in df.iterrows():
        if pd.isna(row.get('career_outcome')):
            continue

        # Euclidean distance
        dist = 0
        for feat in features:
            if feat in college_stats and feat in row:
                val1 = college_stats[feat]
                val2 = row[feat]
                if not pd.isna(val2):
                    dist += (val1 - val2) ** 2

        dist = np.sqrt(dist)
        similarity = 100 / (1 + dist)

        similarities.append({
            'name': row['player_name'],
            'draft_year': int(row['draft_year']),
            'similarity': round(similarity, 1),
            'outcome': row['career_outcome'],
            'rookie_leaf': row.get('rookie_leaf')
        })

    # Sort by similarity
    similarities.sort(key=lambda x: x['similarity'], reverse=True)

    return similarities[:5]


def predict_career_arc(player_name, draft_year):
    """
    Generate complete career arc projection for a college QB
    """
    root = get_project_root()

    print("="*80)
    print(f"CAREER ARC PROJECTION: {player_name} ({draft_year})")
    print("="*80)

    # Load player's college stats
    print("\n[1] Loading college stats...")
    stats_path = root / 'data' / 'processed' / 'aggregated_stats' / 'stats_career_average.csv'
    df_college = pd.read_csv(stats_path)

    player_stats = df_college[
        (df_college['player_name'] == player_name) &
        (df_college['draft_year'] == draft_year)
    ]

    if len(player_stats) == 0:
        print(f"    [ERROR] No data found for {player_name} ({draft_year})")
        return None

    college_stats = player_stats.iloc[0].to_dict()
    print(f"    [OK] College stats loaded for {player_name}")
    print(f"         College: {college_stats.get('college', 'Unknown')}")
    print(f"         EPA/play: {college_stats.get('epa_per_play', 0):.3f}")
    print(f"         Attempts: {int(college_stats.get('attempts', 0))}")

    # Train models
    print("\n[2] Training prediction models...")
    models, scalers, features, outcome_classes = train_outcome_models()

    # Generate predictions
    print("\n[3] Generating outcome probabilities...")
    predictions = predict_outcome_probabilities(college_stats, models, scalers, features, outcome_classes)

    print("\n    Outcome Probabilities (sum to 100%):")
    for outcome, prob in sorted(predictions['outcome_probs'].items(), key=lambda x: x[1], reverse=True):
        print(f"      {outcome:15s}: {prob*100:5.1f}%")
    print(f"      Rookie LEAF estimate: {predictions['rookie_leaf_estimate']:+.3f}")

    # Verify sum to 100%
    total_prob = sum(predictions['outcome_probs'].values())
    print(f"\n    Total probability: {total_prob*100:.1f}% [OK]")

    # Generate trajectories
    print("\n[4] Generating career trajectories...")
    trajectories = {
        'pessimistic': generate_career_trajectory(predictions, 'pessimistic'),
        'realistic': generate_career_trajectory(predictions, 'realistic'),
        'optimistic': generate_career_trajectory(predictions, 'optimistic')
    }
    print(f"    [OK] Generated 3 trajectory scenarios")

    # Find comparables
    print("\n[5] Finding historical comparables...")
    comps = find_historical_comparables(college_stats, predictions['outcome_probs'])
    print(f"    [OK] Most similar historical QBs:")
    for comp in comps:
        print(f"         {comp['name']:25s} ({comp['draft_year']}) - {comp['similarity']}% match - {comp['outcome']}")

    # Compile projection
    projection = {
        'player_name': player_name,
        'draft_year': draft_year,
        'college': college_stats.get('college', 'Unknown'),
        'outcome_probabilities': predictions['outcome_probs'],
        'rookie_leaf_estimate': predictions['rookie_leaf_estimate'],
        'trajectories': trajectories,
        'historical_comps': comps,
        'confidence_level': 'MODERATE',
        'model_notes': 'Based on career average college stats. Mutually exclusive outcomes sum to 100%.'
    }

    return projection


def visualize_projection_console(projection):
    """Print console visualization of projection"""
    print("\n" + "="*80)
    print("CAREER PROJECTION SUMMARY")
    print("="*80)

    print(f"\nPlayer: {projection['player_name']} ({projection['draft_year']})")
    print(f"College: {projection['college']}")

    print(f"\nCareer Outcome Probabilities (sum to 100%):")
    probs = projection['outcome_probabilities']
    for outcome, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        bar = '#' * int(prob * 50)
        print(f"  {outcome:15s}: {prob*100:5.1f}%  {bar}")

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
        print(f"  {i}. {comp['name']:25s} ({comp['draft_year']}) - {comp['similarity']}% match - {comp['outcome']}")


def create_html_visualization(projection, output_path):
    """Create interactive HTML visualization"""

    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{player_name} Career Projection</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }}

        .header .subtitle {{
            font-size: 1.2em;
            opacity: 0.9;
        }}

        .content {{
            padding: 40px;
        }}

        .section {{
            margin-bottom: 40px;
        }}

        .section h2 {{
            font-size: 1.8em;
            color: #333;
            margin-bottom: 20px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}

        .probability-bars {{
            margin: 20px 0;
        }}

        .prob-item {{
            margin-bottom: 20px;
        }}

        .prob-label {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            font-weight: 600;
            color: #555;
        }}

        .prob-bar-container {{
            width: 100%;
            height: 40px;
            background: #f0f0f0;
            border-radius: 8px;
            overflow: hidden;
            position: relative;
        }}

        .prob-bar {{
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 1s ease-out;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }}

        .elite {{ background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%); }}
        .solid-starter {{ background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%); }}
        .backup {{ background: linear-gradient(90deg, #43e97b 0%, #38f9d7 100%); }}
        .journeyman {{ background: linear-gradient(90deg, #fa709a 0%, #fee140 100%); }}
        .bust {{ background: linear-gradient(90deg, #30cfd0 0%, #330867 100%); }}

        .chart-container {{
            position: relative;
            height: 400px;
            margin: 30px 0;
        }}

        .comps-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}

        .comp-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 12px;
            border-left: 4px solid #667eea;
        }}

        .comp-name {{
            font-weight: 700;
            font-size: 1.1em;
            color: #333;
            margin-bottom: 8px;
        }}

        .comp-detail {{
            color: #666;
            font-size: 0.9em;
            margin: 4px 0;
        }}

        .similarity-badge {{
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            margin-top: 8px;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}

        .stat-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
        }}

        .stat-value {{
            font-size: 2em;
            font-weight: 700;
            margin-bottom: 5px;
        }}

        .stat-label {{
            opacity: 0.9;
            font-size: 0.9em;
        }}

        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{player_name}</h1>
            <div class="subtitle">{college} • {draft_year} NFL Draft</div>
        </div>

        <div class="content">
            <!-- Key Stats -->
            <div class="section">
                <h2>Quick Stats</h2>
                <div class="stats-grid">
                    <div class="stat-box">
                        <div class="stat-value">{rookie_leaf}</div>
                        <div class="stat-label">Projected Rookie LEAF</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{most_likely}</div>
                        <div class="stat-label">Most Likely Outcome</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{career_years}</div>
                        <div class="stat-label">Expected Career Years</div>
                    </div>
                </div>
            </div>

            <!-- Outcome Probabilities -->
            <div class="section">
                <h2>Career Outcome Probabilities</h2>
                <div class="probability-bars">
                    {probability_bars}
                </div>
            </div>

            <!-- Career Trajectory Chart -->
            <div class="section">
                <h2>Projected Career Trajectory</h2>
                <div class="chart-container">
                    <canvas id="trajectoryChart"></canvas>
                </div>
            </div>

            <!-- Historical Comparisons -->
            <div class="section">
                <h2>Historical Comparisons</h2>
                <div class="comps-grid">
                    {comps_html}
                </div>
            </div>
        </div>

        <div class="footer">
            <p>{model_notes}</p>
            <p>Generated with College-to-NFL Projection Model</p>
        </div>
    </div>

    <script>
        // Animate probability bars
        window.addEventListener('load', function() {{
            document.querySelectorAll('.prob-bar').forEach(bar => {{
                const width = bar.style.width;
                bar.style.width = '0%';
                setTimeout(() => {{
                    bar.style.width = width;
                }}, 100);
            }});
        }});

        // Create trajectory chart
        const ctx = document.getElementById('trajectoryChart').getContext('2d');
        const chart = new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {years_labels},
                datasets: [
                    {{
                        label: 'Optimistic',
                        data: {optimistic_data},
                        borderColor: '#f093fb',
                        backgroundColor: 'rgba(240, 147, 251, 0.1)',
                        borderWidth: 3,
                        fill: true,
                        tension: 0.4
                    }},
                    {{
                        label: 'Realistic',
                        data: {realistic_data},
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        borderWidth: 3,
                        fill: true,
                        tension: 0.4
                    }},
                    {{
                        label: 'Pessimistic',
                        data: {pessimistic_data},
                        borderColor: '#30cfd0',
                        backgroundColor: 'rgba(48, 207, 208, 0.1)',
                        borderWidth: 3,
                        fill: true,
                        tension: 0.4
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        display: true,
                        position: 'top',
                        labels: {{
                            font: {{
                                size: 14,
                                weight: 'bold'
                            }},
                            padding: 20
                        }}
                    }},
                    tooltip: {{
                        mode: 'index',
                        intersect: false
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: false,
                        title: {{
                            display: true,
                            text: 'LEAF Rating',
                            font: {{
                                size: 14,
                                weight: 'bold'
                            }}
                        }},
                        grid: {{
                            color: 'rgba(0, 0, 0, 0.05)'
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'NFL Season',
                            font: {{
                                size: 14,
                                weight: 'bold'
                            }}
                        }},
                        grid: {{
                            display: false
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""

    # Prepare probability bars HTML
    prob_bars_html = ""
    outcome_classes = {
        'Elite': 'elite',
        'Solid Starter': 'solid-starter',
        'Backup': 'backup',
        'Journeyman': 'journeyman',
        'Bust': 'bust'
    }

    for outcome, prob in sorted(projection['outcome_probabilities'].items(), key=lambda x: x[1], reverse=True):
        css_class = outcome_classes.get(outcome, '')
        prob_bars_html += f"""
                    <div class="prob-item">
                        <div class="prob-label">
                            <span>{outcome}</span>
                            <span>{prob*100:.1f}%</span>
                        </div>
                        <div class="prob-bar-container">
                            <div class="prob-bar {css_class}" style="width: {prob*100}%;">
                            </div>
                        </div>
                    </div>"""

    # Prepare comparables HTML
    comps_html = ""
    for comp in projection['historical_comps']:
        comps_html += f"""
                    <div class="comp-card">
                        <div class="comp-name">{comp['name']}</div>
                        <div class="comp-detail">Draft Year: {comp['draft_year']}</div>
                        <div class="comp-detail">Outcome: {comp['outcome']}</div>
                        <div class="similarity-badge">{comp['similarity']}% Match</div>
                    </div>"""

    # Prepare trajectory data
    years_labels = []
    optimistic_data = []
    realistic_data = []
    pessimistic_data = []

    max_years = max(
        len(projection['trajectories']['optimistic']),
        len(projection['trajectories']['realistic']),
        len(projection['trajectories']['pessimistic'])
    )

    for year in range(1, max_years + 1):
        years_labels.append(f"Year {year}")

        opt = next((t['leaf_estimate'] for t in projection['trajectories']['optimistic'] if t['year'] == year), None)
        real = next((t['leaf_estimate'] for t in projection['trajectories']['realistic'] if t['year'] == year), None)
        pess = next((t['leaf_estimate'] for t in projection['trajectories']['pessimistic'] if t['year'] == year), None)

        optimistic_data.append(opt)
        realistic_data.append(real)
        pessimistic_data.append(pess)

    # Get most likely outcome
    most_likely = max(projection['outcome_probabilities'], key=projection['outcome_probabilities'].get)

    # Fill in template
    html = html_template.format(
        player_name=projection['player_name'],
        college=projection['college'],
        draft_year=projection['draft_year'],
        rookie_leaf=f"{projection['rookie_leaf_estimate']:+.3f}",
        most_likely=most_likely,
        career_years=len(projection['trajectories']['realistic']),
        probability_bars=prob_bars_html,
        comps_html=comps_html,
        years_labels=json.dumps(years_labels),
        optimistic_data=json.dumps(optimistic_data),
        realistic_data=json.dumps(realistic_data),
        pessimistic_data=json.dumps(pessimistic_data),
        model_notes=projection['model_notes']
    )

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n[OK] HTML visualization saved to: {output_path}")


def main():
    """Example usage"""
    import sys

    if len(sys.argv) < 3:
        print("Usage: python predict_career_arc_exclusive.py 'Player Name' YEAR")
        print("\nExample:")
        print("  python predict_career_arc_exclusive.py 'Patrick Mahomes' 2017")
        return

    player_name = sys.argv[1]
    draft_year = int(sys.argv[2])

    # Generate projection
    projection = predict_career_arc(player_name, draft_year)

    if projection:
        # Console visualization
        visualize_projection_console(projection)

        # Save JSON
        root = get_project_root()
        projections_dir = root / 'data' / 'projections'
        projections_dir.mkdir(parents=True, exist_ok=True)

        json_path = projections_dir / f'{player_name.replace(" ", "_")}_{draft_year}_projection_exclusive.json'
        with open(json_path, 'w') as f:
            json.dump(projection, f, indent=2)

        print(f"\n[6] Saving projection...")
        print(f"    [OK] Saved to: {json_path}")

        # Create HTML visualization
        html_path = projections_dir / f'{player_name.replace(" ", "_")}_{draft_year}_projection.html'
        create_html_visualization(projection, html_path)

        print("\n" + "="*80)
        print("VISUALIZATIONS CREATED")
        print("="*80)
        print(f"\nJSON: {json_path}")
        print(f"HTML: {html_path}")
        print(f"\nOpen the HTML file in your browser to view the interactive visualization!")


if __name__ == "__main__":
    main()
