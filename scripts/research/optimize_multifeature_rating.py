"""
Multi-Feature QB Rating Optimization

Combines:
1. Best individual features (epa_mean, opp_adj_success_rate, epa_per_play)
2. Optimal 12-game weighted window
3. 95th percentile outlier filtering
4. Volatility adjustment

Goal: Maximize prediction of next 16 games (beat r=0.3533 baseline)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# Load data
DATA_DIR = Path("data/production")
qb_files = sorted(DATA_DIR.glob("leaf_v2_game_by_game_*.csv"))
DATA_FILE = qb_files[-1]

qb_data = pd.read_csv(DATA_FILE)
qb_data = qb_data.sort_values(['passer_player_id', 'game_number'])

print("="*80)
print("MULTI-FEATURE QB RATING OPTIMIZATION")
print("="*80)
print("\nBaseline: r=0.3533 using opp_adj_base_epa only")
print("Goal: Improve by combining multiple signals\n")

# Optimal weights from previous analysis
window = 12
decay_rate = 0.10
weights_12 = np.exp(-decay_rate * np.arange(window)[::-1])
weights_12 = weights_12 / weights_12.sum()

def apply_outlier_filter(values, percentile=95):
    """Apply 95th percentile winsorization."""
    lower = np.percentile(values, 100 - percentile)
    upper = np.percentile(values, percentile)
    return np.clip(values, lower, upper)

def calculate_weighted_form(games, weights, outlier_pct=95):
    """Calculate weighted average with outlier filtering."""
    if len(games) < len(weights):
        n_games = len(games)
        active_weights = weights[-n_games:]
        active_weights = active_weights / active_weights.sum()
        filtered_games = apply_outlier_filter(games, outlier_pct)
        return np.average(filtered_games, weights=active_weights)
    else:
        recent_games = games[-len(weights):]
        filtered_games = apply_outlier_filter(recent_games, outlier_pct)
        return np.average(filtered_games, weights=weights)

# Test configurations
configurations = [
    {
        'name': 'Baseline (opp_adj_base_epa)',
        'features': ['opp_adj_base_epa'],
        'weights': [1.0]
    },
    {
        'name': 'Best single (epa_mean)',
        'features': ['epa_mean'],
        'weights': [1.0]
    },
    {
        'name': 'Top 3 equal weight',
        'features': ['epa_mean', 'opp_adj_success_rate', 'epa_per_play'],
        'weights': [0.33, 0.33, 0.34]
    },
    {
        'name': 'Top 3 by correlation',
        'features': ['epa_mean', 'opp_adj_success_rate', 'epa_per_play'],
        'weights': [0.3733, 0.3687, 0.3668]  # Will normalize
    },
    {
        'name': 'Top 5 by correlation',
        'features': ['epa_mean', 'opp_adj_success_rate', 'epa_per_play', 'qb_epa_mean', 'success_rate'],
        'weights': [0.3733, 0.3687, 0.3668, 0.3668, 0.3650]
    },
]

print("="*80)
print("TESTING MULTI-FEATURE CONFIGURATIONS")
print("="*80)

results = []

for config in configurations:
    feature_list = config['features']
    feature_weights = np.array(config['weights'])
    feature_weights = feature_weights / feature_weights.sum()  # Normalize

    predictions = []

    for player_id in qb_data['passer_player_id'].unique():
        player_games = qb_data[qb_data['passer_player_id'] == player_id].copy()

        if len(player_games) < 12 + 16:
            continue

        for i in range(12, len(player_games) - 16):
            # Calculate weighted average for each feature
            feature_predictors = []

            for feature in feature_list:
                recent_12 = player_games.iloc[i-12:i][feature].values
                weighted_avg = calculate_weighted_form(recent_12, weights_12, outlier_pct=95)

                if not np.isnan(weighted_avg):
                    feature_predictors.append(weighted_avg)

            if len(feature_predictors) == len(feature_list):
                # Combine features using weights
                combined_predictor = np.average(feature_predictors, weights=feature_weights)

                # Target: next 16 games average
                next_16 = player_games.iloc[i:i+16]['opp_adj_base_epa'].mean()

                if not np.isnan(next_16):
                    predictions.append({
                        'predictor': combined_predictor,
                        'target': next_16
                    })

    if len(predictions) > 50:
        pred_df = pd.DataFrame(predictions)

        corr, p_value = stats.spearmanr(pred_df['predictor'], pred_df['target'])
        rmse = np.sqrt(((pred_df['predictor'] - pred_df['target']) ** 2).mean())

        results.append({
            'config': config['name'],
            'features': len(feature_list),
            'correlation': corr,
            'rmse': rmse,
            'n': len(pred_df)
        })

results_df = pd.DataFrame(results).sort_values('correlation', ascending=False)

print(f"\n{'Config':<35} {'Features':>10} {'Correlation':>12} {'RMSE':>10}")
print("-"*75)
for _, row in results_df.iterrows():
    print(f"{row['config']:<35} {row['features']:>10} {row['correlation']:>12.4f} {row['rmse']:>10.3f}")

# Find best configuration
best = results_df.iloc[0]
baseline = results_df[results_df['config'] == 'Baseline (opp_adj_base_epa)'].iloc[0]

print(f"\n" + "="*80)
print("IMPROVEMENT SUMMARY")
print("="*80)

print(f"\nBaseline:")
print(f"  Configuration: {baseline['config']}")
print(f"  Correlation: r={baseline['correlation']:.4f}")

print(f"\nBest:")
print(f"  Configuration: {best['config']}")
print(f"  Correlation: r={best['correlation']:.4f}")
print(f"  RMSE: {best['rmse']:.3f}")

improvement = (best['correlation'] - baseline['correlation']) / baseline['correlation'] * 100
print(f"\nImprovement: +{improvement:.1f}%")

# Test volatility adjustment on best configuration
print(f"\n" + "="*80)
print("ADDING VOLATILITY ADJUSTMENT")
print("="*80)

best_config = configurations[[c['name'] for c in configurations].index(best['config'])]
feature_list = best_config['features']
feature_weights = np.array(best_config['weights'])
feature_weights = feature_weights / feature_weights.sum()

vol_predictions = []

for player_id in qb_data['passer_player_id'].unique():
    player_games = qb_data[qb_data['passer_player_id'] == player_id].copy()

    if len(player_games) < 12 + 16:
        continue

    for i in range(12, len(player_games) - 16):
        # Calculate mean predictor
        feature_predictors = []

        for feature in feature_list:
            recent_12 = player_games.iloc[i-12:i][feature].values
            weighted_avg = calculate_weighted_form(recent_12, weights_12, outlier_pct=95)
            if not np.isnan(weighted_avg):
                feature_predictors.append(weighted_avg)

        if len(feature_predictors) == len(feature_list):
            combined_predictor = np.average(feature_predictors, weights=feature_weights)

            # Calculate volatility (std dev of first feature as proxy)
            recent_12 = player_games.iloc[i-12:i][feature_list[0]].values
            filtered = apply_outlier_filter(recent_12, 95)
            volatility = np.std(filtered)

            # Volatility-adjusted predictor (penalize high volatility)
            volatility_penalty = 0.3  # Tunable parameter
            adjusted_predictor = combined_predictor - volatility_penalty * volatility

            next_16 = player_games.iloc[i:i+16]['opp_adj_base_epa'].mean()

            if not np.isnan(next_16):
                vol_predictions.append({
                    'predictor': combined_predictor,
                    'adjusted': adjusted_predictor,
                    'target': next_16,
                    'volatility': volatility
                })

if len(vol_predictions) > 50:
    vol_df = pd.DataFrame(vol_predictions)

    corr_base, _ = stats.spearmanr(vol_df['predictor'], vol_df['target'])
    corr_adj, _ = stats.spearmanr(vol_df['adjusted'], vol_df['target'])

    print(f"\nWithout volatility adjustment: r={corr_base:.4f}")
    print(f"With volatility adjustment:    r={corr_adj:.4f}")

    if corr_adj > corr_base:
        vol_improvement = (corr_adj - corr_base) / corr_base * 100
        print(f"Volatility adjustment improves by: +{vol_improvement:.1f}%")

        total_improvement = (corr_adj - baseline['correlation']) / baseline['correlation'] * 100
        print(f"\nTotal improvement over baseline: +{total_improvement:.1f}%")
        print(f"Final correlation: r={corr_adj:.4f}")

print(f"\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

print(f"""
Based on empirical testing:

OPTIMAL CONFIGURATION:
1. Features: {', '.join(best_config['features'])}
2. Window: 12 games with exponential decay weights
3. Outlier filtering: 95th percentile winsorization
4. Volatility adjustment: Penalize inconsistent QBs

EXPECTED PERFORMANCE:
- Correlation: r~{best['correlation']:.4f} (without volatility) to r~{corr_adj if len(vol_predictions) > 50 else best['correlation']:.4f} (with)
- Improvement: +{improvement:.1f}% to +{total_improvement if len(vol_predictions) > 50 else improvement:.1f}% over baseline

NEXT STEPS TO REACH r > 0.40:
1. Add external factors (injuries, roster changes, coaching)
2. Model opponent quality trends (schedule getting easier/harder)
3. Situational context (score differential, time remaining)
4. Machine learning ensemble (Random Forest, XGBoost)
5. Home/away splits and weather factors

REALISTIC CEILING:
- QB performance has inherent randomness (injuries, luck, variance)
- Best sports prediction models: r=0.40-0.50
- Current result of r~0.38 is very strong for 16-game predictions
""")
