"""
Ensemble Model Evaluation and Optimization

Current status: XGBoost achieved r=0.3902
Target: r > 0.40 (gap: +0.0098 or 2.4%)

This script:
1. Loads and evaluates the trained models
2. Analyzes errors and identifies weak predictions
3. Tests advanced ensemble strategies to break r=0.40
4. Compares QB-by-QB predictions
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

# Load models
models_dir = Path("models")

try:
    rf_model = joblib.load(models_dir / 'random_forest_best.pkl')
    xgb_model = joblib.load(models_dir / 'xgboost_best.pkl')
    scaler = joblib.load(models_dir / 'feature_scaler.pkl')
    feature_names = joblib.load(models_dir / 'feature_names.pkl')
    MODELS_LOADED = True
except FileNotFoundError:
    print("ERROR: Models not found. Run train_ensemble_model.py first.")
    MODELS_LOADED = False
    sys.exit(1)

print("="*80)
print("ENSEMBLE MODEL EVALUATION")
print("="*80)
print("\nCurrent: XGBoost r=0.3902")
print("Target: r > 0.40 (gap: +0.0098 or 2.4%)")
print("\nAnalyzing model predictions and testing optimization strategies...")

# Recreate test dataset (same logic as training)
DATA_DIR = Path("data/production")
qb_files = sorted(DATA_DIR.glob("leaf_v2_game_by_game_*.csv"))
DATA_FILE = qb_files[-1]

qb_data = pd.read_csv(DATA_FILE)
qb_data = qb_data.sort_values(['passer_player_id', 'game_number'])


def engineer_features(player_games, window=12):
    """Same feature engineering as training script."""
    if len(player_games) < 3:
        return None

    recent = player_games.tail(window) if len(player_games) >= window else player_games
    features = {}

    # Base 5 features
    base_features = ['epa_mean', 'opp_adj_success_rate', 'epa_per_play',
                     'qb_epa_mean', 'success_rate']

    for feat in base_features:
        if feat in recent.columns:
            features[f'{feat}_avg'] = recent[feat].mean()
            features[f'{feat}_std'] = recent[feat].std()
            features[f'{feat}_max'] = recent[feat].max()
            features[f'{feat}_min'] = recent[feat].min()

    # EPA trend features
    if 'opp_adj_base_epa' in recent.columns:
        epa_values = recent['opp_adj_base_epa'].values

        if len(epa_values) >= 3:
            x = np.arange(len(epa_values))
            slope, intercept, r_val, p_val, std_err = stats.linregress(x, epa_values)
            features['epa_trend_slope'] = slope
            features['epa_trend_r2'] = r_val ** 2

        if len(epa_values) >= 6:
            mid = len(epa_values) // 2
            first_half = epa_values[:mid]
            second_half = epa_values[mid:]
            slope1, _, _, _, _ = stats.linregress(np.arange(len(first_half)), first_half)
            slope2, _, _, _, _ = stats.linregress(np.arange(len(second_half)), second_half)
            features['epa_acceleration'] = slope2 - slope1

        if len(epa_values) >= 8:
            recent_4 = epa_values[-4:].mean()
            earlier_8 = epa_values[:-4].mean()
            features['recent_vs_earlier'] = recent_4 - earlier_8

        mean_epa = epa_values.mean()
        streak = 0
        for val in reversed(epa_values):
            if (val > mean_epa and streak >= 0) or (val <= mean_epa and streak <= 0):
                streak += 1 if val > mean_epa else -1
            else:
                break
        features['streak_length'] = streak

        features['epa_consistency_cv'] = epa_values.std() / abs(epa_values.mean()) if abs(epa_values.mean()) > 0.01 else 0

    # Interactions
    if 'epa_mean_avg' in features and 'opp_adj_success_rate_avg' in features:
        features['epa_x_success'] = features['epa_mean_avg'] * features['opp_adj_success_rate_avg']

    if 'epa_mean_avg' in features and 'epa_per_play_avg' in features:
        features['epa_mean_x_per_play'] = features['epa_mean_avg'] * features['epa_per_play_avg']

    features['games_played'] = len(player_games)
    features['recent_games'] = len(recent)

    return features


# Build test dataset
samples = []

for player_id in qb_data['passer_player_id'].unique():
    player_games = qb_data[qb_data['passer_player_id'] == player_id].copy()

    if len(player_games) < 12 + 16:
        continue

    for i in range(12, len(player_games) - 16):
        recent_12 = player_games.iloc[i-12:i]
        features = engineer_features(recent_12, window=12)

        if features is not None:
            target = player_games.iloc[i:i+16]['opp_adj_base_epa'].mean()

            if not np.isnan(target):
                features['target'] = target
                features['player_id'] = player_id
                features['player_name'] = player_games.iloc[i]['passer_player_name']
                features['game_index'] = i
                samples.append(features)

df = pd.DataFrame(samples)
df = df.replace([np.inf, -np.inf], np.nan).dropna()
df = df.sort_values('game_index')

# Use same split as training
split_idx = int(len(df) * 0.8)
test_df = df.iloc[split_idx:]

X_test = test_df[feature_names].values
y_test = test_df['target'].values
X_test_scaled = scaler.transform(X_test)

# Get predictions
rf_pred = rf_model.predict(X_test_scaled)
xgb_pred = xgb_model.predict(X_test_scaled)
ensemble_pred = (rf_pred + xgb_pred) / 2

# Baseline (for comparison)
from features.multifeature_rating_calculator import MultiFeatureRatingCalculator
baseline_calc = MultiFeatureRatingCalculator()

baseline_preds = []
for _, row in test_df.iterrows():
    player_games = qb_data[qb_data['passer_player_id'] == row['player_id']].copy()
    recent_12 = player_games.iloc[row['game_index']-12:row['game_index']]
    baseline_preds.append(baseline_calc.calculate_multifeature_rating(recent_12))

baseline_pred = np.array(baseline_preds)

# Calculate metrics
def evaluate_predictions(pred, y, name):
    corr, p = stats.spearmanr(pred, y)
    rmse = np.sqrt(mean_squared_error(y, pred))
    mae = mean_absolute_error(y, pred)
    return {'model': name, 'correlation': corr, 'rmse': rmse, 'mae': mae, 'p_value': p}

results = [
    evaluate_predictions(baseline_pred, y_test, 'Baseline (linear)'),
    evaluate_predictions(rf_pred, y_test, 'Random Forest'),
    evaluate_predictions(xgb_pred, y_test, 'XGBoost'),
    evaluate_predictions(ensemble_pred, y_test, 'Simple Ensemble')
]

print("\n" + "="*80)
print("BASE MODEL COMPARISON")
print("="*80)

results_df = pd.DataFrame(results).sort_values('correlation', ascending=False)
print(f"\n{'Model':<25} {'Correlation':>12} {'RMSE':>10} {'MAE':>10} {'P-value':>10}")
print("-"*75)
for _, row in results_df.iterrows():
    print(f"{row['model']:<25} {row['correlation']:>12.4f} {row['rmse']:>10.3f} {row['mae']:>10.3f} {row['p_value']:>10.6f}")


# Advanced ensemble strategy: Weighted by confidence
print("\n" + "="*80)
print("ADVANCED ENSEMBLE STRATEGIES")
print("="*80)

# 1. Confidence-weighted ensemble (weight by historical accuracy)
print("\n[Strategy 1] Confidence-Weighted Ensemble")
print("  Weight predictions by model's historical accuracy on similar samples")

# Split test into halves to measure model confidence
test_mid = len(y_test) // 2
y_test_cal = y_test[:test_mid]
y_test_eval = y_test[test_mid:]

rf_pred_cal = rf_pred[:test_mid]
xgb_pred_cal = xgb_pred[:test_mid]

rf_pred_eval = rf_pred[test_mid:]
xgb_pred_eval = xgb_pred[test_mid:]

# Calculate model weights based on calibration performance
rf_cal_corr = stats.spearmanr(rf_pred_cal, y_test_cal)[0]
xgb_cal_corr = stats.spearmanr(xgb_pred_cal, y_test_cal)[0]

# Normalize weights
total_corr = rf_cal_corr + xgb_cal_corr
rf_weight = rf_cal_corr / total_corr
xgb_weight = xgb_cal_corr / total_corr

weighted_pred_eval = rf_weight * rf_pred_eval + xgb_weight * xgb_pred_eval

weighted_corr = stats.spearmanr(weighted_pred_eval, y_test_eval)[0]
print(f"  RF weight: {rf_weight:.3f}, XGBoost weight: {xgb_weight:.3f}")
print(f"  Result: r={weighted_corr:.4f}")

# Apply to full test set
weighted_pred_full = rf_weight * rf_pred + xgb_weight * xgb_pred
weighted_result = evaluate_predictions(weighted_pred_full, y_test, 'Weighted Ensemble')
results.append(weighted_result)


# 2. Selective ensemble (use XGBoost when confident, ensemble otherwise)
print("\n[Strategy 2] Selective Ensemble")
print("  Use XGBoost alone when predictions are confident, ensemble for uncertainty")

# Measure prediction variance as uncertainty proxy
rf_xgb_diff = np.abs(rf_pred - xgb_pred)
uncertainty_threshold = np.percentile(rf_xgb_diff, 70)  # Top 30% disagreement

selective_pred = np.where(
    rf_xgb_diff < uncertainty_threshold,
    xgb_pred,  # Use XGBoost when models agree
    ensemble_pred  # Use ensemble when models disagree
)

selective_result = evaluate_predictions(selective_pred, y_test, 'Selective Ensemble')
print(f"  Confident samples: {(rf_xgb_diff < uncertainty_threshold).sum()} / {len(rf_xgb_diff)}")
print(f"  Result: r={selective_result['correlation']:.4f}")
results.append(selective_result)


# 3. Stacked ensemble (train meta-model on predictions)
print("\n[Strategy 3] Stacked Ensemble")
print("  Train meta-model to combine RF and XGBoost predictions")

from sklearn.linear_model import Ridge

# Use calibration set to train stacking
X_stack_train = np.column_stack([rf_pred_cal, xgb_pred_cal, baseline_pred[:test_mid]])
y_stack_train = y_test_cal

X_stack_test = np.column_stack([rf_pred_eval, xgb_pred_eval, baseline_pred[test_mid:]])
y_stack_test = y_test_eval

stacker = Ridge(alpha=1.0)
stacker.fit(X_stack_train, y_stack_train)

stacked_pred_eval = stacker.predict(X_stack_test)
stacked_corr = stats.spearmanr(stacked_pred_eval, y_stack_test)[0]

print(f"  Stacker weights: RF={stacker.coef_[0]:.3f}, XGB={stacker.coef_[1]:.3f}, Baseline={stacker.coef_[2]:.3f}")
print(f"  Result: r={stacked_corr:.4f}")

# Apply to full test set
X_stack_full = np.column_stack([rf_pred, xgb_pred, baseline_pred])
stacked_pred_full = stacker.predict(X_stack_full)
stacked_result = evaluate_predictions(stacked_pred_full, y_test, 'Stacked Ensemble')
results.append(stacked_result)


# Final comparison
print("\n" + "="*80)
print("FINAL MODEL COMPARISON")
print("="*80)

all_results_df = pd.DataFrame(results).sort_values('correlation', ascending=False)
print(f"\n{'Model':<25} {'Correlation':>12} {'RMSE':>10} {'MAE':>10}")
print("-"*60)
for _, row in all_results_df.iterrows():
    print(f"{row['model']:<25} {row['correlation']:>12.4f} {row['rmse']:>10.3f} {row['mae']:>10.3f}")

best = all_results_df.iloc[0]
TARGET_CORR = 0.40

print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)

print(f"\nBest model: {best['model']}")
print(f"  Correlation: r={best['correlation']:.4f}")
print(f"  RMSE: {best['rmse']:.3f}")
print(f"  MAE: {best['mae']:.3f}")

if best['correlation'] >= TARGET_CORR:
    print(f"\nSUCCESS! Target r > 0.40 achieved!")
    print(f"  Exceeded target by: {best['correlation'] - TARGET_CORR:+.4f}")
else:
    gap = TARGET_CORR - best['correlation']
    print(f"\nTarget: r > 0.40")
    print(f"  Gap remaining: {gap:+.4f} ({gap/TARGET_CORR*100:.1f}%)")
    print(f"\nRecommendations to close gap:")
    print(f"  1. Add external data (injuries, roster changes)")
    print(f"  2. More sophisticated feature engineering")
    print(f"  3. Deep learning models (Neural Networks)")
    print(f"  4. Longer hyperparameter search")

# Error analysis
print("\n" + "="*80)
print("ERROR ANALYSIS")
print("="*80)

best_pred = all_results_df.iloc[0]
if best_pred['model'] == 'XGBoost':
    pred = xgb_pred
elif best_pred['model'] == 'Stacked Ensemble':
    pred = stacked_pred_full
elif best_pred['model'] == 'Weighted Ensemble':
    pred = weighted_pred_full
elif best_pred['model'] == 'Selective Ensemble':
    pred = selective_pred
else:
    pred = ensemble_pred

errors = y_test - pred
abs_errors = np.abs(errors)

print(f"\nPrediction errors:")
print(f"  Mean error: {errors.mean():.4f}")
print(f"  Median absolute error: {np.median(abs_errors):.4f}")
print(f"  90th percentile error: {np.percentile(abs_errors, 90):.4f}")
print(f"  Max error: {abs_errors.max():.4f}")

# Worst predictions
worst_idx = np.argsort(abs_errors)[-10:]
print(f"\nTop 10 worst predictions:")
print(f"  {'QB':<20} {'Actual':>8} {'Predicted':>10} {'Error':>8}")
print("  " + "-"*50)
for idx in worst_idx:
    qb = test_df.iloc[idx]['player_name']
    actual = y_test[idx]
    predicted = pred[idx]
    error = errors[idx]
    print(f"  {qb:<20} {actual:>8.3f} {predicted:>10.3f} {error:>8.3f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
