"""
Neural Network Diagnostic Analysis

User achieved r=0.5820 with MLPRegressor - suspiciously high but plausible.
This script performs hypercritical analysis to determine if legitimate or data leakage.

Diagnostic checks:
1. Model architecture and training parameters
2. Time-based split test (original method)
3. Player-based split test (rigorous - no player overlap)
4. Data leakage indicators (prediction variance, distribution)
5. Comparison to baseline (r=0.3853)
6. Overfitting evidence (validation scores over time)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

print("="*80)
print("NEURAL NETWORK DIAGNOSTIC ANALYSIS")
print("="*80)
print("\nTarget: Determine if r=0.5820 is legitimate or data leakage")
print("Approach: Hypercritical examination of model, data, and predictions")

# Load model and data preparation tools
models_dir = Path("models")
try:
    mlp_model = joblib.load(models_dir / 'mlp_best.pkl')
    scaler = joblib.load(models_dir / 'feature_scaler.pkl')
    feature_names = joblib.load(models_dir / 'feature_names.pkl')
    print("\n[+] Models loaded successfully")
except Exception as e:
    print(f"\n[-] Error loading models: {e}")
    sys.exit(1)

# Load data
DATA_DIR = Path("data/production")
qb_files = sorted(DATA_DIR.glob("leaf_v2_game_by_game_*.csv"))
DATA_FILE = qb_files[-1]

qb_data = pd.read_csv(DATA_FILE)
qb_data = qb_data.sort_values(['passer_player_id', 'game_number'])

print(f"[+] Data loaded: {len(qb_data)} games from {DATA_FILE.name}")


def engineer_features(player_games, window=12):
    """Same feature engineering as training scripts."""
    if len(player_games) < 3:
        return None
    recent = player_games.tail(window) if len(player_games) >= window else player_games
    features = {}
    base_features = ['epa_mean', 'opp_adj_success_rate', 'epa_per_play', 'qb_epa_mean', 'success_rate']
    for feat in base_features:
        if feat in recent.columns:
            features[f'{feat}_avg'] = recent[feat].mean()
            features[f'{feat}_std'] = recent[feat].std()
            features[f'{feat}_max'] = recent[feat].max()
            features[f'{feat}_min'] = recent[feat].min()
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
    if 'epa_mean_avg' in features and 'opp_adj_success_rate_avg' in features:
        features['epa_x_success'] = features['epa_mean_avg'] * features['opp_adj_success_rate_avg']
    if 'epa_mean_avg' in features and 'epa_per_play_avg' in features:
        features['epa_mean_x_per_play'] = features['epa_mean_avg'] * features['epa_per_play_avg']
    features['games_played'] = len(player_games)
    features['recent_games'] = len(recent)
    return features


# DIAGNOSTIC 1: Model Architecture and Training Parameters
print("\n" + "="*80)
print("DIAGNOSTIC 1: MODEL ARCHITECTURE")
print("="*80)

print(f"\nNeural Network Architecture:")
print(f"  Hidden layers: {mlp_model.hidden_layer_sizes}")
print(f"  Activation: {mlp_model.activation}")
print(f"  Learning rate: {mlp_model.learning_rate} (init={mlp_model.learning_rate_init})")
print(f"  Alpha (L2 penalty): {mlp_model.alpha}")
print(f"  Training iterations completed: {mlp_model.n_iter_}")
print(f"  Max iterations allowed: {mlp_model.max_iter}")
print(f"  Early stopping: {mlp_model.early_stopping}")

if mlp_model.early_stopping:
    print(f"  Validation fraction: {mlp_model.validation_fraction}")

# Check if training converged or hit max iterations
if mlp_model.n_iter_ >= mlp_model.max_iter - 1:
    print("\n  WARNING: Hit max iterations - may not have converged")
else:
    print(f"\n  [+] Converged after {mlp_model.n_iter_} iterations")

# Validation curve analysis (if available)
if hasattr(mlp_model, 'validation_scores_') and mlp_model.validation_scores_ is not None:
    val_scores = mlp_model.validation_scores_
    print(f"\n  Validation scores over time:")
    print(f"    Best validation score: {max(val_scores):.6f}")
    print(f"    Final validation score: {val_scores[-1]:.6f}")

    # Check for overfitting signs
    if len(val_scores) > 10:
        early_avg = np.mean(val_scores[:len(val_scores)//3])
        late_avg = np.mean(val_scores[-len(val_scores)//3:])
        if late_avg < early_avg * 0.95:
            print(f"    WARNING: Validation scores degraded over time (possible overfitting)")
        else:
            print(f"    [+] Validation scores stable or improving")


# DIAGNOSTIC 2: Time-Based Split (Original Method)
print("\n" + "="*80)
print("DIAGNOSTIC 2: TIME-BASED SPLIT TEST")
print("="*80)

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
                features['game_index'] = i
                samples.append(features)

df = pd.DataFrame(samples)
df = df.replace([np.inf, -np.inf], np.nan).dropna()
df = df.sort_values('game_index')

# Same 80/20 split as training
split_idx = int(len(df) * 0.8)
test_df = df.iloc[split_idx:]

X_test = test_df[feature_names].values
y_test = test_df['target'].values
X_test_scaled = scaler.transform(X_test)

mlp_pred = mlp_model.predict(X_test_scaled)
mlp_corr, mlp_p = stats.spearmanr(mlp_pred, y_test)
mlp_rmse = np.sqrt(mean_squared_error(y_test, mlp_pred))
mlp_mae = mean_absolute_error(y_test, mlp_pred)

print(f"\nTime-based split (80/20) results:")
print(f"  Spearman r: {mlp_corr:.4f} (p={mlp_p:.6f})")
print(f"  RMSE: {mlp_rmse:.3f}")
print(f"  MAE: {mlp_mae:.3f}")

if abs(mlp_corr - 0.5820) > 0.01:
    print(f"  WARNING: Correlation {mlp_corr:.4f} differs from reported 0.5820")
else:
    print(f"  [+] Matches reported r=0.5820")


# DIAGNOSTIC 3: Player-Based Split (Rigorous Test)
print("\n" + "="*80)
print("DIAGNOSTIC 3: PLAYER-BASED SPLIT TEST")
print("="*80)
print("More rigorous: Ensure NO player appears in both train and test")

unique_players = df['player_id'].unique()
np.random.seed(42)
test_players = np.random.choice(unique_players, size=int(len(unique_players) * 0.2), replace=False)

train_df_player = df[~df['player_id'].isin(test_players)]
test_df_player = df[df['player_id'].isin(test_players)]

print(f"\nPlayer-based split:")
print(f"  Train: {len(train_df_player)} samples from {len(unique_players) - len(test_players)} players")
print(f"  Test: {len(test_df_player)} samples from {len(test_players)} players")

# Retrain scaler and model on player-based train set
X_train_player = train_df_player[feature_names].values
y_train_player = train_df_player['target'].values
X_test_player = test_df_player[feature_names].values
y_test_player = test_df_player['target'].values

scaler_player = type(scaler)()
X_train_player_scaled = scaler_player.fit_transform(X_train_player)
X_test_player_scaled = scaler_player.transform(X_test_player)

# Use existing model to predict (as-is, not retrained)
mlp_pred_player = mlp_model.predict(X_test_player_scaled)
mlp_corr_player, mlp_p_player = stats.spearmanr(mlp_pred_player, y_test_player)

print(f"\nPlayer-based split results (using original model):")
print(f"  Spearman r: {mlp_corr_player:.4f} (p={mlp_p_player:.6f})")

if mlp_corr_player < 0.30:
    print(f"  CRITICAL: Player-based r={mlp_corr_player:.4f} << time-based r={mlp_corr:.4f}")
    print(f"  VERDICT: Model memorized individual players (severe overfitting)")
elif mlp_corr_player < mlp_corr * 0.7:
    print(f"  WARNING: Player-based r={mlp_corr_player:.4f} significantly lower than time-based")
    print(f"  VERDICT: Model has player-specific bias")
else:
    print(f"  [+] Player-based r={mlp_corr_player:.4f} comparable to time-based r={mlp_corr:.4f}")
    print(f"  [+] Model generalizes across players")


# DIAGNOSTIC 4: Data Leakage Indicators
print("\n" + "="*80)
print("DIAGNOSTIC 4: DATA LEAKAGE INDICATORS")
print("="*80)

# Check prediction variance vs actual variance
pred_std = mlp_pred.std()
actual_std = y_test.std()
variance_ratio = pred_std / actual_std

print(f"\nPrediction variance analysis:")
print(f"  Actual std: {actual_std:.4f}")
print(f"  Predicted std: {pred_std:.4f}")
print(f"  Variance ratio: {variance_ratio:.4f}")

if variance_ratio < 0.3:
    print(f"  CRITICAL: Predictions too narrow - classic data leakage sign")
elif variance_ratio < 0.5:
    print(f"  WARNING: Predictions narrower than expected")
else:
    print(f"  [+] Prediction variance seems reasonable")

# Check prediction distribution
print(f"\nPrediction distribution:")
print(f"  Actual range: [{y_test.min():.3f}, {y_test.max():.3f}]")
print(f"  Predicted range: [{mlp_pred.min():.3f}, {mlp_pred.max():.3f}]")

pred_range = mlp_pred.max() - mlp_pred.min()
actual_range = y_test.max() - y_test.min()
range_ratio = pred_range / actual_range

print(f"  Range ratio: {range_ratio:.4f}")

if range_ratio < 0.3:
    print(f"  CRITICAL: Predictions clustered too tightly")
elif range_ratio < 0.5:
    print(f"  WARNING: Limited prediction range")
else:
    print(f"  [+] Prediction range reasonable")

# Check for suspiciously perfect bins
bins = np.linspace(y_test.min(), y_test.max(), 10)
actual_hist, _ = np.histogram(y_test, bins=bins)
pred_hist, _ = np.histogram(mlp_pred, bins=bins)

# If predictions exactly match actual distribution, that's suspicious
hist_corr = stats.pearsonr(actual_hist, pred_hist)[0]
print(f"\nDistribution similarity: {hist_corr:.4f}")
if hist_corr > 0.95:
    print(f"  CRITICAL: Prediction distribution suspiciously similar to actual")
else:
    print(f"  [+] Distribution similarity reasonable")


# DIAGNOSTIC 5: Baseline Comparison
print("\n" + "="*80)
print("DIAGNOSTIC 5: BASELINE COMPARISON")
print("="*80)

BASELINE_CORR = 0.3853
STACKED_ENSEMBLE_CORR = 0.3946

improvement_vs_baseline = (mlp_corr - BASELINE_CORR) / BASELINE_CORR * 100
improvement_vs_stacked = (mlp_corr - STACKED_ENSEMBLE_CORR) / STACKED_ENSEMBLE_CORR * 100

print(f"\nBaseline (linear multi-feature): r={BASELINE_CORR:.4f}")
print(f"Stacked Ensemble (previous best): r={STACKED_ENSEMBLE_CORR:.4f}")
print(f"Neural Network: r={mlp_corr:.4f}")
print(f"\nImprovement:")
print(f"  vs Baseline: {improvement_vs_baseline:+.1f}%")
print(f"  vs Stacked Ensemble: {improvement_vs_stacked:+.1f}%")

if improvement_vs_baseline > 50:
    print(f"\n  CRITICAL: {improvement_vs_baseline:.1f}% improvement is unrealistic for QB prediction")
    print(f"  Expected improvement: 2-5% at most")
    print(f"  VERDICT: Likely data leakage")
elif improvement_vs_baseline > 20:
    print(f"\n  WARNING: {improvement_vs_baseline:.1f}% improvement is suspiciously high")
    print(f"  VERDICT: Investigate features for leakage")
else:
    print(f"\n  [+] {improvement_vs_baseline:.1f}% improvement is plausible")


# DIAGNOSTIC 6: Error Analysis
print("\n" + "="*80)
print("DIAGNOSTIC 6: ERROR ANALYSIS")
print("="*80)

errors = y_test - mlp_pred
abs_errors = np.abs(errors)

print(f"\nPrediction errors:")
print(f"  Mean error: {errors.mean():.4f}")
print(f"  Median absolute error: {np.median(abs_errors):.4f}")
print(f"  90th percentile error: {np.percentile(abs_errors, 90):.4f}")
print(f"  Max error: {abs_errors.max():.4f}")

# Check for systematic bias
if abs(errors.mean()) > 0.02:
    print(f"  WARNING: Systematic bias detected (mean error = {errors.mean():.4f})")
else:
    print(f"  [+] No systematic bias")

# Worst predictions
worst_idx = np.argsort(abs_errors)[-10:]
print(f"\nTop 10 worst predictions:")
print(f"  {'QB':<20} {'Actual':>8} {'Predicted':>10} {'Error':>8}")
print("  " + "-"*50)
for idx in worst_idx:
    qb = test_df.iloc[idx]['player_id']
    actual = y_test[idx]
    predicted = mlp_pred[idx]
    error = errors[idx]
    print(f"  {str(qb):<20} {actual:>8.3f} {predicted:>10.3f} {error:>8.3f}")


# FINAL VERDICT
print("\n" + "="*80)
print("FINAL VERDICT")
print("="*80)

red_flags = []
green_flags = []

# Check all diagnostics
if mlp_model.n_iter_ >= mlp_model.max_iter - 1:
    red_flags.append("Model hit max iterations without convergence")

if abs(mlp_corr - 0.5820) > 0.01:
    red_flags.append(f"Cannot reproduce r=0.5820 (got r={mlp_corr:.4f})")

if mlp_corr_player < mlp_corr * 0.7:
    red_flags.append(f"Player-based split r={mlp_corr_player:.4f} << time-based r={mlp_corr:.4f}")
else:
    green_flags.append(f"Model generalizes across players")

if variance_ratio < 0.3:
    red_flags.append(f"Prediction variance too narrow ({variance_ratio:.3f})")
elif variance_ratio >= 0.5:
    green_flags.append(f"Prediction variance reasonable ({variance_ratio:.3f})")

if range_ratio < 0.3:
    red_flags.append(f"Prediction range too narrow ({range_ratio:.3f})")
elif range_ratio >= 0.5:
    green_flags.append(f"Prediction range reasonable ({range_ratio:.3f})")

if improvement_vs_baseline > 50:
    red_flags.append(f"Unrealistic improvement: {improvement_vs_baseline:.1f}% vs baseline")
elif improvement_vs_baseline <= 20:
    green_flags.append(f"Plausible improvement: {improvement_vs_baseline:.1f}% vs baseline")

print(f"\nRED FLAGS ({len(red_flags)}):")
if red_flags:
    for flag in red_flags:
        print(f"  [-] {flag}")
else:
    print(f"  None detected")

print(f"\nGREEN FLAGS ({len(green_flags)}):")
if green_flags:
    for flag in green_flags:
        print(f"  [+] {flag}")
else:
    print(f"  None detected")

print(f"\n" + "="*80)
if len(red_flags) >= 3:
    print("VERDICT: LIKELY DATA LEAKAGE OR OVERFITTING")
    print("Recommendation: DO NOT USE this model in production")
elif len(red_flags) >= 1:
    print("VERDICT: SUSPICIOUS - Further investigation needed")
    print("Recommendation: Use with caution, validate on new data")
else:
    print("VERDICT: APPEARS LEGITIMATE")
    print("Recommendation: Further validation recommended before production use")
print("="*80)
