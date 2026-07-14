"""Quick evaluation of all trained models to see final results."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

# Load data
DATA_DIR = Path("data/production")
qb_files = sorted(DATA_DIR.glob("leaf_v2_game_by_game_*.csv"))
DATA_FILE = qb_files[-1]

qb_data = pd.read_csv(DATA_FILE)
qb_data = qb_data.sort_values(['passer_player_id', 'game_number'])

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
                samples.append(features)

df = pd.DataFrame(samples)
df = df.replace([np.inf, -np.inf], np.nan).dropna()
df = df.sort_values('games_played')  # Proxy for time

# Use same split
split_idx = int(len(df) * 0.8)
test_df = df.iloc[split_idx:]

# Load models
models_dir = Path("models")
scaler = joblib.load(models_dir / 'feature_scaler.pkl')
feature_names = joblib.load(models_dir / 'feature_names.pkl')

X_test = test_df[feature_names].values
y_test = test_df['target'].values
X_test_scaled = scaler.transform(X_test)

print("="*80)
print("FINAL RESULTS - ALL MODELS")
print("="*80)

results = []

# Baseline
results.append({'model': 'Baseline (linear multi-feature)', 'correlation': 0.3853, 'rmse': 0.151, 'mae': 0.116})

# Previous best
results.append({'model': 'Stacked Ensemble (previous)', 'correlation': 0.3946, 'rmse': 0.149, 'mae': 0.114})

# Extensive models
try:
    rf_ext = joblib.load(models_dir / 'rf_extensive.pkl')
    rf_pred = rf_ext.predict(X_test_scaled)
    rf_corr = stats.spearmanr(rf_pred, y_test)[0]
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_mae = mean_absolute_error(y_test, rf_pred)
    results.append({'model': 'Random Forest (1000 iter)', 'correlation': rf_corr, 'rmse': rf_rmse, 'mae': rf_mae})
    print(f"\n[+] Random Forest (extensive): r={rf_corr:.4f}")
except Exception as e:
    print(f"\n[-] Random Forest (extensive): {e}")

try:
    xgb_ext = joblib.load(models_dir / 'xgb_extensive.pkl')
    xgb_pred = xgb_ext.predict(X_test_scaled)
    xgb_corr = stats.spearmanr(xgb_pred, y_test)[0]
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    results.append({'model': 'XGBoost GPU (1000 iter)', 'correlation': xgb_corr, 'rmse': xgb_rmse, 'mae': xgb_mae})
    print(f"[+] XGBoost GPU (extensive): r={xgb_corr:.4f}")
except Exception as e:
    print(f"[-] XGBoost GPU (extensive): {e}")

try:
    gb_ext = joblib.load(models_dir / 'gb_extensive.pkl')
    gb_pred = gb_ext.predict(X_test_scaled)
    gb_corr = stats.spearmanr(gb_pred, y_test)[0]
    gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
    gb_mae = mean_absolute_error(y_test, gb_pred)
    results.append({'model': 'GradientBoosting (500 iter)', 'correlation': gb_corr, 'rmse': gb_rmse, 'mae': gb_mae})
    print(f"[+] GradientBoosting (extensive): r={gb_corr:.4f}")
except Exception as e:
    print(f"[-] GradientBoosting (extensive): {e}")

try:
    mlp = joblib.load(models_dir / 'mlp_best.pkl')
    mlp_pred = mlp.predict(X_test_scaled)
    mlp_corr = stats.spearmanr(mlp_pred, y_test)[0]
    mlp_rmse = np.sqrt(mean_squared_error(y_test, mlp_pred))
    mlp_mae = mean_absolute_error(y_test, mlp_pred)
    results.append({'model': 'Neural Network (MLPRegressor)', 'correlation': mlp_corr, 'rmse': mlp_rmse, 'mae': mlp_mae})
    print(f"[+] Neural Network (MLP): r={mlp_corr:.4f}")
except Exception as e:
    print(f"[-] Neural Network (MLP): {e}")

# Final comparison
results_df = pd.DataFrame(results).sort_values('correlation', ascending=False)

print("\n" + "="*80)
print("FINAL MODEL RANKING")
print("="*80)

print(f"\n{'Rank':<6} {'Model':<35} {'Correlation':>12} {'RMSE':>10} {'MAE':>10}")
print("-"*75)
for i, (_, row) in enumerate(results_df.iterrows(), 1):
    print(f"{i:<6} {row['model']:<35} {row['correlation']:>12.4f} {row['rmse']:>10.3f} {row['mae']:>10.3f}")

best = results_df.iloc[0]
TARGET_CORR = 0.40

print("\n" + "="*80)
print("FINAL VERDICT")
print("="*80)

print(f"\nBest model: {best['model']}")
print(f"  Correlation: r={best['correlation']:.4f}")
print(f"  RMSE: {best['rmse']:.3f}")
print(f"  MAE: {best['mae']:.3f}")

if best['correlation'] >= TARGET_CORR:
    print(f"\n*** SUCCESS! Target r > 0.40 ACHIEVED! ***")
    print(f"  Exceeded target by: {best['correlation'] - TARGET_CORR:+.4f}")
else:
    gap = TARGET_CORR - best['correlation']
    print(f"\nTarget: r > {TARGET_CORR:.2f}")
    print(f"  Gap remaining: {gap:+.4f} ({gap/TARGET_CORR*100:.1f}%)")
    print(f"\nImprovement from baseline:")
    print(f"  Baseline: r=0.3853")
    print(f"  Best: r={best['correlation']:.4f}")
    print(f"  Gain: {best['correlation'] - 0.3853:+.4f} ({(best['correlation'] - 0.3853) / 0.3853 * 100:+.1f}%)")
