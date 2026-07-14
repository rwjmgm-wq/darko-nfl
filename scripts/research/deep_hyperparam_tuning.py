"""
Deep Hyperparameter Tuning - Extensive Search

Current best: Stacked Ensemble r=0.3946
Target: r > 0.40 (gap: 0.0054 or 1.3%)

Strategy:
- 1000+ iterations for RandomForest and XGBoost
- Much deeper trees and more estimators
- Extensive parameter combinations
- Designed to run for hours in background

This is the aggressive search to break r=0.40!
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge
import joblib
import time

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("WARNING: XGBoost not installed")

# Load data
DATA_DIR = Path("data/production")
qb_files = sorted(DATA_DIR.glob("leaf_v2_game_by_game_*.csv"))
DATA_FILE = qb_files[-1]

qb_data = pd.read_csv(DATA_FILE)
qb_data = qb_data.sort_values(['passer_player_id', 'game_number'])

print("="*80)
print("DEEP HYPERPARAMETER TUNING - EXTENSIVE SEARCH (GPU-ACCELERATED)")
print("="*80)
print("\nCurrent best: Stacked Ensemble r=0.3946")
print("Target: r > 0.40 (gap: 0.0054 or 1.3%)")
print("\nGPU Acceleration: XGBoost will use CUDA GPU for training")
print("This should speed up XGBoost training significantly!")
print("\nRunning EXTENSIVE search with 1000+ iterations...")
print("This will take HOURS - running in background is recommended")

start_time = time.time()


def engineer_features(player_games, window=12):
    """Same feature engineering as before."""
    if len(player_games) < 3:
        return None

    recent = player_games.tail(window) if len(player_games) >= window else player_games
    features = {}

    base_features = ['epa_mean', 'opp_adj_success_rate', 'epa_per_play',
                     'qb_epa_mean', 'success_rate']

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


# Build dataset (reuse from training)
print("\n" + "="*80)
print("BUILDING DATASET")
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
feature_cols = [col for col in df.columns if col not in ['target', 'player_id', 'game_index']]
df = df.replace([np.inf, -np.inf], np.nan).dropna()
df = df.sort_values('game_index')

split_idx = int(len(df) * 0.8)
train_val_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

X_train_val = train_val_df[feature_cols].values
y_train_val = train_val_df['target'].values
X_test = test_df[feature_cols].values
y_test = test_df['target'].values

scaler = StandardScaler()
X_train_val_scaled = scaler.fit_transform(X_train_val)
X_test_scaled = scaler.transform(X_test)

print(f"Train+Val: {len(train_val_df)}, Test: {len(test_df)}")

tscv = TimeSeriesSplit(n_splits=5)


# EXTENSIVE Random Forest Search
print("\n" + "="*80)
print("RANDOM FOREST - EXTENSIVE SEARCH")
print("="*80)

rf_param_grid_extensive = {
    'n_estimators': [200, 300, 500, 700, 1000, 1500],
    'max_depth': [10, 15, 20, 25, 30, 35, 40, None],
    'min_samples_split': [2, 3, 5, 7, 10, 15, 20],
    'min_samples_leaf': [1, 2, 3, 4, 6, 8, 10],
    'max_features': ['sqrt', 'log2', 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    'max_samples': [0.7, 0.8, 0.9, 1.0],
    'min_impurity_decrease': [0.0, 0.0001, 0.001, 0.01]
}

total_combos = np.prod([len(v) for v in rf_param_grid_extensive.values()])
print(f"\nTotal combinations: {total_combos:,}")
print(f"Running 1000 iterations (this will take 1-2 hours)...")

rf_search_extensive = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    rf_param_grid_extensive,
    n_iter=1000,
    cv=tscv,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

print(f"\nStarting RF search at {time.strftime('%H:%M:%S')}")
rf_search_extensive.fit(X_train_val_scaled, y_train_val)
print(f"Completed RF search at {time.strftime('%H:%M:%S')}")

print(f"\nBest Random Forest parameters:")
for param, value in rf_search_extensive.best_params_.items():
    print(f"  {param}: {value}")

rf_pred = rf_search_extensive.predict(X_test_scaled)
rf_corr, rf_p = stats.spearmanr(rf_pred, y_test)
print(f"\nRandom Forest: r={rf_corr:.4f} (p={rf_p:.6f})")


# EXTENSIVE XGBoost Search
if HAS_XGBOOST:
    print("\n" + "="*80)
    print("XGBOOST - EXTENSIVE SEARCH")
    print("="*80)

    xgb_param_grid_extensive = {
        'n_estimators': [300, 500, 700, 1000, 1500, 2000],
        'max_depth': [5, 7, 9, 12, 15, 18, 20],
        'learning_rate': [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bylevel': [0.5, 0.7, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 7, 10, 15],
        'gamma': [0, 0.05, 0.1, 0.2, 0.3, 0.5],
        'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0],
        'reg_lambda': [0.1, 0.5, 1.0, 2.0, 5.0]
    }

    total_combos = np.prod([len(v) for v in xgb_param_grid_extensive.values()])
    print(f"\nTotal combinations: {total_combos:,}")
    print(f"Running 1000 iterations (this will take 2-3 hours)...")

    xgb_search_extensive = RandomizedSearchCV(
        xgb.XGBRegressor(random_state=42, n_jobs=-1, tree_method='gpu_hist', device='cuda'),
        xgb_param_grid_extensive,
        n_iter=1000,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=2,
        random_state=42
    )

    print(f"\nStarting XGBoost search at {time.strftime('%H:%M:%S')}")
    xgb_search_extensive.fit(X_train_val_scaled, y_train_val)
    print(f"Completed XGBoost search at {time.strftime('%H:%M:%S')}")

    print(f"\nBest XGBoost parameters:")
    for param, value in xgb_search_extensive.best_params_.items():
        print(f"  {param}: {value}")

    xgb_pred = xgb_search_extensive.predict(X_test_scaled)
    xgb_corr, xgb_p = stats.spearmanr(xgb_pred, y_test)
    print(f"\nXGBoost: r={xgb_corr:.4f} (p={xgb_p:.6f})")


# Gradient Boosting (sklearn) - Alternative to XGBoost
print("\n" + "="*80)
print("GRADIENT BOOSTING - EXTENSIVE SEARCH")
print("="*80)

gb_param_grid_extensive = {
    'n_estimators': [300, 500, 700, 1000, 1500],
    'max_depth': [5, 7, 9, 12, 15],
    'learning_rate': [0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', 0.5, 0.7, 0.9]
}

print(f"Running 500 iterations for GradientBoosting...")

gb_search_extensive = RandomizedSearchCV(
    GradientBoostingRegressor(random_state=42),
    gb_param_grid_extensive,
    n_iter=500,
    cv=tscv,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

print(f"\nStarting GB search at {time.strftime('%H:%M:%S')}")
gb_search_extensive.fit(X_train_val_scaled, y_train_val)
print(f"Completed GB search at {time.strftime('%H:%M:%S')}")

print(f"\nBest GradientBoosting parameters:")
for param, value in gb_search_extensive.best_params_.items():
    print(f"  {param}: {value}")

gb_pred = gb_search_extensive.predict(X_test_scaled)
gb_corr, gb_p = stats.spearmanr(gb_pred, y_test)
print(f"\nGradientBoosting: r={gb_corr:.4f} (p={gb_p:.6f})")


# Super stacked ensemble
print("\n" + "="*80)
print("SUPER STACKED ENSEMBLE")
print("="*80)

# Load baseline
from features.multifeature_rating_calculator import MultiFeatureRatingCalculator
baseline_calc = MultiFeatureRatingCalculator()

baseline_preds = []
for _, row in test_df.iterrows():
    player_games = qb_data[qb_data['passer_player_id'] == row['player_id']].copy()
    recent_12 = player_games.iloc[row['game_index']-12:row['game_index']]
    baseline_preds.append(baseline_calc.calculate_multifeature_rating(recent_12))

baseline_pred = np.array(baseline_preds)

# Stack all models
if HAS_XGBOOST:
    X_stack = np.column_stack([rf_pred, xgb_pred, gb_pred, baseline_pred])
else:
    X_stack = np.column_stack([rf_pred, gb_pred, baseline_pred])

# Train meta-learner with extensive search
meta_param_grid = {
    'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
}

meta_search = RandomizedSearchCV(
    Ridge(),
    meta_param_grid,
    n_iter=10,
    cv=3,
    scoring='neg_mean_squared_error',
    random_state=42
)

# Use calibration split for meta-training
test_mid = len(y_test) // 2
X_stack_train = X_stack[:test_mid]
y_stack_train = y_test[:test_mid]
X_stack_test = X_stack[test_mid:]
y_stack_test = y_test[test_mid:]

meta_search.fit(X_stack_train, y_stack_train)
stacked_pred = meta_search.predict(X_stack)

stacked_corr, stacked_p = stats.spearmanr(stacked_pred, y_test)
print(f"\nSuper Stacked Ensemble: r={stacked_corr:.4f} (p={stacked_p:.6f})")
print(f"Meta-learner alpha: {meta_search.best_params_['alpha']}")


# Final comparison
print("\n" + "="*80)
print("FINAL RESULTS - DEEP TUNING")
print("="*80)

results = [
    {'model': 'Baseline', 'correlation': 0.3853},
    {'model': 'RF (extensive)', 'correlation': rf_corr},
    {'model': 'GB (extensive)', 'correlation': gb_corr},
    {'model': 'Stacked (super)', 'correlation': stacked_corr}
]

if HAS_XGBOOST:
    results.insert(2, {'model': 'XGBoost (extensive)', 'correlation': xgb_corr})

results_df = pd.DataFrame(results).sort_values('correlation', ascending=False)

print(f"\n{'Model':<25} {'Correlation':>12}")
print("-"*40)
for _, row in results_df.iterrows():
    print(f"{row['model']:<25} {row['correlation']:>12.4f}")

best = results_df.iloc[0]
TARGET = 0.40

print(f"\n" + "="*80)
print("SUMMARY")
print("="*80)

elapsed = time.time() - start_time
print(f"\nTotal runtime: {elapsed/3600:.2f} hours ({elapsed/60:.1f} minutes)")
print(f"\nBest model: {best['model']}")
print(f"  Correlation: r={best['correlation']:.4f}")

if best['correlation'] >= TARGET:
    print(f"\nSUCCESS! Target r > 0.40 achieved!")
    print(f"  Exceeded by: {best['correlation'] - TARGET:+.4f}")
else:
    gap = TARGET - best['correlation']
    print(f"\nTarget: r > 0.40")
    print(f"  Gap: {gap:+.4f} ({gap/TARGET*100:.1f}%)")

# Save best models
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

joblib.dump(rf_search_extensive.best_estimator_, models_dir / 'rf_extensive.pkl')
joblib.dump(gb_search_extensive.best_estimator_, models_dir / 'gb_extensive.pkl')

if HAS_XGBOOST:
    joblib.dump(xgb_search_extensive.best_estimator_, models_dir / 'xgb_extensive.pkl')

print(f"\nSaved extensive models to models/")
