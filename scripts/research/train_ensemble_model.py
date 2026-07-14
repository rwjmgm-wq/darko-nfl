"""
Machine Learning Ensemble Training

Goal: Push correlation from r=0.3853 to r > 0.40 using non-linear ML models

Current baseline: Multi-feature linear rating achieves r=0.3853
Target: r > 0.40 (+3.8% improvement)

Strategy:
- Use Random Forest and XGBoost to capture non-linear feature interactions
- Extensive hyperparameter tuning (user has powerful hardware)
- Feature engineering: base features + trend features + interactions
- Proper train/validation/test split with time-series awareness
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("WARNING: XGBoost not installed. Install with: pip install xgboost")

# Load data
DATA_DIR = Path("data/production")
qb_files = sorted(DATA_DIR.glob("leaf_v2_game_by_game_*.csv"))
DATA_FILE = qb_files[-1]

qb_data = pd.read_csv(DATA_FILE)
qb_data = qb_data.sort_values(['passer_player_id', 'game_number'])

print("="*80)
print("MACHINE LEARNING ENSEMBLE TRAINING")
print("="*80)
print("\nCurrent baseline: Multi-feature linear rating r=0.3853")
print("Target: r > 0.40 (+3.8% improvement)")
print("\nUsing powerful hardware for extensive hyperparameter tuning...")


def engineer_features(player_games, window=12):
    """
    Engineer comprehensive feature set from recent games.

    Features include:
    - Base performance metrics (5 features)
    - Trend indicators (slope, acceleration)
    - Consistency metrics (std, range)
    - Peak/floor performance
    - Interaction features
    """
    if len(player_games) < 3:
        return None

    recent = player_games.tail(window) if len(player_games) >= window else player_games

    features = {}

    # Base 5 features (means from recent games)
    base_features = ['epa_mean', 'opp_adj_success_rate', 'epa_per_play',
                     'qb_epa_mean', 'success_rate']

    for feat in base_features:
        if feat in recent.columns:
            features[f'{feat}_avg'] = recent[feat].mean()
            features[f'{feat}_std'] = recent[feat].std()
            features[f'{feat}_max'] = recent[feat].max()
            features[f'{feat}_min'] = recent[feat].min()

    # Target EPA trend features
    if 'opp_adj_base_epa' in recent.columns:
        epa_values = recent['opp_adj_base_epa'].values

        # Linear trend
        if len(epa_values) >= 3:
            x = np.arange(len(epa_values))
            slope, intercept, r_val, p_val, std_err = stats.linregress(x, epa_values)
            features['epa_trend_slope'] = slope
            features['epa_trend_r2'] = r_val ** 2

        # Acceleration (change in trend)
        if len(epa_values) >= 6:
            mid = len(epa_values) // 2
            first_half = epa_values[:mid]
            second_half = epa_values[mid:]

            slope1, _, _, _, _ = stats.linregress(np.arange(len(first_half)), first_half)
            slope2, _, _, _, _ = stats.linregress(np.arange(len(second_half)), second_half)
            features['epa_acceleration'] = slope2 - slope1

        # Recent vs earlier
        if len(epa_values) >= 8:
            recent_4 = epa_values[-4:].mean()
            earlier_8 = epa_values[:-4].mean()
            features['recent_vs_earlier'] = recent_4 - earlier_8

        # Streak
        mean_epa = epa_values.mean()
        streak = 0
        for val in reversed(epa_values):
            if (val > mean_epa and streak >= 0) or (val <= mean_epa and streak <= 0):
                streak += 1 if val > mean_epa else -1
            else:
                break
        features['streak_length'] = streak

        # Consistency
        features['epa_consistency_cv'] = epa_values.std() / abs(epa_values.mean()) if abs(epa_values.mean()) > 0.01 else 0

    # Interaction features (multiplicative interactions between top features)
    if 'epa_mean_avg' in features and 'opp_adj_success_rate_avg' in features:
        features['epa_x_success'] = features['epa_mean_avg'] * features['opp_adj_success_rate_avg']

    if 'epa_mean_avg' in features and 'epa_per_play_avg' in features:
        features['epa_mean_x_per_play'] = features['epa_mean_avg'] * features['epa_per_play_avg']

    # Games played (sample size indicator)
    features['games_played'] = len(player_games)
    features['recent_games'] = len(recent)

    return features


# Build dataset
print("\n" + "="*80)
print("FEATURE ENGINEERING")
print("="*80)

samples = []

for player_id in qb_data['passer_player_id'].unique():
    player_games = qb_data[qb_data['passer_player_id'] == player_id].copy()

    if len(player_games) < 12 + 16:
        continue

    for i in range(12, len(player_games) - 16):
        recent_12 = player_games.iloc[i-12:i]

        # Engineer features
        features = engineer_features(recent_12, window=12)

        if features is not None:
            # Target: next 16 games average
            target = player_games.iloc[i:i+16]['opp_adj_base_epa'].mean()

            if not np.isnan(target):
                features['target'] = target
                features['player_id'] = player_id
                features['game_index'] = i
                samples.append(features)

df = pd.DataFrame(samples)

print(f"\nGenerated {len(df)} training samples")
print(f"Feature count: {len([col for col in df.columns if col not in ['target', 'player_id', 'game_index']])}")

# Remove any infinite or NaN features
feature_cols = [col for col in df.columns if col not in ['target', 'player_id', 'game_index']]
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

print(f"After removing NaN/inf: {len(df)} samples")

# Time-aware train/test split (use earlier games for train, later for test)
# Sort by game_index to maintain temporal order
df = df.sort_values('game_index')

# Use 80% for train+validation, 20% for final test
split_idx = int(len(df) * 0.8)
train_val_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

print(f"\nTrain+Val: {len(train_val_df)} samples")
print(f"Test: {len(test_df)} samples")

X_train_val = train_val_df[feature_cols].values
y_train_val = train_val_df['target'].values

X_test = test_df[feature_cols].values
y_test = test_df['target'].values

# Standardize features
scaler = StandardScaler()
X_train_val_scaled = scaler.fit_transform(X_train_val)
X_test_scaled = scaler.transform(X_test)


# Random Forest with extensive hyperparameter tuning
print("\n" + "="*80)
print("TRAINING RANDOM FOREST")
print("="*80)

rf_param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [5, 10, 15, 20, 25, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7]
}

print(f"\nRandom Forest hyperparameter grid:")
print(f"  Total combinations: {np.prod([len(v) for v in rf_param_grid.values()])}")

# Use TimeSeriesSplit for cross-validation (respects temporal order)
tscv = TimeSeriesSplit(n_splits=5)

print(f"  Using TimeSeriesSplit with 5 folds")
print(f"  Running RandomizedSearchCV with 100 iterations...")

rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    rf_param_grid,
    n_iter=100,  # Try 100 random combinations
    cv=tscv,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

rf_search.fit(X_train_val_scaled, y_train_val)

print(f"\nBest Random Forest parameters:")
for param, value in rf_search.best_params_.items():
    print(f"  {param}: {value}")

# Evaluate on test set
rf_pred = rf_search.predict(X_test_scaled)
rf_corr, rf_p = stats.spearmanr(rf_pred, y_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_mae = mean_absolute_error(y_test, rf_pred)

print(f"\nRandom Forest Test Performance:")
print(f"  Spearman r: {rf_corr:.4f} (p={rf_p:.6f})")
print(f"  RMSE: {rf_rmse:.3f}")
print(f"  MAE: {rf_mae:.3f}")

# Feature importance
rf_model = rf_search.best_estimator_
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 10 most important features:")
for i, row in feature_importance.head(10).iterrows():
    print(f"  {i+1}. {row['feature']:<30} {row['importance']:.4f}")


# XGBoost (if available)
if HAS_XGBOOST:
    print("\n" + "="*80)
    print("TRAINING XGBOOST")
    print("="*80)

    xgb_param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 5, 7, 9, 12],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0, 0.1, 0.2, 0.3]
    }

    print(f"\nXGBoost hyperparameter grid:")
    print(f"  Total combinations: {np.prod([len(v) for v in xgb_param_grid.values()])}")

    print(f"  Running RandomizedSearchCV with 100 iterations...")

    xgb_search = RandomizedSearchCV(
        xgb.XGBRegressor(random_state=42, n_jobs=-1, tree_method='hist'),
        xgb_param_grid,
        n_iter=100,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    xgb_search.fit(X_train_val_scaled, y_train_val)

    print(f"\nBest XGBoost parameters:")
    for param, value in xgb_search.best_params_.items():
        print(f"  {param}: {value}")

    # Evaluate on test set
    xgb_pred = xgb_search.predict(X_test_scaled)
    xgb_corr, xgb_p = stats.spearmanr(xgb_pred, y_test)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    xgb_mae = mean_absolute_error(y_test, xgb_pred)

    print(f"\nXGBoost Test Performance:")
    print(f"  Spearman r: {xgb_corr:.4f} (p={xgb_p:.6f})")
    print(f"  RMSE: {xgb_rmse:.3f}")
    print(f"  MAE: {xgb_mae:.3f}")

    # Feature importance
    xgb_model = xgb_search.best_estimator_
    xgb_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTop 10 most important features:")
    for i, row in xgb_importance.head(10).iterrows():
        print(f"  {i+1}. {row['feature']:<30} {row['importance']:.4f}")


# Ensemble (average of RF and XGBoost)
if HAS_XGBOOST:
    print("\n" + "="*80)
    print("ENSEMBLE PREDICTION")
    print("="*80)

    ensemble_pred = (rf_pred + xgb_pred) / 2
    ensemble_corr, ensemble_p = stats.spearmanr(ensemble_pred, y_test)
    ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)

    print(f"\nEnsemble (RF + XGBoost average) Test Performance:")
    print(f"  Spearman r: {ensemble_corr:.4f} (p={ensemble_p:.6f})")
    print(f"  RMSE: {ensemble_rmse:.3f}")
    print(f"  MAE: {ensemble_mae:.3f}")


# Final comparison
print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

BASELINE_CORR = 0.3853

results = [
    {'model': 'Baseline (linear multi-feature)', 'correlation': BASELINE_CORR, 'rmse': 0.151, 'mae': 0.116},
    {'model': 'Random Forest', 'correlation': rf_corr, 'rmse': rf_rmse, 'mae': rf_mae}
]

if HAS_XGBOOST:
    results.append({'model': 'XGBoost', 'correlation': xgb_corr, 'rmse': xgb_rmse, 'mae': xgb_mae})
    results.append({'model': 'Ensemble (RF + XGBoost)', 'correlation': ensemble_corr, 'rmse': ensemble_rmse, 'mae': ensemble_mae})

results_df = pd.DataFrame(results).sort_values('correlation', ascending=False)

print(f"\n{'Model':<35} {'Correlation':>12} {'RMSE':>10} {'MAE':>10}")
print("-"*75)
for _, row in results_df.iterrows():
    print(f"{row['model']:<35} {row['correlation']:>12.4f} {row['rmse']:>10.3f} {row['mae']:>10.3f}")

best_model = results_df.iloc[0]
improvement = (best_model['correlation'] - BASELINE_CORR) / BASELINE_CORR * 100

print(f"\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\nBaseline: r={BASELINE_CORR:.4f}")
print(f"Best ML Model: {best_model['model']}")
print(f"  Correlation: r={best_model['correlation']:.4f}")
print(f"  Improvement: {improvement:+.2f}%")

TARGET_CORR = 0.40
gap = TARGET_CORR - best_model['correlation']

print(f"\nTarget: r > {TARGET_CORR:.2f}")
if best_model['correlation'] >= TARGET_CORR:
    print(f"SUCCESS: Target achieved! (r={best_model['correlation']:.4f})")
else:
    print(f"Gap remaining: {gap:+.4f} ({gap/TARGET_CORR*100:.1f}%)")

# Save models
print("\n" + "="*80)
print("SAVING MODELS")
print("="*80)

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

joblib.dump(rf_search.best_estimator_, models_dir / 'random_forest_best.pkl')
joblib.dump(scaler, models_dir / 'feature_scaler.pkl')
joblib.dump(feature_cols, models_dir / 'feature_names.pkl')

print(f"\nSaved models:")
print(f"  Random Forest: models/random_forest_best.pkl")
print(f"  Scaler: models/feature_scaler.pkl")
print(f"  Features: models/feature_names.pkl")

if HAS_XGBOOST:
    joblib.dump(xgb_search.best_estimator_, models_dir / 'xgboost_best.pkl')
    print(f"  XGBoost: models/xgboost_best.pkl")

print("\nModels can be loaded with joblib.load() for production use")
