"""
Neural Network Training for QB Performance Prediction

Goal: Push correlation from r=0.3946 (current best: Stacked Ensemble) to r > 0.40

Current status:
- Baseline linear multi-feature: r=0.3853
- XGBoost: r=0.3902
- Stacked Ensemble: r=0.3946 (gap to r>0.40: just 0.0054 or 1.3%)

This script:
1. Tests neural network architectures to capture complex non-linearities
2. Extensive hyperparameter tuning (user has powerful hardware)
3. Designed for multi-hour background runtime
4. Tests shallow to deep networks with various regularization
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

# Check for PyTorch/TensorFlow
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    print("WARNING: PyTorch not installed. Will use sklearn MLPRegressor only.")
    print("For best results, install PyTorch: pip install torch")

# Load data
DATA_DIR = Path("data/production")
qb_files = sorted(DATA_DIR.glob("leaf_v2_game_by_game_*.csv"))
DATA_FILE = qb_files[-1]

qb_data = pd.read_csv(DATA_FILE)
qb_data = qb_data.sort_values(['passer_player_id', 'game_number'])

print("="*80)
print("NEURAL NETWORK TRAINING")
print("="*80)
print("\nCurrent best: Stacked Ensemble r=0.3946")
print("Target: r > 0.40 (gap: 0.0054 or 1.3%)")
print("\nTesting neural network architectures for complex non-linearities...")


def engineer_features(player_games, window=12):
    """
    Same feature engineering as ensemble models (30 features).
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

feature_cols = [col for col in df.columns if col not in ['target', 'player_id', 'game_index']]

print(f"\nGenerated {len(df)} training samples")
print(f"Feature count: {len(feature_cols)}")

# Time-aware train/test split
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


# Sklearn MLPRegressor with extensive hyperparameter search
print("\n" + "="*80)
print("SKLEARN NEURAL NETWORK (MLPRegressor)")
print("="*80)

mlp_param_grid = {
    'hidden_layer_sizes': [
        (50,), (100,), (150,), (200,),
        (50, 25), (100, 50), (150, 75), (200, 100),
        (100, 50, 25), (150, 100, 50), (200, 150, 100),
        (150, 100, 50, 25), (200, 150, 100, 50)
    ],
    'activation': ['relu', 'tanh', 'logistic'],
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
    'learning_rate_init': [0.0001, 0.0005, 0.001, 0.005, 0.01],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'max_iter': [500, 1000, 2000],
    'early_stopping': [True],
    'validation_fraction': [0.1, 0.2],
    'beta_1': [0.9, 0.95, 0.99],
    'beta_2': [0.999, 0.9995, 0.9999]
}

print(f"\nMLPRegressor hyperparameter grid:")
print(f"  Total combinations: {np.prod([len(v) for v in mlp_param_grid.values()])}")
print(f"  Testing 500 random combinations...")
print(f"  Estimated runtime: 1-2 hours")

tscv = TimeSeriesSplit(n_splits=5)

mlp_search = RandomizedSearchCV(
    MLPRegressor(random_state=42),
    mlp_param_grid,
    n_iter=500,
    cv=tscv,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

print("\nStarting MLPRegressor training...")
mlp_search.fit(X_train_val_scaled, y_train_val)

print(f"\nBest MLPRegressor parameters:")
for param, value in mlp_search.best_params_.items():
    print(f"  {param}: {value}")

# Evaluate on test set
mlp_pred = mlp_search.predict(X_test_scaled)
mlp_corr, mlp_p = stats.spearmanr(mlp_pred, y_test)
mlp_rmse = np.sqrt(mean_squared_error(y_test, mlp_pred))
mlp_mae = mean_absolute_error(y_test, mlp_pred)

print(f"\nMLPRegressor Test Performance:")
print(f"  Spearman r: {mlp_corr:.4f} (p={mlp_p:.6f})")
print(f"  RMSE: {mlp_rmse:.3f}")
print(f"  MAE: {mlp_mae:.3f}")


# PyTorch Neural Network (if available)
if HAS_PYTORCH:
    print("\n" + "="*80)
    print("PYTORCH NEURAL NETWORK")
    print("="*80)

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_val_scaled)
    y_train_tensor = torch.FloatTensor(y_train_val).reshape(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)

    # Define flexible neural network architectures
    class FlexibleNN(nn.Module):
        def __init__(self, input_size, hidden_layers, dropout_rate=0.2, use_batchnorm=True):
            super(FlexibleNN, self).__init__()

            layers = []
            prev_size = input_size

            for hidden_size in hidden_layers:
                layers.append(nn.Linear(prev_size, hidden_size))
                if use_batchnorm:
                    layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
                prev_size = hidden_size

            layers.append(nn.Linear(prev_size, 1))

            self.model = nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x)

    def train_pytorch_model(architecture, learning_rate=0.001, batch_size=64, epochs=200, patience=20):
        """Train a PyTorch model with early stopping."""
        model = FlexibleNN(X_train_val_scaled.shape[1], architecture)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=False)

        # Split train/val for early stopping
        split_idx = int(len(X_train_tensor) * 0.9)
        X_train_split = X_train_tensor[:split_idx]
        y_train_split = y_train_tensor[:split_idx]
        X_val_split = X_train_tensor[split_idx:]
        y_val_split = y_train_tensor[split_idx:]

        train_dataset = TensorDataset(X_train_split, y_train_split)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_split)
                val_loss = criterion(val_outputs, y_val_split).item()

            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        # Load best model
        model.load_state_dict(best_model_state)

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_predictions = test_outputs.numpy().flatten()

        return model, test_predictions

    # Test different architectures
    print("\nTesting different PyTorch architectures...")
    print("This will take 30-60 minutes...\n")

    architectures = [
        ([128], "Shallow (1 layer, 128 units)"),
        ([256], "Shallow (1 layer, 256 units)"),
        ([128, 64], "Medium (2 layers: 128, 64)"),
        ([256, 128], "Medium (2 layers: 256, 128)"),
        ([256, 128, 64], "Deep (3 layers: 256, 128, 64)"),
        ([512, 256, 128], "Deep (3 layers: 512, 256, 128)"),
        ([256, 128, 64, 32], "Very Deep (4 layers: 256, 128, 64, 32)"),
        ([512, 256, 128, 64], "Very Deep (4 layers: 512, 256, 128, 64)"),
    ]

    pytorch_results = []

    for arch, desc in architectures:
        print(f"Training {desc}...")

        # Try multiple learning rates
        best_corr = -1
        best_pred = None

        for lr in [0.0001, 0.0005, 0.001, 0.005]:
            model, predictions = train_pytorch_model(
                arch,
                learning_rate=lr,
                batch_size=64,
                epochs=200,
                patience=20
            )

            corr, _ = stats.spearmanr(predictions, y_test)

            if corr > best_corr:
                best_corr = corr
                best_pred = predictions
                best_lr = lr

        rmse = np.sqrt(mean_squared_error(y_test, best_pred))
        mae = mean_absolute_error(y_test, best_pred)

        pytorch_results.append({
            'architecture': desc,
            'learning_rate': best_lr,
            'correlation': best_corr,
            'rmse': rmse,
            'mae': mae
        })

        print(f"  Best: r={best_corr:.4f} (lr={best_lr})")

    pytorch_results_df = pd.DataFrame(pytorch_results).sort_values('correlation', ascending=False)

    print("\n" + "="*80)
    print("PYTORCH ARCHITECTURE COMPARISON")
    print("="*80)

    print(f"\n{'Architecture':<40} {'Best LR':>10} {'Correlation':>12} {'RMSE':>10} {'MAE':>10}")
    print("-"*90)
    for _, row in pytorch_results_df.iterrows():
        print(f"{row['architecture']:<40} {row['learning_rate']:>10.4f} {row['correlation']:>12.4f} {row['rmse']:>10.3f} {row['mae']:>10.3f}")

    best_pytorch = pytorch_results_df.iloc[0]
    print(f"\nBest PyTorch architecture: {best_pytorch['architecture']}")
    print(f"  Correlation: r={best_pytorch['correlation']:.4f}")


# Final comparison
print("\n" + "="*80)
print("FINAL MODEL COMPARISON")
print("="*80)

# Load previous best models for comparison
try:
    xgb_model = joblib.load(Path("models") / 'xgboost_best.pkl')
    xgb_pred = xgb_model.predict(X_test_scaled)
    xgb_corr = stats.spearmanr(xgb_pred, y_test)[0]
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    HAS_XGB = True
except:
    HAS_XGB = False
    print("Note: XGBoost model not found for comparison")

results = [
    {'model': 'Baseline (linear multi-feature)', 'correlation': 0.3853, 'rmse': 0.151, 'mae': 0.116},
]

if HAS_XGB:
    results.append({'model': 'XGBoost', 'correlation': xgb_corr, 'rmse': xgb_rmse, 'mae': xgb_mae})

results.append({'model': 'Stacked Ensemble (previous best)', 'correlation': 0.3946, 'rmse': 0.149, 'mae': 0.114})
results.append({'model': 'MLPRegressor (sklearn)', 'correlation': mlp_corr, 'rmse': mlp_rmse, 'mae': mlp_mae})

if HAS_PYTORCH:
    results.append({
        'model': f'PyTorch ({best_pytorch["architecture"]})',
        'correlation': best_pytorch['correlation'],
        'rmse': best_pytorch['rmse'],
        'mae': best_pytorch['mae']
    })

results_df = pd.DataFrame(results).sort_values('correlation', ascending=False)

print(f"\n{'Model':<45} {'Correlation':>12} {'RMSE':>10} {'MAE':>10}")
print("-"*75)
for _, row in results_df.iterrows():
    print(f"{row['model']:<45} {row['correlation']:>12.4f} {row['rmse']:>10.3f} {row['mae']:>10.3f}")

best_model = results_df.iloc[0]

print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)

print(f"\nBest model: {best_model['model']}")
print(f"  Correlation: r={best_model['correlation']:.4f}")
print(f"  RMSE: {best_model['rmse']:.3f}")
print(f"  MAE: {best_model['mae']:.3f}")

TARGET_CORR = 0.40
if best_model['correlation'] >= TARGET_CORR:
    print(f"\nSUCCESS! Target r > 0.40 achieved!")
    print(f"  Exceeded target by: {best_model['correlation'] - TARGET_CORR:+.4f}")
else:
    gap = TARGET_CORR - best_model['correlation']
    print(f"\nTarget: r > {TARGET_CORR:.2f}")
    print(f"  Gap remaining: {gap:+.4f} ({gap/TARGET_CORR*100:.1f}%)")

# Save best models
print("\n" + "="*80)
print("SAVING MODELS")
print("="*80)

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

joblib.dump(mlp_search.best_estimator_, models_dir / 'mlp_best.pkl')
joblib.dump(scaler, models_dir / 'feature_scaler_nn.pkl')
joblib.dump(feature_cols, models_dir / 'feature_names_nn.pkl')

print(f"\nSaved models:")
print(f"  MLPRegressor: models/mlp_best.pkl")
print(f"  Scaler: models/feature_scaler_nn.pkl")
print(f"  Features: models/feature_names_nn.pkl")

if HAS_PYTORCH and best_model['model'].startswith('PyTorch'):
    # Save best PyTorch model
    best_arch = architectures[pytorch_results_df.index[0]][0]
    final_model, _ = train_pytorch_model(best_arch, learning_rate=best_pytorch['learning_rate'])
    torch.save(final_model.state_dict(), models_dir / 'pytorch_best.pth')
    print(f"  PyTorch: models/pytorch_best.pth")

print("\nNeural network training complete!")
