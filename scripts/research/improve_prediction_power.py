"""
Research: How to Improve QB Performance Prediction (r > 0.35)

Current: r=0.3533 using only opponent-adjusted EPA
Goal: Identify additional signals that improve prediction of next 16 games

Potential improvements:
1. Additional performance metrics
2. Context/situation features
3. Supporting cast quality
4. Trend indicators
5. Multi-model ensemble
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
print("IMPROVING QB PERFORMANCE PREDICTION")
print("="*80)
print("\nCurrent baseline: r=0.3533 using opponent-adjusted EPA only")
print("\nTesting additional features available in our data...")

# Check what columns we have
print("\n" + "="*80)
print("AVAILABLE DATA COLUMNS")
print("="*80)
print("\nColumns in our dataset:")
for col in sorted(qb_data.columns):
    print(f"  - {col}")

# Key metrics to explore (from actual available columns)
candidate_features = [
    'opp_adj_base_epa',      # Current baseline
    'opp_adj_epa',
    'opp_adj_cpoe',
    'opp_adj_success_rate',
    'base_epa',
    'epa_mean',
    'epa_per_play',
    'epa_per_play_ctx_adj',
    'cpoe_mean',
    'success_rate',
    'success_rate_ctx_adj',
    'air_yards_per_attempt',
    'yards_per_attempt',
    'completion_pct',
    'td_rate',
    'int_rate',
    'sack_rate',
    'qb_epa_mean',
]

# Filter to features that exist
available_features = [f for f in candidate_features if f in qb_data.columns]

print(f"\n" + "="*80)
print(f"AVAILABLE FEATURES FOR TESTING ({len(available_features)})")
print("="*80)
for feat in available_features:
    print(f"  - {feat}")

# Test each feature individually
print(f"\n" + "="*80)
print("INDIVIDUAL FEATURE PREDICTIVE POWER")
print("="*80)
print("\nTesting each feature's correlation with next 16 games performance...")

feature_results = []

for feature in available_features:
    predictions = []

    for player_id in qb_data['passer_player_id'].unique():
        player_games = qb_data[qb_data['passer_player_id'] == player_id].copy()

        if len(player_games) < 12 + 16:
            continue

        for i in range(12, len(player_games) - 16):
            # Predictor: simple average of last 12 games for this feature
            recent_12 = player_games.iloc[i-12:i][feature].mean()

            # Target: next 16 games average of opp_adj_base_epa
            next_16_epa = player_games.iloc[i:i+16]['opp_adj_base_epa'].mean()

            if not np.isnan(recent_12) and not np.isnan(next_16_epa):
                predictions.append({
                    'predictor': recent_12,
                    'target': next_16_epa
                })

    if len(predictions) > 50:
        pred_df = pd.DataFrame(predictions)
        corr, p_value = stats.spearmanr(pred_df['predictor'], pred_df['target'])

        feature_results.append({
            'feature': feature,
            'correlation': corr,
            'p_value': p_value,
            'n': len(pred_df)
        })

results_df = pd.DataFrame(feature_results).sort_values('correlation', ascending=False)

print(f"\n{'Feature':<30} {'Correlation':>12} {'P-value':>10} {'N':>8}")
print("-"*70)
for _, row in results_df.iterrows():
    print(f"{row['feature']:<30} {row['correlation']:>12.4f} {row['p_value']:>10.4f} {row['n']:>8.0f}")

# Find features better than baseline
baseline_corr = results_df[results_df['feature'] == 'opp_adj_base_epa']['correlation'].iloc[0]
print(f"\n" + "="*80)
print("FEATURES BETTER THAN BASELINE")
print("="*80)
print(f"\nBaseline (opp_adj_base_epa): r={baseline_corr:.4f}")

better_features = results_df[results_df['correlation'] > baseline_corr]
if len(better_features) > 0:
    print(f"\nFeatures with BETTER predictive power:")
    for _, row in better_features.iterrows():
        improvement = (row['correlation'] - baseline_corr) / baseline_corr * 100
        print(f"  {row['feature']:<30} r={row['correlation']:.4f} (+{improvement:.1f}%)")
else:
    print("\nNo individual feature beats opp_adj_base_epa alone.")

# Test multi-feature combinations
print(f"\n" + "="*80)
print("MULTI-FEATURE COMBINATIONS")
print("="*80)
print("\nTesting if combining features improves prediction...")

# Test top 3 features combined
top_3_features = results_df.head(3)['feature'].tolist()
print(f"\nTop 3 features: {top_3_features}")

# Simple ensemble: average of predictions from top 3 features
combo_predictions = []

for player_id in qb_data['passer_player_id'].unique():
    player_games = qb_data[qb_data['passer_player_id'] == player_id].copy()

    if len(player_games) < 12 + 16:
        continue

    for i in range(12, len(player_games) - 16):
        predictors = []

        for feature in top_3_features:
            recent_12 = player_games.iloc[i-12:i][feature].mean()
            if not np.isnan(recent_12):
                # Standardize to same scale as EPA
                std_predictor = (recent_12 - qb_data[feature].mean()) / qb_data[feature].std()
                predictors.append(std_predictor * qb_data['opp_adj_base_epa'].std() + qb_data['opp_adj_base_epa'].mean())

        if len(predictors) == len(top_3_features):
            combo_predictor = np.mean(predictors)
            next_16_epa = player_games.iloc[i:i+16]['opp_adj_base_epa'].mean()

            if not np.isnan(next_16_epa):
                combo_predictions.append({
                    'predictor': combo_predictor,
                    'target': next_16_epa
                })

if len(combo_predictions) > 50:
    combo_df = pd.DataFrame(combo_predictions)
    combo_corr, combo_p = stats.spearmanr(combo_df['predictor'], combo_df['target'])

    print(f"\nCombination of top 3 features:")
    print(f"  Correlation: r={combo_corr:.4f}")
    print(f"  Baseline: r={baseline_corr:.4f}")

    if combo_corr > baseline_corr:
        improvement = (combo_corr - baseline_corr) / baseline_corr * 100
        print(f"  Improvement: +{improvement:.1f}%")
    else:
        print(f"  No improvement over baseline")

# Test adding volatility/consistency metrics
print(f"\n" + "="*80)
print("CONSISTENCY/VOLATILITY FEATURES")
print("="*80)
print("\nTesting if recent performance stability predicts future...")

volatility_predictions = []

for player_id in qb_data['passer_player_id'].unique():
    player_games = qb_data[qb_data['passer_player_id'] == player_id].copy()

    if len(player_games) < 12 + 16:
        continue

    for i in range(12, len(player_games) - 16):
        recent_12 = player_games.iloc[i-12:i]['opp_adj_base_epa'].values

        # Predictors
        mean_epa = recent_12.mean()
        std_epa = recent_12.std()

        # Target
        next_16_epa = player_games.iloc[i:i+16]['opp_adj_base_epa'].mean()

        if not np.isnan(mean_epa) and not np.isnan(std_epa) and not np.isnan(next_16_epa):
            volatility_predictions.append({
                'mean': mean_epa,
                'std': std_epa,
                'target': next_16_epa
            })

if len(volatility_predictions) > 50:
    vol_df = pd.DataFrame(volatility_predictions)

    # Test if low volatility QBs are more predictable
    corr_mean, _ = stats.spearmanr(vol_df['mean'], vol_df['target'])
    corr_std, _ = stats.spearmanr(vol_df['std'], vol_df['target'])

    print(f"\nMean EPA correlation: r={corr_mean:.4f}")
    print(f"Std Dev EPA correlation: r={corr_std:.4f}")

    # Combined predictor: mean adjusted for volatility
    vol_df['adjusted'] = vol_df['mean'] - 0.5 * vol_df['std']  # Penalize volatility
    corr_adj, _ = stats.spearmanr(vol_df['adjusted'], vol_df['target'])

    print(f"Volatility-adjusted EPA: r={corr_adj:.4f}")

    if corr_adj > corr_mean:
        improvement = (corr_adj - corr_mean) / corr_mean * 100
        print(f"  => Adjusting for volatility improves by {improvement:.1f}%")

print(f"\n" + "="*80)
print("RECOMMENDATIONS FOR IMPROVEMENT")
print("="*80)

print("""
Based on available data in our dataset, here are ways to improve r:

1. MULTI-FACTOR MODELS
   - Combine top predictive features
   - Weight them by their individual correlation
   - Test ensemble methods

2. CONTEXTUAL ADJUSTMENTS
   - Opponent quality trends (getting easier/harder schedule)
   - Home/away splits
   - Weather/climate factors

3. SUPPORTING CAST METRICS (if available)
   - O-line quality ratings
   - Receiver quality/separation
   - Running game support
   - Team play-calling tendencies

4. ADVANCED FEATURES
   - EPA trend (improving vs declining)
   - Consistency score (penalize volatility)
   - Situational splits (3rd down, red zone, pressure)

5. MACHINE LEARNING
   - Random forest to find non-linear relationships
   - Feature importance analysis
   - Cross-validation for robustness

REALISTIC EXPECTATIONS:
- QB performance is inherently noisy (injuries, game script, luck)
- r=0.35 is actually quite good for 16-game predictions
- Best sports prediction models typically max out at r=0.4-0.5
- Diminishing returns: getting from 0.35 to 0.40 requires much more effort
""")

print(f"\n" + "="*80)
print("NEXT STEPS")
print("="*80)

best_single_feature = results_df.iloc[0]
print(f"""
1. Investigate why '{best_single_feature['feature']}' has r={best_single_feature['correlation']:.4f}
2. Test optimal weighting of top 3-5 features
3. Add external data sources (injuries, roster changes, coaching)
4. Build ensemble model combining multiple signals
5. Validate on holdout data to ensure no overfitting
""")
