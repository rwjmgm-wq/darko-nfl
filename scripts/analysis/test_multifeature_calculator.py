"""
Test Multi-Feature Rating Calculator

Validates that the new calculator achieves expected performance:
- r=0.3853 correlation with next 16 games
- +9.1% improvement over baseline
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pandas as pd
import numpy as np
from scipy import stats
from features.multifeature_rating_calculator import MultiFeatureRatingCalculator

# Load data
DATA_DIR = Path("data/production")
qb_files = sorted(DATA_DIR.glob("leaf_v2_game_by_game_*.csv"))
DATA_FILE = qb_files[-1]

print("="*80)
print("TESTING MULTI-FEATURE RATING CALCULATOR")
print("="*80)
print(f"\nLoading data from: {DATA_FILE.name}")

qb_data = pd.read_csv(DATA_FILE)
qb_data = qb_data.sort_values(['passer_player_id', 'game_number'])

print(f"Loaded {len(qb_data)} games for {qb_data['passer_player_id'].nunique()} QBs")

# Initialize calculator
calculator = MultiFeatureRatingCalculator(
    window_size=12,
    decay_rate=0.10,
    outlier_percentile=95.0,
    volatility_penalty=0.3,
    apply_volatility_adjustment=True
)

print("\n" + "="*80)
print("CALCULATOR CONFIGURATION")
print("="*80)

config = calculator.get_configuration_summary()
print(f"\nFeatures ({len(config['features'])}):")
for i, (feature, weight) in enumerate(zip(config['features'], config['feature_weights']), 1):
    print(f"  {i}. {feature:<25} weight={weight:.4f}")

print(f"\nWindow configuration:")
print(f"  Size: {config['window_size']} games")
print(f"  Decay rate: {config['decay_rate']}")
print(f"  Outlier filtering: {config['outlier_percentile']}th percentile")

print(f"\nGame weights (exponential decay):")
print(f"  Newest 3 games:  {', '.join([f'{w:.4f}' for w in config['game_weights_first_3']])}")
print(f"  Oldest 3 games:  {', '.join([f'{w:.4f}' for w in config['game_weights_last_3']])}")

print(f"\nVolatility adjustment:")
print(f"  Enabled: {config['apply_volatility_adjustment']}")
print(f"  Penalty factor: {config['volatility_penalty']}")

# Test prediction performance
print("\n" + "="*80)
print("TESTING PREDICTIVE PERFORMANCE")
print("="*80)
print("\nPredicting next 16 games from last 12 games using multi-feature rating...")

predictions = []

for player_id in qb_data['passer_player_id'].unique():
    player_games = qb_data[qb_data['passer_player_id'] == player_id].copy()

    if len(player_games) < 12 + 16:
        continue

    for i in range(12, len(player_games) - 16):
        # Use last 12 games to calculate predictor
        recent_12_games = player_games.iloc[i-12:i]

        predictor = calculator.calculate_multifeature_rating(recent_12_games)

        # Target: next 16 games average
        next_16_epa = player_games.iloc[i:i+16]['opp_adj_base_epa'].mean()

        if not np.isnan(predictor) and not np.isnan(next_16_epa):
            predictions.append({
                'predictor': predictor,
                'target': next_16_epa
            })

pred_df = pd.DataFrame(predictions)

print(f"\nGenerated {len(pred_df)} predictions")

# Calculate performance metrics
corr, p_value = stats.spearmanr(pred_df['predictor'], pred_df['target'])
rmse = np.sqrt(((pred_df['predictor'] - pred_df['target']) ** 2).mean())
mae = np.abs(pred_df['predictor'] - pred_df['target']).mean()

print(f"\n" + "="*80)
print("PERFORMANCE RESULTS")
print("="*80)

print(f"\nSpearman Correlation: r={corr:.4f} (p={p_value:.6f})")
print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")

# Compare to baseline
BASELINE_CORR = 0.3533
TARGET_CORR = 0.3853

improvement = (corr - BASELINE_CORR) / BASELINE_CORR * 100

print(f"\n" + "="*80)
print("COMPARISON TO BASELINE")
print("="*80)

print(f"\nBaseline (single feature): r={BASELINE_CORR:.4f}")
print(f"Multi-feature calculator:  r={corr:.4f}")
print(f"Improvement: {improvement:+.1f}%")

print(f"\nTarget performance: r={TARGET_CORR:.4f}")
if abs(corr - TARGET_CORR) < 0.01:
    print(f"SUCCESS: Achieved target within 1% (r={corr:.4f})")
elif corr >= TARGET_CORR:
    print(f"SUCCESS: Exceeded target! (r={corr:.4f} vs {TARGET_CORR:.4f})")
else:
    diff = TARGET_CORR - corr
    diff_pct = diff / TARGET_CORR * 100
    print(f"BELOW TARGET: Short by {diff:.4f} ({diff_pct:.1f}%)")

# Test on specific QBs
print(f"\n" + "="*80)
print("SAMPLE QB RATINGS")
print("="*80)

sample_qbs = ['Patrick Mahomes', 'Joe Burrow', 'Josh Allen', 'Sam Darnold', 'Trevor Lawrence']
print(f"\nCalculating ratings for sample QBs...")

for qb_name in sample_qbs:
    qb_games = qb_data[qb_data['passer_player_name'].str.contains(qb_name, na=False, case=False)]

    if len(qb_games) == 0:
        continue

    rating_info = calculator.calculate_multifeature_rating(qb_games, return_components=True)

    if isinstance(rating_info, dict):
        print(f"\n{qb_name}:")
        print(f"  Games: {len(qb_games)}")
        print(f"  Multi-feature rating: {rating_info['rating']:+.3f}")
        print(f"  Combined (pre-volatility): {rating_info['combined_rating']:+.3f}")
        print(f"  Volatility: {rating_info['volatility']:.3f}")
        print(f"  Volatility penalty: {rating_info['volatility_penalty']:+.3f}")

        # Show top 2 contributing features
        feature_contributions = []
        for feature in calculator.features:
            contrib = rating_info.get(f'{feature}_rating', np.nan)
            if not np.isnan(contrib):
                feature_contributions.append((feature, contrib))

        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        print(f"  Top features:")
        for feat, val in feature_contributions[:2]:
            print(f"    {feat}: {val:+.3f}")

print(f"\n" + "="*80)
print("TEST COMPLETE")
print("="*80)

if abs(corr - TARGET_CORR) <= 0.01 or corr >= TARGET_CORR:
    print("\nSUCCESS: Multi-feature calculator validated!")
    print(f"Achieved r={corr:.4f} (target: r={TARGET_CORR:.4f})")
else:
    print("\nWARNING: Performance below target")
    print("Check feature availability and data quality")
