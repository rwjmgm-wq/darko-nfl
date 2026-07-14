"""
Empirical Optimization of QB Rating Weights

Tests different configurations to maximize predictive power:
1. Different weighting schemes (uniform, linear, exponential, custom)
2. Outlier filtering (winsorization at different percentiles)
3. Combinations of both

Goal: Find weights that best predict next 16 games performance
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from itertools import product

# Load QB data
DATA_DIR = Path("data/production")
qb_files = sorted(DATA_DIR.glob("leaf_v2_game_by_game_*.csv"))
DATA_FILE = qb_files[-1]

qb_data = pd.read_csv(DATA_FILE)
qb_data = qb_data.rename(columns={
    'passer_player_id': 'player_id',
    'passer_player_name': 'player_name',
    'opp_adj_base_epa': 'rating'
})

qb_data = qb_data.sort_values(['player_id', 'game_number'])

print("="*80)
print("EMPIRICAL OPTIMIZATION: QB RATING WEIGHTS")
print("="*80)

def apply_outlier_filter(values, percentile=None):
    """Apply winsorization to extreme values."""
    if percentile is None:
        return values

    lower = np.percentile(values, 100 - percentile)
    upper = np.percentile(values, percentile)
    return np.clip(values, lower, upper)

def calculate_weighted_form(games, weights, outlier_percentile=None):
    """Calculate weighted average with optional outlier filtering."""
    if len(games) < len(weights):
        n_games = len(games)
        active_weights = weights[-n_games:]
        if active_weights.sum() > 0:
            active_weights = active_weights / active_weights.sum()
        else:
            return np.nan
        filtered_games = apply_outlier_filter(games, outlier_percentile)
        return np.average(filtered_games, weights=active_weights)
    else:
        recent_games = games[-len(weights):]
        filtered_games = apply_outlier_filter(recent_games, outlier_percentile)
        return np.average(filtered_games, weights=weights)

def generate_weight_schemes(window=12):
    """Generate different weighting schemes to test."""
    schemes = {}

    # 1. Uniform (simple average)
    schemes['uniform'] = np.ones(window) / window

    # 2. Linear decay (newest = X%, oldest = Y%)
    for newest_pct in [0.25, 0.30, 0.35, 0.40, 0.45]:
        linear = np.linspace(0.01, newest_pct, window)
        linear = linear / linear.sum()
        schemes[f'linear_{int(newest_pct*100)}'] = linear[::-1]

    # 3. Exponential decay (different decay rates)
    for decay in [0.10, 0.15, 0.20, 0.25, 0.30]:
        expo = np.exp(-decay * np.arange(window)[::-1])
        expo = expo / expo.sum()
        schemes[f'expo_{int(decay*100)}'] = expo

    # 4. Step function (recent N games weighted heavily)
    for cutoff in [4, 6, 8]:
        step = np.ones(window)
        step[-cutoff:] = 3.0  # Triple weight for recent games
        step = step / step.sum()
        schemes[f'step_{cutoff}'] = step

    # 5. Proposed scheme (35%, 25%, 17%, 12%, 7%, 3%, 1%)
    proposed = np.array([0.35, 0.25, 0.17, 0.12, 0.07, 0.03, 0.01] + [0.0] * (window - 7))
    proposed = proposed / proposed.sum()
    schemes['proposed'] = proposed

    return schemes

# Generate all weight schemes
weight_schemes = generate_weight_schemes(window=12)

print(f"\nTesting {len(weight_schemes)} weighting schemes...")
print("\nWeight schemes:")
for name, weights in list(weight_schemes.items())[:5]:
    print(f"  {name:15s}: newest={weights[0]:.3f}, oldest={weights[-1]:.3f}")

# Test outlier filtering levels
outlier_levels = [None, 95, 97, 99]  # None = no filtering, 95 = cap at 5th/95th percentile

print(f"\nTesting {len(outlier_levels)} outlier filtering levels...")
print("  None: No filtering")
print("  95: Cap extreme 5% (winsorize at 5th/95th percentile)")
print("  97: Cap extreme 3%")
print("  99: Cap extreme 1%")

# Test all combinations
print(f"\n{'='*80}")
print(f"TESTING {len(weight_schemes) * len(outlier_levels)} CONFIGURATIONS")
print(f"{'='*80}")

results = []

for scheme_name, weights in weight_schemes.items():
    for outlier_pct in outlier_levels:

        predictions = []

        for player_id in qb_data['player_id'].unique():
            player_games = qb_data[qb_data['player_id'] == player_id].copy()

            if len(player_games) < 12 + 16:
                continue

            for i in range(12, len(player_games) - 16):
                recent_window = player_games.iloc[i-12:i]['rating'].values

                # Calculate predictor with this configuration
                predictor = calculate_weighted_form(recent_window, weights, outlier_pct)

                # Outcome: next 16 games average
                next_16 = player_games.iloc[i:i+16]['rating'].mean()

                if not np.isnan(predictor):
                    predictions.append({
                        'predictor': predictor,
                        'next_16': next_16
                    })

        if len(predictions) > 50:
            pred_df = pd.DataFrame(predictions)

            corr, p_value = stats.spearmanr(pred_df['predictor'], pred_df['next_16'])
            rmse = np.sqrt(((pred_df['predictor'] - pred_df['next_16']) ** 2).mean())
            mae = np.abs(pred_df['predictor'] - pred_df['next_16']).mean()

            results.append({
                'scheme': scheme_name,
                'outlier_filter': outlier_pct if outlier_pct else 'none',
                'correlation': corr,
                'rmse': rmse,
                'mae': mae,
                'n_predictions': len(pred_df)
            })

results_df = pd.DataFrame(results)

# Sort by correlation (best predictor)
results_df = results_df.sort_values('correlation', ascending=False)

print("\n" + "="*80)
print("TOP 20 CONFIGURATIONS (by correlation)")
print("="*80)

print(f"\n{'Rank':<6} {'Scheme':<15} {'Filter':<8} {'Correlation':>12} {'RMSE':>10} {'MAE':>10}")
print("-"*70)
for i, (_, row) in enumerate(results_df.head(20).iterrows(), 1):
    print(f"{i:<6} {row['scheme']:<15} {str(row['outlier_filter']):<8} "
          f"{row['correlation']:>12.4f} {row['rmse']:>10.3f} {row['mae']:>10.3f}")

# Find best configuration
best = results_df.iloc[0]

print("\n" + "="*80)
print("OPTIMAL CONFIGURATION")
print("="*80)

print(f"\nBest weighting scheme: {best['scheme']}")
print(f"Best outlier filter: {best['outlier_filter']}")
print(f"Correlation: r={best['correlation']:.4f}")
print(f"RMSE: {best['rmse']:.3f}")
print(f"MAE: {best['mae']:.3f}")

# Show the optimal weights
optimal_weights = weight_schemes[best['scheme']]
print(f"\nOptimal weights (newest to oldest):")
for i, w in enumerate(optimal_weights[:8], 1):
    print(f"  Game {i}: {w:.3f} ({w*100:.1f}%)")

# Compare proposed vs optimal
proposed_row = results_df[results_df['scheme'] == 'proposed'].iloc[0] if 'proposed' in results_df['scheme'].values else None

if proposed_row is not None:
    print(f"\n" + "="*80)
    print("PROPOSED vs OPTIMAL")
    print("="*80)

    print(f"\nProposed (35%, 25%, 17%...):")
    print(f"  Correlation: r={proposed_row['correlation']:.4f}")
    print(f"  RMSE: {proposed_row['rmse']:.3f}")
    print(f"  Rank: #{(results_df['correlation'] > proposed_row['correlation']).sum() + 1}")

    improvement = (best['correlation'] - proposed_row['correlation']) / proposed_row['correlation'] * 100

    if improvement > 1:
        print(f"\nOptimal is {improvement:.1f}% better than proposed")
    elif improvement < -1:
        print(f"\nProposed is actually {-improvement:.1f}% better than 'optimal' (overfitting?)")
    else:
        print(f"\nProposed is within 1% of optimal (essentially equivalent)")

# Analyze outlier filtering impact
print(f"\n" + "="*80)
print("OUTLIER FILTERING IMPACT")
print("="*80)

for scheme in ['uniform', 'proposed', best['scheme']]:
    if scheme not in weight_schemes:
        continue

    print(f"\n{scheme.upper()} scheme:")
    scheme_results = results_df[results_df['scheme'] == scheme].sort_values('correlation', ascending=False)

    for _, row in scheme_results.iterrows():
        improvement_pct = (row['correlation'] - scheme_results.iloc[-1]['correlation']) / scheme_results.iloc[-1]['correlation'] * 100
        print(f"  Filter {str(row['outlier_filter']):4s}: r={row['correlation']:.4f} ({improvement_pct:+.1f}% vs no filter)")

# Test if 95th percentile filtering is universally beneficial
print(f"\n" + "="*80)
print("SHOULD WE ALWAYS USE OUTLIER FILTERING?")
print("="*80)

for outlier_level in [None, 95, 97, 99]:
    level_results = results_df[results_df['outlier_filter'] == (outlier_level if outlier_level else 'none')]
    avg_corr = level_results['correlation'].mean()
    max_corr = level_results['correlation'].max()

    filter_name = f"{outlier_level}th percentile" if outlier_level else "No filtering"
    print(f"\n{filter_name}:")
    print(f"  Average correlation: {avg_corr:.4f}")
    print(f"  Best correlation: {max_corr:.4f}")
    print(f"  # of schemes in top 10: {len(level_results.head(10))}")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

print(f"\nBased on {len(results_df)} tested configurations:")
print(f"1. Use '{best['scheme']}' weighting scheme")
print(f"2. Apply {best['outlier_filter']}th percentile winsorization" if best['outlier_filter'] != 'none' else "2. No outlier filtering needed")
print(f"3. This achieves r={best['correlation']:.4f} prediction of next 16 games")

# Show optimal weights in detail
print(f"\nOptimal weight distribution:")
optimal_weights = weight_schemes[best['scheme']]
for i in range(min(12, len(optimal_weights))):
    print(f"  Game -{11-i:2d} (t-{11-i}): {optimal_weights[i]:6.3f} ({optimal_weights[i]*100:5.1f}%)")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
