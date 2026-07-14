"""
Trend and Momentum Indicators Research

Goal: Push correlation from r=0.3853 to r > 0.40 by adding trend features

Current baseline: Multi-feature rating (5 features) achieves r=0.3853
Target: r > 0.40 (+3.8% improvement needed)

Potential trend indicators:
1. Momentum: Linear trend of recent games (improving vs declining)
2. Acceleration: Change in trend (speeding up or slowing down)
3. Hot/cold streaks: Consecutive positive/negative performances
4. Consistency: Standard deviation, range, coefficient of variation
5. Peak performance: Best game in last N games
6. Floor: Worst game in last N games
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

qb_data = pd.read_csv(DATA_FILE)
qb_data = qb_data.sort_values(['passer_player_id', 'game_number'])

print("="*80)
print("TREND AND MOMENTUM INDICATORS RESEARCH")
print("="*80)
print("\nCurrent baseline: Multi-feature rating r=0.3853")
print("Target: r > 0.40 (+3.8% improvement)")
print("\nTesting trend-based features to capture momentum and consistency...")


def calculate_trend_features(games, window=12):
    """
    Calculate trend and momentum features from recent games.

    Args:
        games: Array of game values (sorted oldest to newest)
        window: Number of recent games to analyze

    Returns:
        Dictionary of trend features
    """
    if len(games) < 3:
        return None

    recent = games[-window:] if len(games) >= window else games
    n = len(recent)

    # 1. Linear trend (slope)
    x = np.arange(n)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent)

    # 2. Acceleration (change in trend between first half and second half)
    if n >= 6:
        mid = n // 2
        first_half = recent[:mid]
        second_half = recent[mid:]

        slope_first, _, _, _, _ = stats.linregress(np.arange(len(first_half)), first_half)
        slope_second, _, _, _, _ = stats.linregress(np.arange(len(second_half)), second_half)
        acceleration = slope_second - slope_first
    else:
        acceleration = 0.0

    # 3. Hot/cold streak (consecutive positive/negative games)
    mean_val = np.mean(recent)
    streak = 0
    for val in reversed(recent):
        if (val > mean_val and streak >= 0) or (val <= mean_val and streak <= 0):
            streak += 1 if val > mean_val else -1
        else:
            break

    # 4. Consistency metrics
    std_dev = np.std(recent)
    range_val = np.max(recent) - np.min(recent)
    coef_var = std_dev / abs(mean_val) if abs(mean_val) > 0.01 else 0.0

    # 5. Peak and floor
    peak = np.max(recent)
    floor = np.min(recent)

    # 6. Recent vs earlier performance (last 4 vs first 8 games in 12-game window)
    if n >= 12:
        recent_4 = np.mean(recent[-4:])
        earlier_8 = np.mean(recent[-12:-4])
        recent_vs_earlier = recent_4 - earlier_8
    elif n >= 6:
        recent_4 = np.mean(recent[-min(4, n//3):])
        earlier_8 = np.mean(recent[:-(n//3)])
        recent_vs_earlier = recent_4 - earlier_8
    else:
        recent_vs_earlier = 0.0

    return {
        'trend_slope': slope,
        'trend_acceleration': acceleration,
        'streak_length': streak,
        'consistency_std': std_dev,
        'consistency_range': range_val,
        'consistency_cv': coef_var,
        'peak_performance': peak,
        'floor_performance': floor,
        'recent_vs_earlier': recent_vs_earlier
    }


# Test trend features individually
print("\n" + "="*80)
print("TESTING INDIVIDUAL TREND FEATURES")
print("="*80)

trend_feature_names = [
    'trend_slope',
    'trend_acceleration',
    'streak_length',
    'consistency_std',
    'consistency_range',
    'consistency_cv',
    'peak_performance',
    'floor_performance',
    'recent_vs_earlier'
]

trend_results = []

for feature_name in trend_feature_names:
    predictions = []

    for player_id in qb_data['passer_player_id'].unique():
        player_games = qb_data[qb_data['passer_player_id'] == player_id].copy()

        if len(player_games) < 12 + 16:
            continue

        for i in range(12, len(player_games) - 16):
            recent_12 = player_games.iloc[i-12:i]['opp_adj_base_epa'].values

            # Calculate trend features
            trend_features = calculate_trend_features(recent_12, window=12)

            if trend_features is not None:
                predictor = trend_features[feature_name]

                # Target: next 16 games average
                next_16 = player_games.iloc[i:i+16]['opp_adj_base_epa'].mean()

                if not np.isnan(predictor) and not np.isnan(next_16) and not np.isinf(predictor):
                    predictions.append({
                        'predictor': predictor,
                        'target': next_16
                    })

    if len(predictions) > 50:
        pred_df = pd.DataFrame(predictions)
        corr, p_value = stats.spearmanr(pred_df['predictor'], pred_df['target'])

        trend_results.append({
            'feature': feature_name,
            'correlation': corr,
            'p_value': p_value,
            'n': len(pred_df)
        })

trend_results_df = pd.DataFrame(trend_results).sort_values('correlation', ascending=False)

print(f"\n{'Feature':<25} {'Correlation':>12} {'P-value':>10} {'N':>8}")
print("-"*65)
for _, row in trend_results_df.iterrows():
    print(f"{row['feature']:<25} {row['correlation']:>12.4f} {row['p_value']:>10.6f} {row['n']:>8.0f}")


# Test adding trend features to baseline multi-feature model
print("\n" + "="*80)
print("COMBINING TREND FEATURES WITH BASELINE")
print("="*80)

# Initialize baseline calculator
baseline_calc = MultiFeatureRatingCalculator(
    window_size=12,
    decay_rate=0.10,
    outlier_percentile=95.0,
    volatility_penalty=0.3,
    apply_volatility_adjustment=True
)

print("\nBaseline: 5-feature multi-feature rating (r=0.3853)")
print("\nTesting combinations with top trend features...")

# Get top 3 trend features
top_trend_features = trend_results_df.head(3)['feature'].tolist()

print(f"\nTop 3 trend features: {', '.join(top_trend_features)}")

# Test different combinations
test_configs = [
    {
        'name': 'Baseline only',
        'use_baseline': True,
        'use_trends': []
    },
    {
        'name': 'Baseline + trend_slope',
        'use_baseline': True,
        'use_trends': ['trend_slope']
    },
    {
        'name': 'Baseline + top 2 trends',
        'use_baseline': True,
        'use_trends': top_trend_features[:2]
    },
    {
        'name': 'Baseline + top 3 trends',
        'use_baseline': True,
        'use_trends': top_trend_features
    }
]

combo_results = []

for config in test_configs:
    predictions = []

    for player_id in qb_data['passer_player_id'].unique():
        player_games = qb_data[qb_data['passer_player_id'] == player_id].copy()

        if len(player_games) < 12 + 16:
            continue

        for i in range(12, len(player_games) - 16):
            recent_12_games = player_games.iloc[i-12:i]

            # Baseline multi-feature rating
            if config['use_baseline']:
                baseline_rating = baseline_calc.calculate_multifeature_rating(recent_12_games)
            else:
                baseline_rating = 0.0

            # Trend features
            trend_values = []
            if len(config['use_trends']) > 0:
                recent_12_epa = recent_12_games['opp_adj_base_epa'].values
                trend_features = calculate_trend_features(recent_12_epa, window=12)

                if trend_features is not None:
                    for trend_name in config['use_trends']:
                        trend_values.append(trend_features[trend_name])

            # Combine: baseline + average of trend features (normalized)
            if len(trend_values) > 0 and not np.isnan(baseline_rating):
                # Normalize trend values to same scale as baseline
                # Simple approach: standardize and add with lower weight
                trend_contribution = 0.0
                for tv in trend_values:
                    if not np.isnan(tv) and not np.isinf(tv):
                        # Small weight for trend features (20% of baseline weight)
                        trend_contribution += tv * 0.05 / len(trend_values)

                combined_predictor = baseline_rating + trend_contribution
            else:
                combined_predictor = baseline_rating

            # Target
            next_16 = player_games.iloc[i:i+16]['opp_adj_base_epa'].mean()

            if not np.isnan(combined_predictor) and not np.isnan(next_16):
                predictions.append({
                    'predictor': combined_predictor,
                    'target': next_16
                })

    if len(predictions) > 50:
        pred_df = pd.DataFrame(predictions)
        corr, p_value = stats.spearmanr(pred_df['predictor'], pred_df['target'])
        rmse = np.sqrt(((pred_df['predictor'] - pred_df['target']) ** 2).mean())

        combo_results.append({
            'config': config['name'],
            'features': len(config['use_trends']),
            'correlation': corr,
            'rmse': rmse,
            'n': len(pred_df)
        })

combo_results_df = pd.DataFrame(combo_results).sort_values('correlation', ascending=False)

print(f"\n{'Configuration':<30} {'Trend Feat':>12} {'Correlation':>12} {'RMSE':>10}")
print("-"*75)
for _, row in combo_results_df.iterrows():
    print(f"{row['config']:<30} {row['features']:>12} {row['correlation']:>12.4f} {row['rmse']:>10.3f}")


# Summary and recommendations
print("\n" + "="*80)
print("SUMMARY AND RECOMMENDATIONS")
print("="*80)

baseline_corr = combo_results_df[combo_results_df['config'] == 'Baseline only']['correlation'].iloc[0]
best_combo = combo_results_df.iloc[0]

improvement = (best_combo['correlation'] - baseline_corr) / baseline_corr * 100
absolute_improvement = best_combo['correlation'] - baseline_corr

print(f"\nBaseline (5-feature multi-feature): r={baseline_corr:.4f}")
print(f"Best with trend features: r={best_combo['correlation']:.4f}")
print(f"Improvement: {improvement:+.2f}% ({absolute_improvement:+.4f})")

TARGET_CORR = 0.40
gap_to_target = TARGET_CORR - best_combo['correlation']
gap_pct = gap_to_target / TARGET_CORR * 100

print(f"\nTarget: r > {TARGET_CORR:.2f}")
print(f"Gap remaining: {gap_to_target:+.4f} ({gap_pct:.1f}%)")

print("\n" + "="*80)
print("FINDINGS")
print("="*80)

print("""
TREND FEATURE ANALYSIS:
1. Most predictive trend features:
""")

for i, (_, row) in enumerate(trend_results_df.head(3).iterrows(), 1):
    print(f"   {i}. {row['feature']:<25} r={row['correlation']:.4f}")

print(f"""
2. Impact of adding trend features:
   - Best combination: {best_combo['config']}
   - Correlation: r={best_combo['correlation']:.4f}
   - Change from baseline: {improvement:+.2f}%

3. Key insights:
   - Trend features {'DO' if improvement > 0.5 else 'DO NOT'} provide meaningful improvement
   - {'Strong' if abs(trend_results_df.iloc[0]['correlation']) > 0.10 else 'Weak'} individual predictive power
   - {'Worth pursuing' if improvement > 1.0 else 'Limited benefit - explore other approaches'}

NEXT STEPS TO REACH r > 0.40:
1. {'[+] Continue' if improvement > 1.0 else '[-] Skip'} trend features - {'add to production' if improvement > 1.0 else 'limited benefit'}
2. Test contextual splits (home/away, weather, rest days)
3. Model schedule difficulty (opponent quality trends)
4. Integrate external data (injuries, roster changes, coaching)
5. Try machine learning ensemble (Random Forest, XGBoost)

REALISTIC ASSESSMENT:
- Current progress: r={best_combo['correlation']:.4f}
- Gap to r=0.40: {gap_to_target:+.4f} ({gap_pct:.1f}%)
- QB performance has inherent randomness
- r=0.40-0.45 may be ceiling for this prediction task
""")
