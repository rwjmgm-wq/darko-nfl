"""
Contextual Splits Analysis

Goal: Push correlation from r=0.3853 to r > 0.40 by modeling contextual factors

Current baseline: Multi-feature rating (5 features) achieves r=0.3853
Target: r > 0.40 (+3.8% improvement needed)

Potential contextual factors:
1. Home vs Away performance splits
2. Rest days (short vs normal rest)
3. Weather conditions (if available)
4. Time of season (early vs late)
5. Score differential / game script

Strategy: Test whether QBs who perform differently in different contexts
are more/less predictable in future games.
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
print("CONTEXTUAL SPLITS ANALYSIS")
print("="*80)
print("\nCurrent baseline: Multi-feature rating r=0.3853")
print("Target: r > 0.40 (+3.8% improvement)")
print("\nAnalyzing contextual performance variations...")

# Check available contextual columns
print("\n" + "="*80)
print("AVAILABLE CONTEXTUAL DATA")
print("="*80)

contextual_cols = ['home_away', 'location', 'roof', 'surface', 'temp', 'wind',
                   'week', 'season', 'game_type', 'posteam', 'defteam']

available_context = [col for col in contextual_cols if col in qb_data.columns]
print(f"\nAvailable contextual columns ({len(available_context)}):")
for col in available_context:
    unique_vals = qb_data[col].nunique()
    print(f"  - {col:<20} ({unique_vals} unique values)")


def calculate_context_features(player_games, window=12):
    """
    Calculate performance split features.

    Args:
        player_games: DataFrame of player's games
        window: Window size for recent games

    Returns:
        Dictionary of context features
    """
    if len(player_games) < window:
        recent = player_games
    else:
        recent = player_games.tail(window)

    features = {}

    # 1. Home vs Away split (if available)
    if 'home_away' in recent.columns or 'location' in recent.columns:
        loc_col = 'home_away' if 'home_away' in recent.columns else 'location'

        home_games = recent[recent[loc_col] == 'home']
        away_games = recent[recent[loc_col] == 'away']

        if len(home_games) > 0 and len(away_games) > 0:
            home_avg = home_games['opp_adj_base_epa'].mean()
            away_avg = away_games['opp_adj_base_epa'].mean()
            home_away_diff = home_avg - away_avg
            features['home_away_split'] = home_away_diff
        else:
            features['home_away_split'] = 0.0
    else:
        features['home_away_split'] = 0.0

    # 2. Dome vs Outdoor split (if available)
    if 'roof' in recent.columns:
        dome_games = recent[recent['roof'].isin(['dome', 'closed'])]
        outdoor_games = recent[recent['roof'].isin(['outdoors', 'open'])]

        if len(dome_games) > 0 and len(outdoor_games) > 0:
            dome_avg = dome_games['opp_adj_base_epa'].mean()
            outdoor_avg = outdoor_games['opp_adj_base_epa'].mean()
            features['dome_outdoor_split'] = dome_avg - outdoor_avg
        else:
            features['dome_outdoor_split'] = 0.0
    else:
        features['dome_outdoor_split'] = 0.0

    # 3. Early vs late season performance
    if 'week' in recent.columns:
        early_games = recent[recent['week'] <= 9]
        late_games = recent[recent['week'] > 9]

        if len(early_games) > 0 and len(late_games) > 0:
            early_avg = early_games['opp_adj_base_epa'].mean()
            late_avg = late_games['opp_adj_base_epa'].mean()
            features['early_late_split'] = late_avg - early_avg
        else:
            features['early_late_split'] = 0.0
    else:
        features['early_late_split'] = 0.0

    # 4. Weather sensitivity (if temp/wind available)
    if 'temp' in recent.columns:
        # Define cold as < 40F
        cold_games = recent[recent['temp'] < 40]
        warm_games = recent[recent['temp'] >= 40]

        if len(cold_games) > 0 and len(warm_games) > 0:
            cold_avg = cold_games['opp_adj_base_epa'].mean()
            warm_avg = warm_games['opp_adj_base_epa'].mean()
            features['cold_warm_split'] = warm_avg - cold_avg
        else:
            features['cold_warm_split'] = 0.0
    else:
        features['cold_warm_split'] = 0.0

    if 'wind' in recent.columns:
        # Define windy as > 10 mph
        windy_games = recent[recent['wind'] > 10]
        calm_games = recent[recent['wind'] <= 10]

        if len(windy_games) > 0 and len(calm_games) > 0:
            windy_avg = windy_games['opp_adj_base_epa'].mean()
            calm_avg = calm_games['opp_adj_base_epa'].mean()
            features['wind_split'] = calm_avg - windy_avg
        else:
            features['wind_split'] = 0.0
    else:
        features['wind_split'] = 0.0

    return features


# Test contextual features individually
print("\n" + "="*80)
print("TESTING INDIVIDUAL CONTEXTUAL FEATURES")
print("="*80)

context_results = []

for player_id in qb_data['passer_player_id'].unique():
    player_games = qb_data[qb_data['passer_player_id'] == player_id].copy()

    if len(player_games) < 12 + 16:
        continue

    for i in range(12, len(player_games) - 16):
        recent_12 = player_games.iloc[i-12:i]

        # Calculate context features
        context_features = calculate_context_features(recent_12, window=12)

        # Target: next 16 games
        next_16 = player_games.iloc[i:i+16]['opp_adj_base_epa'].mean()

        if not np.isnan(next_16):
            result = {'target': next_16}
            result.update(context_features)
            context_results.append(result)

context_df = pd.DataFrame(context_results)

print(f"\nGenerated {len(context_df)} context feature samples")

# Test correlation of each contextual feature
print(f"\n{'Contextual Feature':<25} {'Correlation':>12} {'P-value':>10}")
print("-"*55)

context_feature_corrs = []

for feature in [col for col in context_df.columns if col != 'target']:
    # Filter out NaN and inf values
    valid = context_df[[feature, 'target']].dropna()
    valid = valid[~np.isinf(valid[feature])]

    if len(valid) > 50:
        corr, p_value = stats.spearmanr(valid[feature], valid['target'])
        print(f"{feature:<25} {corr:>12.4f} {p_value:>10.6f}")

        context_feature_corrs.append({
            'feature': feature,
            'correlation': abs(corr),
            'signed_corr': corr
        })

context_corr_df = pd.DataFrame(context_feature_corrs).sort_values('correlation', ascending=False)


# Test combining contextual features with baseline
print("\n" + "="*80)
print("COMBINING CONTEXTUAL FEATURES WITH BASELINE")
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
print("\nTesting combinations with contextual features...")

# Test configurations
if len(context_corr_df) > 0:
    top_context_feature = context_corr_df.iloc[0]['feature']

    test_configs = [
        {
            'name': 'Baseline only',
            'use_baseline': True,
            'use_context': []
        },
        {
            'name': f'Baseline + {top_context_feature}',
            'use_baseline': True,
            'use_context': [top_context_feature]
        },
        {
            'name': 'Baseline + all context',
            'use_baseline': True,
            'use_context': context_corr_df['feature'].tolist()
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
                baseline_rating = baseline_calc.calculate_multifeature_rating(recent_12_games)

                # Context features
                context_values = []
                if len(config['use_context']) > 0:
                    context_features = calculate_context_features(recent_12_games, window=12)

                    for context_name in config['use_context']:
                        if context_name in context_features:
                            context_values.append(context_features[context_name])

                # Combine: baseline + weighted context features
                if len(context_values) > 0 and not np.isnan(baseline_rating):
                    # Small weight for context features (10% of baseline)
                    context_contribution = 0.0
                    for cv in context_values:
                        if not np.isnan(cv) and not np.isinf(cv):
                            context_contribution += cv * 0.02 / len(context_values)

                    combined_predictor = baseline_rating + context_contribution
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
                'features': len(config['use_context']),
                'correlation': corr,
                'rmse': rmse,
                'n': len(pred_df)
            })

    if len(combo_results) > 0:
        combo_results_df = pd.DataFrame(combo_results).sort_values('correlation', ascending=False)

        print(f"\n{'Configuration':<35} {'Context Feat':>13} {'Correlation':>12} {'RMSE':>10}")
        print("-"*80)
        for _, row in combo_results_df.iterrows():
            print(f"{row['config']:<35} {row['features']:>13} {row['correlation']:>12.4f} {row['rmse']:>10.3f}")

        # Summary
        print("\n" + "="*80)
        print("SUMMARY AND RECOMMENDATIONS")
        print("="*80)

        baseline_corr = combo_results_df[combo_results_df['config'] == 'Baseline only']['correlation'].iloc[0]
        best_combo = combo_results_df.iloc[0]

        improvement = (best_combo['correlation'] - baseline_corr) / baseline_corr * 100
        absolute_improvement = best_combo['correlation'] - baseline_corr

        print(f"\nBaseline (5-feature multi-feature): r={baseline_corr:.4f}")
        print(f"Best with contextual features: r={best_combo['correlation']:.4f}")
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
CONTEXTUAL SPLITS ANALYSIS:
1. Most predictive contextual features:
""")

        for i, (_, row) in enumerate(context_corr_df.head(3).iterrows(), 1):
            print(f"   {i}. {row['feature']:<25} r={row['signed_corr']:+.4f}")

        print(f"""
2. Impact of adding contextual features:
   - Best combination: {best_combo['config']}
   - Correlation: r={best_combo['correlation']:.4f}
   - Change from baseline: {improvement:+.2f}%

3. Key insights:
   - Contextual features {'DO' if improvement > 0.5 else 'DO NOT'} provide meaningful improvement
   - {'Strong' if len(context_corr_df) > 0 and context_corr_df.iloc[0]['correlation'] > 0.05 else 'Weak'} individual predictive power
   - {'Worth pursuing' if improvement > 1.0 else 'Limited benefit - explore other approaches'}

NEXT STEPS TO REACH r > 0.40:
1. {'[+] Continue' if improvement > 1.0 else '[-] Skip'} contextual splits - {'add to production' if improvement > 1.0 else 'limited benefit'}
2. Model schedule difficulty (opponent quality trends)
3. Integrate external data (injuries, roster changes, coaching)
4. Try machine learning ensemble (Random Forest, XGBoost)

REALISTIC ASSESSMENT:
- Current progress: r={best_combo['correlation']:.4f}
- Gap to r=0.40: {gap_to_target:+.4f} ({gap_pct:.1f}%)
- May need ML ensemble or external data to reach target
- r=0.40-0.45 likely ceiling for feature engineering alone
""")

else:
    print("\nNo contextual features available for testing")
