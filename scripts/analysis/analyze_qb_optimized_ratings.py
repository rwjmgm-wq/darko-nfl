"""
Optimized QB Rating System

Based on empirical analysis from optimize_rating_weights.py and multi-feature research:
1. Current Form (Single-Feature): 12-game weighted average with exponential decay
   - Weights: Gradual increase from oldest (4.5%) to newest (13.6%)
   - Outlier filtering: 95th percentile winsorization
   - Achieves r=0.3533 correlation with next 16 games

2. Multi-Feature Rating (NEW): 5-feature ensemble
   - Features: epa_mean, opp_adj_success_rate, epa_per_play, qb_epa_mean, success_rate
   - Correlation-weighted combination
   - Volatility penalty for inconsistent performance
   - Achieves r=0.3853 correlation with next 16 games (+9.1% improvement)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pandas as pd
import numpy as np
from features.multifeature_rating_calculator import MultiFeatureRatingCalculator

# Load QB data
DATA_DIR = Path("data/production")
qb_files = sorted(DATA_DIR.glob("leaf_v2_game_by_game_*.csv"))
DATA_FILE = qb_files[-1]

qb_data = pd.read_csv(DATA_FILE)
qb_data = qb_data.rename(columns={
    'passer_player_id': 'player_id',
    'passer_player_name': 'player_name',
    'opp_adj_base_epa': 'raw_rating'
})

qb_data = qb_data.sort_values(['player_id', 'game_number'])

print("="*80)
print("OPTIMIZED QB RATING SYSTEM")
print("="*80)
print("\n[1] Single-Feature Current Form: 12-game weighted average")
print("    Optimal window size: 12 games (data-driven)")
print("    Weighting: Exponential decay (expo_10)")
print("    Outlier filtering: 95th percentile winsorization")
print("    Predictive power: r=0.3533 (best of 60 tested configurations)")

print("\n[2] Multi-Feature Rating: 5-feature ensemble")
print("    Features: epa_mean, opp_adj_success_rate, epa_per_play, qb_epa_mean, success_rate")
print("    Correlation-weighted combination + volatility penalty")
print("    Predictive power: r=0.3853 (+9.1% improvement)")

# Define optimal weights (exponential decay with rate 0.10)
window = 12
decay_rate = 0.10
weights_12 = np.exp(-decay_rate * np.arange(window)[::-1])
weights_12 = weights_12 / weights_12.sum()

print(f"\nOptimal weights (oldest to newest):")
for i in range(window):
    print(f"  Game t-{window-i-1:2d}: {weights_12[i]:.4f} ({weights_12[i]*100:5.2f}%)")

def apply_outlier_filter(values, percentile=95):
    """Apply 95th percentile winsorization to extreme values."""
    lower = np.percentile(values, 100 - percentile)
    upper = np.percentile(values, percentile)
    return np.clip(values, lower, upper)

def calculate_weighted_form(games, weights, outlier_percentile=95):
    """Calculate weighted average with outlier filtering."""
    if len(games) < len(weights):
        # Early career: use available games
        n_games = len(games)
        active_weights = weights[-n_games:]
        active_weights = active_weights / active_weights.sum()
        filtered_games = apply_outlier_filter(games, outlier_percentile)
        return np.average(filtered_games, weights=active_weights)
    else:
        # Use last N games matching weights length
        recent_games = games[-len(weights):]
        filtered_games = apply_outlier_filter(recent_games, outlier_percentile)
        return np.average(filtered_games, weights=weights)

# Initialize multi-feature calculator
print("\n" + "="*80)
print("INITIALIZING MULTI-FEATURE CALCULATOR")
print("="*80)

multifeature_calc = MultiFeatureRatingCalculator(
    window_size=12,
    decay_rate=0.10,
    outlier_percentile=95.0,
    volatility_penalty=0.3,
    apply_volatility_adjustment=True
)

config = multifeature_calc.get_configuration_summary()
print(f"\nMulti-feature calculator configuration:")
print(f"  Features: {', '.join(config['features'])}")
print(f"  Window: {config['window_size']} games")
print(f"  Outlier filtering: {config['outlier_percentile']}th percentile")
print(f"  Volatility penalty: {config['volatility_penalty']}")

# Calculate current form for each QB
print("\n" + "="*80)
print("CALCULATING RATINGS")
print("="*80)

results = []

for player_id in qb_data['player_id'].unique():
    player_games = qb_data[qb_data['player_id'] == player_id].copy()

    # Single-feature current form: optimal weighted 12-game average
    ratings = player_games['raw_rating'].values
    current_form_12w = calculate_weighted_form(ratings, weights_12, outlier_percentile=95)

    # Multi-feature rating
    multifeature_rating = multifeature_calc.calculate_multifeature_rating(player_games)

    # For comparison: simple averages
    recent_12_simple = player_games.tail(12)['raw_rating'].mean()
    recent_16 = player_games.tail(16)['raw_rating'].mean()
    career_avg = player_games['raw_rating'].mean()

    results.append({
        'player_id': player_id,
        'player_name': player_games['player_name'].iloc[0],
        'total_games': len(player_games),
        'multifeature_rating': multifeature_rating,  # NEW: Multi-feature rating
        'current_form_12w': current_form_12w,  # Single-feature optimal weighted
        'recent_12_simple': recent_12_simple,
        'recent_16': recent_16,
        'career_avg': career_avg,
        'latest_season': player_games['season'].iloc[-1],
        'latest_game': player_games['raw_rating'].iloc[-1]
    })

results_df = pd.DataFrame(results)
results_df = results_df[results_df['total_games'] >= 30].copy()

print(f"Qualified QBs (30+ games): {len(results_df)}")

# Define tiers based on weighted current form
percentiles = results_df['current_form_12w'].quantile([0.20, 0.60, 0.85])

def classify_tier(rating):
    if rating >= percentiles[0.85]:
        return 'Elite'
    elif rating >= percentiles[0.60]:
        return 'Good'
    elif rating >= percentiles[0.20]:
        return 'Average'
    else:
        return 'Below Average'

results_df['tier'] = results_df['current_form_12w'].apply(classify_tier)

print(f"\nCurrent Form thresholds (12-game optimal weighted):")
print(f"  Elite:   > {percentiles[0.85]:+.3f} (top 15%)")
print(f"  Good:    > {percentiles[0.60]:+.3f} (top 40%)")
print(f"  Average: > {percentiles[0.20]:+.3f} (middle 40%)")
print(f"  Below:   < {percentiles[0.20]:+.3f} (bottom 20%)")

# TOP 20 QBs BY MULTI-FEATURE RATING
print(f"\n{'='*80}")
print("TOP 20 QBs BY MULTI-FEATURE RATING (5-feature ensemble)")
print(f"{'='*80}")

top_20_multifeature = results_df.nlargest(20, 'multifeature_rating')
print(f"\n{'Rank':<6} {'QB':<25} {'Multi-Feat':>12} {'Single-Feat':>12} {'Career':>10} {'Tier':>15}")
print("-"*95)
for i, (_, qb) in enumerate(top_20_multifeature.iterrows(), 1):
    print(f"{i:<6} {qb['player_name']:<25} {qb['multifeature_rating']:>12.3f} {qb['current_form_12w']:>12.3f} "
          f"{qb['career_avg']:>10.3f} {qb['tier']:>15}")

# TOP 20 QBs BY SINGLE-FEATURE (for comparison)
print(f"\n{'='*80}")
print("TOP 20 QBs BY SINGLE-FEATURE RATING (12-game optimal weighted)")
print(f"{'='*80}")

top_20 = results_df.nlargest(20, 'current_form_12w')
print(f"\n{'Rank':<6} {'QB':<25} {'Single-Feat':>12} {'Multi-Feat':>12} {'Career':>10} {'Tier':>15}")
print("-"*95)
for i, (_, qb) in enumerate(top_20.iterrows(), 1):
    print(f"{i:<6} {qb['player_name']:<25} {qb['current_form_12w']:>12.3f} {qb['multifeature_rating']:>12.3f} "
          f"{qb['career_avg']:>10.3f} {qb['tier']:>15}")

# Compare weighting methods
print(f"\n{'='*80}")
print("WEIGHTING IMPACT: Optimal vs Simple Average")
print(f"{'='*80}")

results_df['weight_impact'] = results_df['current_form_12w'] - results_df['recent_12_simple']

print(f"\nMost BOOSTED by optimal weighting (strong recent trend):")
boosted = results_df.nlargest(10, 'weight_impact')
print(f"\n{'QB':<25} {'Weighted':>10} {'Simple':>10} {'Boost':>10} {'Tier':>15}")
print("-"*80)
for _, qb in boosted.iterrows():
    print(f"{qb['player_name']:<25} {qb['current_form_12w']:>10.3f} {qb['recent_12_simple']:>10.3f} "
          f"{qb['weight_impact']:>10.3f} {qb['tier']:>15}")

print(f"\nMost PENALIZED by optimal weighting (declining recent trend):")
penalized = results_df.nsmallest(10, 'weight_impact')
print(f"\n{'QB':<25} {'Weighted':>10} {'Simple':>10} {'Penalty':>10} {'Tier':>15}")
print("-"*80)
for _, qb in penalized.iterrows():
    print(f"{qb['player_name']:<25} {qb['current_form_12w']:>10.3f} {qb['recent_12_simple']:>10.3f} "
          f"{qb['weight_impact']:>10.3f} {qb['tier']:>15}")

# SAM DARNOLD
print(f"\n{'='*80}")
print("SAM DARNOLD - MULTI-FEATURE ANALYSIS")
print(f"{'='*80}")

darnold = results_df[results_df['player_name'].str.contains('Darnold', na=False)]
if len(darnold) > 0:
    d = darnold.iloc[0]

    print(f"\n[1] MULTI-FEATURE RATING (5-feature ensemble):")
    print(f"    Rating: {d['multifeature_rating']:+.3f}")
    print(f"    Tier: {d['tier']}")
    multifeature_rank = (results_df['multifeature_rating'] > d['multifeature_rating']).sum() + 1
    print(f"    Rank: #{multifeature_rank} out of {len(results_df)}")

    print(f"\n[2] SINGLE-FEATURE RATING (12-game optimal weighted):")
    print(f"    Rating: {d['current_form_12w']:+.3f}")
    print(f"    Tier: {d['tier']}")
    single_rank = (results_df['current_form_12w'] > d['current_form_12w']).sum() + 1
    print(f"    Rank: #{single_rank} out of {len(results_df)}")

    print(f"\nCOMPARISON OF METHODS:")
    print(f"  Multi-feature rating:     {d['multifeature_rating']:+.3f} (r=0.3853, best predictor)")
    print(f"  Single-feature weighted:  {d['current_form_12w']:+.3f} (r=0.3533)")
    print(f"  12-game simple avg:       {d['recent_12_simple']:+.3f}")
    print(f"  16-game simple avg:       {d['recent_16']:+.3f}")
    print(f"  Career average:           {d['career_avg']:+.3f}")

    multifeature_improvement = d['multifeature_rating'] - d['current_form_12w']
    print(f"\n  Multi-feature improvement: {multifeature_improvement:+.3f}")

    print(f"\nWEIGHTING IMPACT:")
    print(f"  Impact from optimal weighting: {d['weight_impact']:+.3f}")
    if d['weight_impact'] > 0.01:
        print(f"  => Positive recent trend! Latest games stronger than earlier in window")
    elif d['weight_impact'] < -0.01:
        print(f"  => Declining trend. Latest games weaker than earlier in window")
    else:
        print(f"  => Stable. No major trend")

    # Get last 12 games to show trajectory
    darnold_games = qb_data[qb_data['player_name'].str.contains('Darnold', na=False)].copy()
    last_12 = darnold_games.tail(12)

    print(f"\nLAST 12 GAMES (newest first, with outlier filtering applied):")
    print(f"  {'Game':>4} {'Season':>8} {'Week':>6} {'Raw Rating':>12} {'Filtered':>12} {'Weight':>8}")
    print("  " + "-"*60)

    # Apply outlier filtering to show impact
    last_12_ratings = last_12['raw_rating'].values
    filtered_ratings = apply_outlier_filter(last_12_ratings, 95)

    for i, (_, game) in enumerate(last_12.iloc[::-1].iterrows()):
        weight_pct = weights_12[i] * 100
        raw = game['raw_rating']
        filtered = filtered_ratings[len(filtered_ratings) - 1 - i]
        filtered_marker = " *" if abs(raw - filtered) > 0.01 else ""
        print(f"  {i+1:>4} {int(game['season']):>8} {int(game['week']):>6} "
              f"{raw:>12.3f} {filtered:>12.3f}{filtered_marker} {weight_pct:>7.2f}%")

    if any(abs(last_12_ratings - filtered_ratings) > 0.01):
        print("\n  * = Outlier filtered (capped at 5th/95th percentile)")

    # Threshold comparison
    print(f"\nTHRESHOLD COMPARISON:")
    print(f"  Elite threshold:  {percentiles[0.85]:+.3f}")
    print(f"  Good threshold:   {percentiles[0.60]:+.3f}")
    print(f"  Darnold:          {d['current_form_12w']:+.3f}")

    if d['current_form_12w'] >= percentiles[0.85]:
        print(f"  => ELITE tier!")
    elif d['current_form_12w'] >= percentiles[0.60]:
        print(f"  => GOOD tier!")
        gap_to_elite = percentiles[0.85] - d['current_form_12w']
        print(f"  => Needs +{gap_to_elite:.3f} to reach Elite")
    else:
        gap = percentiles[0.60] - d['current_form_12w']
        print(f"  => Needs +{gap:.3f} to reach Good tier")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")
print("""
KEY TAKEAWAYS:
1. Multi-feature rating (5 features) achieves r=0.3853 (+9.1% over baseline)
   - Combines: epa_mean, opp_adj_success_rate, epa_per_play, qb_epa_mean, success_rate
   - Correlation-weighted combination with volatility penalty
   - BEST PREDICTOR of next 16 games performance

2. Single-feature current form achieves r=0.3533
   - 12-game window is optimal (empirically validated)
   - Exponential decay weighting (expo_10) best captures predictive signal
   - 95th percentile outlier filtering smooths extreme performances
   - 47% better than steep weighting schemes (35% on newest game)

3. Why multi-feature approach works:
   - Captures different aspects of QB performance (efficiency, success rate, volume)
   - Reduces noise by averaging across multiple correlated signals
   - Volatility penalty rewards consistent performers
   - Better separation between Elite/Good/Average tiers

RECOMMENDATION:
- Use MULTI-FEATURE RATING for current performance evaluation
- Best predictor of future performance (r=0.3853)
- More robust than single-feature approaches
""")
