"""
Dual QB Rating System

1. CURRENT FORM: Raw EPA with light smoothing (EWMA span=3-5)
   - Used for: Early career thresholds, current performance evaluation
   - Fast-updating, reflects actual recent play

2. CAREER RATING: Weighted recent performance (last 51 games heavy, exponential decay)
   - Used for: Overall tier classification, contract decisions
   - Balances current form with proven track record
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load QB data
DATA_DIR = Path("data/production")
qb_files = sorted(DATA_DIR.glob("leaf_v2_game_by_game_*.csv"))
DATA_FILE = qb_files[-1]

qb_data = pd.read_csv(DATA_FILE)
qb_data = qb_data.rename(columns={
    'passer_player_id': 'player_id',
    'passer_player_name': 'player_name',
    'opp_adj_base_epa': 'raw_epa'  # Opponent-adjusted but NOT Kalman filtered
})

# Calculate CURRENT FORM rating (light EWMA smoothing, span=5)
qb_data = qb_data.sort_values(['player_id', 'game_number'])
qb_data['current_form'] = qb_data.groupby('player_id')['raw_epa'].transform(
    lambda x: x.ewm(span=5, adjust=False).mean()
)

print("="*80)
print("DUAL QB RATING SYSTEM ANALYSIS")
print("="*80)
print(f"\nData range: {qb_data['season'].min()}-{qb_data['season'].max()}")
print(f"Total QB-games: {len(qb_data):,}")

def calculate_career_rating(player_games, window=51, decay_rate=0.02):
    """
    Calculate career rating with recent games (last 51) weighted most heavily.
    Uses raw EPA, not Kalman.
    """
    if len(player_games) == 0:
        return np.nan

    ratings = player_games['current_form'].values
    n_games = len(ratings)

    # Exponential decay from most recent
    games_back = np.arange(n_games)[::-1]
    weights = np.exp(-decay_rate * games_back)

    # Boost recent window to full weight
    if n_games > window:
        weights[-window:] = 1.0

    # Normalize
    weights = weights / weights.sum() * n_games

    return np.average(ratings, weights=weights)

# Calculate both ratings for each QB
print("\n" + "="*80)
print("CALCULATING DUAL RATINGS")
print("="*80)

results = []

for player_id in qb_data['player_id'].unique():
    player_games = qb_data[qb_data['player_id'] == player_id].copy()

    # Current form: Last 16 games average (unweighted recent performance)
    recent_16 = player_games.tail(16)['current_form'].mean()

    # Current form: Last 8 games (very recent)
    recent_8 = player_games.tail(8)['current_form'].mean()

    # Career rating: Weighted over all games
    career_rating = calculate_career_rating(player_games)

    # Raw averages for comparison
    career_avg_raw = player_games['raw_epa'].mean()
    recent_51_avg = player_games.tail(51)['current_form'].mean()

    results.append({
        'player_id': player_id,
        'player_name': player_games['player_name'].iloc[0],
        'total_games': len(player_games),
        'current_form_8': recent_8,
        'current_form_16': recent_16,
        'career_rating': career_rating,
        'recent_51_avg': recent_51_avg,
        'career_avg_raw': career_avg_raw,
        'latest_season': player_games['season'].iloc[-1],
        'latest_raw_epa': player_games['raw_epa'].iloc[-1]
    })

results_df = pd.DataFrame(results)

# Filter to QBs with 30+ games
results_df = results_df[results_df['total_games'] >= 30].copy()

print(f"Qualified QBs (30+ games): {len(results_df)}")

# Define tiers based on CURRENT FORM (16 games) for active evaluation
# Use career rating as tiebreaker/verification
percentiles_form = results_df['current_form_16'].quantile([0.20, 0.60, 0.85])
percentiles_career = results_df['career_rating'].quantile([0.20, 0.60, 0.85])

print(f"\nCURRENT FORM thresholds (last 16 games):")
print(f"  Elite:   > {percentiles_form[0.85]:+.3f} (top 15%)")
print(f"  Good:    > {percentiles_form[0.60]:+.3f} (top 40%)")
print(f"  Average: > {percentiles_form[0.20]:+.3f} (middle 40%)")
print(f"  Below:   < {percentiles_form[0.20]:+.3f} (bottom 20%)")

print(f"\nCAREER RATING thresholds (weighted all games):")
print(f"  Elite:   > {percentiles_career[0.85]:+.3f}")
print(f"  Good:    > {percentiles_career[0.60]:+.3f}")
print(f"  Average: > {percentiles_career[0.20]:+.3f}")
print(f"  Below:   < {percentiles_career[0.20]:+.3f}")

# Classify by current form (what matters NOW)
def classify_current_tier(row):
    form = row['current_form_16']
    if form >= percentiles_form[0.85]:
        return 'Elite'
    elif form >= percentiles_form[0.60]:
        return 'Good'
    elif form >= percentiles_form[0.20]:
        return 'Average'
    else:
        return 'Below Average'

results_df['current_tier'] = results_df.apply(classify_current_tier, axis=1)

# TOP QBs by current form
print(f"\n{'='*80}")
print("TOP 20 QBs BY CURRENT FORM (Last 16 Games)")
print(f"{'='*80}")

top_20_form = results_df.nlargest(20, 'current_form_16')
print(f"\n{'Rank':<6} {'QB':<25} {'Form 16':>10} {'Form 8':>10} {'Career':>10} {'Tier':>15}")
print("-"*80)
for i, (_, qb) in enumerate(top_20_form.iterrows(), 1):
    print(f"{i:<6} {qb['player_name']:<25} {qb['current_form_16']:>10.3f} {qb['current_form_8']:>10.3f} "
          f"{qb['career_rating']:>10.3f} {qb['current_tier']:>15}")

# Hot hand: QBs with Form 8 >> Form 16 (heating up)
print(f"\n{'='*80}")
print("HEATING UP (Form 8 > Form 16 + Career)")
print(f"{'='*80}")

results_df['heating_up'] = results_df['current_form_8'] - results_df['current_form_16']
heating = results_df.nlargest(10, 'heating_up')
print(f"\n{'QB':<25} {'Form 8':>10} {'Form 16':>10} {'Career':>10} {'Heating':>10} {'Tier':>15}")
print("-"*80)
for _, qb in heating.iterrows():
    print(f"{qb['player_name']:<25} {qb['current_form_8']:>10.3f} {qb['current_form_16']:>10.3f} "
          f"{qb['career_rating']:>10.3f} {qb['heating_up']:>10.3f} {qb['current_tier']:>15}")

# Cold streak: Form 8 << Form 16 (cooling off)
print(f"\n{'='*80}")
print("COOLING OFF (Form 8 < Form 16)")
print(f"{'='*80}")

cooling = results_df.nsmallest(10, 'heating_up')
print(f"\n{'QB':<25} {'Form 8':>10} {'Form 16':>10} {'Career':>10} {'Change':>10} {'Tier':>15}")
print("-"*80)
for _, qb in cooling.iterrows():
    print(f"{qb['player_name']:<25} {qb['current_form_8']:>10.3f} {qb['current_form_16']:>10.3f} "
          f"{qb['career_rating']:>10.3f} {qb['heating_up']:>10.3f} {qb['current_tier']:>15}")

# Biggest gap: Career reputation vs current form
print(f"\n{'='*80}")
print("CAREER REPUTATION vs CURRENT FORM")
print(f"{'='*80}")

results_df['career_gap'] = results_df['current_form_16'] - results_df['career_rating']

print(f"\nPlaying ABOVE career reputation:")
overperforming = results_df.nlargest(10, 'career_gap')
print(f"\n{'QB':<25} {'Form 16':>10} {'Career':>10} {'Gap':>10} {'Tier':>15}")
print("-"*80)
for _, qb in overperforming.iterrows():
    print(f"{qb['player_name']:<25} {qb['current_form_16']:>10.3f} {qb['career_rating']:>10.3f} "
          f"{qb['career_gap']:>10.3f} {qb['current_tier']:>15}")

print(f"\nPlaying BELOW career reputation:")
underperforming = results_df.nsmallest(10, 'career_gap')
print(f"\n{'QB':<25} {'Form 16':>10} {'Career':>10} {'Gap':>10} {'Tier':>15}")
print("-"*80)
for _, qb in underperforming.iterrows():
    print(f"{qb['player_name']:<25} {qb['current_form_16']:>10.3f} {qb['career_rating']:>10.3f} "
          f"{qb['career_gap']:>10.3f} {qb['current_tier']:>15}")

# SAM DARNOLD ANALYSIS
print(f"\n{'='*80}")
print("SAM DARNOLD - DUAL RATING ANALYSIS")
print(f"{'='*80}")

darnold = results_df[results_df['player_name'].str.contains('Darnold', na=False)]
if len(darnold) > 0:
    d = darnold.iloc[0]

    print(f"\nCURRENT FORM (What he's doing NOW):")
    print(f"  Last 8 games:  {d['current_form_8']:+.3f}")
    print(f"  Last 16 games: {d['current_form_16']:+.3f}")
    print(f"  Current tier (based on Form 16): {d['current_tier']}")

    print(f"\nCAREER EVALUATION:")
    print(f"  Career rating (weighted): {d['career_rating']:+.3f}")
    print(f"  Career avg (raw EPA): {d['career_avg_raw']:+.3f}")
    print(f"  Recent 51 games: {d['recent_51_avg']:+.3f}")

    print(f"\nREPUTATION GAP:")
    print(f"  Current form - Career: {d['career_gap']:+.3f}")
    if d['career_gap'] > 0.10:
        print(f"  => Playing SIGNIFICANTLY above career reputation!")
    elif d['career_gap'] > 0.05:
        print(f"  => Playing above career reputation")

    print(f"\nTHRESHOLD COMPARISON:")
    print(f"  Elite threshold (Form 16): {percentiles_form[0.85]:+.3f}")
    print(f"  Good threshold (Form 16): {percentiles_form[0.60]:+.3f}")
    print(f"  Darnold Form 16: {d['current_form_16']:+.3f}")

    if d['current_form_16'] >= percentiles_form[0.85]:
        print(f"  => ELITE current form!")
    elif d['current_form_16'] >= percentiles_form[0.60]:
        print(f"  => GOOD current form!")
    else:
        gap = percentiles_form[0.60] - d['current_form_16']
        print(f"  => Needs +{gap:.3f} to reach Good tier")

    # Show 2024-2025 trajectory
    darnold_games = qb_data[qb_data['player_name'].str.contains('Darnold', na=False)].copy()
    recent_trajectory = darnold_games.tail(26)

    print(f"\n2024-2025 TRAJECTORY (Last 26 games):")
    print(f"  Avg Current Form: {recent_trajectory['current_form'].mean():+.3f}")
    print(f"  Avg Raw EPA: {recent_trajectory['raw_epa'].mean():+.3f}")

    darnold_2025 = darnold_games[darnold_games['season'] == 2025]
    print(f"\n2025 ONLY ({len(darnold_2025)} games):")
    print(f"  Avg Current Form: {darnold_2025['current_form'].mean():+.3f}")
    print(f"  Avg Raw EPA: {darnold_2025['raw_epa'].mean():+.3f}")

    # Game by game 2025
    print(f"\n2025 Game-by-Game:")
    print(f"  {'Week':<6} {'Raw EPA':>10} {'Form (EWMA)':>15}")
    print("  " + "-"*35)
    for _, game in darnold_2025.sort_values('week').iterrows():
        print(f"  {int(game['week']):<6} {game['raw_epa']:>10.3f} {game['current_form']:>15.3f}")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")
print("""
INTERPRETATION GUIDE:

CURRENT FORM (16 games): What matters RIGHT NOW
  - Use for: Current rankings, fantasy, game predictions
  - Fast-updating, reflects recent performance

CAREER RATING (weighted): Long-term evaluation
  - Use for: Contract decisions, historical comparisons
  - Balances hot/cold streaks with proven track record

GAP Analysis:
  - Current >> Career: Breakout season, improved situation
  - Current << Career: Decline, injury, bad situation
""")
