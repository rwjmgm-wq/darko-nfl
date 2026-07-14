"""
QB Early Career Thresholds - Current Form Edition

Uses RAW EPA with light EWMA smoothing (span=5) instead of Kalman filtering.
This reflects ACTUAL performance at early career milestones, not heavily smoothed estimates.

Identifies thresholds that predict eventual Elite/Good/Average/Below Average outcomes.
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
    'opp_adj_base_epa': 'raw_rating'  # Opponent-adjusted but NOT Kalman filtered
})

# Calculate current form (light EWMA smoothing, span=5)
qb_data = qb_data.sort_values(['player_id', 'game_number'])
qb_data['current_form'] = qb_data.groupby('player_id')['raw_rating'].transform(
    lambda x: x.ewm(span=5, adjust=False).mean()
)

# Filter to QBs with 30+ games
qb_summary = qb_data.groupby('player_id').agg({
    'current_form': 'mean',
    'game_number': 'max',
    'player_name': 'first'
}).reset_index()
qb_summary.columns = ['player_id', 'career_avg', 'total_games', 'player_name']

qualified_qbs = qb_summary[qb_summary['total_games'] >= 30]['player_id']
qb_data = qb_data[qb_data['player_id'].isin(qualified_qbs)].copy()

# Define tiers based on career average current form
qb_outcomes = qb_data.groupby('player_id').agg({
    'current_form': 'mean',
    'game_number': 'max',
    'player_name': 'first'
}).reset_index()
qb_outcomes.columns = ['player_id', 'career_rating', 'total_games', 'player_name']

percentiles = qb_outcomes['career_rating'].quantile([0.20, 0.60, 0.85])
qb_outcomes['tier'] = pd.cut(
    qb_outcomes['career_rating'],
    bins=[-np.inf, percentiles[0.20], percentiles[0.60], percentiles[0.85], np.inf],
    labels=['Below Average', 'Average', 'Good', 'Elite']
)

print("="*80)
print("QB EARLY CAREER THRESHOLDS - CURRENT FORM EDITION")
print("="*80)
print(f"\nAnalyzing {len(qualified_qbs)} QBs with 30+ games")
print(f"Using RAW EPA with light EWMA smoothing (span=5)")
print(f"\nData range: {qb_data['season'].min()}-{qb_data['season'].max()}")

print(f"\nTier thresholds (career current form):")
print(f"  Elite:   > {percentiles[0.85]:+.3f} (top 15%)")
print(f"  Good:    > {percentiles[0.60]:+.3f} (top 40%)")
print(f"  Average: > {percentiles[0.20]:+.3f} (middle 40%)")
print(f"  Below:   < {percentiles[0.20]:+.3f} (bottom 20%)")

print(f"\n{'='*80}")
print("TIER DISTRIBUTION")
print(f"{'='*80}")
print(qb_outcomes['tier'].value_counts().sort_index())

# MILESTONE ANALYSIS
print(f"\n{'='*80}")
print("MILESTONE CURRENT FORM BY EVENTUAL TIER")
print(f"{'='*80}")

milestones = [4, 8, 12, 16, 20, 24, 28, 32]

milestone_results = []
for milestone in milestones:
    milestone_games = qb_data[qb_data['game_number'] == milestone].copy()
    milestone_merged = milestone_games.merge(
        qb_outcomes[['player_id', 'tier', 'career_rating']],
        on='player_id'
    )

    if len(milestone_merged) > 5:
        milestone_results.append({
            'game': milestone,
            'elite_avg': milestone_merged[milestone_merged['tier'] == 'Elite']['current_form'].mean(),
            'elite_min': milestone_merged[milestone_merged['tier'] == 'Elite']['current_form'].min(),
            'elite_max': milestone_merged[milestone_merged['tier'] == 'Elite']['current_form'].max(),
            'good_avg': milestone_merged[milestone_merged['tier'] == 'Good']['current_form'].mean(),
            'average_avg': milestone_merged[milestone_merged['tier'] == 'Average']['current_form'].mean(),
            'below_avg': milestone_merged[milestone_merged['tier'] == 'Below Average']['current_form'].mean(),
        })

milestone_df = pd.DataFrame(milestone_results)

print(f"\n{'Game':<8} {'Elite Avg':>12} {'Elite Min':>12} {'Elite Max':>12} {'Good Avg':>12} {'Avg Tier':>12} {'Below Avg':>12}")
print("-"*90)
for _, row in milestone_df.iterrows():
    print(f"{int(row['game']):<8} {row['elite_avg']:>12.3f} {row['elite_min']:>12.3f} {row['elite_max']:>12.3f} "
          f"{row['good_avg']:>12.3f} {row['average_avg']:>12.3f} {row['below_avg']:>12.3f}")

# ELITE QB MINIMUM REQUIREMENTS
print(f"\n{'='*80}")
print("ELITE QB MINIMUM REQUIREMENTS")
print(f"{'='*80}")
print(f"\n{'Game':<8} {'Elite Min':>12} {'Below This':>12} {'% Elite Below 0':>18}")
print("-"*60)

for milestone in milestones:
    milestone_games = qb_data[qb_data['game_number'] == milestone].copy()
    milestone_merged = milestone_games.merge(
        qb_outcomes[['player_id', 'tier']],
        on='player_id'
    )

    elite_qbs = milestone_merged[milestone_merged['tier'] == 'Elite']

    if len(elite_qbs) > 0:
        elite_min = elite_qbs['current_form'].min()
        below_min = len(milestone_merged[milestone_merged['current_form'] < elite_min])
        pct_below_zero = (elite_qbs['current_form'] < 0.0).sum() / len(elite_qbs) * 100

        print(f"{milestone:<8} {elite_min:>12.3f} {below_min:>12} {pct_below_zero:>18.1f}%")

# BUST IDENTIFICATION
print(f"\n{'='*80}")
print("BUST IDENTIFICATION THRESHOLDS")
print(f"{'='*80}")

for milestone in [8, 16, 24, 32]:
    milestone_games = qb_data[qb_data['game_number'] == milestone].copy()
    milestone_merged = milestone_games.merge(
        qb_outcomes[['player_id', 'tier', 'player_name']],
        on='player_id'
    )

    if len(milestone_merged) < 5:
        continue

    elite_qbs = milestone_merged[milestone_merged['tier'] == 'Elite']
    below_qbs = milestone_merged[milestone_merged['tier'] == 'Below Average']

    if len(elite_qbs) > 0 and len(below_qbs) > 0:
        elite_min = elite_qbs['current_form'].min()
        below_max = below_qbs['current_form'].max()

        print(f"\nGAME {milestone}:")
        print(f"  Elite minimum: {elite_min:+.3f}")
        print(f"  Below Average maximum: {below_max:+.3f}")
        print(f"  'Bust Zone': below {below_max:+.3f}")

# SPECIFIC THRESHOLDS
print(f"\n{'='*80}")
print("ACTIONABLE THRESHOLDS BY MILESTONE")
print(f"{'='*80}")

print("""
Based on CURRENT FORM ratings (raw EPA + light EWMA):

GAME 8 EVALUATION:
  Elite potential: >= elite_min from analysis
  Good potential: >= good_avg
  Warning signs: < average_avg
  Bust territory: < below_max

GAME 16 EVALUATION:
  Elite track: >= elite_min
  Solid starter: >= good_avg
  Concerns: < average_avg

GAME 32 EVALUATION (End of Year 2):
  Established Elite: >= elite_min
  Quality starter: >= good_avg
  Replacement level: < average_avg
""")

# Print specific recommendations
print(f"\nSPECIFIC NUMERIC THRESHOLDS:")
for milestone in [8, 16, 32]:
    row = milestone_df[milestone_df['game'] == milestone].iloc[0]
    print(f"\nGame {milestone}:")
    print(f"  Elite minimum: {row['elite_min']:+.3f}")
    print(f"  Elite average: {row['elite_avg']:+.3f}")
    print(f"  Good average: {row['good_avg']:+.3f}")
    print(f"  Average tier: {row['average_avg']:+.3f}")
    print(f"  Below average: {row['below_avg']:+.3f}")

# SAM DARNOLD CHECK
print(f"\n{'='*80}")
print("SAM DARNOLD - EARLY CAREER TRAJECTORY")
print(f"{'='*80}")

darnold_games = qb_data[qb_data['player_name'].str.contains('Darnold', na=False)].sort_values('game_number')

if len(darnold_games) > 0:
    print(f"\nEarly Career Current Form (vs thresholds):")
    print(f"{'Game':<8} {'Darnold':>12} {'Elite Min':>12} {'Good Avg':>12} {'Assessment':>20}")
    print("-"*70)

    for milestone in [8, 16, 24, 32]:
        darnold_game = darnold_games[darnold_games['game_number'] == milestone]
        thresh_row = milestone_df[milestone_df['game'] == milestone]

        if len(darnold_game) > 0 and len(thresh_row) > 0:
            d_rating = darnold_game['current_form'].iloc[0]
            elite_min = thresh_row['elite_min'].iloc[0]
            good_avg = thresh_row['good_avg'].iloc[0]

            if d_rating >= elite_min:
                assessment = "Elite trajectory"
            elif d_rating >= good_avg:
                assessment = "Good trajectory"
            else:
                assessment = "Below thresholds"

            print(f"{milestone:<8} {d_rating:>12.3f} {elite_min:>12.3f} {good_avg:>12.3f} {assessment:>20}")

    # Current status
    recent_16 = darnold_games.tail(16)
    recent_8 = darnold_games.tail(8)

    print(f"\nCURRENT STATUS (2024-2025):")
    print(f"  Last 16 games avg: {recent_16['current_form'].mean():+.3f}")
    print(f"  Last 8 games avg: {recent_8['current_form'].mean():+.3f}")
    print(f"  Career avg: {darnold_games['current_form'].mean():+.3f}")

    # Check current vs thresholds
    current_form_16 = recent_16['current_form'].mean()
    current_elite_thresh = percentiles[0.85]
    current_good_thresh = percentiles[0.60]

    print(f"\n  Current Form Tier:")
    if current_form_16 >= current_elite_thresh:
        print(f"    ELITE (above {current_elite_thresh:+.3f})")
    elif current_form_16 >= current_good_thresh:
        print(f"    GOOD (above {current_good_thresh:+.3f})")
    else:
        print(f"    Average/Below (threshold: {current_good_thresh:+.3f})")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")
