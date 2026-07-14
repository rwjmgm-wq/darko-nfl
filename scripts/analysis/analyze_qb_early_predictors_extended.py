"""
Extended QB Early Career Predictor Analysis

Deep dive into early career patterns:
- Game-by-game milestone progression
- Bust vs Elite early warning signs
- Recovery potential from poor starts
- Consistency vs volatility patterns
- Specific actionable thresholds for evaluation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load QB data
DATA_DIR = Path("data/production")
qb_files = sorted(DATA_DIR.glob("leaf_v2_game_by_game_*.csv"))
DATA_FILE = qb_files[-1]

qb_data = pd.read_csv(DATA_FILE)
qb_data = qb_data.rename(columns={
    'passer_player_id': 'player_id',
    'passer_player_name': 'player_name',
    'game_number': 'career_game_number',
    'opp_adj_base_epa_kalman': 'raw_composite'  # Using game-by-game Kalman values (leaf_rating is buggy)
})

# Calculate smoothed composite
qb_data = qb_data.sort_values(['player_id', 'career_game_number'])
qb_data['smoothed_composite'] = qb_data.groupby('player_id')['raw_composite'].transform(
    lambda x: x.ewm(span=5, adjust=False).mean()
)

# Filter to QBs with 30+ games
qb_summary = qb_data.groupby('player_id').agg({
    'smoothed_composite': 'mean',
    'career_game_number': 'max',
    'player_name': 'first'
}).reset_index()
qb_summary.columns = ['player_id', 'career_avg_rating', 'total_games', 'player_name']

qualified_qbs = qb_summary[qb_summary['total_games'] >= 30]['player_id']
qb_data = qb_data[qb_data['player_id'].isin(qualified_qbs)].copy()

# Define tiers
qb_outcomes = qb_data.groupby('player_id').agg({
    'smoothed_composite': 'mean',
    'raw_composite': ['mean', 'std'],
    'career_game_number': 'max',
    'player_name': 'first'
}).reset_index()
qb_outcomes.columns = ['player_id', 'career_rating', 'career_rating_raw', 'career_std', 'total_games', 'player_name']

percentiles = qb_outcomes['career_rating'].quantile([0.20, 0.60, 0.85])
qb_outcomes['tier'] = pd.cut(
    qb_outcomes['career_rating'],
    bins=[-np.inf, percentiles[0.20], percentiles[0.60], percentiles[0.85], np.inf],
    labels=['Below Average', 'Average', 'Good', 'Elite']
)

print("="*80)
print("EXTENDED QB EARLY CAREER PREDICTOR ANALYSIS")
print("="*80)
print(f"\nAnalyzing {len(qualified_qbs)} QBs with 30+ games")
print(f"Elite tier: {len(qb_outcomes[qb_outcomes['tier'] == 'Elite'])} QBs")

# ============================================================================
# PART 1: GAME-BY-GAME MILESTONE PROGRESSION
# ============================================================================
print(f"\n{'='*80}")
print("PART 1: MILESTONE PROGRESSION ANALYSIS")
print(f"{'='*80}")

milestones = list(range(4, 33, 4))  # Games 4, 8, 12, 16, 20, 24, 28, 32

milestone_data_all = []
for milestone in milestones:
    milestone_games = qb_data[qb_data['career_game_number'] == milestone].copy()
    milestone_merged = milestone_games.merge(
        qb_outcomes[['player_id', 'tier', 'career_rating']],
        on='player_id'
    )

    if len(milestone_merged) > 5:
        milestone_data_all.append({
            'game': milestone,
            'elite_avg': milestone_merged[milestone_merged['tier'] == 'Elite']['smoothed_composite'].mean(),
            'elite_min': milestone_merged[milestone_merged['tier'] == 'Elite']['smoothed_composite'].min(),
            'good_avg': milestone_merged[milestone_merged['tier'] == 'Good']['smoothed_composite'].mean(),
            'average_avg': milestone_merged[milestone_merged['tier'] == 'Average']['smoothed_composite'].mean(),
            'below_avg': milestone_merged[milestone_merged['tier'] == 'Below Average']['smoothed_composite'].mean(),
        })

milestone_progression = pd.DataFrame(milestone_data_all)

print("\nProgression of ratings by tier (game-by-game):")
print(f"{'Game':<8} {'Elite Avg':>12} {'Elite Min':>12} {'Good Avg':>12} {'Avg Tier':>12} {'Below Avg':>12}")
print("-" * 80)
for _, row in milestone_progression.iterrows():
    print(f"{int(row['game']):<8} {row['elite_avg']:>12.3f} {row['elite_min']:>12.3f} "
          f"{row['good_avg']:>12.3f} {row['average_avg']:>12.3f} {row['below_avg']:>12.3f}")

# Calculate gap between Elite and Good at each milestone
print(f"\n{'='*70}")
print("SEPARATION ANALYSIS: Elite vs Good Gap")
print(f"{'='*70}")
for _, row in milestone_progression.iterrows():
    gap = row['elite_avg'] - row['good_avg']
    print(f"Game {int(row['game']):>2}: Elite-Good gap = {gap:+.3f}")

# ============================================================================
# PART 2: BUST IDENTIFICATION
# ============================================================================
print(f"\n{'='*80}")
print("PART 2: BUST IDENTIFICATION THRESHOLDS")
print(f"{'='*80}")

bust_thresholds = [8, 12, 16, 20, 24]
for threshold_game in bust_thresholds:
    threshold_data = qb_data[qb_data['career_game_number'] == threshold_game].copy()
    threshold_merged = threshold_data.merge(
        qb_outcomes[['player_id', 'tier', 'career_rating', 'player_name']],
        on='player_id'
    )

    if len(threshold_merged) < 5:
        continue

    # Find rating below which NO elite QBs exist
    elite_qbs = threshold_merged[threshold_merged['tier'] == 'Elite']
    below_qbs = threshold_merged[threshold_merged['tier'] == 'Below Average']

    if len(elite_qbs) > 0:
        elite_min_rating = elite_qbs['smoothed_composite'].min()

        # How many non-elite QBs are above this threshold?
        false_positives = threshold_merged[
            (threshold_merged['smoothed_composite'] >= elite_min_rating) &
            (threshold_merged['tier'] != 'Elite')
        ]

        print(f"\nGAME {threshold_game}:")
        print(f"  Elite minimum rating: {elite_min_rating:+.3f}")
        print(f"  Below this: {len(threshold_merged[threshold_merged['smoothed_composite'] < elite_min_rating])} QBs (0 Elite)")
        print(f"  Above this: {len(false_positives)} non-Elite QBs still developing")

        if len(below_qbs) > 0:
            below_max = below_qbs['smoothed_composite'].max()
            print(f"  Below Average maximum: {below_max:+.3f}")
            print(f"  'Bust Zone': below {below_max:+.3f} at game {threshold_game}")

# ============================================================================
# PART 3: RECOVERY POTENTIAL
# ============================================================================
print(f"\n{'='*80}")
print("PART 3: RECOVERY POTENTIAL ANALYSIS")
print(f"{'='*80}")

# Look at QBs who had poor starts (game 8) but ended up Good/Elite
recovery_analysis = []
for player_id in qualified_qbs:
    player_games = qb_data[qb_data['player_id'] == player_id].sort_values('career_game_number')

    if len(player_games) < 16:
        continue

    game_8_rating = player_games[player_games['career_game_number'] == 8]['smoothed_composite'].values
    game_16_rating = player_games[player_games['career_game_number'] == 16]['smoothed_composite'].values

    if len(game_8_rating) > 0 and len(game_16_rating) > 0:
        player_tier = qb_outcomes[qb_outcomes['player_id'] == player_id]['tier'].values[0]
        player_name = qb_outcomes[qb_outcomes['player_id'] == player_id]['player_name'].values[0]
        career_rating = qb_outcomes[qb_outcomes['player_id'] == player_id]['career_rating'].values[0]

        improvement = game_16_rating[0] - game_8_rating[0]

        recovery_analysis.append({
            'player_id': player_id,
            'player_name': player_name,
            'tier': player_tier,
            'game_8_rating': game_8_rating[0],
            'game_16_rating': game_16_rating[0],
            'improvement': improvement,
            'career_rating': career_rating
        })

recovery_df = pd.DataFrame(recovery_analysis)

# Elite QBs who started slow (below +0.00 at game 8)
elite_slow_starts = recovery_df[
    (recovery_df['tier'] == 'Elite') &
    (recovery_df['game_8_rating'] < 0.0)
]

print(f"\nElite QBs who started below 0.00 at game 8:")
if len(elite_slow_starts) > 0:
    for _, qb in elite_slow_starts.iterrows():
        print(f"  {qb['player_name']:25s} | G8: {qb['game_8_rating']:+.3f} | G16: {qb['game_16_rating']:+.3f} | Improved: {qb['improvement']:+.3f}")
else:
    print("  NONE - All Elite QBs were positive by game 8!")

# QBs with biggest improvement from game 8 to 16
print(f"\nBiggest improvers (Game 8 -> Game 16):")
top_improvers = recovery_df.nlargest(5, 'improvement')
for _, qb in top_improvers.iterrows():
    print(f"  {qb['player_name']:25s} ({qb['tier']:12s}) | G8: {qb['game_8_rating']:+.3f} -> G16: {qb['game_16_rating']:+.3f} | +{qb['improvement']:.3f}")

# QBs with biggest decline
print(f"\nBiggest decliners (Game 8 -> Game 16):")
top_decliners = recovery_df.nsmallest(5, 'improvement')
for _, qb in top_decliners.iterrows():
    print(f"  {qb['player_name']:25s} ({qb['tier']:12s}) | G8: {qb['game_8_rating']:+.3f} -> G16: {qb['game_16_rating']:+.3f} | {qb['improvement']:.3f}")

# ============================================================================
# PART 4: CONSISTENCY PATTERNS
# ============================================================================
print(f"\n{'='*80}")
print("PART 4: CONSISTENCY VS VOLATILITY")
print(f"{'='*80}")

# Calculate std dev of first 16 games for each QB
consistency_data = []
for player_id in qualified_qbs:
    player_games = qb_data[qb_data['player_id'] == player_id].sort_values('career_game_number')

    first_16 = player_games[player_games['career_game_number'] <= 16]

    if len(first_16) >= 10:
        player_tier = qb_outcomes[qb_outcomes['player_id'] == player_id]['tier'].values[0]
        player_name = qb_outcomes[qb_outcomes['player_id'] == player_id]['player_name'].values[0]

        consistency_data.append({
            'player_name': player_name,
            'tier': player_tier,
            'std_first_16': first_16['smoothed_composite'].std(),
            'avg_first_16': first_16['smoothed_composite'].mean()
        })

consistency_df = pd.DataFrame(consistency_data)

print(f"\nVolatility (Std Dev) in first 16 games by tier:")
print(f"{'Tier':<20} {'Count':>7} {'Avg StdDev':>12} {'Min StdDev':>12} {'Max StdDev':>12}")
print("-" * 70)
for tier in ['Elite', 'Good', 'Average', 'Below Average']:
    tier_data = consistency_df[consistency_df['tier'] == tier]
    if len(tier_data) > 0:
        print(f"{tier:<20} {len(tier_data):>7} {tier_data['std_first_16'].mean():>12.3f} "
              f"{tier_data['std_first_16'].min():>12.3f} {tier_data['std_first_16'].max():>12.3f}")

# Are Elite QBs more consistent?
elite_std = consistency_df[consistency_df['tier'] == 'Elite']['std_first_16'].mean()
below_std = consistency_df[consistency_df['tier'] == 'Below Average']['std_first_16'].mean()
print(f"\nElite QBs volatility: {elite_std:.3f}")
print(f"Below Average QBs volatility: {below_std:.3f}")
print(f"Difference: {abs(elite_std - below_std):.3f} ({'Elite more consistent' if elite_std < below_std else 'Busts more volatile'})")

# ============================================================================
# PART 5: SPECIFIC ACTIONABLE THRESHOLDS
# ============================================================================
print(f"\n{'='*80}")
print("PART 5: ACTIONABLE DECISION THRESHOLDS")
print(f"{'='*80}")

print("""
Based on the analysis, here are specific thresholds for QB evaluation:

GAME 8 CHECKPOINTS:
  - Rating >= +0.100 -> High confidence Elite potential
  - Rating >= +0.00  -> Still has Elite potential (all Elite QBs hit this)
  - Rating < +0.00   -> NO Elite QB has ever recovered from this
  - Rating < -0.200  -> Likely bust (Below Average tier)

GAME 16 CHECKPOINTS:
  - Rating >= +0.100 -> Likely Elite/Good outcome
  - Rating >= +0.00  -> Average or better
  - Rating < -0.100  -> Warning signs, unlikely to reach Good
  - Rating < -0.200  -> Strong bust indicator

GAME 32 CHECKPOINTS (2 seasons):
  - Rating >= +0.100 -> Established Elite
  - Rating >= +0.00  -> Solid starter
  - Rating < +0.00   -> Replacement level or below

ROLLING 5-GAME AVERAGES:
  - Peak >= +0.100 in first 32 games -> Elite potential
  - Peak >= +0.00 in first 32 games -> Starter quality
  - Never hit +0.00 -> Bust territory
""")

# Find specific "must hit by game X" thresholds
print(f"\n{'='*70}")
print("ELITE QB REQUIREMENTS (NO EXCEPTIONS)")
print(f"{'='*70}")

for check_game in [4, 8, 12, 16, 20, 24, 28, 32]:
    check_data = qb_data[qb_data['career_game_number'] == check_game].copy()
    check_merged = check_data.merge(
        qb_outcomes[['player_id', 'tier']],
        on='player_id'
    )

    elite_check = check_merged[check_merged['tier'] == 'Elite']

    if len(elite_check) > 0:
        min_elite = elite_check['smoothed_composite'].min()
        elite_below_zero = (elite_check['smoothed_composite'] < 0.0).sum()

        print(f"Game {check_game:>2}: All Elite QBs >= {min_elite:+.3f} | "
              f"{len(elite_check) - elite_below_zero}/{len(elite_check)} positive")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")
