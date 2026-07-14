"""
Analyze Early Career Predictors for QB Success

Identifies thresholds and indicators in early career that predict eventual:
- Elite status (top tier QBs)
- Good status (above average starters)
- Average status (replacement level)
- Bust status (below replacement)

Answers questions like:
- "If a QB doesn't have X EPA by game 20, can they still be elite?"
- "What's the minimum rolling 5-game average needed to reach good status?"
- "Which early career windows are most predictive?"
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats

# Load QB composite game-by-game data
DATA_DIR = Path("data/production")
qb_files = sorted(DATA_DIR.glob("leaf_v2_game_by_game_*.csv"))

if not qb_files:
    raise FileNotFoundError("No QB game-by-game data found. Run generate_qb_composite.py first.")

DATA_FILE = qb_files[-1]  # Most recent

print("="*80)
print("QB EARLY CAREER PREDICTOR ANALYSIS")
print("="*80)

# Load data
qb_data = pd.read_csv(DATA_FILE)

# Rename columns to match expected names
qb_data = qb_data.rename(columns={
    'passer_player_id': 'player_id',
    'passer_player_name': 'player_name',
    'opp_adj_base_epa_kalman': 'raw_composite'  # Using game-by-game Kalman values (leaf_rating is buggy)
})

# Calculate smoothed composite using EWMA (span=5 for QBs, matching visualizer)
qb_data = qb_data.sort_values(['player_id', 'career_game_number'])
qb_data['smoothed_composite'] = qb_data.groupby('player_id')['raw_composite'].transform(
    lambda x: x.ewm(span=5, adjust=False).mean()
)

print(f"\nLoaded {len(qb_data):,} QB game records")
print(f"Unique QBs: {qb_data['player_id'].nunique()}")

# Filter to QBs with reasonable sample (30+ games to measure career outcome)
qb_summary = qb_data.groupby('player_id').agg({
    'smoothed_composite': 'mean',
    'career_game_number': 'max',
    'player_name': 'first'
}).reset_index()
qb_summary.columns = ['player_id', 'career_avg_rating', 'total_games', 'player_name']

# Keep QBs with 30+ games (2+ seasons as starter)
qualified_qbs = qb_summary[qb_summary['total_games'] >= 30]['player_id']

qb_data = qb_data[qb_data['player_id'].isin(qualified_qbs)].copy()
print(f"Filtered to {len(qualified_qbs)} qualified QBs with 30+ games ({len(qb_data):,} games)")

# Define outcome categories based on career average rating
# Using career smoothed composite as measure of career success
qb_outcomes = qb_data.groupby('player_id').agg({
    'smoothed_composite': 'mean',
    'raw_composite': 'mean',
    'career_game_number': 'max',
    'player_name': 'first'
}).reset_index()
qb_outcomes.columns = ['player_id', 'career_rating', 'career_rating_raw', 'total_games', 'player_name']

# Define tiers using percentiles (more robust than fixed z-scores)
# Elite: Top 15%
# Good: Top 40% (but not Elite)
# Average: Middle 40%
# Below: Bottom 20%
percentiles = qb_outcomes['career_rating'].quantile([0.20, 0.60, 0.85])
qb_outcomes['tier'] = pd.cut(
    qb_outcomes['career_rating'],
    bins=[-np.inf, percentiles[0.20], percentiles[0.60], percentiles[0.85], np.inf],
    labels=['Below Average', 'Average', 'Good', 'Elite']
)

print(f"\nTier thresholds (career rating):")
print(f"  Elite:   > {percentiles[0.85]:.3f} (top 15%)")
print(f"  Good:    > {percentiles[0.60]:.3f} (top 40%)")
print(f"  Average: > {percentiles[0.20]:.3f} (middle 40%)")
print(f"  Below:   < {percentiles[0.20]:.3f} (bottom 20%)")

print(f"\n{'='*80}")
print("CAREER OUTCOME DISTRIBUTION")
print(f"{'='*80}")
print(qb_outcomes['tier'].value_counts().sort_index())

print(f"\n{'='*80}")
print("EXAMPLES BY TIER")
print(f"{'='*80}")

for tier in ['Elite', 'Good', 'Average', 'Below Average']:
    tier_qbs = qb_outcomes[qb_outcomes['tier'] == tier].nlargest(3, 'career_rating')
    print(f"\n{tier}:")
    for _, qb in tier_qbs.iterrows():
        print(f"  {qb['player_name']:25s} | Rating: {qb['career_rating']:+.3f} | Games: {int(qb['total_games'])}")

# Now analyze early career metrics for each tier
print(f"\n{'='*80}")
print("EARLY CAREER THRESHOLD ANALYSIS")
print(f"{'='*80}")

# Test different early career windows
windows = [
    (1, 8, "First 8 games (half season)"),
    (1, 16, "First 16 games (1 season)"),
    (1, 32, "First 32 games (2 seasons)"),
    (17, 32, "Games 17-32 (2nd season)"),
]

rolling_windows = [3, 5, 8, 16]

# Analyze each window
for start_game, end_game, window_name in windows:
    print(f"\n{'='*70}")
    print(f"WINDOW: {window_name}")
    print(f"{'='*70}")

    # Get stats for this window
    window_data = qb_data[
        (qb_data['career_game_number'] >= start_game) &
        (qb_data['career_game_number'] <= end_game)
    ].copy()

    if len(window_data) == 0:
        continue

    # Calculate average rating in this window for each QB
    window_summary = window_data.groupby('player_id').agg({
        'smoothed_composite': 'mean',
        'raw_composite': ['mean', 'max', 'min', 'std'],
        'career_game_number': 'count'
    }).reset_index()

    window_summary.columns = ['player_id', 'avg_rating', 'avg_raw', 'max_raw', 'min_raw', 'std_raw', 'games_in_window']

    # Merge with outcomes
    window_analysis = window_summary.merge(qb_outcomes[['player_id', 'tier', 'career_rating']], on='player_id')

    # Filter to QBs who played enough in this window
    min_games_required = min(5, end_game - start_game + 1)
    window_analysis = window_analysis[window_analysis['games_in_window'] >= min_games_required]

    if len(window_analysis) == 0:
        print(f"Insufficient data for this window")
        continue

    # Stats by tier
    print(f"\nAverage Performance by Eventual Tier:")
    print(f"{'Tier':<20} {'Count':>7} {'Avg Rating':>12} {'Max Game':>10} {'Std Dev':>10}")
    print("-" * 70)

    for tier in ['Elite', 'Good', 'Average', 'Below Average']:
        tier_data = window_analysis[window_analysis['tier'] == tier]
        if len(tier_data) > 0:
            print(f"{tier:<20} {len(tier_data):>7} "
                  f"{tier_data['avg_rating'].mean():>12.3f} "
                  f"{tier_data['max_raw'].mean():>10.3f} "
                  f"{tier_data['std_raw'].mean():>10.3f}")

    # Find thresholds
    print(f"\nTHRESHOLD ANALYSIS:")

    # What % of Elite QBs hit certain thresholds?
    elite_qbs = window_analysis[window_analysis['tier'] == 'Elite']

    if len(elite_qbs) > 0:
        thresholds = [0.0, 0.25, 0.5, 0.75, 1.0]
        print(f"\n% of Elite QBs exceeding threshold:")
        for threshold in thresholds:
            pct = (elite_qbs['avg_rating'] >= threshold).sum() / len(elite_qbs) * 100
            print(f"  Rating >= {threshold:+.2f}: {pct:.1f}%")

        # Find minimum threshold (what's the worst Elite QB's early performance?)
        min_elite = elite_qbs['avg_rating'].min()
        max_elite = elite_qbs['avg_rating'].max()
        print(f"\nElite QB range: {min_elite:+.3f} to {max_elite:+.3f}")
        print(f"Elite QB avg:   {elite_qbs['avg_rating'].mean():+.3f}")

    # Predictive power: correlation between window performance and career outcome
    corr, p_value = stats.spearmanr(window_analysis['avg_rating'], window_analysis['career_rating'])
    print(f"\nPredictive Power (correlation with career rating): r={corr:.3f} (p={p_value:.4f})")

    # Classification accuracy: can we predict tier from this window?
    # Simple threshold approach: what threshold maximizes elite identification?
    best_threshold = None
    best_accuracy = 0

    for test_threshold in np.linspace(-0.5, 1.5, 21):
        predicted_elite = window_analysis['avg_rating'] >= test_threshold
        actual_elite = window_analysis['tier'] == 'Elite'

        if predicted_elite.sum() > 0:
            precision = (predicted_elite & actual_elite).sum() / predicted_elite.sum()
            recall = (predicted_elite & actual_elite).sum() / actual_elite.sum()

            if recall > 0.5:  # Only consider thresholds that catch >50% of elite QBs
                if precision > best_accuracy:
                    best_accuracy = precision
                    best_threshold = test_threshold

    if best_threshold is not None:
        print(f"\nBest threshold for Elite identification: {best_threshold:+.3f}")
        print(f"  Precision: {best_accuracy:.1%} (of QBs above threshold, % that became Elite)")
        predicted = window_analysis['avg_rating'] >= best_threshold
        actual = window_analysis['tier'] == 'Elite'
        recall = (predicted & actual).sum() / actual.sum()
        print(f"  Recall:    {recall:.1%} (of Elite QBs, % identified by threshold)")

# Rolling average analysis
print(f"\n{'='*80}")
print("ROLLING AVERAGE ANALYSIS")
print(f"{'='*80}")

for window_size in rolling_windows:
    print(f"\n{'='*70}")
    print(f"ROLLING {window_size}-GAME AVERAGE")
    print(f"{'='*70}")

    # Calculate rolling average for each QB
    qb_data_sorted = qb_data.sort_values(['player_id', 'career_game_number'])
    qb_data_sorted[f'rolling_{window_size}'] = qb_data_sorted.groupby('player_id')['smoothed_composite'].transform(
        lambda x: x.rolling(window=window_size, min_periods=window_size).mean()
    )

    # For each QB, find their best rolling average in first 32 games
    early_best_rolling = qb_data_sorted[qb_data_sorted['career_game_number'] <= 32].groupby('player_id').agg({
        f'rolling_{window_size}': 'max'
    }).reset_index()
    early_best_rolling.columns = ['player_id', 'best_early_rolling']

    # Merge with outcomes
    rolling_analysis = early_best_rolling.merge(
        qb_outcomes[['player_id', 'tier', 'career_rating', 'player_name']],
        on='player_id'
    )

    # Remove NaNs
    rolling_analysis = rolling_analysis.dropna()

    if len(rolling_analysis) == 0:
        print("Insufficient data")
        continue

    print(f"\nBest {window_size}-game average in first 32 games by tier:")
    print(f"{'Tier':<20} {'Count':>7} {'Avg Best':>12} {'Min Best':>12} {'Max Best':>12}")
    print("-" * 70)

    for tier in ['Elite', 'Good', 'Average', 'Below Average']:
        tier_data = rolling_analysis[rolling_analysis['tier'] == tier]
        if len(tier_data) > 0:
            print(f"{tier:<20} {len(tier_data):>7} "
                  f"{tier_data['best_early_rolling'].mean():>12.3f} "
                  f"{tier_data['best_early_rolling'].min():>12.3f} "
                  f"{tier_data['best_early_rolling'].max():>12.3f}")

    # Key finding: what's the minimum rolling average needed to become elite?
    elite_rolling = rolling_analysis[rolling_analysis['tier'] == 'Elite']
    if len(elite_rolling) > 0:
        min_elite_rolling = elite_rolling['best_early_rolling'].min()
        elite_below_threshold = elite_rolling[elite_rolling['best_early_rolling'] < 0.5]

        print(f"\nMinimum {window_size}-game average for Elite QB: {min_elite_rolling:+.3f}")
        print(f"Elite QBs who never hit +0.50 in first 32 games: {len(elite_below_threshold)}/{len(elite_rolling)}")

        if len(elite_below_threshold) > 0:
            print(f"Examples:")
            for _, qb in elite_below_threshold.iterrows():
                print(f"  {qb['player_name']:25s} | Best {window_size}-game: {qb['best_early_rolling']:+.3f} | Career: {qb['career_rating']:+.3f}")

# Game-by-game milestone analysis
print(f"\n{'='*80}")
print("MILESTONE GAME ANALYSIS")
print(f"{'='*80}")

milestone_games = [8, 16, 20, 24, 32]

for milestone in milestone_games:
    print(f"\n{'='*70}")
    print(f"PERFORMANCE AT GAME {milestone}")
    print(f"{'='*70}")

    # Get rating at specific game number
    milestone_data = qb_data[qb_data['career_game_number'] == milestone].copy()

    if len(milestone_data) < 10:
        print("Insufficient data")
        continue

    # Merge with outcomes
    milestone_analysis = milestone_data.merge(
        qb_outcomes[['player_id', 'tier', 'career_rating']],
        on='player_id'
    )

    print(f"\nSmoothed rating at game {milestone} by eventual tier:")
    print(f"{'Tier':<20} {'Count':>7} {'Avg Rating':>12} {'Min Rating':>12}")
    print("-" * 70)

    for tier in ['Elite', 'Good', 'Average', 'Below Average']:
        tier_data = milestone_analysis[milestone_analysis['tier'] == tier]
        if len(tier_data) > 0:
            print(f"{tier:<20} {len(tier_data):>7} "
                  f"{tier_data['smoothed_composite'].mean():>12.3f} "
                  f"{tier_data['smoothed_composite'].min():>12.3f}")

# Summary of key findings
print(f"\n{'='*80}")
print("KEY FINDINGS SUMMARY")
print(f"{'='*80}")

print("""
This analysis identifies early career thresholds that predict QB success.
Use these insights to evaluate young QBs and project their likely career trajectory.

KEY QUESTIONS ANSWERED:
1. What early windows are most predictive?
2. What minimum thresholds separate elite from non-elite?
3. Can QBs recover from poor early performance?
4. Which metrics/windows best identify future stars?

See detailed results above for specific thresholds and examples.
""")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")
