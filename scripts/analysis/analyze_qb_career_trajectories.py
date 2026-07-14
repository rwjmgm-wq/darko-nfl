"""
Analyze QB Career Trajectory Patterns

Examines how QB LEAF ratings change over career games to find:
- Improvement rates for early career QBs
- Peak performance windows
- Decline rates for late career QBs
- Position-specific patterns

This will inform data-driven prediction algorithms for QB visualizer.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Load QB LEAF game-by-game data
DATA_FILE = Path("data/production/leaf_v2_game_by_game_20251105.csv")

print("="*80)
print("QB CAREER TRAJECTORY ANALYSIS")
print("="*80)

# Load data
qb_data = pd.read_csv(DATA_FILE)
print(f"\nLoaded {len(qb_data):,} QB game records")

# Filter to QBs with reasonable sample (30+ attempts per game on average)
qb_summary = qb_data.groupby('passer_player_id').agg({
    'attempts': 'mean',
    'game_number': 'count'
}).reset_index()
qb_summary.columns = ['passer_player_id', 'avg_attempts', 'total_games']

# Keep QBs with 20+ avg attempts and 20+ games (starter threshold)
qualified_qbs = qb_summary[
    (qb_summary['avg_attempts'] >= 20) &
    (qb_summary['total_games'] >= 20)
]['passer_player_id']

qb_data = qb_data[qb_data['passer_player_id'].isin(qualified_qbs)].copy()
print(f"Filtered to {len(qualified_qbs)} qualified starting QBs ({len(qb_data):,} games)")

print(f"\n{'='*80}")
print(f"QB CAREER TRAJECTORY ANALYSIS")
print(f"{'='*80}")

# Group by career game bins
bins = [0, 16, 32, 48, 64, 500]
labels = ['Games 1-16\n(Year 1)', 'Games 17-32\n(Year 2)', 'Games 33-48\n(Year 3)',
          'Games 49-64\n(Year 4)', 'Games 65+\n(Veteran)']

qb_data['game_bin'] = pd.cut(qb_data['game_number'], bins=bins, labels=labels, right=False)

# Calculate average LEAF rating by career game bin
trajectory = qb_data.groupby('game_bin', observed=True)['leaf_rating'].agg(['mean', 'median', 'std', 'count'])

print(f"\n{'='*60}")
print(f"Average LEAF Rating by Career Stage")
print(f"{'='*60}")
print(trajectory)

# Find year-over-year changes
print(f"\n{'='*60}")
print(f"Year-over-Year Changes")
print(f"{'='*60}")

for i in range(len(trajectory) - 1):
    current_avg = trajectory.iloc[i]['mean']
    next_avg = trajectory.iloc[i+1]['mean']
    change = next_avg - current_avg
    pct_change = (change/trajectory.iloc[i]['mean']*100) if trajectory.iloc[i]['mean'] != 0 else 0
    print(f"{labels[i]:20} -> {labels[i+1]:20}: {change:+.4f} ({pct_change:.1f}% relative)")

# Analyze by career game number (more granular)
qb_data['game_group'] = (qb_data['game_number'] // 8) * 8  # 8-game bins
game_trajectory = qb_data.groupby('game_group')['leaf_rating'].agg(['mean', 'count'])
game_trajectory = game_trajectory[game_trajectory['count'] >= 30]  # Min 30 games for reliability

# Find peak performance window
peak_idx = game_trajectory['mean'].idxmax()
peak_games = f"{int(peak_idx)}-{int(peak_idx)+7}"
peak_rating = game_trajectory.loc[peak_idx, 'mean']

print(f"\n{'='*60}")
print(f"Peak Performance")
print(f"{'='*60}")
print(f"Peak Games: {peak_games}")
print(f"Peak Rating: {peak_rating:.4f}")

# Calculate improvement rate (games 0-48) - QBs develop slower
early_trajectory = game_trajectory[game_trajectory.index <= 48]
if len(early_trajectory) >= 3:
    x = early_trajectory.index.values
    y = early_trajectory['mean'].values
    slope_early, intercept_early, r_early, _, _ = stats.linregress(x, y)
    print(f"\nEarly Career (Games 0-48):")
    print(f"  Improvement per 16 games: {slope_early * 16:+.4f}")
    print(f"  R-squared: {r_early**2:.3f}")

# Calculate mid-career rate (games 48-96)
mid_trajectory = game_trajectory[(game_trajectory.index >= 48) & (game_trajectory.index < 96)]
if len(mid_trajectory) >= 3:
    x = mid_trajectory.index.values
    y = mid_trajectory['mean'].values
    slope_mid, intercept_mid, r_mid, _, _ = stats.linregress(x, y)
    print(f"\nMid Career (Games 48-96):")
    print(f"  Change per 16 games: {slope_mid * 16:+.4f}")
    print(f"  R-squared: {r_mid**2:.3f}")

# Calculate decline rate (games 96+) - veteran stage
late_trajectory = game_trajectory[game_trajectory.index >= 96]
if len(late_trajectory) >= 3:
    x = late_trajectory.index.values
    y = late_trajectory['mean'].values
    slope_late, intercept_late, r_late, _, _ = stats.linregress(x, y)
    print(f"\nLate Career (Games 96+):")
    print(f"  Change per 16 games: {slope_late * 16:+.4f}")
    print(f"  R-squared: {r_late**2:.3f}")

# Analyze individual QB trajectories
print(f"\n{'='*60}")
print(f"Individual QB Patterns (50+ games)")
print(f"{'='*60}")

qb_patterns = []

for qb_id in qb_data['passer_player_id'].unique():
    qb_games = qb_data[qb_data['passer_player_id'] == qb_id].copy()

    if len(qb_games) < 50:
        continue

    qb_games = qb_games.sort_values('game_number')

    # Split into early (first 32 games) and late (games 32+)
    early_games = qb_games[qb_games['game_number'] < 32]
    late_games = qb_games[qb_games['game_number'] >= 32]

    if len(early_games) >= 10 and len(late_games) >= 10:
        early_avg = early_games['leaf_rating'].mean()
        late_avg = late_games['leaf_rating'].mean()
        change = late_avg - early_avg

        qb_patterns.append({
            'player_name': qb_games['passer_player_name'].iloc[0],
            'total_games': len(qb_games),
            'early_avg': early_avg,
            'late_avg': late_avg,
            'change': change,
            'improved': change > 0.05  # 0.05 threshold for meaningful improvement
        })

patterns_df = pd.DataFrame(qb_patterns)

if len(patterns_df) > 0:
    improvers = patterns_df[patterns_df['improved']].sort_values('change', ascending=False).head(10)
    decliners = patterns_df[~patterns_df['improved']].sort_values('change').head(10)

    print(f"\nTop 10 Improvers (Early -> Late Career):")
    for _, row in improvers.iterrows():
        print(f"  {row['player_name']:25} {row['early_avg']:+.3f} -> {row['late_avg']:+.3f} ({row['change']:+.3f})")

    print(f"\nTop 10 Decliners (Early -> Late Career):")
    for _, row in decliners.iterrows():
        print(f"  {row['player_name']:25} {row['early_avg']:+.3f} -> {row['late_avg']:+.3f} ({row['change']:+.3f})")

    # Calculate overall improvement/decline percentages
    pct_improved = (patterns_df['improved'].sum() / len(patterns_df)) * 100
    avg_change_improvers = patterns_df[patterns_df['improved']]['change'].mean()
    avg_change_decliners = patterns_df[~patterns_df['improved']]['change'].mean()

    print(f"\n{'='*60}")
    print(f"Summary Statistics")
    print(f"{'='*60}")
    print(f"QBs with 50+ games: {len(patterns_df)}")
    print(f"Improved from early to late: {pct_improved:.1f}%")
    print(f"Average improvement: {avg_change_improvers:+.3f}")
    print(f"Average decline: {avg_change_decliners:+.3f}")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('QB LEAF Career Trajectory Analysis', fontsize=16, fontweight='bold')

# Plot 1: Average by career stage
ax1 = axes[0, 0]
trajectory_plot = trajectory.reset_index()
ax1.bar(range(len(trajectory_plot)), trajectory_plot['mean'], yerr=trajectory_plot['std'],
        capsize=5, alpha=0.7, color='#e74c3c')
ax1.set_xticks(range(len(trajectory_plot)))
ax1.set_xticklabels(trajectory_plot['game_bin'], rotation=45, ha='right')
ax1.set_ylabel('Average LEAF Rating')
ax1.set_title('Average Rating by Career Stage')
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Granular trajectory (8-game bins)
ax2 = axes[0, 1]
game_trajectory_plot = game_trajectory.reset_index()
ax2.plot(game_trajectory_plot['game_group'], game_trajectory_plot['mean'],
         marker='o', linewidth=2, markersize=6, color='#e74c3c')
ax2.set_xlabel('Career Game Number')
ax2.set_ylabel('Average LEAF Rating')
ax2.set_title('Detailed Career Trajectory (8-Game Bins)')
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.axvline(x=peak_idx, color='blue', linestyle=':', alpha=0.5, label=f'Peak: Games {peak_games}')
ax2.grid(alpha=0.3)
ax2.legend()

# Plot 3: Distribution of early->late changes
ax3 = axes[1, 0]
if len(patterns_df) > 0:
    ax3.hist(patterns_df['change'], bins=30, alpha=0.7, edgecolor='black', color='#e74c3c')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No Change')
    ax3.axvline(x=patterns_df['change'].mean(), color='blue', linestyle='--', linewidth=2, label='Mean Change')
    ax3.set_xlabel('Rating Change (Late - Early Career)')
    ax3.set_ylabel('Number of QBs')
    ax3.set_title('Distribution of Career Changes (50+ games)')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

# Plot 4: Sample count by career games
ax4 = axes[1, 1]
game_trajectory_count = game_trajectory.reset_index()
ax4.bar(game_trajectory_count['game_group'], game_trajectory_count['count'],
        alpha=0.7, color='#e74c3c')
ax4.set_xlabel('Career Game Number')
ax4.set_ylabel('Number of Game Records')
ax4.set_title('Sample Size by Career Stage')
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()

# Save figure
output_file = "career_trajectory_qb.png"
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\nSaved visualization to: {output_file}")
plt.close()

print(f"\n{'='*80}")
print("KEY FINDINGS FOR QB PREDICTIONS")
print(f"{'='*80}")
print("Use these empirical rates for data-driven QB career predictions:")
if len(early_trajectory) >= 3:
    print(f"  Early Career (Games 0-48): {slope_early * 16:+.4f} per season")
if len(mid_trajectory) >= 3:
    print(f"  Mid Career (Games 48-96): {slope_mid * 16:+.4f} per season")
if len(late_trajectory) >= 3:
    print(f"  Late Career (Games 96+): {slope_late * 16:+.4f} per season")
print(f"\nPeak: Games {peak_games} (Rating: {peak_rating:.3f})")
print(f"{'='*80}")
