"""
Analyze TE Career Trajectory Patterns

Examines how TE receiving ratings change over career games to find:
- Improvement rates for early career TEs
- Peak performance windows
- Decline rates for late career TEs

Note: This analyzes RECEIVING performance only. Blocking development is not measured.

This will inform data-driven prediction algorithms for TE visualizer.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Load TE composite game-by-game data
DATA_DIR = Path("data/production")
te_files = sorted(DATA_DIR.glob("te_composite_game_by_game_*.csv"))

if not te_files:
    raise FileNotFoundError("No TE game-by-game data found. Run generate_te_game_by_game.py first.")

DATA_FILE = te_files[-1]  # Most recent

print("="*80)
print("TE CAREER TRAJECTORY ANALYSIS")
print("="*80)
print("NOTE: Analyzing RECEIVING performance only (blocking not measured)")

# Load data
te_data = pd.read_csv(DATA_FILE)
print(f"\nLoaded {len(te_data):,} TE game records")

# Filter to TEs with reasonable sample (15+ targets per game on average)
te_summary = te_data.groupby('receiver_player_id').agg({
    'targets': 'mean',
    'career_game_number': 'count'
}).reset_index()
te_summary.columns = ['receiver_player_id', 'avg_targets', 'total_games']

# Keep TEs with 2+ avg targets and 20+ games (regular contributor threshold)
qualified_tes = te_summary[
    (te_summary['avg_targets'] >= 2) &
    (te_summary['total_games'] >= 20)
]['receiver_player_id']

te_data = te_data[te_data['receiver_player_id'].isin(qualified_tes)].copy()
print(f"Filtered to {len(qualified_tes)} qualified TEs ({len(te_data):,} games)")

print(f"\n{'='*80}")
print(f"TE CAREER TRAJECTORY ANALYSIS")
print(f"{'='*80}")

# Group by career game bins
bins = [0, 16, 32, 48, 64, 500]
labels = ['Games 1-16\n(Year 1)', 'Games 17-32\n(Year 2)', 'Games 33-48\n(Year 3)',
          'Games 49-64\n(Year 4)', 'Games 65+\n(Veteran)']

te_data['game_bin'] = pd.cut(te_data['career_game_number'], bins=bins, labels=labels, right=False)

# Calculate average composite rating by career game bin
trajectory = te_data.groupby('game_bin', observed=True)['smoothed_composite'].agg(['mean', 'median', 'std', 'count'])

print(f"\n{'='*60}")
print(f"Average Composite Rating by Career Stage")
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
te_data['game_group'] = (te_data['career_game_number'] // 8) * 8  # 8-game bins
game_trajectory = te_data.groupby('game_group')['smoothed_composite'].agg(['mean', 'count'])
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

# Calculate improvement rate (games 0-48) - TEs develop steadily
early_trajectory = game_trajectory[game_trajectory.index <= 48]
if len(early_trajectory) >= 3:
    x = early_trajectory.index.values
    y = early_trajectory['mean'].values
    slope_early, intercept_early, r_early, _, _ = stats.linregress(x, y)
    print(f"\nEarly Career (Games 0-48):")
    print(f"  Improvement per 16 games: {slope_early * 16:+.4f}")
    print(f"  R-squared: {r_early**2:.3f}")

# Calculate mid-career rate (games 48-80)
mid_trajectory = game_trajectory[(game_trajectory.index >= 48) & (game_trajectory.index < 80)]
if len(mid_trajectory) >= 3:
    x = mid_trajectory.index.values
    y = mid_trajectory['mean'].values
    slope_mid, intercept_mid, r_mid, _, _ = stats.linregress(x, y)
    print(f"\nMid Career (Games 48-80):")
    print(f"  Change per 16 games: {slope_mid * 16:+.4f}")
    print(f"  R-squared: {r_mid**2:.3f}")

# Calculate decline rate (games 80+) - late career stage
late_trajectory = game_trajectory[game_trajectory.index >= 80]
if len(late_trajectory) >= 3:
    x = late_trajectory.index.values
    y = late_trajectory['mean'].values
    slope_late, intercept_late, r_late, _, _ = stats.linregress(x, y)
    print(f"\nLate Career (Games 80+):")
    print(f"  Change per 16 games: {slope_late * 16:+.4f}")
    print(f"  R-squared: {r_late**2:.3f}")

# Analyze individual TE trajectories
print(f"\n{'='*60}")
print(f"Individual TE Patterns (40+ games)")
print(f"{'='*60}")

te_patterns = []

for te_id in te_data['receiver_player_id'].unique():
    te_games = te_data[te_data['receiver_player_id'] == te_id].copy()

    if len(te_games) < 40:
        continue

    te_games = te_games.sort_values('career_game_number')

    # Split into early (first 24 games) and late (games 24+)
    early_games = te_games[te_games['career_game_number'] < 24]
    late_games = te_games[te_games['career_game_number'] >= 24]

    if len(early_games) >= 10 and len(late_games) >= 10:
        early_avg = early_games['smoothed_composite'].mean()
        late_avg = late_games['smoothed_composite'].mean()
        change = late_avg - early_avg

        te_patterns.append({
            'player_name': te_games['receiver_player_name'].iloc[0],
            'total_games': len(te_games),
            'early_avg': early_avg,
            'late_avg': late_avg,
            'change': change,
            'improved': change > 0.05  # 0.05 threshold for meaningful improvement
        })

patterns_df = pd.DataFrame(te_patterns)

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
    print(f"TEs with 40+ games: {len(patterns_df)}")
    print(f"Improved from early to late: {pct_improved:.1f}%")
    print(f"Average improvement: {avg_change_improvers:+.3f}")
    print(f"Average decline: {avg_change_decliners:+.3f}")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('TE Receiving Career Trajectory Analysis', fontsize=16, fontweight='bold')

# Plot 1: Average by career stage
ax1 = axes[0, 0]
trajectory_plot = trajectory.reset_index()
ax1.bar(range(len(trajectory_plot)), trajectory_plot['mean'], yerr=trajectory_plot['std'],
        capsize=5, alpha=0.7, color='#27ae60')
ax1.set_xticks(range(len(trajectory_plot)))
ax1.set_xticklabels(trajectory_plot['game_bin'], rotation=45, ha='right')
ax1.set_ylabel('Average Composite Rating')
ax1.set_title('Average Rating by Career Stage')
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Granular trajectory (8-game bins)
ax2 = axes[0, 1]
game_trajectory_plot = game_trajectory.reset_index()
ax2.plot(game_trajectory_plot['game_group'], game_trajectory_plot['mean'],
         marker='o', linewidth=2, markersize=6, color='#27ae60')
ax2.set_xlabel('Career Game Number')
ax2.set_ylabel('Average Composite Rating')
ax2.set_title('Detailed Career Trajectory (8-Game Bins)')
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.axvline(x=peak_idx, color='blue', linestyle=':', alpha=0.5, label=f'Peak: Games {peak_games}')
ax2.grid(alpha=0.3)
ax2.legend()

# Plot 3: Distribution of early->late changes
ax3 = axes[1, 0]
if len(patterns_df) > 0:
    ax3.hist(patterns_df['change'], bins=20, alpha=0.7, edgecolor='black', color='#27ae60')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No Change')
    ax3.axvline(x=patterns_df['change'].mean(), color='blue', linestyle='--', linewidth=2, label='Mean Change')
    ax3.set_xlabel('Rating Change (Late - Early Career)')
    ax3.set_ylabel('Number of TEs')
    ax3.set_title('Distribution of Career Changes (40+ games)')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

# Plot 4: Sample count by career games
ax4 = axes[1, 1]
game_trajectory_count = game_trajectory.reset_index()
ax4.bar(game_trajectory_count['game_group'], game_trajectory_count['count'],
        alpha=0.7, color='#27ae60')
ax4.set_xlabel('Career Game Number')
ax4.set_ylabel('Number of Game Records')
ax4.set_title('Sample Size by Career Stage')
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()

# Save figure
output_file = "career_trajectory_te.png"
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\nSaved visualization to: {output_file}")
plt.close()

print(f"\n{'='*80}")
print("KEY FINDINGS FOR TE PREDICTIONS")
print(f"{'='*80}")
print("Use these empirical rates for data-driven TE career predictions:")
if len(early_trajectory) >= 3:
    print(f"  Early Career (Games 0-48): {slope_early * 16:+.4f} per season")
if len(mid_trajectory) >= 3:
    print(f"  Mid Career (Games 48-80): {slope_mid * 16:+.4f} per season")
if len(late_trajectory) >= 3:
    print(f"  Late Career (Games 80+): {slope_late * 16:+.4f} per season")
print(f"\nPeak: Games {peak_games} (Rating: {peak_rating:.3f})")
print(f"\nNote: These rates reflect RECEIVING development only")
print(f"{'='*80}")
