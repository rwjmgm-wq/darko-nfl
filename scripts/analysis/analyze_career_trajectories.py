"""
Analyze Career Trajectory Patterns

Examines how player ratings change over career games to find:
- Improvement rates for early career players
- Peak performance windows
- Decline rates for late career players
- Position-specific patterns

This will inform data-driven prediction algorithms.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Load data
DATA_DIR = Path("data/production")

print("="*80)
print("CAREER TRAJECTORY ANALYSIS")
print("="*80)

# Load WR data
wr_files = sorted(DATA_DIR.glob("wr_composite_game_by_game_*.csv"))
wr_data = pd.read_csv(wr_files[-1])
print(f"\nLoaded {len(wr_data):,} WR game records")

# Load RB data
rb_files = sorted(DATA_DIR.glob("rb_composite_game_by_game_*.csv"))
rb_data = pd.read_csv(rb_files[-1])
print(f"Loaded {len(rb_data):,} RB game records")

def analyze_position_trajectory(data, position_name, player_id_col):
    """Analyze career trajectory patterns for a position."""

    print(f"\n{'='*80}")
    print(f"{position_name} CAREER TRAJECTORY ANALYSIS")
    print(f"{'='*80}")

    # Group by career game bins (0-15, 16-31, 32-47, 48-63, 64+)
    bins = [0, 16, 32, 48, 64, 200]
    labels = ['Games 1-16\n(Rookie)', 'Games 17-32\n(Year 2)', 'Games 33-48\n(Year 3)',
              'Games 49-64\n(Year 4)', 'Games 65+\n(Veteran)']

    data['game_bin'] = pd.cut(data['career_game_number'], bins=bins, labels=labels, right=False)

    # Calculate average rating by career game bin
    trajectory = data.groupby('game_bin', observed=True)['smoothed_composite'].agg(['mean', 'median', 'std', 'count'])

    print(f"\n{'='*60}")
    print(f"Average Rating by Career Stage")
    print(f"{'='*60}")
    print(trajectory)

    # Find improvement/decline rates
    print(f"\n{'='*60}")
    print(f"Year-over-Year Changes")
    print(f"{'='*60}")

    for i in range(len(trajectory) - 1):
        current_avg = trajectory.iloc[i]['mean']
        next_avg = trajectory.iloc[i+1]['mean']
        change = next_avg - current_avg
        print(f"{labels[i]:20} -> {labels[i+1]:20}: {change:+.4f} ({change/trajectory.iloc[i]['mean']*100 if trajectory.iloc[i]['mean'] != 0 else 0:.1f}% relative)")

    # Analyze by career game number (more granular)
    # Group into smaller bins
    data['game_group'] = (data['career_game_number'] // 8) * 8  # 8-game bins
    game_trajectory = data.groupby('game_group')['smoothed_composite'].agg(['mean', 'count'])
    game_trajectory = game_trajectory[game_trajectory['count'] >= 50]  # Min 50 games for reliability

    # Find peak performance window
    peak_idx = game_trajectory['mean'].idxmax()
    peak_games = f"{int(peak_idx)}-{int(peak_idx)+7}"
    peak_rating = game_trajectory.loc[peak_idx, 'mean']

    print(f"\n{'='*60}")
    print(f"Peak Performance")
    print(f"{'='*60}")
    print(f"Peak Games: {peak_games}")
    print(f"Peak Rating: {peak_rating:.4f}")

    # Calculate improvement rate (games 0-32)
    early_trajectory = game_trajectory[game_trajectory.index <= 32]
    if len(early_trajectory) >= 3:
        x = early_trajectory.index.values
        y = early_trajectory['mean'].values
        slope_early, intercept_early, r_early, _, _ = stats.linregress(x, y)
        print(f"\nEarly Career (Games 0-32):")
        print(f"  Improvement per 16 games: {slope_early * 16:+.4f}")
        print(f"  R-squared: {r_early**2:.3f}")

    # Calculate decline rate (games 48+)
    late_trajectory = game_trajectory[game_trajectory.index >= 48]
    if len(late_trajectory) >= 3:
        x = late_trajectory.index.values
        y = late_trajectory['mean'].values
        slope_late, intercept_late, r_late, _, _ = stats.linregress(x, y)
        print(f"\nLate Career (Games 48+):")
        print(f"  Decline per 16 games: {slope_late * 16:+.4f}")
        print(f"  R-squared: {r_late**2:.3f}")

    # Analyze individual player trajectories
    print(f"\n{'='*60}")
    print(f"Individual Player Patterns (50+ games)")
    print(f"{'='*60}")

    player_patterns = []

    for player_id in data[player_id_col].unique():
        player_games = data[data[player_id_col] == player_id].copy()

        if len(player_games) < 50:
            continue

        player_games = player_games.sort_values('career_game_number')

        # Split into early (first 32 games) and late (games 32+)
        early_games = player_games[player_games['career_game_number'] < 32]
        late_games = player_games[player_games['career_game_number'] >= 32]

        if len(early_games) >= 10 and len(late_games) >= 10:
            early_avg = early_games['smoothed_composite'].mean()
            late_avg = late_games['smoothed_composite'].mean()
            change = late_avg - early_avg

            player_patterns.append({
                'player_name': player_games['receiver_player_name' if position_name == 'WR' else 'player_name'].iloc[0],
                'total_games': len(player_games),
                'early_avg': early_avg,
                'late_avg': late_avg,
                'change': change,
                'improved': change > 0.2
            })

    patterns_df = pd.DataFrame(player_patterns)

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
        print(f"Players with 50+ games: {len(patterns_df)}")
        print(f"Improved from early to late: {pct_improved:.1f}%")
        print(f"Average improvement: {avg_change_improvers:+.3f}")
        print(f"Average decline: {avg_change_decliners:+.3f}")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{position_name} Career Trajectory Analysis', fontsize=16, fontweight='bold')

    # Plot 1: Average by career stage
    ax1 = axes[0, 0]
    trajectory_plot = trajectory.reset_index()
    ax1.bar(range(len(trajectory_plot)), trajectory_plot['mean'], yerr=trajectory_plot['std'],
            capsize=5, alpha=0.7, color='#2ecc71' if position_name == 'WR' else '#9b59b6')
    ax1.set_xticks(range(len(trajectory_plot)))
    ax1.set_xticklabels(trajectory_plot['game_bin'], rotation=45, ha='right')
    ax1.set_ylabel('Average Smoothed Composite')
    ax1.set_title('Average Rating by Career Stage')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Granular trajectory (8-game bins)
    ax2 = axes[0, 1]
    game_trajectory_plot = game_trajectory.reset_index()
    ax2.plot(game_trajectory_plot['game_group'], game_trajectory_plot['mean'],
             marker='o', linewidth=2, markersize=6, color='#2ecc71' if position_name == 'WR' else '#9b59b6')
    ax2.set_xlabel('Career Game Number')
    ax2.set_ylabel('Average Smoothed Composite')
    ax2.set_title('Detailed Career Trajectory (8-Game Bins)')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=peak_idx, color='red', linestyle=':', alpha=0.5, label=f'Peak: Games {peak_games}')
    ax2.grid(alpha=0.3)
    ax2.legend()

    # Plot 3: Distribution of early→late changes
    ax3 = axes[1, 0]
    if len(patterns_df) > 0:
        ax3.hist(patterns_df['change'], bins=30, alpha=0.7, edgecolor='black',
                color='#2ecc71' if position_name == 'WR' else '#9b59b6')
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No Change')
        ax3.axvline(x=patterns_df['change'].mean(), color='blue', linestyle='--', linewidth=2, label='Mean Change')
        ax3.set_xlabel('Rating Change (Late - Early Career)')
        ax3.set_ylabel('Number of Players')
        ax3.set_title('Distribution of Career Changes (50+ games)')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)

    # Plot 4: Sample count by career games
    ax4 = axes[1, 1]
    game_trajectory_count = game_trajectory.reset_index()
    ax4.bar(game_trajectory_count['game_group'], game_trajectory_count['count'],
            alpha=0.7, color='#2ecc71' if position_name == 'WR' else '#9b59b6')
    ax4.set_xlabel('Career Game Number')
    ax4.set_ylabel('Number of Game Records')
    ax4.set_title('Sample Size by Career Stage')
    ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_file = f"career_trajectory_{position_name.lower()}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_file}")
    plt.close()

    return {
        'position': position_name,
        'trajectory': trajectory,
        'game_trajectory': game_trajectory,
        'peak_games': peak_games,
        'peak_rating': peak_rating,
        'early_slope': slope_early if len(early_trajectory) >= 3 else None,
        'late_slope': slope_late if len(late_trajectory) >= 3 else None,
        'patterns': patterns_df if len(patterns_df) > 0 else None
    }

# Analyze both positions
wr_results = analyze_position_trajectory(wr_data, 'WR', 'receiver_player_id')
rb_results = analyze_position_trajectory(rb_data, 'RB', 'player_id')

# Compare positions
print(f"\n{'='*80}")
print("CROSS-POSITION COMPARISON")
print(f"{'='*80}")

print(f"\nPeak Performance:")
print(f"  WR: {wr_results['peak_games']:15} (rating: {wr_results['peak_rating']:.4f})")
print(f"  RB: {rb_results['peak_games']:15} (rating: {rb_results['peak_rating']:.4f})")

if wr_results['early_slope'] and rb_results['early_slope']:
    print(f"\nEarly Career Improvement (per 16 games):")
    print(f"  WR: {wr_results['early_slope']*16:+.4f}")
    print(f"  RB: {rb_results['early_slope']*16:+.4f}")

if wr_results['late_slope'] and rb_results['late_slope']:
    print(f"\nLate Career Decline (per 16 games):")
    print(f"  WR: {wr_results['late_slope']*16:+.4f}")
    print(f"  RB: {rb_results['late_slope']*16:+.4f}")

print(f"\n{'='*80}")
print("COMPLETE")
print(f"{'='*80}")
