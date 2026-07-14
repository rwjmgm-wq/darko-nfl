"""
Analyze DT (Interior Defensive Line) Career Trajectory Patterns

Examines how DT ratings change over career games to find:
- Improvement rates for early career DTs
- Peak performance windows
- Decline rates for late career DTs

This will inform data-driven prediction algorithms for DT visualizer.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Load DT composite game-by-game data
DATA_DIR = Path("data/production")
dt_files = sorted(DATA_DIR.glob("dt_composite_game_by_game_*.csv"))

if not dt_files:
    raise FileNotFoundError("No DT game-by-game data found. Run generate_dt_game_by_game.py first.")

DATA_FILE = dt_files[-1]  # Most recent

print("="*80)
print("DT (INTERIOR DL) CAREER TRAJECTORY ANALYSIS")
print("="*80)

# Load data
dt_data = pd.read_csv(DATA_FILE)
print(f"\nLoaded {len(dt_data):,} DT game records")
print(f"Unique DTs: {dt_data['player_id'].nunique()}")

# Filter to DTs with reasonable sample (regular contributors)
dt_summary = dt_data.groupby('player_id').agg({
    'sacks': 'sum',
    'pressures': 'sum',
    'career_game_number': 'count'
}).reset_index()
dt_summary.columns = ['player_id', 'total_sacks', 'total_pressures', 'total_games']

# Keep DTs with 20+ games (1+ season threshold)
qualified_dts = dt_summary[dt_summary['total_games'] >= 20]['player_id']

dt_data = dt_data[dt_data['player_id'].isin(qualified_dts)].copy()
print(f"Filtered to {len(qualified_dts)} qualified DTs ({len(dt_data):,} games)")
print(f"Average games per DT: {len(dt_data) / len(qualified_dts):.1f}")

print(f"\n{'='*80}")
print(f"CAREER STAGE ANALYSIS")
print(f"{'='*80}")

# Group by career game bins (16 game seasons)
bins = [0, 16, 32, 48, 64, 80, 500]
labels = ['Games 1-16\n(Year 1)', 'Games 17-32\n(Year 2)', 'Games 33-48\n(Year 3)',
          'Games 49-64\n(Year 4)', 'Games 65-80\n(Year 5)', 'Games 81+\n(Veteran)']

dt_data['game_bin'] = pd.cut(dt_data['career_game_number'], bins=bins, labels=labels, right=False)

# Calculate average composite rating by career game bin
trajectory = dt_data.groupby('game_bin', observed=True)['smoothed_composite'].agg(['mean', 'median', 'std', 'count'])

print(f"\n{'='*70}")
print(f"Average Composite Rating by Career Stage")
print(f"{'='*70}")
print(trajectory)

# Find year-over-year changes
print(f"\n{'='*70}")
print(f"Year-over-Year Changes")
print(f"{'='*70}")

for i in range(len(trajectory) - 1):
    current_avg = trajectory.iloc[i]['mean']
    next_avg = trajectory.iloc[i+1]['mean']
    change = next_avg - current_avg
    pct_change = (change/abs(trajectory.iloc[i]['mean'])*100) if trajectory.iloc[i]['mean'] != 0 else 0
    print(f"{labels[i]:25} -> {labels[i+1]:25}: {change:+.4f} ({pct_change:+.1f}%)")

# Analyze by career game number (more granular - 8 game bins)
dt_data['game_group'] = (dt_data['career_game_number'] // 8) * 8
game_trajectory = dt_data.groupby('game_group')['smoothed_composite'].agg(['mean', 'count'])
game_trajectory = game_trajectory[game_trajectory['count'] >= 50]  # Min 50 games for reliability

# Find peak performance window
peak_idx = game_trajectory['mean'].idxmax()
peak_games = f"{int(peak_idx)}-{int(peak_idx)+7}"
peak_rating = game_trajectory.loc[peak_idx, 'mean']

print(f"\n{'='*70}")
print(f"Peak Performance Window")
print(f"{'='*70}")
print(f"Peak Games: {peak_games}")
print(f"Peak Rating: {peak_rating:.4f}")

# Calculate improvement rate for rookie contract (games 0-48)
early_trajectory = game_trajectory[game_trajectory.index <= 48]
if len(early_trajectory) >= 3:
    x = early_trajectory.index.values
    y = early_trajectory['mean'].values
    slope_early, intercept_early, r_early, _, _ = stats.linregress(x, y)

    print(f"\nEarly Career - Rookie Contract (Games 0-48):")
    print(f"  Improvement per 16 games (1 season): {slope_early * 16:+.4f}")
    print(f"  Improvement over 48 games (3 seasons): {slope_early * 48:+.4f}")
    print(f"  R-squared: {r_early**2:.3f}")

# Calculate mid-career rate (games 48-80) - prime years
mid_trajectory = game_trajectory[(game_trajectory.index >= 48) & (game_trajectory.index < 80)]
if len(mid_trajectory) >= 3:
    x = mid_trajectory.index.values
    y = mid_trajectory['mean'].values
    slope_mid, intercept_mid, r_mid, _, _ = stats.linregress(x, y)

    print(f"\nMid Career - Prime Years (Games 48-80):")
    print(f"  Change per 16 games (1 season): {slope_mid * 16:+.4f}")
    print(f"  Change over 32 games (2 seasons): {slope_mid * 32:+.4f}")
    print(f"  R-squared: {r_mid**2:.3f}")

# Calculate decline rate (games 80+) - veteran stage
late_trajectory = game_trajectory[game_trajectory.index >= 80]
if len(late_trajectory) >= 3:
    x = late_trajectory.index.values
    y = late_trajectory['mean'].values
    slope_late, intercept_late, r_late, _, _ = stats.linregress(x, y)

    print(f"\nLate Career - Veteran Stage (Games 80+):")
    print(f"  Change per 16 games (1 season): {slope_late * 16:+.4f}")
    print(f"  R-squared: {r_late**2:.3f}")

# Analyze individual DT trajectories
print(f"\n{'='*70}")
print(f"Individual DT Patterns")
print(f"{'='*70}")

dt_patterns = []

for dt_id in dt_data['player_id'].unique():
    dt_games = dt_data[dt_data['player_id'] == dt_id].copy()

    if len(dt_games) < 40:
        continue

    dt_games = dt_games.sort_values('career_game_number')

    # Split into early (first 32 games = 2 years) and late (games 32+)
    early_games = dt_games[dt_games['career_game_number'] <= 32]
    late_games = dt_games[dt_games['career_game_number'] > 32]

    if len(early_games) >= 16 and len(late_games) >= 16:
        early_avg = early_games['smoothed_composite'].mean()
        late_avg = late_games['smoothed_composite'].mean()
        improvement = late_avg - early_avg

        dt_patterns.append({
            'player_id': dt_id,
            'player_name': dt_games['player_name'].iloc[0],
            'total_games': len(dt_games),
            'early_rating': early_avg,
            'late_rating': late_avg,
            'improvement': improvement,
            'improved': improvement > 0
        })

patterns_df = pd.DataFrame(dt_patterns)

if len(patterns_df) > 0:
    improved_pct = (patterns_df['improved'].sum() / len(patterns_df) * 100)
    print(f"\nDTs who improved from Year 1-2 to Year 3+: {improved_pct:.1f}%")
    print(f"  ({patterns_df['improved'].sum()} of {len(patterns_df)} DTs)")
    print(f"\nAverage early rating (Games 1-32): {patterns_df['early_rating'].mean():+.3f}")
    print(f"Average late rating (Games 33+): {patterns_df['late_rating'].mean():+.3f}")
    print(f"Average improvement: {patterns_df['improvement'].mean():+.3f}")

# Analyze post-peak performance
print(f"\n{'='*70}")
print(f"Post-Peak Analysis")
print(f"{'='*70}")

# DTs who peaked and continued playing
peak_game_num = int(peak_idx)
post_peak_dts = dt_data[dt_data['career_game_number'] > peak_game_num].copy()

if len(post_peak_dts) > 0:
    # Group by player and see if they maintained/improved after peak
    post_peak_summary = post_peak_dts.groupby('player_id').agg({
        'smoothed_composite': 'mean',
        'career_game_number': 'count'
    }).reset_index()
    post_peak_summary.columns = ['player_id', 'post_peak_rating', 'post_peak_games']

    # Filter to DTs with 16+ post-peak games
    post_peak_summary = post_peak_summary[post_peak_summary['post_peak_games'] >= 16]

    maintained_performance = (post_peak_summary['post_peak_rating'] >= peak_rating * 0.9).sum()
    pct_maintained = (maintained_performance / len(post_peak_summary) * 100) if len(post_peak_summary) > 0 else 0

    print(f"DTs with 16+ games after peak ({peak_games}): {len(post_peak_summary)}")
    print(f"DTs maintaining 90%+ of peak rating: {maintained_performance} ({pct_maintained:.1f}%)")
    print(f"Average post-peak rating: {post_peak_summary['post_peak_rating'].mean():+.3f}")
    print(f"Peak rating: {peak_rating:+.3f}")
    print(f"Average decline: {(post_peak_summary['post_peak_rating'].mean() - peak_rating):.3f}")

# Summary for visualizer predictions
print(f"\n{'='*70}")
print(f"SUMMARY FOR VISUALIZER PREDICTIONS")
print(f"{'='*70}")

if len(early_trajectory) >= 3:
    improvement_per_season = slope_early * 16
    print(f"\nEarly Career (Games 0-48):")
    print(f"  Recommended adjustment: +{improvement_per_season:.4f} per season")
    print(f"  (Linear improvement during rookie contract)")

if len(mid_trajectory) >= 3:
    mid_change_per_season = slope_mid * 16
    print(f"\nMid Career (Games 48-80):")
    print(f"  Recommended adjustment: {mid_change_per_season:+.4f} per season")
    print(f"  (Peak plateau or slight improvement)")

if len(late_trajectory) >= 3:
    decline_per_season = slope_late * 16
    print(f"\nLate Career (Games 80+):")
    print(f"  Recommended adjustment: {decline_per_season:.4f} per season")
    print(f"  (Veteran decline)")

print(f"\nPeak Window: Games {peak_games}")
print(f"Regression to mean: 20% per year (standard)")

print(f"\n{'='*70}")
print(f"ANALYSIS COMPLETE")
print(f"{'='*70}")
