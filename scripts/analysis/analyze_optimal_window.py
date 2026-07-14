"""
Determine Optimal Window for QB Performance Evaluation

Tests different window sizes (5, 10, 16, 20, 32 games) to see which best predicts:
1. Next game performance
2. Next 16 games performance (rest of season)

This will tell us if 20 games is the right threshold for overcoming prior data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# Load QB data
DATA_DIR = Path("data/production")
qb_files = sorted(DATA_DIR.glob("leaf_v2_game_by_game_*.csv"))
DATA_FILE = qb_files[-1]

qb_data = pd.read_csv(DATA_FILE)
qb_data = qb_data.rename(columns={
    'passer_player_id': 'player_id',
    'passer_player_name': 'player_name',
    'opp_adj_base_epa': 'rating'
})

# Sort by player and game
qb_data = qb_data.sort_values(['player_id', 'game_number'])

print("="*80)
print("OPTIMAL WINDOW SIZE ANALYSIS")
print("="*80)
print("\nTesting which recent window best predicts future performance...")

# Test different window sizes
windows = [5, 8, 10, 12, 16, 20, 24, 32, 51]

results = []

for window_size in windows:
    print(f"\nTesting {window_size}-game window...")

    predictions = []

    for player_id in qb_data['player_id'].unique():
        player_games = qb_data[qb_data['player_id'] == player_id].copy()

        # Need at least window_size + 16 games for testing
        if len(player_games) < window_size + 16:
            continue

        # For each possible prediction point (after window_size games, before last 16)
        for i in range(window_size, len(player_games) - 16):
            # Predictor: Average of last window_size games
            recent_window = player_games.iloc[i-window_size:i]['rating'].mean()

            # Outcome 1: Next game
            next_game = player_games.iloc[i]['rating']

            # Outcome 2: Next 16 games average
            next_16 = player_games.iloc[i:i+16]['rating'].mean()

            predictions.append({
                'window_size': window_size,
                'predictor': recent_window,
                'next_game': next_game,
                'next_16': next_16
            })

    pred_df = pd.DataFrame(predictions)

    if len(pred_df) > 50:
        # Correlation with next game
        corr_next, p_next = stats.spearmanr(pred_df['predictor'], pred_df['next_game'])

        # Correlation with next 16 games
        corr_16, p_16 = stats.spearmanr(pred_df['predictor'], pred_df['next_16'])

        # RMSE for next 16 games
        rmse_16 = np.sqrt(((pred_df['predictor'] - pred_df['next_16']) ** 2).mean())

        results.append({
            'window': window_size,
            'n_predictions': len(pred_df),
            'corr_next_game': corr_next,
            'p_next_game': p_next,
            'corr_next_16': corr_16,
            'p_next_16': p_16,
            'rmse_next_16': rmse_16
        })

        print(f"  Correlation with next game: r={corr_next:.3f} (p={p_next:.4f})")
        print(f"  Correlation with next 16 games: r={corr_16:.3f} (p={p_16:.4f})")
        print(f"  RMSE (next 16 games): {rmse_16:.3f}")

results_df = pd.DataFrame(results)

print("\n" + "="*80)
print("SUMMARY: Which window best predicts future performance?")
print("="*80)

print(f"\n{'Window':<8} {'N':>8} {'Next Game r':>14} {'Next 16 r':>14} {'RMSE 16':>12}")
print("-"*70)
for _, row in results_df.iterrows():
    print(f"{int(row['window']):<8} {int(row['n_predictions']):>8} "
          f"{row['corr_next_game']:>14.3f} {row['corr_next_16']:>14.3f} {row['rmse_next_16']:>12.3f}")

# Find optimal windows
best_next_game = results_df.loc[results_df['corr_next_game'].idxmax()]
best_next_16 = results_df.loc[results_df['corr_next_16'].idxmax()]
best_rmse = results_df.loc[results_df['rmse_next_16'].idxmin()]

print("\n" + "="*80)
print("OPTIMAL WINDOWS")
print("="*80)

print(f"\nBest for predicting NEXT GAME:")
print(f"  Window size: {int(best_next_game['window'])} games")
print(f"  Correlation: r={best_next_game['corr_next_game']:.3f}")

print(f"\nBest for predicting NEXT 16 GAMES (rest of season):")
print(f"  Window size: {int(best_next_16['window'])} games")
print(f"  Correlation: r={best_next_16['corr_next_16']:.3f}")
print(f"  RMSE: {best_next_16['rmse_next_16']:.3f}")

print(f"\nLowest RMSE for next 16 games:")
print(f"  Window size: {int(best_rmse['window'])} games")

# Check if 20 is reasonable
window_20 = results_df[results_df['window'] == 20].iloc[0] if 20 in results_df['window'].values else None

if window_20 is not None:
    print(f"\nYour proposed 20-game window:")
    print(f"  Next game correlation: r={window_20['corr_next_game']:.3f}")
    print(f"  Next 16 games correlation: r={window_20['corr_next_16']:.3f}")
    print(f"  RMSE: {window_20['rmse_next_16']:.3f}")

    # Compare to best
    if window_20['corr_next_16'] >= best_next_16['corr_next_16'] * 0.95:
        print(f"  => 20-game window is within 95% of optimal!")
    else:
        pct_diff = (1 - window_20['corr_next_16'] / best_next_16['corr_next_16']) * 100
        print(f"  => {pct_diff:.1f}% worse than optimal {int(best_next_16['window'])}-game window")

# Test for diminishing returns
print("\n" + "="*80)
print("DIMINISHING RETURNS ANALYSIS")
print("="*80)
print("\nHow much does adding more games improve prediction?")

for i in range(len(results_df) - 1):
    curr = results_df.iloc[i]
    next_row = results_df.iloc[i + 1]

    improvement = next_row['corr_next_16'] - curr['corr_next_16']
    games_added = next_row['window'] - curr['window']
    improvement_per_game = improvement / games_added if games_added > 0 else 0

    print(f"  {int(curr['window']):>2} -> {int(next_row['window']):>2} games: "
          f"+{improvement:+.4f} correlation (+{improvement_per_game:+.5f} per game)")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

# Find point of diminishing returns (< 0.001 improvement per game)
for i in range(len(results_df) - 1):
    curr = results_df.iloc[i]
    next_row = results_df.iloc[i + 1]

    improvement = next_row['corr_next_16'] - curr['corr_next_16']
    games_added = next_row['window'] - curr['window']
    improvement_per_game = improvement / games_added if games_added > 0 else 0

    if improvement_per_game < 0.001 and curr['window'] >= 10:
        print(f"\nDiminishing returns start after {int(curr['window'])} games")
        print(f"Beyond this, adding more history provides minimal predictive value.")
        break

print(f"\nFor CURRENT PERFORMANCE evaluation:")
print(f"  Recommended window: {int(best_next_16['window'])} games")
print(f"  This maximizes prediction of rest-of-season performance")

if window_20 is not None and window_20['corr_next_16'] >= best_next_16['corr_next_16'] * 0.98:
    print(f"\n20-game window is acceptable (within 2% of optimal)")
    print(f"Good balance of responsiveness and stability")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
