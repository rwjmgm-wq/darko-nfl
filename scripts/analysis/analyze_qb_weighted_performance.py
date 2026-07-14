"""
QB Performance Analysis with Weighted Recent Performance

Elite status should reflect CURRENT ability, not career average.
This script uses:
1. Last 51 games (3 seasons) weighted most heavily
2. Exponential decay for older games
3. Exception: Proven Elite status if 3+ seasons historically above threshold
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
    'opp_adj_base_epa_kalman': 'rating'
})

# Sort by player and game number
qb_data = qb_data.sort_values(['player_id', 'game_number'])

print("="*80)
print("QB WEIGHTED PERFORMANCE ANALYSIS")
print("="*80)
print(f"\nData range: {qb_data['season'].min()}-{qb_data['season'].max()}")
print(f"Total QB-games: {len(qb_data):,}")

def calculate_weighted_rating(player_games, window=51, decay_rate=0.03):
    """
    Calculate weighted rating with recent games weighted heaviest.

    Args:
        player_games: DataFrame of player's games sorted by game_number
        window: Number of recent games to weight most heavily (default 51 = ~3 seasons)
        decay_rate: Exponential decay rate for older games (default 0.03)

    Returns:
        Weighted average rating
    """
    if len(player_games) == 0:
        return np.nan

    # Get ratings in chronological order
    ratings = player_games['rating'].values
    n_games = len(ratings)

    # Create weights: exponential decay from most recent game
    # Most recent game has weight 1.0
    # Each game back in time has weight exp(-decay_rate * distance)
    games_back = np.arange(n_games)[::-1]  # 0 for most recent, n-1 for oldest
    weights = np.exp(-decay_rate * games_back)

    # Boost recent window (last 51 games) to full weight
    if n_games > window:
        weights[-window:] = 1.0

    # Normalize weights to sum to n_games (so weighted avg is comparable to simple avg)
    weights = weights / weights.sum() * n_games

    # Calculate weighted average
    weighted_avg = np.average(ratings, weights=weights)

    return weighted_avg

def check_proven_elite(player_games, threshold=0.05, min_seasons=3):
    """
    Check if player has proven Elite status (3+ seasons above threshold historically).

    Args:
        player_games: DataFrame of player's games
        threshold: Rating threshold to consider "Elite season" (default +0.05)
        min_seasons: Minimum seasons above threshold (default 3)

    Returns:
        Boolean: True if player has proven Elite history
    """
    # Group by season and calculate season averages
    season_avgs = player_games.groupby('season')['rating'].mean()

    # Count seasons above threshold
    elite_seasons = (season_avgs >= threshold).sum()

    return elite_seasons >= min_seasons

# Calculate weighted ratings for each QB
print("\n" + "="*80)
print("CALCULATING WEIGHTED RATINGS")
print("="*80)

weighted_results = []

for player_id in qb_data['player_id'].unique():
    player_games = qb_data[qb_data['player_id'] == player_id].copy()

    # Get most recent games (for current status)
    current_games = player_games.tail(51)  # Last 51 games (~3 seasons)

    # Calculate metrics
    weighted_rating = calculate_weighted_rating(player_games)
    recent_avg = current_games['rating'].mean()
    career_avg = player_games['rating'].mean()
    proven_elite = check_proven_elite(player_games)

    weighted_results.append({
        'player_id': player_id,
        'player_name': player_games['player_name'].iloc[0],
        'total_games': len(player_games),
        'weighted_rating': weighted_rating,
        'recent_51_avg': recent_avg,
        'career_avg': career_avg,
        'proven_elite': proven_elite,
        'current_rating': player_games['rating'].iloc[-1],
        'latest_season': player_games['season'].iloc[-1]
    })

weighted_df = pd.DataFrame(weighted_results)

# Filter to QBs with 30+ games for meaningful comparison
weighted_df = weighted_df[weighted_df['total_games'] >= 30].copy()

print(f"Qualified QBs (30+ games): {len(weighted_df)}")

# Define tiers based on WEIGHTED RATING (not career average)
# Elite: Top 15% OR proven Elite with recent performance above threshold
# Good: Top 40%
# Average: Middle 40%
# Below: Bottom 20%

percentiles = weighted_df['weighted_rating'].quantile([0.20, 0.60, 0.85])

def classify_tier(row):
    """Classify QB tier based on weighted rating and proven Elite status."""
    rating = row['weighted_rating']
    proven = row['proven_elite']
    recent = row['recent_51_avg']

    # Proven Elite QBs get benefit of doubt if recent performance is decent
    if proven and recent >= 0.00:
        return 'Elite'

    # Otherwise use weighted rating percentiles
    if rating >= percentiles[0.85]:
        return 'Elite'
    elif rating >= percentiles[0.60]:
        return 'Good'
    elif rating >= percentiles[0.20]:
        return 'Average'
    else:
        return 'Below Average'

weighted_df['tier'] = weighted_df.apply(classify_tier, axis=1)

print(f"\nTier thresholds (weighted rating):")
print(f"  Elite:   > {percentiles[0.85]:.3f} (top 15%) OR proven Elite with recent >= 0.00")
print(f"  Good:    > {percentiles[0.60]:.3f} (top 40%)")
print(f"  Average: > {percentiles[0.20]:.3f} (middle 40%)")
print(f"  Below:   < {percentiles[0.20]:.3f} (bottom 20%)")

print(f"\n{'='*80}")
print("TIER DISTRIBUTION")
print(f"{'='*80}")
print(weighted_df['tier'].value_counts().sort_index())

# Show proven Elite QBs
print(f"\n{'='*80}")
print("PROVEN ELITE QBs (3+ Elite seasons historically)")
print(f"{'='*80}")
proven_elite_qbs = weighted_df[weighted_df['proven_elite'] == True].sort_values('weighted_rating', ascending=False)
print(f"\n{len(proven_elite_qbs)} QBs with proven Elite history:\n")
print(f"{'QB':<25} {'Weighted':>10} {'Recent 51':>10} {'Career':>10} {'Tier':>15}")
print("-"*80)
for _, qb in proven_elite_qbs.iterrows():
    print(f"{qb['player_name']:<25} {qb['weighted_rating']:>10.3f} {qb['recent_51_avg']:>10.3f} "
          f"{qb['career_avg']:>10.3f} {qb['tier']:>15}")

# Compare weighted vs career classification
print(f"\n{'='*80}")
print("TOP 20 QBs BY WEIGHTED RATING")
print(f"{'='*80}")

top_20 = weighted_df.nlargest(20, 'weighted_rating')
print(f"\n{'Rank':<6} {'QB':<25} {'Weighted':>10} {'Recent 51':>10} {'Career':>10} {'Tier':>15}")
print("-"*80)
for i, (_, qb) in enumerate(top_20.iterrows(), 1):
    print(f"{i:<6} {qb['player_name']:<25} {qb['weighted_rating']:>10.3f} {qb['recent_51_avg']:>10.3f} "
          f"{qb['career_avg']:>10.3f} {qb['tier']:>15}")

# Show biggest movers (QBs where weighted >> career or career >> weighted)
print(f"\n{'='*80}")
print("BIGGEST IMPROVERS (Weighted Rating >> Career Average)")
print(f"{'='*80}")

weighted_df['improvement'] = weighted_df['weighted_rating'] - weighted_df['career_avg']
improvers = weighted_df.nlargest(10, 'improvement')
print(f"\n{'QB':<25} {'Career':>10} {'Weighted':>10} {'Change':>10} {'Recent 51':>10} {'Tier':>15}")
print("-"*80)
for _, qb in improvers.iterrows():
    print(f"{qb['player_name']:<25} {qb['career_avg']:>10.3f} {qb['weighted_rating']:>10.3f} "
          f"{qb['improvement']:>10.3f} {qb['recent_51_avg']:>10.3f} {qb['tier']:>15}")

print(f"\n{'='*80}")
print("BIGGEST DECLINERS (Career Average >> Weighted Rating)")
print(f"{'='*80}")

decliners = weighted_df.nsmallest(10, 'improvement')
print(f"\n{'QB':<25} {'Career':>10} {'Weighted':>10} {'Change':>10} {'Recent 51':>10} {'Tier':>15}")
print("-"*80)
for _, qb in decliners.iterrows():
    print(f"{qb['player_name']:<25} {qb['career_avg']:>10.3f} {qb['weighted_rating']:>10.3f} "
          f"{qb['improvement']:>10.3f} {qb['recent_51_avg']:>10.3f} {qb['tier']:>15}")

# Special focus on Sam Darnold
print(f"\n{'='*80}")
print("SAM DARNOLD DETAILED ANALYSIS")
print(f"{'='*80}")

darnold = weighted_df[weighted_df['player_name'].str.contains('Darnold', na=False)]
if len(darnold) > 0:
    d = darnold.iloc[0]
    print(f"\nTotal games: {d['total_games']}")
    print(f"Career average: {d['career_avg']:+.3f}")
    print(f"Weighted rating: {d['weighted_rating']:+.3f}")
    print(f"Recent 51 games: {d['recent_51_avg']:+.3f}")
    print(f"Current rating (latest): {d['current_rating']:+.3f}")
    print(f"Improvement (weighted - career): {d['improvement']:+.3f}")
    print(f"Proven Elite status: {d['proven_elite']}")
    print(f"Classification: {d['tier']}")

    # Show recent trajectory (2024-2025)
    darnold_games = qb_data[qb_data['player_name'].str.contains('Darnold', na=False)].copy()
    recent_darnold = darnold_games[darnold_games['season'] >= 2024].sort_values('game_number')

    print(f"\n2024-2025 Performance ({len(recent_darnold)} games):")
    print(f"  Average rating: {recent_darnold['rating'].mean():+.3f}")
    print(f"  2024 only: {darnold_games[darnold_games['season'] == 2024]['rating'].mean():+.3f}")
    print(f"  2025 only: {darnold_games[darnold_games['season'] == 2025]['rating'].mean():+.3f}")

    # Compare to thresholds
    print(f"\nThreshold Analysis:")
    print(f"  Elite threshold (85th percentile): {percentiles[0.85]:+.3f}")
    print(f"  Good threshold (60th percentile): {percentiles[0.60]:+.3f}")
    print(f"  Darnold weighted rating: {d['weighted_rating']:+.3f}")

    if d['weighted_rating'] >= percentiles[0.85]:
        print(f"  => Darnold is ABOVE Elite threshold!")
    elif d['weighted_rating'] >= percentiles[0.60]:
        print(f"  => Darnold is in Good tier (above 60th percentile)")
    else:
        gap_to_good = percentiles[0.60] - d['weighted_rating']
        print(f"  => Darnold needs +{gap_to_good:.3f} to reach Good tier")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")
