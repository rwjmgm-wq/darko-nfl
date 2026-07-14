"""
Quick investigation of leaf_rating calculation issue.
"""
import pandas as pd

df = pd.read_csv('data/production/leaf_v2_game_by_game_20251105.csv')

print("="*80)
print("INVESTIGATING leaf_rating CALCULATION")
print("="*80)

# Check if leaf_rating is just the last game's opp_adj_base_epa_kalman
print("\nHypothesis: leaf_rating is the FINAL opp_adj_base_epa_kalman value for each player")
print("\nChecking sample QBs:")

for qb_name in ['T.Brady', 'P.Mahomes', 'J.Allen', 'S.Darnold']:
    qb_data = df[df['passer_player_name'].str.contains(qb_name.split('.')[-1], na=False)]

    if len(qb_data) > 0:
        # Sort by game number to get progression
        qb_data = qb_data.sort_values('game_number')

        first_game = qb_data.iloc[0]
        last_game = qb_data.iloc[-1]

        print(f"\n{qb_data.iloc[0]['passer_player_name']}:")
        print(f"  Total games: {len(qb_data)}")
        print(f"  First game:")
        print(f"    - leaf_rating: {first_game['leaf_rating']:.6f}")
        print(f"    - opp_adj_base_epa_kalman: {first_game['opp_adj_base_epa_kalman']:.6f}")
        print(f"  Last game:")
        print(f"    - leaf_rating: {last_game['leaf_rating']:.6f}")
        print(f"    - opp_adj_base_epa_kalman: {last_game['opp_adj_base_epa_kalman']:.6f}")
        print(f"  Match (leaf_rating == last opp_adj_base_epa_kalman): {abs(last_game['leaf_rating'] - last_game['opp_adj_base_epa_kalman']) < 0.0001}")

# Check how many unique leaf_rating values each QB has
print("\n" + "="*80)
print("Checking variability of leaf_rating across all QBs:")
print("="*80)

qb_variability = df.groupby('passer_player_name').agg({
    'leaf_rating': ['nunique', 'std'],
    'opp_adj_base_epa_kalman': ['nunique', 'std'],
    'game_number': 'max'
}).reset_index()

qb_variability.columns = ['QB', 'leaf_nunique', 'leaf_std', 'kalman_nunique', 'kalman_std', 'total_games']

# Find QBs with only 1 unique leaf_rating (like Darnold)
constant_rating = qb_variability[qb_variability['leaf_nunique'] == 1]

print(f"\nQBs with constant leaf_rating (like Darnold): {len(constant_rating)}/{len(qb_variability)}")
print(f"QBs with varying leaf_rating: {len(qb_variability) - len(constant_rating)}/{len(qb_variability)}")

if len(constant_rating) > 0:
    print("\nSample QBs with CONSTANT leaf_rating:")
    print(constant_rating.head(10).to_string(index=False))

print("\nSample QBs with VARYING leaf_rating:")
varying_rating = qb_variability[qb_variability['leaf_nunique'] > 1].head(10)
print(varying_rating.to_string(index=False))

# Summary
print("\n" + "="*80)
print("DIAGNOSIS:")
print("="*80)

all_constant = (qb_variability['leaf_nunique'] == 1).all()
if all_constant:
    print("\nALL QBs have constant leaf_rating - this is by design!")
    print("The 'leaf_rating' column appears to be the FINAL career rating for each QB.")
    print("For game-by-game analysis, use 'opp_adj_base_epa_kalman' instead.")
else:
    pct_constant = (qb_variability['leaf_nunique'] == 1).sum() / len(qb_variability) * 100
    print(f"\n{pct_constant:.1f}% of QBs have constant leaf_rating.")
    print("This is likely a data quality issue in the pipeline.")
