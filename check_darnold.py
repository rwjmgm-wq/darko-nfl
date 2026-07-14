"""Check Sam Darnold's actual early career trajectory"""
import pandas as pd

df = pd.read_csv('data/production/leaf_v2_game_by_game_20251105.csv')
darnold = df[df['passer_player_name'].str.contains('Darnold', na=False)].sort_values('game_number')

print("="*70)
print("SAM DARNOLD CAREER TRAJECTORY")
print("="*70)
print("\nEarly Career Milestones (using correct opp_adj_base_epa_kalman):")
print(f"{'Game':<6} {'Rating':>10} {'Season':>8} {'Week':>6} {'EPA/play':>10}")
print("-"*70)

for milestone in [4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64, 72, 80]:
    game = darnold[darnold['game_number'] == milestone]
    if len(game) > 0:
        g = game.iloc[0]
        print(f"{milestone:<6} {g['opp_adj_base_epa_kalman']:>10.3f} {int(g['season']):>8} {int(g['week']):>6} {g['epa_per_play']:>10.3f}")

# Calculate career avg
print(f"\n{'='*70}")
print("Career Summary:")
print(f"  Total games: {len(darnold)}")
print(f"  Career avg (Kalman): {darnold['opp_adj_base_epa_kalman'].mean():.3f}")
print(f"  Career avg (raw EPA): {darnold['epa_per_play'].mean():.3f}")
print(f"  Current rating (latest): {darnold['opp_adj_base_epa_kalman'].iloc[-1]:.3f}")

# 2024 performance
darnold_2024 = darnold[darnold['season'] == 2024]
if len(darnold_2024) > 0:
    print(f"\n2024 Season ({len(darnold_2024)} games):")
    print(f"  Avg rating: {darnold_2024['opp_adj_base_epa_kalman'].mean():.3f}")
    print(f"  Raw EPA/play: {darnold_2024['epa_per_play'].mean():.3f}")

# Check if he's improving
early_career = darnold[darnold['game_number'] <= 32]
recent_games = darnold[darnold['game_number'] > 56]
if len(recent_games) > 0:
    print(f"\nImprovement:")
    print(f"  First 32 games avg: {early_career['opp_adj_base_epa_kalman'].mean():.3f}")
    print(f"  Games 57+ avg: {recent_games['opp_adj_base_epa_kalman'].mean():.3f}")
    print(f"  Improvement: {recent_games['opp_adj_base_epa_kalman'].mean() - early_career['opp_adj_base_epa_kalman'].mean():+.3f}")
