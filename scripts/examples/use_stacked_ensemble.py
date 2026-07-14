"""
Example: Using the Stacked Ensemble Predictor

Demonstrates how to use the production-ready stacked ensemble
to predict QB performance for the next 16 games.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pandas as pd
from features.stacked_ensemble_predictor import StackedEnsemblePredictor

print("="*80)
print("STACKED ENSEMBLE PREDICTOR - USAGE EXAMPLE")
print("="*80)

# Load the predictor
print("\n[Step 1] Loading stacked ensemble...")
try:
    predictor = StackedEnsemblePredictor.load('models/stacked_ensemble')
    print("[+] Model loaded successfully!")
except FileNotFoundError:
    print("[-] Models not found. Run train_stacked_ensemble.py first.")
    sys.exit(1)

# Show model info
info = predictor.get_model_info()
print(f"\n[Step 2] Model Information:")
print(f"  Architecture: {info['architecture']}")
print(f"  Target: {info['target']}")
print(f"  Features: {info['features']}")
print(f"  Window: {info['window']} games")

# Load QB data
print(f"\n[Step 3] Loading QB data...")
DATA_DIR = Path("data/production")
qb_files = sorted(DATA_DIR.glob("leaf_v2_game_by_game_*.csv"))
DATA_FILE = qb_files[-1]

qb_data = pd.read_csv(DATA_FILE)
qb_data = qb_data.sort_values(['passer_player_id', 'game_number'])

print(f"[+] Loaded {len(qb_data)} games from {DATA_FILE.name}")

# Example 1: Single QB prediction with components
print(f"\n[Step 4] Example 1: Single QB Prediction")
print("-" * 80)

# Get a QB with sufficient game history
sample_qbs = []
for player_id in qb_data['passer_player_id'].unique():
    player_games = qb_data[qb_data['passer_player_id'] == player_id]
    if len(player_games) >= 12:
        sample_qbs.append({
            'player_id': player_id,
            'player_name': player_games['passer_player_name'].iloc[0],
            'games': len(player_games),
            'recent_epa': player_games.tail(12)['opp_adj_base_epa'].mean()
        })

sample_qbs_df = pd.DataFrame(sample_qbs).sort_values('recent_epa', ascending=False)

# Pick top 5 QBs
print(f"\nPredicting for top 5 QBs by recent EPA:")
for i, (_, qb) in enumerate(sample_qbs_df.head(5).iterrows(), 1):
    player_games = qb_data[qb_data['passer_player_id'] == qb['player_id']].tail(12)

    # Get prediction with components
    pred = predictor.predict(player_games, return_components=True)

    print(f"\n{i}. {qb['player_name']} ({qb['games']} games)")
    print(f"   Stacked Ensemble: {pred['prediction']:+.3f}")
    print(f"   Components:")
    print(f"     Random Forest: {pred['rf_pred']:+.3f}")
    print(f"     XGBoost:       {pred['xgb_pred']:+.3f}")
    print(f"     Baseline:      {pred['baseline_pred']:+.3f}")

# Example 2: Batch prediction for all QBs
print(f"\n[Step 5] Example 2: Batch Prediction")
print("-" * 80)

# Prepare data for batch prediction
players_data = {}
for player_id in qb_data['passer_player_id'].unique():
    player_games = qb_data[qb_data['passer_player_id'] == player_id]
    if len(player_games) >= 12:
        players_data[player_id] = player_games.tail(12)

print(f"\nPredicting for {len(players_data)} QBs with 12+ games...")

# Batch predict
predictions_df = predictor.predict_batch(players_data, return_components=False)

# Add player names
name_map = dict(zip(qb_data['passer_player_id'], qb_data['passer_player_name']))
predictions_df['player_name'] = predictions_df['player_id'].map(name_map)

# Show top 10 and bottom 10
predictions_df = predictions_df.sort_values('prediction', ascending=False)

print(f"\nTop 10 projected QBs (next 16 games):")
print(f"{'Rank':<6} {'QB':<25} {'Prediction':>12}")
print("-" * 50)
for i, (_, qb) in enumerate(predictions_df.head(10).iterrows(), 1):
    print(f"{i:<6} {qb['player_name']:<25} {qb['prediction']:>12.3f}")

print(f"\nBottom 10 projected QBs (next 16 games):")
print(f"{'Rank':<6} {'QB':<25} {'Prediction':>12}")
print("-" * 50)
for i, (_, qb) in enumerate(predictions_df.tail(10).iterrows(), 1):
    print(f"{i:<6} {qb['player_name']:<25} {qb['prediction']:>12.3f}")

# Example 3: Individual QB lookup
print(f"\n[Step 6] Example 3: Interactive Lookup")
print("-" * 80)

qb_names = ['Patrick Mahomes', 'Josh Allen', 'Lamar Jackson', 'Jalen Hurts']

print(f"\nLooking up specific QBs:")
for qb_name in qb_names:
    matches = qb_data[qb_data['passer_player_name'].str.contains(qb_name, case=False, na=False)]

    if len(matches) > 0:
        player_id = matches['passer_player_id'].iloc[0]
        player_games = qb_data[qb_data['passer_player_id'] == player_id].tail(12)

        if len(player_games) >= 3:
            pred = predictor.predict(player_games, return_components=True)

            print(f"\n{qb_name}:")
            print(f"  Recent games: {len(player_games)}")
            print(f"  Recent EPA avg: {player_games['opp_adj_base_epa'].mean():+.3f}")
            print(f"  Predicted next 16 games: {pred['prediction']:+.3f}")
        else:
            print(f"\n{qb_name}: Insufficient data ({len(player_games)} games)")
    else:
        print(f"\n{qb_name}: Not found in dataset")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"""
The stacked ensemble predictor combines:
1. Random Forest (captures non-linear feature interactions)
2. XGBoost (gradient boosting for sequential patterns)
3. Baseline multi-feature rating (interpretable 5-feature system)

Meta-learner weights: RF={info['stacker_params']['alpha']}

USAGE:
1. Single prediction: predictor.predict(player_games)
2. With components: predictor.predict(player_games, return_components=True)
3. Batch prediction: predictor.predict_batch(players_dict)

REQUIREMENTS:
- Minimum 3 games (recommended 12+ for best accuracy)
- Features: epa_mean, opp_adj_success_rate, epa_per_play, qb_epa_mean, success_rate

PRODUCTION READY:
- Validated on held-out test data
- Saved to models/stacked_ensemble/
- Load with: StackedEnsemblePredictor.load('models/stacked_ensemble')
""")
