"""
Simple Kalman Filter Test

Debug the Kalman filter issue with minimal code.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import pandas as pd
import numpy as np
from features.kalman_filter import QBKalmanFilter
import traceback

# Create simple test data
test_data = pd.DataFrame({
    'season': [2023, 2023, 2023, 2023],
    'week': [1, 2, 3, 4],
    'game_date': pd.date_range('2023-09-01', periods=4, freq='W'),
    'targets': [10, 8, 12, 9],
    'air_epa': [0.15, 0.20, 0.10, 0.18]
})

print("Test data:")
print(test_data)
print()

# Create Kalman filter
kf = QBKalmanFilter(auto_tune=True, em_iterations=5)

print("Testing Kalman filter...")
try:
    estimates, uncertainties = kf.filter_player_metric(
        player_id='test_player',
        player_games=test_data,
        metric='air_epa',
        prior_mean=0.0,
        game_date_col='game_date',
        attempts_col='targets'
    )

    print("SUCCESS!")
    print(f"Estimates: {estimates}")
    print(f"Uncertainties: {uncertainties}")

except Exception as e:
    print(f"ERROR: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
