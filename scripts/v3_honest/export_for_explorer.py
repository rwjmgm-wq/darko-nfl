"""
Export LEAF v3 ratings in the schema the qb-leaf-explorer Dash app consumes.

Writes into the qb-leaf-explorer repo's data/production/:
  leaf_v3_game_by_game_YYYYMMDD.csv
      (season, week, game_date, passer_player_id, player_name, posteam,
       game_number, leaf_rating, opp_adj_base_epa_kalman,
       opp_adj_base_epa_uncertainty, plays, age)
  leaf_v3_current_ratings_YYYYMMDD.csv
      (player_id, player_name, last_season, last_week, leaf_rating,
       leaf_uncertainty, total_games, total_attempts)
  leaf_v3_params.json (calibration + model parameters, for the app's
       prediction function)

QBs only: career attempts >= 100 (kills trick-play artifacts like a safety
with one career pass sitting atop the old current-ratings file).
"""

import json
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
APP_DATA = ROOT / 'qb-leaf-explorer' / 'data' / 'production'
STAMP = date.today().strftime('%Y%m%d')


def main():
    df = pd.read_csv(ROOT / 'data' / 'production' / 'leaf_v3_ratings.csv')
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values(['passer_player_id', 'game_date']).reset_index(drop=True)

    career_plays = df.groupby('passer_player_id')['plays'].transform('sum')
    df = df[career_plays >= 100].copy()
    df['game_number'] = df.groupby('passer_player_id').cumcount() + 1

    games = pd.DataFrame({
        'season': df['season'],
        'week': df['week'],
        'game_date': df['game_date'],
        'passer_player_id': df['passer_player_id'],
        'player_name': df['passer_player_name'],
        'posteam': df['posteam'], 'defteam': df['defteam'],
        'game_number': df['game_number'],
        'leaf_rating': df['leaf_v3'],
        'opp_adj_base_epa_kalman': df['k_informed'],
        'opp_adj_base_epa_uncertainty': np.sqrt(df['k_informed_var']),
        'game_epa': df['adj_epa'],
        'plays': df['plays'],
        'age': df['age'],
    })
    # 6dp on floats: full repr differs in the last bits between numpy builds,
    # so an unrounded export rewrites all ~12k rows on every weekly CI run
    # (~5MB of git history a week for data that barely changed). 6 decimals is
    # far more precision than any displayed or downstream use needs.
    games_path = APP_DATA / f'leaf_v3_game_by_game_{STAMP}.csv'
    games.to_csv(games_path, index=False, float_format='%.6f')
    print(f'[OK] {len(games):,} rows -> {games_path.name}')

    last = df.groupby('passer_player_id').last()
    agg = df.groupby('passer_player_id').agg(total_games=('game_id', 'nunique'),
                                             total_attempts=('plays', 'sum'))
    current = pd.DataFrame({
        'player_id': last.index,
        'player_name': last['passer_player_name'].values,
        'last_season': last['season'].values,
        'last_week': last['week'].values,
        'leaf_rating': last['leaf_v3'].values,
        'leaf_uncertainty': np.sqrt(last['k_informed_var'].values),
        'total_games': agg['total_games'].values,
        'total_attempts': agg['total_attempts'].values,
    })
    cur_path = APP_DATA / f'leaf_v3_current_ratings_{STAMP}.csv'
    current.to_csv(cur_path, index=False, float_format='%.6f')
    print(f'[OK] {len(current)} QBs -> {cur_path.name}')

    shutil.copy(ROOT / 'data' / 'production' / 'leaf_v3_params.json',
                APP_DATA / 'leaf_v3_params.json')
    print('[OK] params copied')


if __name__ == '__main__':
    main()
