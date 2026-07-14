"""
A3: participation + charting extraction (2016-2025).

Merges nflverse participation (lineups + NGS charting) with pbp QB plays.

Outputs:
  data/raw/qb_game_pressure_features.csv - per QB-game:
      charted_dropbacks, pressure_rate, epa_clean, epa_pressure,
      clean_n, pressure_n, tt_mean
  data/raw/rapm_plays.parquet - play-level (season, game_id, play_id,
      passer_player_id, qb_epa, offense_players) for the RAPM stage
"""

import io
from pathlib import Path

import nfl_data_py as nfl
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).parent.parent.parent
SEASONS = range(2016, 2026)
PART_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp_participation/pbp_participation_{yr}.parquet"


def main():
    game_rows, play_rows = [], []
    for season in SEASONS:
        print(f'[{season}] participation...', flush=True)
        r = requests.get(PART_URL.format(yr=season), timeout=180)
        part = pd.read_parquet(io.BytesIO(r.content))
        part = part[['nflverse_game_id', 'play_id', 'offense_players',
                     'was_pressure', 'time_to_throw']]

        print(f'[{season}] pbp...', flush=True)
        pbp = nfl.import_pbp_data([season], downcast=False, cache=False)
        p = pbp[(pbp['qb_dropback'] == 1) & pbp['passer_player_id'].notna()][
            ['game_id', 'play_id', 'passer_player_id', 'qb_epa', 'season']].dropna(subset=['qb_epa'])

        m = p.merge(part, left_on=['game_id', 'play_id'],
                    right_on=['nflverse_game_id', 'play_id'], how='inner')
        m['was_pressure'] = pd.to_numeric(m['was_pressure'], errors='coerce')
        charted = m[m['was_pressure'].notna()].copy()
        print(f'  merged {len(m):,} dropbacks, {len(charted):,} with pressure charting', flush=True)

        g = charted.groupby(['game_id', 'passer_player_id']).agg(
            charted_dropbacks=('qb_epa', 'size'),
            pressure_rate=('was_pressure', 'mean'),
            tt_mean=('time_to_throw', 'mean'),
        ).reset_index()
        clean = (charted[charted.was_pressure == 0].groupby(['game_id', 'passer_player_id'])
                 .agg(epa_clean=('qb_epa', 'mean'), clean_n=('qb_epa', 'size')).reset_index())
        press = (charted[charted.was_pressure == 1].groupby(['game_id', 'passer_player_id'])
                 .agg(epa_pressure=('qb_epa', 'mean'), pressure_n=('qb_epa', 'size')).reset_index())
        g = g.merge(clean, on=['game_id', 'passer_player_id'], how='left')
        g = g.merge(press, on=['game_id', 'passer_player_id'], how='left')
        g['season'] = season
        game_rows.append(g)

        play_rows.append(m[['season', 'game_id', 'play_id', 'passer_player_id',
                            'qb_epa', 'offense_players']])
        del pbp, p, m, charted

    games = pd.concat(game_rows, ignore_index=True)
    games.to_csv(ROOT / 'data' / 'raw' / 'qb_game_pressure_features.csv', index=False)
    plays = pd.concat(play_rows, ignore_index=True)
    plays.to_parquet(ROOT / 'data' / 'raw' / 'rapm_plays.parquet', index=False)
    print(f'[OK] {len(games):,} QB-games, {len(plays):,} plays saved')


if __name__ == '__main__':
    main()
