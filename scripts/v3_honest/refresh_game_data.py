"""
Refresh the LEAF v3 engine's base game data from fresh nflverse pbp.

Replaces the stale leaf_v2 game file (frozen Nov 2025, missing the back half
of the 2025 season) as the engine input. Output schema matches what
build_leaf_v3.load_games() consumes.

Output: data/production/qb_games_base.csv
"""

from pathlib import Path

import nfl_data_py as nfl
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
SEASONS = range(2006, 2026)


def main():
    frames = []
    for season in SEASONS:
        print(f'[{season}] downloading...', flush=True)
        df = nfl.import_pbp_data([season], downcast=False, cache=False)
        p = df[(df['qb_dropback'] == 1) & df['passer_player_id'].notna()][
            ['game_id', 'game_date', 'season', 'week', 'passer_player_id',
             'passer_player_name', 'posteam', 'defteam', 'qb_epa', 'cpoe',
             'success', 'pass_attempt']].copy()
        p = p.dropna(subset=['qb_epa'])
        g = p.groupby(['game_id', 'game_date', 'season', 'week',
                       'passer_player_id', 'passer_player_name', 'posteam', 'defteam']).agg(
            qb_epa_mean=('qb_epa', 'mean'),
            qb_epa_sum=('qb_epa', 'sum'),
            qb_epa_count=('qb_epa', 'size'),
            cpoe_mean=('cpoe', 'mean'),
            success_mean=('success', 'mean'),
            attempts=('pass_attempt', 'sum'),
        ).reset_index()
        frames.append(g)
        print(f'  {len(g)} QB-games', flush=True)
        del df, p

    out = pd.concat(frames, ignore_index=True)
    out_path = ROOT / 'data' / 'production' / 'qb_games_base.csv'
    out.to_csv(out_path, index=False)
    print(f'[OK] {len(out):,} QB-games ({out.season.min()}-{out.season.max()}) -> {out_path}')


if __name__ == '__main__':
    main()
