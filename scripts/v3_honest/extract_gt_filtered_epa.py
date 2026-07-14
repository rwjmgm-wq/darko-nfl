"""
A2/W1: per-game QB EPA with garbage time removed (wp < 0.05 or > 0.95).

Output: data/raw/qb_game_gt_epa.csv
  (game_id, passer_player_id, gt_epa_mean, gt_plays, all_epa_mean, all_plays)
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
            ['game_id', 'passer_player_id', 'qb_epa', 'wp']].dropna(subset=['qb_epa'])
        p['competitive'] = p['wp'].between(0.05, 0.95)

        allg = p.groupby(['game_id', 'passer_player_id']).agg(
            all_epa_mean=('qb_epa', 'mean'), all_plays=('qb_epa', 'size')).reset_index()
        gt = p[p['competitive']].groupby(['game_id', 'passer_player_id']).agg(
            gt_epa_mean=('qb_epa', 'mean'), gt_plays=('qb_epa', 'size')).reset_index()
        g = allg.merge(gt, on=['game_id', 'passer_player_id'], how='left')
        frames.append(g)
        print(f'  {len(g)} QB-games', flush=True)
        del df, p

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(ROOT / 'data' / 'raw' / 'qb_game_gt_epa.csv', index=False)
    print(f'[OK] {len(out)} QB-games -> qb_game_gt_epa.csv')


if __name__ == '__main__':
    main()
