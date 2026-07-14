"""
A1/L7: per-game QB play-level features from nflfastR pbp (2006-2025).

Extracted per QB-game:
  dropbacks, sacks, sack_epa, scrambles, scramble_epa,
  cpoe_all, cpoe_deep (air_yards >= 15), deep_att, short_att,
  air_epa_pp (QB throw value), yac_epa_pp (receiver after-catch value),
  qb_hit_rate, home

Output: data/raw/qb_game_play_features.csv
"""

from pathlib import Path

import nfl_data_py as nfl
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
SEASONS = range(2006, 2026)

COLS = ['season', 'week', 'game_id', 'home_team', 'posteam', 'passer_player_id',
        'qb_dropback', 'qb_scramble', 'sack', 'qb_hit', 'pass_attempt',
        'complete_pass', 'air_yards', 'cpoe', 'qb_epa', 'air_epa', 'yac_epa',
        'comp_air_epa', 'comp_yac_epa']


def main():
    frames = []
    for season in SEASONS:
        print(f'[{season}] downloading...', flush=True)
        df = nfl.import_pbp_data([season], downcast=False, cache=False)
        cols = [c for c in COLS if c in df.columns]
        p = df[cols].copy()
        p = p[(p['qb_dropback'] == 1) & p['passer_player_id'].notna()]

        p['deep'] = (p['air_yards'] >= 15).astype(float)
        p['is_home'] = (p['posteam'] == p['home_team']).astype(float)

        g = p.groupby(['season', 'week', 'game_id', 'passer_player_id']).agg(
            dropbacks=('qb_dropback', 'sum'),
            sacks=('sack', 'sum'),
            scrambles=('qb_scramble', 'sum'),
            qb_hits=('qb_hit', 'sum'),
            attempts=('pass_attempt', 'sum'),
            cpoe_all=('cpoe', 'mean'),
            qb_epa_sum=('qb_epa', 'sum'),
            home=('is_home', 'first'),
        ).reset_index()

        # sack / scramble EPA
        for flag, name in [('sack', 'sack_epa'), ('qb_scramble', 'scramble_epa')]:
            s = (p[p[flag] == 1].groupby(['game_id', 'passer_player_id'])['qb_epa']
                 .sum().rename(name).reset_index())
            g = g.merge(s, on=['game_id', 'passer_player_id'], how='left')

        # deep vs short CPOE and attempt mix
        deep = (p[p['deep'] == 1].groupby(['game_id', 'passer_player_id'])
                .agg(cpoe_deep=('cpoe', 'mean'), deep_att=('pass_attempt', 'sum')).reset_index())
        g = g.merge(deep, on=['game_id', 'passer_player_id'], how='left')

        # air vs YAC EPA on completions (receiver contribution split)
        comp = (p[p['complete_pass'] == 1].groupby(['game_id', 'passer_player_id'])
                .agg(comp_air_epa=('comp_air_epa', 'sum'), comp_yac_epa=('comp_yac_epa', 'sum'))
                .reset_index())
        g = g.merge(comp, on=['game_id', 'passer_player_id'], how='left')

        frames.append(g)
        print(f'  {len(g)} QB-games', flush=True)
        del df, p

    out = pd.concat(frames, ignore_index=True)
    for c in ['sack_epa', 'scramble_epa', 'cpoe_deep', 'deep_att', 'comp_air_epa', 'comp_yac_epa']:
        if c in out.columns:
            out[c] = out[c].fillna(0.0) if c != 'cpoe_deep' else out[c]
    out_path = ROOT / 'data' / 'raw' / 'qb_game_play_features.csv'
    out.to_csv(out_path, index=False)
    print(f'[OK] {len(out)} QB-games -> {out_path}')


if __name__ == '__main__':
    main()
