"""
LEAF v3.4 base dataset builder (audit finding C: input-population ambiguity).

Differences from the v3 builder (scripts/v3_honest/refresh_game_data.py), which
is left untouched for reproducibility:

  * `season_type` is carried explicitly, taken from nflverse (REG/POST) rather
    than inferred from a week-number heuristic.
  * Player position is joined from player_meta and QB passers are retained by
    default; every dropped row is COUNTED AND REPORTED, never silently binned.
  * Production/evaluation default is regular season only. Postseason can be
    included with --include-postseason, which writes a separately named file so
    a postseason-inclusive analysis can never be mistaken for the default.
  * passer_player_name is not a grouping key (see v3 deviation 5); grouping is
    on identity with the modal name attached.

Outputs (never overwrites v3 artifacts):
  data/production/qb_games_base_v34.csv            (default: REG only, QB only)
  data/production/qb_games_base_v34_withpost.csv   (--include-postseason)
  data/production/qb_games_base_v34_dropreport.csv (what was excluded and why)

Usage:
  python scripts/v3_4/build_base_v34.py [--include-postseason] [--keep-non-qb]
"""

import argparse
from pathlib import Path

import nfl_data_py as nfl
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
SEASONS = range(2006, 2026)
PROD = ROOT / 'data' / 'production'


def output_paths(include_postseason: bool, keep_non_qb: bool):
    """Resolve output paths so that EVERY population combination writes to a
    distinct file. Both the season-type choice and the passer-population choice
    are encoded, so no non-default run can overwrite the default artifact.

    default (REG, QB only)      -> qb_games_base_v34.csv
    REG, all passers            -> qb_games_base_v34_allpassers.csv
    REG+POST, QB only           -> qb_games_base_v34_withpost.csv
    REG+POST, all passers       -> qb_games_base_v34_withpost_allpassers.csv
    """
    suffix = ('_withpost' if include_postseason else '') + \
             ('_allpassers' if keep_non_qb else '')
    stem = f'qb_games_base_v34{suffix}'
    return PROD / f'{stem}.csv', PROD / f'{stem}_dropreport.csv'


def build(include_postseason: bool, keep_non_qb: bool):
    meta = pd.read_csv(ROOT / 'data' / 'raw' / 'player_meta.csv')
    pos = meta.set_index('gsis_id')['position']

    frames, drops = [], []
    for season in SEASONS:
        print(f'[{season}] downloading...', flush=True)
        df = nfl.import_pbp_data([season], downcast=False, cache=False)
        cols = ['game_id', 'game_date', 'season', 'season_type', 'week',
                'passer_player_id', 'passer_player_name', 'posteam', 'defteam',
                'qb_epa', 'cpoe', 'success', 'pass_attempt']
        p = df[(df['qb_dropback'] == 1) & df['passer_player_id'].notna()][cols].copy()
        n_raw_plays = len(p)
        p = p.dropna(subset=['qb_epa'])

        # ---- population filters, each counted before it is applied ----
        p['position'] = p['passer_player_id'].map(pos)
        n_missing_pos = int(p['position'].isna().sum())
        if not keep_non_qb:
            is_qb = p['position'].eq('QB')
            # missing position is NOT silently dropped: report and exclude,
            # since an unknown passer cannot be certified as a QB.
            bad = p[~is_qb]
            for (posn), grp in bad.groupby(bad['position'].fillna('<missing>')):
                drops.append({'season': season, 'reason': 'non_qb_passer',
                              'position': posn, 'plays': len(grp),
                              'players': grp['passer_player_id'].nunique()})
            p = p[is_qb]

        if not include_postseason:
            post = p['season_type'].ne('REG')
            if post.any():
                drops.append({'season': season, 'reason': 'postseason',
                              'position': 'QB', 'plays': int(post.sum()),
                              'players': int(p.loc[post, 'passer_player_id'].nunique())})
            p = p[~post]

        # ---- aggregate to QB-game (identity keys only) ----
        g = p.groupby(['game_id', 'game_date', 'season', 'season_type', 'week',
                       'passer_player_id', 'posteam', 'defteam']).agg(
            qb_epa_mean=('qb_epa', 'mean'), qb_epa_sum=('qb_epa', 'sum'),
            qb_epa_count=('qb_epa', 'size'), cpoe_mean=('cpoe', 'mean'),
            success_mean=('success', 'mean'), attempts=('pass_attempt', 'sum'),
        ).reset_index()

        clean = p[~p['passer_player_name'].str.contains(r'\(3rd QB\)', na=False)]
        names = (clean.groupby('passer_player_id')['passer_player_name']
                 .agg(lambda s: s.mode().iat[0] if len(s.mode()) else s.iat[0]))
        fallback = p.groupby('passer_player_id')['passer_player_name'].first()
        g['passer_player_name'] = (g['passer_player_id'].map(names)
                                   .fillna(g['passer_player_id'].map(fallback)))
        g = g[['game_id', 'game_date', 'season', 'season_type', 'week',
               'passer_player_id', 'passer_player_name', 'posteam', 'defteam',
               'qb_epa_mean', 'qb_epa_sum', 'qb_epa_count', 'cpoe_mean',
               'success_mean', 'attempts']]
        frames.append(g)
        print(f'    {len(g)} QB-games from {n_raw_plays} dropbacks '
              f'({n_missing_pos} missing-position plays)', flush=True)
        del df, p

    out = pd.concat(frames, ignore_index=True)
    path, dr_path = output_paths(include_postseason, keep_non_qb)
    out.to_csv(path, index=False, float_format='%.6f')

    dr = pd.DataFrame(drops)
    dr.to_csv(dr_path, index=False)

    print(f'\n[OK] {len(out):,} QB-games -> {path.name}')
    print(f'     season_type values: {out.season_type.value_counts().to_dict()}')
    print(f'     split QB-games (should be 0): '
          f'{int(out.duplicated(["game_id", "passer_player_id"]).sum())}')
    if len(dr):
        print(f'\n[drop report] -> {dr_path.name}')
        print(dr.groupby('reason')[['plays', 'players']].sum().to_string())
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--include-postseason', action='store_true',
                    help='write a separately named postseason-inclusive file')
    ap.add_argument('--keep-non-qb', action='store_true',
                    help='retain non-QB passers (trick plays, punters)')
    a = ap.parse_args()
    build(a.include_postseason, a.keep_non_qb)
