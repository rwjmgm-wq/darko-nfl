"""
A1 (v3.1) build + single frozen evaluation, per SPEC_LEAF_V3.md Amendment A1.

- L7: Kalman states for play-level components (tuned on train era <= 2018,
  observation noise scaled by the component's own sample size per game).
- L5/L6: season-pair features from v31_build_t1_features.py.
- Expanded fusion: OLS on train-era pairs; frozen; scored ONCE on 2019-2025.
- Ablations reported for every layer. Cluster bootstrap (QB) for delta-r CIs.

Known limitation (logged): scramble EPA is unavailable (scramble plays carry
rusher_player_id, not passer_player_id); scrambles remain part of qb_epa only.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from build_leaf_v3 import kalman_pass  # noqa: E402

TRAIN_MAX = 2018
RNG = np.random.default_rng(42)

# component -> (value construction, per-game weight column)
COMPONENTS = {
    'cpoe_all': ('cpoe_all', 'attempts'),
    'cpoe_deep': ('cpoe_deep', 'deep_att'),
    'deep_rate': ('deep_rate', 'attempts'),
    'sack_rate': ('sack_rate', 'dropbacks'),
    'qb_hit_rate': ('qb_hit_rate', 'dropbacks'),
    'air_epa_pp': ('air_epa_pp', 'attempts'),
    'yac_epa_pp': ('yac_epa_pp', 'attempts'),
}


def tune_component(df, col, wcol, prior_mean, r_grid):
    train_mask = (df['season'] <= TRAIN_MAX).values
    best = None
    for q_mult in [0.001, 0.005, 0.02]:
        for r0 in r_grid:
            q = q_mult * r0
            se = wsum = 0.0
            for _, g in df.groupby('passer_player_id', sort=False):
                v = g[col].values; w = g[wcol].values
                tr = train_mask[g.index.values]
                if not tr.any():
                    continue
                pri, _, _ = kalman_pass(v, w, q, r0, 0.02 * r0 / np.mean(w[w > 0].mean() or 1), prior_mean)
                m = tr & (w > 0)
                se += (w[m] * (v[m] - pri[m]) ** 2).sum(); wsum += w[m].sum()
            mse = se / wsum
            if best is None or mse < best[0]:
                best = (mse, q, r0)
    return best[1], best[2]


def run_component_filters(df):
    train = df[df['season'] <= TRAIN_MAX]
    states = {}
    for name, (col, wcol) in COMPONENTS.items():
        prior = np.average(train[col].fillna(0), weights=train[wcol].clip(lower=0.5))
        var_scale = train[col].var()
        r_grid = [var_scale * 20, var_scale * 60, var_scale * 150]
        q, r0 = tune_component(df, col, wcol, prior, r_grid)
        post = np.empty(len(df))
        for _, g in df.groupby('passer_player_id', sort=False):
            i = g.index.values
            _, p, _ = kalman_pass(g[col].values, g[wcol].values, q, r0, var_scale, prior)
            post[i] = p
        df[f'k_{name}'] = post
        states[name] = {'q': q, 'r': r0, 'prior': prior}
        print(f'  {name:12s} prior={prior:+.3f} q={q:.4g} r={r0:.4g}')
    return df


def cluster_boot_diff(d, col_a, col_b, n=2000):
    qbs = d['passer_player_id'].unique()
    by = {q: d[d.passer_player_id == q] for q in qbs}
    ds = []
    for _ in range(n):
        pick = RNG.choice(qbs, len(qbs), replace=True)
        s = pd.concat([by[q] for q in pick])
        ds.append(s[col_a].corr(s['target']) - s[col_b].corr(s['target']))
    return np.mean(ds), np.percentile(ds, [2.5, 97.5])


def main():
    print('=' * 70)
    print('A1 (v3.1) BUILD + FROZEN EVALUATION')
    print('=' * 70)

    base = pd.read_csv(ROOT / 'data' / 'production' / 'leaf_v3_ratings.csv')
    play = pd.read_csv(ROOT / 'data' / 'raw' / 'qb_game_play_features.csv')

    play['deep_att'] = play['deep_att'].fillna(0)
    play['deep_rate'] = play['deep_att'] / play['attempts'].clip(lower=1)
    play['sack_rate'] = play['sacks'] / play['dropbacks'].clip(lower=1)
    play['qb_hit_rate'] = play['qb_hits'] / play['dropbacks'].clip(lower=1)
    play['air_epa_pp'] = play['comp_air_epa'] / play['attempts'].clip(lower=1)
    play['yac_epa_pp'] = play['comp_yac_epa'] / play['attempts'].clip(lower=1)
    play['cpoe_all'] = play['cpoe_all'].fillna(0)
    play['cpoe_deep'] = play['cpoe_deep'].fillna(0)
    play['deep_att'] = play['deep_att'].clip(lower=0.5)

    keep = ['game_id', 'passer_player_id', 'dropbacks', 'attempts', 'deep_att'] + \
           [c for c in ['cpoe_all', 'cpoe_deep', 'deep_rate', 'sack_rate', 'qb_hit_rate',
                        'air_epa_pp', 'yac_epa_pp']]
    df = base.merge(play[keep], on=['game_id', 'passer_player_id'], how='inner')
    df = df.sort_values(['passer_player_id', 'game_date']).reset_index(drop=True)
    print(f'merged: {len(df)} QB-games ({len(base) - len(df)} base games without play features)')

    print('\n[L7] component Kalman states (train-era tuned)')
    df = run_component_filters(df)

    # end-of-season-Y component states onto T1 pairs
    pairs = pd.read_csv(ROOT / 'data' / 'processed' / 'v31_t1_pairs.csv')
    last = (df.groupby(['passer_player_id', 'season']).last()
            [[f'k_{n}' for n in COMPONENTS]].reset_index().rename(columns={'season': 'y'}))
    pairs = pairs.merge(last, on=['passer_player_id', 'y'], how='inner')
    print(f'\nT1 pairs with all layers: {len(pairs)}')

    train = pairs[pairs.y + 1 <= TRAIN_MAX].copy()
    test = pairs[pairs.y + 1 >= 2019].copy()
    print(f'train {len(train)} | frozen test {len(test)}')

    ABLATIONS = {
        'LEAF v3 (baseline)': ['k_only'],
        '+ L5 (change/missed)': ['team_change', 'games_missed'],
        '+ L6 (schedule)': ['sched_effect'],
        '+ L7 (components)': [f'k_{n}' for n in COMPONENTS],
        'v3.1 (all layers)': ['team_change', 'games_missed', 'sched_effect'] + [f'k_{n}' for n in COMPONENTS],
    }

    print('\n[FROZEN TEST-ERA RESULTS] target = next-season EPA/play')
    lines = ['\n## Amendment A1 results (frozen test era, single evaluation)\n',
             '| Model | test r | test RMSE | dr vs LEAF v3 [95% CI] |', '|---|---|---|---|']
    for name, extra in ABLATIONS.items():
        feats = ['leaf_v3'] + [f for f in extra if f != 'k_only']
        X = np.column_stack([np.ones(len(train))] + [train[f].values for f in feats])
        beta = np.linalg.lstsq(X, train['target'].values, rcond=None)[0]
        for d in (train, test):
            Xd = np.column_stack([np.ones(len(d))] + [d[f].values for f in feats])
            d[f'pred_{name}'] = Xd @ beta
        r = test[f'pred_{name}'].corr(test['target'])
        rmse = np.sqrt(((test[f'pred_{name}'] - test['target']) ** 2).mean())
        if name == 'LEAF v3 (baseline)':
            dtxt = '-'
            print(f'  {name:22s} r={r:+.4f}  RMSE={rmse:.4f}')
        else:
            dm, (dlo, dhi) = cluster_boot_diff(test, f'pred_{name}', 'pred_LEAF v3 (baseline)')
            star = ' *' if dlo > 0 else ''
            dtxt = f'{dm:+.4f} [{dlo:+.4f}, {dhi:+.4f}]{star}'
            print(f'  {name:22s} r={r:+.4f}  RMSE={rmse:.4f}  dr={dtxt}')
        lines.append(f'| {name} | {r:+.4f} | {rmse:.4f} | {dtxt} |')

    lines.append('\nSkill-only ceiling: r = 0.52. Scramble EPA unavailable '
                 '(rusher-attributed); logged limitation.\n')
    with open(ROOT / 'docs' / 'LEAF_V3_RESULTS.md', 'a', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    pairs.to_csv(ROOT / 'data' / 'processed' / 'v31_t1_pairs_full.csv', index=False)
    print('\n[OK] appended to docs/LEAF_V3_RESULTS.md')


if __name__ == '__main__':
    main()
