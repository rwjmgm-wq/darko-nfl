"""
A2 (v3.2) signal-extraction campaign - build + single frozen evaluation.
Per SPEC_LEAF_V3.md Amendment A2. All tuning on seasons <= 2018.

W1 garbage-time-filtered EPA observations (needs qb_game_gt_epa.csv)
W2 robust Kalman (innovation winsorization at c * sqrt(S))
W3 experience-dependent process noise (q_early first G games, q_late after)
W4 variance-aware nonlinear fusion (state^2, state x variance)
W5 stack over {LEAF v3, prev-season, last-12}
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))

TRAIN_MAX = 2018
RNG = np.random.default_rng(42)


def kalman2(values, weights, q_arr, r_play, p0, prior, drift=None, clip_c=None):
    """Scalar Kalman with per-game process noise and optional innovation clipping."""
    n = len(values)
    pri = np.empty(n); post = np.empty(n); var = np.empty(n)
    m, P = prior, p0
    for i in range(n):
        if drift is not None:
            m = m + drift[i]
        P = P + q_arr[i]
        pri[i] = m
        R = r_play / max(weights[i], 0.5)
        S = P + R
        innov = values[i] - m
        if clip_c is not None:
            lim = clip_c * np.sqrt(S)
            innov = np.clip(innov, -lim, lim)
        K = P / S
        m = m + K * innov
        P = (1 - K) * P
        post[i] = m; var[i] = P
    return pri, post, var


def load_all():
    df = pd.read_csv(ROOT / 'data' / 'production' / 'leaf_v3_ratings.csv')
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values(['passer_player_id', 'game_date']).reset_index(drop=True)
    params = json.load(open(ROOT / 'data' / 'production' / 'leaf_v3_params.json'))
    gt_path = ROOT / 'data' / 'raw' / 'qb_game_gt_epa.csv'
    if gt_path.exists():
        gt = pd.read_csv(gt_path)
        df = df.merge(gt[['game_id', 'passer_player_id', 'gt_epa_mean', 'gt_plays']],
                      on=['game_id', 'passer_player_id'], how='left')
        # fall back to full-game EPA when no competitive plays existed
        df['gt_plays'] = df['gt_plays'].fillna(0)
        low = df['gt_plays'] < 5
        df.loc[low, 'gt_epa_mean'] = df.loc[low, 'epa']
        df.loc[low, 'gt_plays'] = df.loc[low, 'plays']
        df['gt_adj_epa'] = df['gt_epa_mean'] - df['def_rating']
        print(f'GT data merged: {(~low).mean() * 100:.0f}% of games have 5+ competitive plays')
    return df, params


def drift_arrays(df, params):
    dr = params['age_drift']
    def per_game(age):
        if np.isnan(age):
            return 0.0
        if age < 25:
            return dr['u25'] / 17
        if age < 32:
            return dr['25_32'] / 17
        return dr['o32'] / 17
    return df['age'].map(per_game).values


def run_filter(df, obs_col, w_col, q_fn, r_play, p0, params, drift_all, clip_c=None):
    """Run a filter variant for every QB; returns posterior state array."""
    a_pr = params['rookie_prior']['intercept']; b_pr = params['rookie_prior']['log_pick_slope']
    post = np.empty(len(df))
    for pid, g in df.groupby('passer_player_id', sort=False):
        i = g.index.values
        n = len(g)
        q_arr = q_fn(np.arange(n))
        prior = a_pr + b_pr * g['log_pick'].iloc[0]
        _, p, _ = kalman2(g[obs_col].values, g[w_col].values, q_arr, r_play, p0,
                          prior, drift=drift_all[i], clip_c=clip_c)
        post[i] = p
    return post


def one_step_mse(df, post_col_fn, train_mask):
    """Weighted one-step MSE of prior predictions on train era (uses pri internally)."""
    # convenience: recompute with pri capture
    pass


def tune_variant(df, params, drift_all, obs_col, w_col, mode):
    """Tune a filter variant on train-era one-step-ahead weighted MSE."""
    train_mask = (df['season'] <= TRAIN_MAX).values
    a_pr = params['rookie_prior']['intercept']; b_pr = params['rookie_prior']['log_pick_slope']

    def score(q_fn, r_play, p0, clip_c):
        se = wsum = 0.0
        for pid, g in df.groupby('passer_player_id', sort=False):
            i = g.index.values
            tr = train_mask[i]
            if not tr.any():
                continue
            q_arr = q_fn(np.arange(len(g)))
            prior = a_pr + b_pr * g['log_pick'].iloc[0]
            pri, _, _ = kalman2(g[obs_col].values, g[w_col].values, q_arr, r_play, p0,
                                prior, drift=drift_all[i], clip_c=clip_c)
            w = g[w_col].values[tr]
            se += (w * (g[obs_col].values[tr] - pri[tr]) ** 2).sum()
            wsum += w.sum()
        return se / wsum

    base_q, r_play, p0 = 5e-5, 1.0, 0.005
    if mode == 'robust':
        best = None
        for q in [2e-5, 5e-5, 2e-4]:
            for c in [1.5, 2.0, 2.5]:
                m = score(lambda n: np.full(len(n), q), r_play, p0, c)
                if best is None or m < best[0]:
                    best = (m, q, c)
        print(f'  W2 robust: q={best[1]:g}, clip_c={best[2]} (wMSE={best[0]:.4f})')
        return {'q_fn': (lambda n, q=best[1]: np.full(len(n), q)), 'r_play': r_play,
                'p0': p0, 'clip_c': best[2]}
    if mode == 'exp_q':
        best = None
        for qe in [2e-4, 8e-4, 3e-3]:
            for ql in [2e-5, 5e-5]:
                for G in [16, 32, 48]:
                    def qf(n, qe=qe, ql=ql, G=G):
                        return np.where(n < G, qe, ql)
                    m = score(qf, r_play, p0, None)
                    if best is None or m < best[0]:
                        best = (m, qe, ql, G)
        print(f'  W3 exp-q: q_early={best[1]:g}, q_late={best[2]:g}, G={best[3]} (wMSE={best[0]:.4f})')
        return {'q_fn': (lambda n, qe=best[1], ql=best[2], G=best[3]: np.where(n < G, qe, ql)),
                'r_play': r_play, 'p0': p0, 'clip_c': None}
    if mode == 'gt':
        best = None
        for q in [2e-5, 5e-5, 2e-4]:
            m = score(lambda n: np.full(len(n), q), r_play, p0, None)
            if best is None or m < best[0]:
                best = (m, q)
        print(f'  W1 gt-filtered: q={best[1]:g} (wMSE={best[0]:.4f})')
        return {'q_fn': (lambda n, q=best[1]: np.full(len(n), q)), 'r_play': r_play,
                'p0': p0, 'clip_c': None}


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
    print('A2 (v3.2) SIGNAL-EXTRACTION CAMPAIGN')
    print('=' * 70)
    df, params = load_all()
    drift_all = drift_arrays(df, params)
    has_gt = 'gt_adj_epa' in df.columns

    print('\n[tuning on train era]')
    variants = {}
    variants['W2_robust'] = tune_variant(df, params, drift_all, 'adj_epa', 'plays', 'robust')
    variants['W3_expq'] = tune_variant(df, params, drift_all, 'adj_epa', 'plays', 'exp_q')
    if has_gt:
        variants['W1_gt'] = tune_variant(df, params, drift_all, 'gt_adj_epa', 'gt_plays', 'gt')
        # combined observation + robust + exp-q
        variants['W123'] = dict(variants['W3_expq'])
        variants['W123']['clip_c'] = variants['W2_robust']['clip_c']

    print('\n[running filters]')
    state_cols = {}
    for name, v in variants.items():
        obs, w = ('gt_adj_epa', 'gt_plays') if (name in ('W1_gt', 'W123') and has_gt) else ('adj_epa', 'plays')
        df[f's_{name}'] = run_filter(df, obs, w, v['q_fn'], v['r_play'], v['p0'],
                                     params, drift_all, clip_c=v['clip_c'])
        state_cols[name] = f's_{name}'

    # season-pair table
    pairs = pd.read_csv(ROOT / 'data' / 'processed' / 'v31_t1_pairs.csv')
    last = df.groupby(['passer_player_id', 'season']).last()[list(state_cols.values())].reset_index()
    last = last.rename(columns={'season': 'y'})
    pairs = pairs.merge(last, on=['passer_player_id', 'y'], how='inner')
    train = pairs[pairs.y + 1 <= TRAIN_MAX].copy()
    test = pairs[pairs.y + 1 >= 2019].copy()
    print(f'\npairs: train {len(train)} | frozen test {len(test)}')

    # models: each variant alone (linear map fit on train), W4, W5, combined
    MODELS = {'LEAF v3 (baseline)': ['leaf_v3']}
    for name, col in state_cols.items():
        MODELS[name] = [col]
    MODELS['W4_nonlinear'] = ['k_informed', 'k_informed_sq', 'k_informed_x_var']
    MODELS['W5_stack'] = ['leaf_v3', 'b1_expanding', 'b4_last12']
    best_state = state_cols.get('W123', state_cols['W3_expq'])
    MODELS['v3.2 combined'] = [best_state, f'{best_state}_sq', f'{best_state}_x_var',
                               'b1_expanding', 'b4_last12']

    for d in (train, test):
        d['k_informed_sq'] = d['k_informed'] ** 2
        d['k_informed_x_var'] = d['k_informed'] * d['k_informed_var']
        d[f'{best_state}_sq'] = d[best_state] ** 2
        d[f'{best_state}_x_var'] = d[best_state] * d['k_informed_var']

    print('\n[FROZEN TEST-ERA RESULTS] target = next-season EPA/play')
    lines = ['\n## Amendment A2 results (frozen test era, single evaluation)\n',
             '| Model | test r | test RMSE | dr vs LEAF v3 [95% CI] |', '|---|---|---|---|']
    for name, feats in MODELS.items():
        X = np.column_stack([np.ones(len(train))] + [train[f].values for f in feats])
        beta = np.linalg.lstsq(X, train['target'].values, rcond=None)[0]
        for d in (train, test):
            Xd = np.column_stack([np.ones(len(d))] + [d[f].values for f in feats])
            d[f'pred_{name}'] = Xd @ beta
        r = test[f'pred_{name}'].corr(test['target'])
        rmse = np.sqrt(((test[f'pred_{name}'] - test['target']) ** 2).mean())
        if name == 'LEAF v3 (baseline)':
            dtxt = '-'
            print(f'  {name:20s} r={r:+.4f}  RMSE={rmse:.4f}')
        else:
            dm, (dlo, dhi) = cluster_boot_diff(test, f'pred_{name}', 'pred_LEAF v3 (baseline)')
            star = ' *' if dlo > 0 else ''
            dtxt = f'{dm:+.4f} [{dlo:+.4f}, {dhi:+.4f}]{star}'
            print(f'  {name:20s} r={r:+.4f}  RMSE={rmse:.4f}  dr={dtxt}')
        lines.append(f'| {name} | {r:+.4f} | {rmse:.4f} | {dtxt} |')

    with open(ROOT / 'docs' / 'LEAF_V3_RESULTS.md', 'a', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n[OK] appended to docs/LEAF_V3_RESULTS.md')


if __name__ == '__main__':
    main()
