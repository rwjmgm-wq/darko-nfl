"""
A3 (v3.3) new-information campaign - single frozen evaluation.
Per SPEC_LEAF_V3.md Amendment A3.

P1: Kalman states for pressure-split metrics. Noise settings FIXED by variance
    reasoning (declared, not tuned - the 2016-2018 window is too thin to tune):
      epa_clean:     r=1.0 per play (EPA-like), weight clean_n
      epa_pressure:  r=1.5 per play, weight pressure_n
      pressure_rate: r=0.16 (binomial p~0.28), weight charted_dropbacks
      tt_mean:       r=0.36 (per-play sd ~0.6s), weight charted_dropbacks
    q = 5e-5 (v3 setting), p0 = component variance, priors = 2016-2018 means.
P2: walk-forward QB RAPM (qb_rapm_by_season.csv), value at end of season Y.

Fusion: ridge (alpha=1.0, declared) on standardized features, trained on pairs
with Y >= 2016 and Y+1 <= 2018 (~60-70 pairs). ONE frozen pass on Y+1 >= 2019
pairs with Y >= 2016; LEAF v3 re-scored on the same restricted pairs.
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

P1_SPECS = {
    'epa_clean': {'w': 'clean_n', 'r': 1.0},
    'epa_pressure': {'w': 'pressure_n', 'r': 1.5},
    'pressure_rate': {'w': 'charted_dropbacks', 'r': 0.16},
    'tt_mean': {'w': 'charted_dropbacks', 'r': 0.36},
}
Q_FIXED = 5e-5


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
    print('A3 (v3.3) NEW-INFORMATION CAMPAIGN - FROZEN EVALUATION')
    print('=' * 70)

    press = pd.read_csv(ROOT / 'data' / 'raw' / 'qb_game_pressure_features.csv')
    base = pd.read_csv(ROOT / 'data' / 'production' / 'leaf_v3_ratings.csv')
    df = base.merge(press.drop(columns=['season']), on=['game_id', 'passer_player_id'], how='inner')
    df = df.sort_values(['passer_player_id', 'game_date']).reset_index(drop=True)
    print(f'{len(df)} QB-games with charting (2016+)')

    # P1 Kalman states, fixed noise
    train = df[df['season'] <= TRAIN_MAX]
    for col, spec in P1_SPECS.items():
        w = df[spec['w']].fillna(0).clip(lower=0.5).values
        v = df[col].values.copy()
        prior = np.average(train[col].dropna(),
                           weights=train.loc[train[col].notna(), spec['w']].clip(lower=0.5))
        v = np.where(np.isnan(v), prior, v)
        p0 = np.nanvar(train[col])
        post = np.empty(len(df))
        for pid, g in df.groupby('passer_player_id', sort=False):
            i = g.index.values
            _, p, _ = kalman_pass(v[i], w[i], Q_FIXED, spec['r'], p0, prior)
            post[i] = p
        df[f'k_{col}'] = post
        print(f'  P1 {col}: prior={prior:+.3f}')

    # season-pair table (restricted era)
    pairs = pd.read_csv(ROOT / 'data' / 'processed' / 'v31_t1_pairs.csv')
    last = df.groupby(['passer_player_id', 'season']).last()[
        [f'k_{c}' for c in P1_SPECS]].reset_index().rename(columns={'season': 'y'})
    pairs = pairs.merge(last, on=['passer_player_id', 'y'], how='inner')

    rapm = pd.read_csv(ROOT / 'data' / 'processed' / 'qb_rapm_by_season.csv')
    pairs = pairs.merge(rapm, on=['passer_player_id', 'y'], how='left')
    pairs['rapm'] = pairs['rapm'].fillna(0.0)
    pairs['has_rapm'] = rapm['rapm'].notna().mean()

    train_p = pairs[(pairs.y >= 2016) & (pairs.y + 1 <= TRAIN_MAX)].copy()
    test_p = pairs[(pairs.y >= 2016) & (pairs.y + 1 >= 2019)].copy()
    print(f'\npairs (Y >= 2016): train {len(train_p)} | frozen test {len(test_p)}')

    MODELS = {
        'LEAF v3 (restricted)': ['leaf_v3'],
        'P1 pressure states': ['leaf_v3', 'k_epa_clean', 'k_epa_pressure', 'k_pressure_rate', 'k_tt_mean'],
        'P2 RAPM': ['leaf_v3', 'rapm'],
        'v3.3 (P1+P2)': ['leaf_v3', 'k_epa_clean', 'k_epa_pressure', 'k_pressure_rate', 'k_tt_mean', 'rapm'],
        'P1 clean-only': ['leaf_v3', 'k_epa_clean'],
    }

    print('\n[FROZEN TEST-ERA RESULTS] target = next-season EPA/play (pairs with Y >= 2016)')
    lines = ['\n## Amendment A3 results (frozen test era, single evaluation)\n',
             f'Pairs restricted to Y >= 2016 (charting era): train {len(train_p)}, test {len(test_p)}.\n',
             '| Model | test r | test RMSE | dr vs LEAF v3 [95% CI] |', '|---|---|---|---|']
    for name, feats in MODELS.items():
        mu, sd = train_p[feats].mean(), train_p[feats].std().replace(0, 1)
        Xtr = ((train_p[feats] - mu) / sd).values
        ytr = train_p['target'].values
        # ridge alpha=1.0 (declared)
        A = Xtr.T @ Xtr + 1.0 * np.eye(len(feats))
        b = np.linalg.solve(A, Xtr.T @ (ytr - ytr.mean()))
        for d in (train_p, test_p):
            Xd = ((d[feats] - mu) / sd).values
            d[f'pred_{name}'] = ytr.mean() + Xd @ b
        r = test_p[f'pred_{name}'].corr(test_p['target'])
        rmse = np.sqrt(((test_p[f'pred_{name}'] - test_p['target']) ** 2).mean())
        if name == 'LEAF v3 (restricted)':
            dtxt = '-'
            print(f'  {name:22s} r={r:+.4f}  RMSE={rmse:.4f}')
        else:
            dm, (dlo, dhi) = cluster_boot_diff(test_p, f'pred_{name}', 'pred_LEAF v3 (restricted)')
            star = ' *' if dlo > 0 else ''
            dtxt = f'{dm:+.4f} [{dlo:+.4f}, {dhi:+.4f}]{star}'
            print(f'  {name:22s} r={r:+.4f}  RMSE={rmse:.4f}  dr={dtxt}')
        lines.append(f'| {name} | {r:+.4f} | {rmse:.4f} | {dtxt} |')

    # descriptive: standardized train-era coefficients of the full model
    feats = MODELS['v3.3 (P1+P2)']
    mu, sd = train_p[feats].mean(), train_p[feats].std().replace(0, 1)
    Xtr = ((train_p[feats] - mu) / sd).values
    ytr = train_p['target'].values
    b = np.linalg.solve(Xtr.T @ Xtr + np.eye(len(feats)), Xtr.T @ (ytr - ytr.mean()))
    coef_txt = ', '.join(f'{f}={c:+.3f}' for f, c in zip(feats, b))
    print(f'\n  v3.3 standardized train coefs: {coef_txt}')
    lines.append(f'\nStandardized train coefficients (v3.3): {coef_txt}\n')

    with open(ROOT / 'docs' / 'LEAF_V3_RESULTS.md', 'a', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n[OK] appended to docs/LEAF_V3_RESULTS.md')


if __name__ == '__main__':
    main()
