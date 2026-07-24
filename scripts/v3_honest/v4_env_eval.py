"""
Amendment A4 frozen evaluation (docs/SPEC_LEAF_V3.md, A4). ONE PASS.

LEAF v4 = walk-forward OLS: target ~ leaf_v3 + E1..E5 (continuous covariates
standardized on training data only). Honesty control: LEAF v3-recal = the
identical walk-forward OLS with leaf_v3 as the sole feature. Primary metric:
dr(v4 - v3-recal) on the pooled 2019-2025 T1 pair set, cluster bootstrap by QB.

Appends results to docs/LEAF_V3_RESULTS.md.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from evaluate_leaf_v3 import cluster_boot_r, cluster_boot_diff  # noqa: E402

TEST_MIN, TEST_MAX = 2019, 2025
TRAIN_MIN_TARGET = 2008
ENV = ['e1_new_team', 'e2_coach_change', 'e3_ret_rec_epa',
       'e4_team_pass_env', 'e5_sched_def']
CONT = ['e3_ret_rec_epa', 'e4_team_pass_env', 'e5_sched_def']


def build_pairs(df, min_target_season):
    """T1 pair construction, identical to evaluate_leaf_v3.build_t1 but with a
    parameterized minimum target season so training pairs exist."""
    rows = []
    for pid, g in df.groupby('passer_player_id'):
        seasons = g.groupby('season').agg(plays=('plays', 'sum')).reset_index()
        for y in seasons['season']:
            if seasons.loc[seasons.season == y, 'plays'].iloc[0] < 150:
                continue
            gy1 = g[g.season == y + 1]
            if gy1['plays'].sum() < 150 or y + 1 < min_target_season:
                continue
            gy = g[g.season == y]
            rows.append({'passer_player_id': pid, 'y': y,
                         'target': np.average(gy1['epa'], weights=gy1['plays']),
                         'leaf_v3': gy['leaf_v3'].iloc[-1]})
    return pd.DataFrame(rows)


def fit_ols(X, yv):
    Xd = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xd, yv, rcond=None)
    return beta


def predict(beta, X):
    return np.column_stack([np.ones(len(X)), X]) @ beta


def main():
    df = pd.read_csv(ROOT / 'data' / 'production' / 'leaf_v3_ratings.csv')
    pairs = build_pairs(df, TRAIN_MIN_TARGET)
    env = pd.read_csv(ROOT / 'data' / 'production' / 'v4_env_features.csv')
    d = pairs.merge(env, on=['passer_player_id', 'y'], how='left')

    # deviation 2 (spec): pairs whose Y+1 team came from the first-game
    # fallback are excluded everywhere — a QB absent from the Week-1 roster
    # may have been unsigned at the forecast origin (e.g. Flacco 2022->2023)
    n_before = (d['y'] + 1 >= TEST_MIN).sum()
    fb = d['team_fallback'] == 1
    n_fb_test = int((fb & (d['y'] + 1 >= TEST_MIN)).sum())
    n_fb_train = int((fb & (d['y'] + 1 < TEST_MIN)).sum())
    d = d[~fb].dropna(subset=ENV).reset_index(drop=True)
    n_after = (d['y'] + 1 >= TEST_MIN).sum()
    print(f'pairs: {len(d)} total; test-era pairs {n_after} of {n_before} '
          f'(fallback exclusions: {n_fb_train} train, {n_fb_test} test)')

    d['v3_recal'] = np.nan
    d['v4_env'] = np.nan
    coef_rows = []
    for s in range(TEST_MIN, TEST_MAX + 1):
        tr = d[(d['y'] + 1 <= s - 1)]
        te_idx = d.index[d['y'] + 1 == s]
        if not len(te_idx):
            continue
        mu = tr[CONT].mean()
        sd = tr[CONT].std().replace(0, 1)

        def design(sub):
            X = sub[['leaf_v3'] + ENV].copy()
            X[CONT] = (X[CONT] - mu) / sd
            return X.values

        b4 = fit_ols(design(tr), tr['target'].values)
        b3 = fit_ols(tr[['leaf_v3']].values, tr['target'].values)
        d.loc[te_idx, 'v4_env'] = predict(b4, design(d.loc[te_idx]))
        d.loc[te_idx, 'v3_recal'] = predict(b3, d.loc[te_idx, ['leaf_v3']].values)
        coef_rows.append({'target_season': s, 'n_train': len(tr),
                          'intercept': b4[0], 'leaf_v3': b4[1],
                          **{k: b4[2 + i] for i, k in enumerate(ENV)},
                          'recal_int': b3[0], 'recal_slope': b3[1]})

    test = d[(d['y'] + 1 >= TEST_MIN)].dropna(subset=['v4_env', 'v3_recal'])
    lines = ['\n## Amendment A4 results — environment layer (frozen pass, '
             f'{TEST_MIN}-{TEST_MAX})\n',
             'Provenance: leaf_v3_ratings.csv was regenerated 2026-07-14 '
             '(weekly data refresh), so the T1 universe here is 200 pairs / '
             '60 QBs vs the published 199 / 61; LEAF v3 raw re-scores '
             'slightly differently than the frozen table for that reason '
             'alone. Team-fallback pairs excluded per deviation 2 '
             f'({n_fb_train} train, {n_fb_test} test).\n',
             f'T1 pairs scored: n = {len(test)} '
             f'({test.passer_player_id.nunique()} QBs). '
             'All three predictors scored on this identical set.\n',
             '| Predictor | r | 95% CI | RMSE |', '|---|---|---|---|']
    print(f'\ntest n = {len(test)} ({test.passer_player_id.nunique()} QBs)')
    for name, col in [('LEAF v3 (raw)', 'leaf_v3'),
                      ('LEAF v3-recal (control)', 'v3_recal'),
                      ('LEAF v4 env', 'v4_env')]:
        r = test[col].corr(test['target'])
        lo, hi = cluster_boot_r(test, col, 'target')
        rmse = np.sqrt(((test[col] - test['target']) ** 2).mean())
        print(f'  {name:26s} r={r:+.3f} [{lo:+.3f},{hi:+.3f}]  RMSE={rmse:.3f}')
        lines.append(f'| {name} | {r:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {rmse:.3f} |')

    # deviation 3: point dr is the PLUG-IN correlation difference; the
    # cluster bootstrap supplies only the CI
    dprim = test['v4_env'].corr(test['target']) - test['v3_recal'].corr(test['target'])
    dsec = test['v4_env'].corr(test['target']) - test['leaf_v3'].corr(test['target'])
    _, (plo, phi) = cluster_boot_diff(test, 'v4_env', 'v3_recal', 'target')
    _, (slo, shi) = cluster_boot_diff(test, 'v4_env', 'leaf_v3', 'target')
    verdict = ('DECISIVE' if plo > 0 else
               'SUGGESTIVE' if dprim >= 0.02 else 'NULL')
    print(f'\n  PRIMARY  dr(v4 - v3-recal) = {dprim:+.3f} [{plo:+.3f},{phi:+.3f}]')
    print(f'  secondary dr(v4 - v3 raw)  = {dsec:+.3f} [{slo:+.3f},{shi:+.3f}]')
    print(f'  VERDICT (per pre-registered criteria): {verdict}')
    lines.append(f'\n**Primary: dr(v4 − v3-recal) = {dprim:+.3f} '
                 f'[{plo:+.3f}, {phi:+.3f}]. Secondary: dr(v4 − v3 raw) = '
                 f'{dsec:+.3f} [{slo:+.3f}, {shi:+.3f}]. '
                 f'Verdict: {verdict}.**\n')

    # deviation 4: disclosure-only breakdowns on the same pass — where does
    # any gain live? Between-season (league-trend) or within-season (QB env)?
    lines.append('Per-target-season breakdown:\n')
    lines.append('| Season | n | r v3-recal | r v4 | dr |')
    lines.append('|---|---|---|---|---|')
    print('\n  per-season:')
    for s, g in test.groupby(test['y'] + 1):
        rr, rv = g['v3_recal'].corr(g['target']), g['v4_env'].corr(g['target'])
        print(f'    {int(s)}: n={len(g):3d}  recal={rr:+.3f}  v4={rv:+.3f}  dr={rv-rr:+.3f}')
        lines.append(f'| {int(s)} | {len(g)} | {rr:+.3f} | {rv:+.3f} | {rv - rr:+.3f} |')
    dm = test.copy()
    for c in ['v3_recal', 'v4_env', 'target']:
        dm[c] = dm[c] - dm.groupby(dm['y'] + 1)[c].transform('mean')
    rw_recal = dm['v3_recal'].corr(dm['target'])
    rw_v4 = dm['v4_env'].corr(dm['target'])
    print(f'  season-demeaned pooled: recal={rw_recal:+.3f}  v4={rw_v4:+.3f}  '
          f'dr={rw_v4 - rw_recal:+.3f}')
    lines.append(f'\nSeason-demeaned (within-season) pooled r: v3-recal '
                 f'{rw_recal:+.3f}, v4 {rw_v4:+.3f}, dr {rw_v4 - rw_recal:+.3f} '
                 '— isolates QB-specific environment signal from league-trend '
                 'tracking.\n')

    lines.append('Walk-forward coefficients (env standardized on train):\n')
    cf = pd.DataFrame(coef_rows).round(4)
    lines.append('| ' + ' | '.join(cf.columns) + ' |')
    lines.append('|' + '---|' * len(cf.columns))
    for _, rrow in cf.iterrows():
        lines.append('| ' + ' | '.join(str(v) for v in rrow.values) + ' |')
    lines.append('')
    print('\n' + cf.round(4).to_string(index=False))

    out = ROOT / 'docs' / 'LEAF_V3_RESULTS.md'
    with open(out, 'a', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'\n[OK] appended -> {out}')


if __name__ == '__main__':
    main()
