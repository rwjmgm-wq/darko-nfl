"""
LEAF v3 frozen test-era evaluation (docs/SPEC_LEAF_V3.md).

Test era: 2019-2025, never used for any tuning. Targets:
  T1: next-season EPA/play (both seasons >= 150 plays)
  T2: next-16-games EPA/play at non-overlapping checkpoints (every 8th game)

All predictors are walk-forward values from leaf_v3_ratings.csv.
Inference: cluster bootstrap by QB (2000 resamples) on r and on the
difference vs the best baseline. Also: 80% predictive-interval coverage.

Output: docs/LEAF_V3_RESULTS.md (table section) + console.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
TEST_MIN_SEASON = 2019
RNG = np.random.default_rng(42)

PREDICTORS = {
    'B1 expanding mean': 'b1_expanding',
    'B2 prev-season EPA': 'b2_prev_season',
    'B3 EWMA (hl=3)': 'b3_ewma',
    'B4 last-12 mean': 'b4_last12',
    'V1 Kalman raw EPA': 'k_epa',
    'V2 Kalman opp-adj': 'k_adj_epa',
    'V3 + priors/age': 'k_informed',
    'LEAF v3 (fused)': 'leaf_v3',
}


def cluster_boot_r(d, col, target, n=2000):
    qbs = d['passer_player_id'].unique()
    by = {q: d[d.passer_player_id == q] for q in qbs}
    rs = []
    for _ in range(n):
        pick = RNG.choice(qbs, len(qbs), replace=True)
        s = pd.concat([by[q] for q in pick])
        if s[col].std() > 0 and s[target].std() > 0:
            rs.append(s[col].corr(s[target]))
    return np.percentile(rs, [2.5, 97.5])


def cluster_boot_diff(d, col_a, col_b, target, n=2000):
    qbs = d['passer_player_id'].unique()
    by = {q: d[d.passer_player_id == q] for q in qbs}
    ds = []
    for _ in range(n):
        pick = RNG.choice(qbs, len(qbs), replace=True)
        s = pd.concat([by[q] for q in pick])
        ds.append(s[col_a].corr(s[target]) - s[col_b].corr(s[target]))
    return np.mean(ds), np.percentile(ds, [2.5, 97.5])


def build_t1(df):
    """Season pairs: predictor = last game's rating in season Y, target = Y+1 EPA."""
    rows = []
    for pid, g in df.groupby('passer_player_id'):
        seasons = g.groupby('season').agg(plays=('plays', 'sum')).reset_index()
        for y in seasons['season']:
            if seasons.loc[seasons.season == y, 'plays'].iloc[0] < 150:
                continue
            gy1 = g[g.season == y + 1]
            if gy1['plays'].sum() < 150 or y + 1 < TEST_MIN_SEASON:
                continue
            gy = g[g.season == y]
            row = {'passer_player_id': pid, 'y': y,
                   'target': np.average(gy1['epa'], weights=gy1['plays'])}
            for name, col in PREDICTORS.items():
                row[col] = gy[col].iloc[-1] if col != 'b2_prev_season' else np.average(gy['epa'], weights=gy['plays'])
            # interval ingredients
            row['k_var'] = gy['k_informed_var'].iloc[-1]
            rows.append(row)
    return pd.DataFrame(rows).dropna(subset=[c for c in PREDICTORS.values()])


def build_t2(df):
    """Non-overlapping checkpoints: every 8th game with >=16 future games, future starts in test era."""
    rows = []
    for pid, g in df.groupby('passer_player_id'):
        v = g.reset_index(drop=True)
        for i in range(8, len(v) - 16, 8):
            fut = v.iloc[i + 1:i + 17]
            if fut['season'].iloc[0] < TEST_MIN_SEASON:
                continue
            row = {'passer_player_id': pid,
                   'target': np.average(fut['epa'], weights=fut['plays'])}
            for name, col in PREDICTORS.items():
                row[col] = v.loc[i, col]
            rows.append(row)
    return pd.DataFrame(rows).dropna(subset=[c for c in PREDICTORS.values()])


def report(d, label, lines):
    print(f'\n{label}: n = {len(d)} ({d.passer_player_id.nunique()} QBs)')
    lines.append(f'### {label} (n = {len(d)}, {d.passer_player_id.nunique()} QBs)\n')
    lines.append('| Predictor | r | 95% CI | RMSE |')
    lines.append('|---|---|---|---|')
    stats = {}
    for name, col in PREDICTORS.items():
        r = d[col].corr(d['target'])
        lo, hi = cluster_boot_r(d, col, 'target')
        rmse = np.sqrt(((d[col] - d['target']) ** 2).mean())
        stats[name] = r
        print(f'  {name:22s} r={r:+.3f} [{lo:+.3f},{hi:+.3f}]  RMSE={rmse:.3f}')
        lines.append(f'| {name} | {r:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {rmse:.3f} |')
    lines.append('')

    best_base = max([(v, k) for k, v in stats.items() if k.startswith('B')])
    base_col = PREDICTORS[best_base[1]]
    diff, (dlo, dhi) = cluster_boot_diff(d, 'leaf_v3', base_col, 'target')
    verdict = 'EXCLUDES zero' if dlo > 0 else 'includes zero'
    print(f'  LEAF v3 vs best baseline ({best_base[1]}): dr = {diff:+.3f} [{dlo:+.3f}, {dhi:+.3f}] ({verdict})')
    lines.append(f'**LEAF v3 vs best baseline ({best_base[1]}):** dr = {diff:+.3f} '
                 f'[{dlo:+.3f}, {dhi:+.3f}] — CI {verdict}.\n')


def main():
    df = pd.read_csv(ROOT / 'data' / 'production' / 'leaf_v3_ratings.csv')
    lines = ['## Frozen test-era results (2019–2025)\n',
             'All predictors walk-forward; all hyperparameters tuned on 2006–2018 only.\n']

    t1 = build_t1(df)
    report(t1, 'T1: next-season EPA/play', lines)

    # 80% predictive interval coverage for T1 using informed Kalman state
    resid_var_extra = np.var(t1['target'] - t1['k_informed'])  # not used for width; report raw state var coverage
    z80 = 1.2816
    width = z80 * np.sqrt(t1['k_var'])
    cover_state = ((t1['target'] >= t1['k_informed'] - width) & (t1['target'] <= t1['k_informed'] + width)).mean()
    print(f'\n  80% interval coverage using state variance alone: {cover_state * 100:.0f}% '
          f'(under-coverage expected: target has its own sampling noise)')
    lines.append(f'80% predictive-interval coverage (state variance only): {cover_state * 100:.0f}%. '
                 'State variance alone under-covers because the target season has its own sampling noise; '
                 'a full predictive interval adds target noise variance.\n')

    t2 = build_t2(df)
    report(t2, 'T2: next-16-games EPA/play', lines)

    out = ROOT / 'docs' / 'LEAF_V3_RESULTS.md'
    header = ['# LEAF v3 Results\n', 'See docs/SPEC_LEAF_V3.md for the pre-registered design.\n']
    out.write_text('\n'.join(header + lines))
    print(f'\n[OK] -> {out}')


if __name__ == '__main__':
    main()
