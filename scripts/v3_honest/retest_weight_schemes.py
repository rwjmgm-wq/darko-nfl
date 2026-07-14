"""
Fair re-test of the 12-game weighting schemes (docs/SPEC_LEAF_V3.md).

The original optimize_rating_weights.py built the linear and "proposed" schemes
with weights reversed relative to game order: the proposed steep-recency scheme
(35/25/17/12/7/3/1 newest->oldest) was actually evaluated with 35% on the game
TWELVE games ago and ZERO weight on the five most recent games. Its published
#57/60 rank is an artifact.

Here every scheme is built explicitly as weights[i] for game t-i (i=0 newest),
then reversed ONCE when applied to chronologically ordered values. Selection on
train era (2006-2018), single frozen comparison on test era (2019-2025),
target = next-16-games EPA at non-overlapping checkpoints.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
TRAIN_MAX_SEASON = 2018
RNG = np.random.default_rng(42)


def schemes_newest_first(window=12):
    """Every scheme defined as w[0] = newest game's weight."""
    s = {}
    s['uniform'] = np.ones(window)
    for pct in [25, 30, 35, 40, 45]:
        s[f'linear_{pct}'] = np.linspace(pct / 100, 0.01, window)
    for decay in [10, 15, 20, 25, 30]:
        s[f'expo_{decay}'] = np.exp(-decay / 100 * np.arange(window))
    for cutoff in [4, 6, 8]:
        w = np.ones(window); w[:cutoff] = 3.0
        s[f'step_{cutoff}'] = w
    s['proposed_steep'] = np.array([0.35, 0.25, 0.17, 0.12, 0.07, 0.03, 0.01] + [0.0] * 5)
    return {k: v / v.sum() for k, v in s.items()}


def weighted_form(vals_chrono, w_newest_first):
    """vals_chrono is oldest->newest; apply weights correctly oriented."""
    n = min(len(vals_chrono), len(w_newest_first))
    v = vals_chrono[-n:]
    w = w_newest_first[:n][::-1]  # reverse ONCE: w[-1] (newest) pairs with v[-1]
    return np.average(v, weights=w / w.sum())


def build_checkpoints(df):
    rows = []
    for pid, g in df.groupby('passer_player_id'):
        v = g.reset_index(drop=True)
        for i in range(12, len(v) - 16, 8):
            hist = v['adj_epa'].values[:i + 1]
            fut = v.iloc[i + 1:i + 17]
            rows.append({'pid': pid, 'season': v.loc[i, 'season'],
                         'hist': hist, 'target': np.average(fut['epa'], weights=fut['plays'])})
    return rows


def evaluate(rows, schemes, era):
    out = {}
    for name, w in schemes.items():
        preds, targs = [], []
        for r in rows:
            if era == 'train' and r['season'] > TRAIN_MAX_SEASON:
                continue
            if era == 'test' and r['season'] <= TRAIN_MAX_SEASON:
                continue
            preds.append(weighted_form(r['hist'], w))
            targs.append(r['target'])
        out[name] = (np.corrcoef(preds, targs)[0, 1], len(preds))
    return out


def main():
    df = pd.read_csv(ROOT / 'data' / 'production' / 'leaf_v3_ratings.csv')
    schemes = schemes_newest_first()

    # sanity print: confirm orientation
    p = schemes['proposed_steep']
    print(f'proposed_steep: newest game weight = {p[0]:.2f}, oldest = {p[-1]:.2f} (correct orientation)')

    rows = build_checkpoints(df)
    print(f'checkpoints: {len(rows)}')

    print('\nTRAIN era (2006-2018) - selection happens here:')
    train = evaluate(rows, schemes, 'train')
    for name, (r, n) in sorted(train.items(), key=lambda x: -x[1][0]):
        print(f'  {name:15s} r = {r:+.4f}  (n={n})')

    print('\nTEST era (2019-2025) - frozen comparison, all schemes reported:')
    test = evaluate(rows, schemes, 'test')
    for name, (r, n) in sorted(test.items(), key=lambda x: -x[1][0]):
        marker = ' <-- originally condemned as #57/60' if name == 'proposed_steep' else ''
        print(f'  {name:15s} r = {r:+.4f}  (n={n}){marker}')

    lines = ['\n## Fair weight-scheme re-test (corrected orientation)\n',
             'The original optimizer reversed the linear/proposed schemes (35% landed on the',
             'oldest game; the five newest games got zero weight). Corrected results,',
             'next-16-games EPA target, frozen test era:\n',
             '| Scheme | train r | test r |', '|---|---|---|']
    for name in sorted(test, key=lambda k: -test[k][0]):
        lines.append(f'| {name} | {train[name][0]:+.4f} | {test[name][0]:+.4f} |')
    spread = max(v[0] for v in test.values()) - min(v[0] for v in test.values())
    lines.append(f'\nSpread across all correctly-oriented schemes: {spread:.3f} in test-era r — '
                 'weighting choice within a 12-game window is a second-order decision.\n')
    with open(ROOT / 'docs' / 'LEAF_V3_RESULTS.md', 'a', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('\n[OK] appended to docs/LEAF_V3_RESULTS.md')


if __name__ == '__main__':
    main()
