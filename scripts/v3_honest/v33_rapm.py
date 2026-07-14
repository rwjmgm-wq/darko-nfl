"""
A3/P2: walk-forward QB RAPM (teammate-adjusted per-play value).

For each season Y (2016-2025): fit ridge regression of play EPA on
QB indicators + offensive-teammate indicators, using ALL plays from
2016 through Y (strictly walk-forward when consumed for predicting Y+1).
The QB coefficient is his per-play value with supporting cast held fixed.

Ridge alpha chosen by 2-fold CV WITHIN the fitting window (legitimate:
no future information). Output: data/processed/qb_rapm_by_season.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

ROOT = Path(__file__).parent.parent.parent
ALPHAS = [200.0, 1000.0, 5000.0]


def build_matrix(plays, qb_index, mate_index):
    n = len(plays)
    rows_q = np.arange(n)
    cols_q = plays['qb_col'].values
    data_q = np.ones(n)

    rows_m, cols_m = [], []
    for i, (mates, qb) in enumerate(zip(plays['offense_players'], plays['passer_player_id'])):
        if not isinstance(mates, str):
            continue
        for pid in mates.split(';'):
            if pid and pid != qb and pid in mate_index:
                rows_m.append(i)
                cols_m.append(len(qb_index) + mate_index[pid])
    X = sparse.csr_matrix(
        (np.concatenate([data_q, np.ones(len(rows_m))]),
         (np.concatenate([rows_q, np.array(rows_m)]),
          np.concatenate([cols_q, np.array(cols_m)]))),
        shape=(n, len(qb_index) + len(mate_index)))
    return X


def fit_window(plays):
    """Fit RAPM on a window of plays; returns {qb_id: coef}."""
    qbs = plays['passer_player_id'].value_counts()
    qbs = qbs[qbs >= 100].index  # QB needs 100+ plays in window
    plays = plays[plays['passer_player_id'].isin(qbs)].reset_index(drop=True)

    mates = {}
    for m in plays['offense_players']:
        if isinstance(m, str):
            for pid in m.split(';'):
                mates[pid] = mates.get(pid, 0) + 1
    mate_ids = [p for p, c in mates.items() if c >= 200 and p not in set(qbs)]

    qb_index = {q: i for i, q in enumerate(qbs)}
    mate_index = {p: i for i, p in enumerate(mate_ids)}
    plays['qb_col'] = plays['passer_player_id'].map(qb_index)

    X = build_matrix(plays, qb_index, mate_index)
    y = plays['qb_epa'].values

    # alpha by 2-fold CV within window
    best = None
    kf = KFold(2, shuffle=True, random_state=42)
    for a in ALPHAS:
        errs = []
        for tr, te in kf.split(y):
            m = Ridge(alpha=a, fit_intercept=True, solver='sparse_cg')
            m.fit(X[tr], y[tr])
            errs.append(((m.predict(X[te]) - y[te]) ** 2).mean())
        mse = np.mean(errs)
        if best is None or mse < best[0]:
            best = (mse, a)
    alpha = best[1]

    m = Ridge(alpha=alpha, fit_intercept=True, solver='sparse_cg')
    m.fit(X, y)
    coefs = {q: m.coef_[i] for q, i in qb_index.items()}
    return coefs, alpha, len(plays), len(mate_ids)


def main():
    plays = pd.read_parquet(ROOT / 'data' / 'raw' / 'rapm_plays.parquet')
    print(f'{len(plays):,} plays, seasons {plays.season.min()}-{plays.season.max()}')

    rows = []
    for y in sorted(plays['season'].unique()):
        window = plays[plays['season'] <= y]
        coefs, alpha, n, n_mates = fit_window(window)
        for qb, c in coefs.items():
            rows.append({'passer_player_id': qb, 'y': y, 'rapm': c})
        print(f'  through {y}: {n:,} plays, {len(coefs)} QBs, {n_mates} teammates, alpha={alpha:g}')

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / 'data' / 'processed' / 'qb_rapm_by_season.csv', index=False)
    print(f'[OK] {len(out)} QB-season RAPM values')


if __name__ == '__main__':
    main()
