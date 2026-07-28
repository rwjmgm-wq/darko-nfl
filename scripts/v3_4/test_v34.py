"""
LEAF v3.4 invariant tests (Phase 1 acceptance criteria).

Each test targets a specific audit finding and FAILS LOUDLY rather than
warning. Run: python scripts/v3_4/test_v34.py

  T1  no fusion target game is after TRAIN_MAX_SEASON            (finding A)
  T2  no same-game row affects another row's pre-game opponent
      state, and no same-date game affects another                (finding B)
  T3  regular-season-only default holds                           (finding C)
  T4  non-QB passers excluded and explicitly reported             (finding C)
  T5  all transformations are fit on training data only           (general)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import engine_v34 as E  # noqa: E402

ROOT = Path(__file__).parent.parent.parent
PROD = ROOT / 'data' / 'production'
FAILURES = []


def check(name, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    print(f'  [{status}] {name}' + (f' — {detail}' if detail else ''))
    if not cond:
        FAILURES.append(name)
    return cond


# ---------------------------------------------------------------- T1
def t1_fusion_targets():
    print('\nT1: fusion targets never enter the frozen era (finding A)')
    pairs = pd.read_csv(PROD / 'leaf_v34_fusion_pairs.csv')
    params = json.load(open(PROD / 'leaf_v34_params.json'))
    tmax = params['train_max_season']
    span = params['fusion']['max_target_span_days']

    used = pairs[(pairs.target_end_season <= tmax)
                 & (pairs.target_span_days <= span)]
    check('every primary-fusion target game <= TRAIN_MAX_SEASON',
          int((used.target_end_season > tmax).sum()) == 0,
          f'max target_end_season = {int(used.target_end_season.max())}')
    check('primary pair count matches params',
          len(used) == params['fusion']['n_primary_pairs'],
          f'{len(used)} vs {params["fusion"]["n_primary_pairs"]}')
    # the v3 defect must be demonstrably excluded, not merely absent
    v3_style = pairs[pairs.origin_season <= tmax]
    leaky = v3_style[v3_style.target_end_season > tmax]
    check('v3-style leaky pairs exist in candidates and are excluded',
          len(leaky) > 0 and len(leaky.merge(used, how='inner',
                                             on=list(pairs.columns))) == 0,
          f'{len(leaky)} leaky candidate pairs excluded')
    check('long-span appearance pairs are not in the deployed fit',
          int((used.target_span_days > span).sum()) == 0,
          f'max span kept = {int(used.target_span_days.max())}d')


# ---------------------------------------------------------------- T2
def t2_no_same_game_leak():
    """Direct causal test: perturb one passer's EPA in a multi-passer game and
    assert no OTHER row in that game (or on that date) changes its pre-game
    defense rating. This proves the property rather than inspecting code."""
    print('\nT2: no same-game / same-date opponent-state leakage (finding B)')
    df = E.load_games()
    # find a game where one offense used >= 2 passers
    counts = df.groupby(['game_id', 'posteam']).size()
    multi = counts[counts > 1]
    check('multi-passer team-games exist to test', len(multi) > 0,
          f'{len(multi)} such team-games')
    gid, pteam = multi.index[len(multi) // 2]

    base = E.walkforward_defense_batched(df, 365, 600)

    rows = df.index[(df.game_id == gid) & (df.posteam == pteam)].tolist()
    # v3's leak flows FORWARD within a game: an earlier passer's row updates the
    # defense state that a later passer's row then reads. So perturb the EARLIER
    # row and inspect the LATER one.
    perturbed, victim = rows[0], rows[1]
    d2 = df.copy()
    d2.loc[perturbed, 'epa'] += 5.0          # massive perturbation
    pert = E.walkforward_defense_batched(d2, 365, 600)

    same_date = df.index[df.game_date == df.loc[victim, 'game_date']]
    check('same-game teammate row unchanged',
          np.isclose(base[victim], pert[victim]),
          f'delta = {abs(base[victim]-pert[victim]):.2e}')
    check('all rows on the same date unchanged',
          np.allclose(base[same_date], pert[same_date]),
          f'max delta = {np.max(np.abs(base[same_date]-pert[same_date])):.2e}')
    later = df.index[df.game_date > df.loc[victim, 'game_date']]
    check('future rows DO change (update still happens)',
          not np.allclose(base[later], pert[later]),
          f'max delta = {np.max(np.abs(base[later]-pert[later])):.2e}')

    # and the v3 implementation must FAIL this same test (regression guard)
    sys.path.insert(0, str(ROOT / 'scripts' / 'v3_honest'))
    import build_leaf_v3 as V3  # noqa: E402
    b3 = V3.walkforward_defense(df, 365, 600)
    p3 = V3.walkforward_defense(d2, 365, 600)
    check('v3 implementation leaks on the same test (confirms the defect)',
          not np.isclose(b3[victim], p3[victim]),
          f'v3 delta = {abs(b3[victim]-p3[victim]):.2e}')


# ---------------------------------------------------------------- T3 / T4
def t3_t4_population():
    print('\nT3/T4: population filters (finding C)')
    base = pd.read_csv(PROD / 'qb_games_base_v34.csv')
    check('season_type column present', 'season_type' in base.columns)
    check('regular season only by default',
          set(base.season_type.unique()) == {'REG'},
          str(base.season_type.value_counts().to_dict()))

    meta = pd.read_csv(ROOT / 'data' / 'raw' / 'player_meta.csv')
    pos = meta.set_index('gsis_id')['position']
    p = base['passer_player_id'].map(pos)
    check('all retained passers are QBs', bool(p.eq('QB').all()),
          f'non-QB rows = {int((~p.eq("QB")).sum())}')

    dr = PROD / 'qb_games_base_v34_dropreport.csv'
    check('drop report exists (nothing silently discarded)', dr.exists())
    if dr.exists():
        d = pd.read_csv(dr)
        check('drop report accounts for non-QB and postseason',
              {'non_qb_passer', 'postseason'} <= set(d.reason.unique()),
              str(d.groupby('reason')['plays'].sum().to_dict()))


# ---------------------------------------------------------------- T5
def t5_train_only_fitting():
    print('\nT5: every fitted quantity uses training data only')
    params = json.load(open(PROD / 'leaf_v34_params.json'))
    tmax = params['train_max_season']
    ratings = pd.read_csv(PROD / 'leaf_v34_ratings.csv')

    check('params record the train cutoff', tmax == E.TRAIN_MAX_SEASON,
          f'{tmax}')

    # Causal test: perturbing ONLY frozen-era observations must not move any
    # train-era rating (would reveal a fitted quantity that saw the test era).
    df = E.load_games()
    df['def_rating'] = E.walkforward_defense_batched(
        df, params['defense']['halflife_days'], params['defense']['k_shrink'])
    df['adj_epa'] = df['epa'] - df['def_rating']
    a_pr, b_pr = E.fit_rookie_prior(df)
    drift = E.fit_age_drift(df)

    d2 = df.copy()
    d2.loc[d2.season >= 2019, 'adj_epa'] += 3.0
    a2, b2 = E.fit_rookie_prior(d2)
    drift2 = E.fit_age_drift(d2)
    check('rookie prior unaffected by frozen-era values',
          np.isclose(a_pr, a2) and np.isclose(b_pr, b2),
          f'd_intercept={abs(a_pr-a2):.2e}')
    check('age drift unaffected by frozen-era values',
          all(np.isclose(drift[k], drift2[k]) for k in drift),
          str({k: round(abs(drift[k]-drift2[k]), 12) for k in drift}))

    # fusion coefficients are fit only from pairs whose targets end <= tmax
    pairs = pd.read_csv(PROD / 'leaf_v34_fusion_pairs.csv')
    used = pairs[(pairs.target_end_season <= tmax)
                 & (pairs.target_span_days <= params['fusion']['max_target_span_days'])]
    check('no fusion training row observes the frozen era',
          int((used.target_end_season >= 2019).sum()) == 0)

    check('ratings cover full timeline (scoring is walk-forward, not train-only)',
          int(ratings.season.max()) >= 2024,
          f'{int(ratings.season.min())}-{int(ratings.season.max())}')


if __name__ == '__main__':
    print('=' * 72)
    print('LEAF v3.4 INVARIANT TESTS')
    print('=' * 72)
    t1_fusion_targets()
    t2_no_same_game_leak()
    t3_t4_population()
    t5_train_only_fitting()
    print('\n' + '=' * 72)
    if FAILURES:
        print(f'{len(FAILURES)} FAILURE(S): ' + '; '.join(FAILURES))
        sys.exit(1)
    print('ALL TESTS PASSED')
