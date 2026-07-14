"""
V3 Age Backfill.

age_at_draft was the strongest college signal but was missing for 60% of QBs
(nflverse draft data only). Fill order:

1. nflverse draft picks 'age' (exact, already used)      -> source='draft'
2. nflverse players birth_date (exact; covers everyone
   who reached the NFL): age at draft = (draft_year-04-25) - birth_date
                                                          -> source='birth_date'
3. CFBD roster class year for remaining (mostly pre-draft prospects):
   age ~ 18.7 + class_year at final college season        -> source='class_year'
   (flagged estimate; noise ~ +/-1 year from redshirts)

Output: data/raw/qb_ages.csv (player_name, draft_year, age_at_draft, age_source)
"""

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).parent.parent


def norm_name(n):
    n = str(n).lower().strip()
    n = re.sub(r'\s+(jr|sr|ii|iii|iv|v)\.?$', '', n)
    n = re.sub(r"[^a-z\s]", '', n)
    return re.sub(r'\s+', ' ', n)


def cfbd_key():
    for line in (ROOT / '.env').read_text().splitlines():
        if line.startswith('CFBD_API_KEY'):
            return line.split('=', 1)[1].strip()


def main():
    feats = pd.read_csv(ROOT / 'data' / 'processed' / 'v3_features.csv')
    qbs = feats[['player_name', 'draft_year', 'college', 'seasons_played']].copy()
    qbs['name_key'] = qbs['player_name'].map(norm_name)
    qbs['age_at_draft'] = np.nan
    qbs['age_source'] = None

    # --- source 1: nflverse draft picks (exact)
    picks = pd.read_csv(ROOT / 'data' / 'raw' / 'draft_picks_with_age.csv')
    picks['name_key'] = picks['player_name'].map(norm_name)
    picks = picks.dropna(subset=['age']).drop_duplicates(subset=['name_key', 'draft_year'])
    m = qbs.merge(picks[['name_key', 'draft_year', 'age']], on=['name_key', 'draft_year'], how='left')
    qbs['age_at_draft'] = m['age'].values
    qbs.loc[qbs['age_at_draft'].notna(), 'age_source'] = 'draft'
    print(f'[1] draft file:      {qbs.age_at_draft.notna().sum()}/{len(qbs)}')

    # --- source 2: nflverse players birth_date (exact)
    import nfl_data_py as nfl
    players = nfl.import_players()
    pq = players[(players['position'] == 'QB') & players['birth_date'].notna()].copy()
    pq['name_key'] = pq['display_name'].map(norm_name)
    pq['birth_date'] = pd.to_datetime(pq['birth_date'], errors='coerce')
    pq = pq.dropna(subset=['birth_date'])

    by_key = pq.groupby('name_key')
    for i, row in qbs[qbs['age_at_draft'].isna()].iterrows():
        if row['name_key'] not in by_key.groups:
            continue
        cands = by_key.get_group(row['name_key']).copy()
        draft_date = pd.Timestamp(int(row['draft_year']), 4, 25)
        cands['age'] = (draft_date - cands['birth_date']).dt.days / 365.25
        # disambiguate same-name QBs: prefer matching nflverse draft_year,
        # else require a plausible draft age
        exact = cands[cands['draft_year'] == row['draft_year']]
        pick_from = exact if len(exact) else cands[(cands['age'] >= 20) & (cands['age'] <= 27)]
        if len(pick_from) == 1:
            qbs.loc[i, 'age_at_draft'] = round(pick_from['age'].iloc[0], 2)
            qbs.loc[i, 'age_source'] = 'birth_date'
        elif len(pick_from) > 1:
            print(f'    [AMBIG] {row.player_name} ({row.draft_year}): {len(pick_from)} candidates, skipped')
    print(f'[2] + birth_date:    {qbs.age_at_draft.notna().sum()}/{len(qbs)}')

    # --- source 3: CFBD roster class year (estimate) for the rest
    need = qbs[qbs['age_at_draft'].isna()].copy()
    need['final_season'] = need['draft_year'] - 1
    pairs = need[['college', 'final_season']].drop_duplicates()
    print(f'[3] roster fetch for {len(pairs)} (team, season) pairs...')
    key = cfbd_key()
    roster_cache = {}
    for _, pr in pairs.iterrows():
        team, yr = pr['college'], int(pr['final_season'])
        try:
            r = requests.get('https://api.collegefootballdata.com/roster',
                             params={'team': team, 'year': yr},
                             headers={'Authorization': f'Bearer {key}'}, timeout=30)
            if r.ok:
                roster_cache[(team, yr)] = r.json()
        except requests.RequestException:
            pass
        time.sleep(0.2)

    n_est = 0
    for i, row in need.iterrows():
        roster = roster_cache.get((row['college'], int(row['final_season'])), [])
        for pl in roster:
            full = f"{pl.get('firstName', '')} {pl.get('lastName', '')}"
            if norm_name(full) == row['name_key'] and pl.get('year'):
                qbs.loc[i, 'age_at_draft'] = round(18.7 + float(pl['year']), 1)
                qbs.loc[i, 'age_source'] = 'class_year'
                n_est += 1
                break
    print(f'[3] + class_year:    {qbs.age_at_draft.notna().sum()}/{len(qbs)} ({n_est} estimates)')

    out = ROOT / 'data' / 'raw' / 'qb_ages.csv'
    qbs[['player_name', 'draft_year', 'age_at_draft', 'age_source']].to_csv(out, index=False)
    print(f'\n[OK] -> {out}')
    print(qbs['age_source'].value_counts(dropna=False).to_string())


if __name__ == '__main__':
    main()
