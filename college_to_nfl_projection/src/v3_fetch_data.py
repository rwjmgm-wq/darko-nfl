"""
V3 Data Fetch.

1. Game-level QB PPA with opponents, garbage time EXCLUDED (CFBD /ppa/players/games,
   available 2013+) -> data/raw/qb_game_ppa.csv
2. Season player stats, passing + rushing categories, 2003-2025
   (completion %, rushing volume/production) -> data/raw/player_season_stats.csv
3. NFL contracts (nflverse/OTC) for the second-contract target -> data/raw/contracts_qb.csv
4. Draft picks WITH age at draft -> data/raw/draft_picks_with_age.csv
"""

import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).parent.parent
PPA_YEARS = range(2013, 2026)
STAT_YEARS = range(2003, 2026)


def cfbd_key():
    for line in (ROOT / '.env').read_text().splitlines():
        if line.startswith('CFBD_API_KEY'):
            return line.split('=', 1)[1].strip()
    raise RuntimeError('CFBD_API_KEY not found')


def cfbd_get(endpoint, params, key):
    r = requests.get(f'https://api.collegefootballdata.com{endpoint}',
                     params=params, headers={'Authorization': f'Bearer {key}'}, timeout=60)
    r.raise_for_status()
    return r.json()


def fetch_game_ppa(key):
    rows = []
    for year in PPA_YEARS:
        n_year = 0
        for season_type, weeks in [('regular', range(1, 18)), ('postseason', range(1, 3))]:
            for week in weeks:
                try:
                    data = cfbd_get('/ppa/players/games',
                                    {'year': year, 'week': week, 'position': 'QB',
                                     'seasonType': season_type, 'excludeGarbageTime': 'true'}, key)
                except requests.HTTPError as e:
                    print(f'  [WARN] {year} {season_type} wk{week}: {e}')
                    continue
                for g in data:
                    ppa = g.get('averagePPA') or {}
                    rows.append({
                        'season': g.get('season', year),
                        'week': g.get('week', week),
                        'season_type': season_type,
                        'player': g.get('name'),
                        'player_id': g.get('id'),
                        'team': g.get('team'),
                        'opponent': g.get('opponent'),
                        'ppa_all': ppa.get('all'),
                        'ppa_pass': ppa.get('pass'),
                        'ppa_rush': ppa.get('rush'),
                    })
                    n_year += 1
                time.sleep(0.25)
        print(f'  {year}: {n_year} QB-games')
    df = pd.DataFrame(rows)
    out = ROOT / 'data' / 'raw' / 'qb_game_ppa.csv'
    df.to_csv(out, index=False)
    print(f'[OK] {len(df):,} QB-games -> {out}')


def fetch_season_stats(key):
    rows = []
    for year in STAT_YEARS:
        for cat in ['passing', 'rushing']:
            try:
                data = cfbd_get('/stats/player/season', {'year': year, 'category': cat}, key)
            except requests.HTTPError as e:
                print(f'  [WARN] {year} {cat}: {e}')
                continue
            for s in data:
                rows.append({
                    'season': s.get('season', year),
                    'player': s.get('player'),
                    'player_id': s.get('playerId'),
                    'team': s.get('team'),
                    'position': s.get('position'),
                    'category': cat,
                    'stat_type': s.get('statType'),
                    'stat': s.get('stat'),
                })
            time.sleep(0.25)
        print(f'  {year}: done')
    df = pd.DataFrame(rows)
    out = ROOT / 'data' / 'raw' / 'player_season_stats.csv'
    df.to_csv(out, index=False)
    print(f'[OK] {len(df):,} stat rows -> {out}')


def fetch_contracts():
    import nfl_data_py as nfl
    df = nfl.import_contracts()
    qb = df[df['position'] == 'QB'].copy()
    out = ROOT / 'data' / 'raw' / 'contracts_qb.csv'
    qb.to_csv(out, index=False)
    print(f'[OK] {len(qb)} QB contracts -> {out}')


def fetch_draft_age():
    import nfl_data_py as nfl
    df = nfl.import_draft_picks()
    qb = df[df['position'] == 'QB'].copy()
    keep = [c for c in ['season', 'round', 'pick', 'pfr_player_name', 'college', 'age',
                        'pfr_player_id', 'gsis_id'] if c in qb.columns]
    qb = qb[keep].rename(columns={'season': 'draft_year', 'pfr_player_name': 'player_name'})
    out = ROOT / 'data' / 'raw' / 'draft_picks_with_age.csv'
    qb.to_csv(out, index=False)
    print(f'[OK] {len(qb)} draft picks (age non-null: {qb["age"].notna().sum()}) -> {out}')


def main():
    key = cfbd_key()
    print('[1] Game-level QB PPA (2013-2025, garbage time excluded)...')
    fetch_game_ppa(key)
    print('\n[2] Season player stats (passing + rushing, 2003-2025)...')
    fetch_season_stats(key)
    print('\n[3] NFL contracts...')
    fetch_contracts()
    print('\n[4] Draft picks with age...')
    fetch_draft_age()


if __name__ == '__main__':
    main()
