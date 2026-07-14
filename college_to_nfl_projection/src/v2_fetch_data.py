"""
V2 Data Fetch: schedules, SP+ top-up, and full draft pick history.

Fetches the inputs the corrected opponent adjustment needs:
1. CFBD /games for 2003-2025 -> data/raw/schedules_all_years.csv
   (per-game opponents so schedule strength reflects each QB's actual opponents,
   not the national average - the bug in apply_sp_plus_to_careers.py)
2. CFBD /ratings/sp for seasons missing from sp_plus_all_years.csv (2025)
   -> data/processed/sp_plus_historical/sp_plus_extra_years.csv
3. nfl_data_py draft picks (all years) -> data/raw/draft_picks_all.csv
   (log(draft pick) baseline needs picks for 2007-2014 classes too)
"""

import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).parent.parent
SEASONS = range(2003, 2026)


def cfbd_key():
    for line in (ROOT / '.env').read_text().splitlines():
        if line.startswith('CFBD_API_KEY'):
            return line.split('=', 1)[1].strip()
    raise RuntimeError('CFBD_API_KEY not found in .env')


def cfbd_get(endpoint, params, key):
    r = requests.get(
        f'https://api.collegefootballdata.com{endpoint}',
        params=params,
        headers={'Authorization': f'Bearer {key}'},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def pick(d, *names):
    """Return the first present key (API has used both snake_case and camelCase)."""
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return None


def fetch_schedules(key):
    out_path = ROOT / 'data' / 'raw' / 'schedules_all_years.csv'
    rows = []
    for year in SEASONS:
        games = cfbd_get('/games', {'year': year, 'seasonType': 'both'}, key)
        n = 0
        for g in games:
            home = pick(g, 'home_team', 'homeTeam')
            away = pick(g, 'away_team', 'awayTeam')
            if not home or not away:
                continue
            rows.append({
                'season': pick(g, 'season') or year,
                'week': pick(g, 'week'),
                'season_type': pick(g, 'season_type', 'seasonType'),
                'home_team': home,
                'away_team': away,
                'home_division': pick(g, 'home_division', 'homeClassification'),
                'away_division': pick(g, 'away_division', 'awayClassification'),
                'completed': pick(g, 'completed'),
            })
            n += 1
        print(f'  {year}: {n} games')
        time.sleep(0.5)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f'[OK] Saved {len(df):,} games -> {out_path}')
    return df


def fetch_sp_plus_extra(key):
    existing = pd.read_csv(ROOT / 'data' / 'processed' / 'sp_plus_historical' / 'sp_plus_all_years.csv')
    have = set(existing['season'].unique())
    need = [y for y in SEASONS if y not in have]
    if not need:
        print('[OK] SP+ already covers all seasons')
        return
    rows = []
    for year in need:
        data = cfbd_get('/ratings/sp', {'year': year}, key)
        for t in data:
            if pick(t, 'team') is None:
                continue
            rows.append({
                'season': year,
                'team': t['team'],
                'conference': pick(t, 'conference'),
                'overall_rating': pick(t, 'rating'),
                'offense_rating': (t.get('offense') or {}).get('rating'),
                'defense_rating': (t.get('defense') or {}).get('rating'),
                'special_teams_rating': (t.get('specialTeams') or {}).get('rating'),
            })
        print(f'  SP+ {year}: {sum(1 for r in rows if r["season"] == year)} teams')
        time.sleep(0.5)
    out_path = ROOT / 'data' / 'processed' / 'sp_plus_historical' / 'sp_plus_extra_years.csv'
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f'[OK] Saved extra SP+ seasons {need} -> {out_path}')


def fetch_draft_picks():
    import nfl_data_py as nfl
    df = nfl.import_draft_picks()
    qbs = df[df['position'] == 'QB'][['season', 'round', 'pick', 'pfr_player_name', 'college']].copy()
    qbs = qbs.rename(columns={'season': 'draft_year', 'pfr_player_name': 'player_name'})
    out_path = ROOT / 'data' / 'raw' / 'draft_picks_all.csv'
    qbs.to_csv(out_path, index=False)
    print(f'[OK] Saved {len(qbs)} QB draft picks ({qbs.draft_year.min()}-{qbs.draft_year.max()}) -> {out_path}')


def main():
    key = cfbd_key()
    print('[1] Fetching schedules 2003-2025...')
    fetch_schedules(key)
    print('\n[2] Topping up SP+ seasons...')
    fetch_sp_plus_extra(key)
    print('\n[3] Fetching full QB draft pick history...')
    fetch_draft_picks()


if __name__ == '__main__':
    main()
