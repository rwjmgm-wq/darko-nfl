"""
Amendment A4: environment-layer covariates (docs/SPEC_LEAF_V3.md, A4).

Builds data/production/v4_env_features.csv keyed (passer_player_id, y) where
y is the predictor season and the covariates describe the Y+1 environment as
knowable at Y+1 kickoff:

  e1_new_team      QB's Y+1 team != team of his last Y game
  e2_coach_change  Y+1 team's Week-1 HC != that franchise's final Y HC
  e3_ret_rec_epa   sum of season-Y receiving EPA over WR/TE/RB on the Y+1
                   team's Week-1 roster, / 17
  e4_team_pass_env Y+1 team's season-Y team pass EPA/dropback
  e5_sched_def     mean season-Y defensive EPA/dropback allowed across the
                   Y+1 team's scheduled regular-season opponents

Proxies (disclosed in spec): Y+1 team from Y+1 Week-1 roster, fallback = first
Y+1 game posteam (fallback count reported); Y+1 coach from first scheduled
Y+1 game. Franchise moves mapped to stable codes.

Run once; no target data is touched here.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import nfl_data_py as nfl

ROOT = Path(__file__).parent.parent.parent
FRANCHISE = {'STL': 'LA', 'SD': 'LAC', 'OAK': 'LV',
             # PFR-style codes in older roster files
             'CLV': 'CLE', 'BLT': 'BAL', 'SL': 'LA', 'ARZ': 'ARI', 'HST': 'HOU'}
FIRST_Y, LAST_Y1 = 2006, 2025   # predictor seasons 2006-2024, targets 2007-2025


def fmap(s):
    return s.replace(FRANCHISE)


def main():
    ratings = pd.read_csv(ROOT / 'data' / 'production' / 'leaf_v3_ratings.csv')
    ratings['posteam'] = fmap(ratings['posteam'])

    # candidate pairs: QB appears in both Y and Y+1 (eval applies its own filter)
    seasons_by_pid = ratings.groupby('passer_player_id')['season'].agg(set)
    pairs = [(pid, y) for pid, ss in seasons_by_pid.items()
             for y in sorted(ss) if (y + 1) in ss and FIRST_Y <= y and y + 1 <= LAST_Y1]
    print(f'candidate pairs: {len(pairs)}')

    # last-Y-game team per (pid, season)
    r = ratings.sort_values(['passer_player_id', 'game_date'])
    last_team = r.groupby(['passer_player_id', 'season'])['posteam'].last()
    first_team = r.groupby(['passer_player_id', 'season'])['posteam'].first()

    # ---- schedules: coaches + opponents ----
    print('loading schedules...')
    sched = nfl.import_schedules(list(range(FIRST_Y, LAST_Y1 + 1)))
    sched = sched.dropna(subset=['gameday'])
    home = sched[['season', 'gameday', 'game_type', 'home_team', 'away_team', 'home_coach']].rename(
        columns={'home_team': 'team', 'away_team': 'opp', 'home_coach': 'coach'})
    away = sched[['season', 'gameday', 'game_type', 'away_team', 'home_team', 'away_coach']].rename(
        columns={'away_team': 'team', 'home_team': 'opp', 'away_coach': 'coach'})
    tg = pd.concat([home, away], ignore_index=True)
    tg['team'] = fmap(tg['team'])
    tg['opp'] = fmap(tg['opp'])
    tg = tg.sort_values(['season', 'team', 'gameday'])
    last_coach = tg.dropna(subset=['coach']).groupby(['season', 'team'])['coach'].last()
    first_coach = tg.dropna(subset=['coach']).groupby(['season', 'team'])['coach'].first()
    reg_opps = tg[tg['game_type'] == 'REG'].groupby(['season', 'team'])['opp'].agg(list)

    # ---- team-season pass offense/defense from qb_games_base ----
    base = pd.read_csv(ROOT / 'data' / 'production' / 'qb_games_base.csv')
    base['posteam'] = fmap(base['posteam'])
    base['defteam'] = fmap(base['defteam'])
    off = base.groupby(['season', 'posteam']).apply(
        lambda g: g['qb_epa_sum'].sum() / g['qb_epa_count'].sum())
    dfn = base.groupby(['season', 'defteam']).apply(
        lambda g: g['qb_epa_sum'].sum() / g['qb_epa_count'].sum())

    # ---- receiving EPA by player-season ----
    print('loading weekly player stats...')
    wk = nfl.import_weekly_data(list(range(FIRST_Y, LAST_Y1)))  # seasons Y only
    pid_col = 'player_id' if 'player_id' in wk.columns else 'gsis_id'
    rec_epa = wk.dropna(subset=['receiving_epa']).groupby(
        [pid_col, 'season'])['receiving_epa'].sum()

    # ---- Week-1 rosters per Y+1 season ----
    print('loading weekly rosters (week 1)...')
    wk1 = {}
    for y1 in range(FIRST_Y + 1, LAST_Y1 + 1):
        try:
            ro = nfl.import_weekly_rosters([y1])
            rpid = 'gsis_id' if 'gsis_id' in ro.columns else 'player_id'
            w1 = ro[ro['week'] == ro['week'].min()][['team', 'position', rpid]].rename(
                columns={rpid: 'pid'}).dropna(subset=['pid'])
            w1['team'] = fmap(w1['team'])
            wk1[y1] = w1
            print(f'  {y1}: {len(w1)} roster rows')
        except Exception as e:
            wk1[y1] = None
            print(f'  [WARN] {y1} rosters unavailable: {e}')

    rows, n_fallback, n_coach_missing = [], 0, 0
    for pid, y in pairs:
        y1 = y + 1
        team_y = last_team.get((pid, y))

        # Y+1 team: Week-1 roster primary, first-game fallback (flagged —
        # a QB absent from the Week-1 roster may have been unsigned at the
        # forecast origin, so his Y+1 team is post-origin information)
        team_y1, fallback = None, 0
        ro = wk1.get(y1)
        if ro is not None:
            hit = ro[ro['pid'] == pid]
            if len(hit):
                team_y1 = hit['team'].iloc[0]
        if team_y1 is None:
            team_y1 = first_team.get((pid, y1))
            fallback = 1
            n_fallback += 1

        e1 = int(team_y1 != team_y)

        c_prev = last_coach.get((y, team_y1))
        c_new = first_coach.get((y1, team_y1))
        if c_prev is None or c_new is None:
            e2 = 0
            n_coach_missing += 1
        else:
            e2 = int(c_new != c_prev)

        e3 = np.nan
        if ro is not None:
            corps = ro[(ro['team'] == team_y1) & (ro['position'].isin(['WR', 'TE', 'RB']))]
            e3 = sum(rec_epa.get((p, y), 0.0) for p in corps['pid']) / 17.0

        e4 = off.get((y, team_y1), np.nan)

        opps = reg_opps.get((y1, team_y1))
        e5 = np.nan
        if opps:
            vals = [dfn.get((y, o)) for o in opps]
            vals = [v for v in vals if v is not None and not pd.isna(v)]
            if vals:
                e5 = float(np.mean(vals))

        rows.append({'passer_player_id': pid, 'y': y, 'y1_team': team_y1,
                     'team_fallback': fallback,
                     'e1_new_team': e1, 'e2_coach_change': e2,
                     'e3_ret_rec_epa': e3, 'e4_team_pass_env': e4,
                     'e5_sched_def': e5})

    out = pd.DataFrame(rows)
    path = ROOT / 'data' / 'production' / 'v4_env_features.csv'
    out.to_csv(path, index=False)
    print(f'\n[OK] {len(out)} pairs -> {path.name}')
    print(f'team fallbacks (first-game proxy): {n_fallback}')
    print(f'coach lookups missing: {n_coach_missing}')
    print(f'e1 new-team rate: {out.e1_new_team.mean():.3f}')
    print(f'e2 coach-change rate: {out.e2_coach_change.mean():.3f}')
    print(f'e3 non-null: {out.e3_ret_rec_epa.notna().mean():.3f}  '
          f'e4: {out.e4_team_pass_env.notna().mean():.3f}  '
          f'e5: {out.e5_sched_def.notna().mean():.3f}')


if __name__ == '__main__':
    main()
