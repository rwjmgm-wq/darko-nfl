"""
H3 landing-spot covariates (SPEC_V3.md, H3). Draft-day information only.

  l1_team_pass_epa_prev  drafting team's QB-dropback EPA/play, draft_year-1
  l2_hc_new              Week-1 HC of draft_year != final HC of draft_year-1
  l3_qb_churn3           distinct primary passers, draft_year-3..-1

Teams from nflverse draft picks joined on (draft_year, pick). Output:
data/processed/v3_landing_features.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import nfl_data_py as nfl

ROOT = Path(__file__).parent.parent
PARENT = ROOT.parent
FRANCHISE = {'STL': 'LA', 'SL': 'LA', 'LAR': 'LA', 'SD': 'LAC', 'OAK': 'LV',
             'ARZ': 'ARI', 'BLT': 'BAL', 'CLV': 'CLE', 'HST': 'HOU',
             # PFR-style codes in nflverse draft picks
             'GNB': 'GB', 'KAN': 'KC', 'NOR': 'NO', 'NWE': 'NE',
             'SDG': 'LAC', 'SFO': 'SF', 'TAM': 'TB', 'LVR': 'LV'}


def fmap(s):
    return s.replace(FRANCHISE)


def main():
    feats = pd.read_csv(ROOT / 'data' / 'processed' / 'v3_features.csv')
    qbs = feats[feats['pick'].notna() & (feats['draft_year'] >= 2007)][
        ['player_name', 'draft_year', 'pick']].copy()
    qbs['pick'] = qbs['pick'].astype(int)
    print(f'drafted QBs to match: {len(qbs)}')

    picks = nfl.import_draft_picks(list(range(2007, 2026)))
    picks = picks[['season', 'pick', 'team', 'pfr_player_name']].rename(
        columns={'season': 'draft_year'})
    picks['team'] = fmap(picks['team'])
    picks['pick'] = picks['pick'].astype(int)
    m = qbs.merge(picks, on=['draft_year', 'pick'], how='left')
    n_miss = m['team'].isna().sum()
    print(f'team join misses: {n_miss}')
    if n_miss:
        print(m[m['team'].isna()][['player_name', 'draft_year', 'pick']].to_string())
    print('team codes:', sorted(m['team'].dropna().unique()))

    # L1: team dropback EPA/play in draft_year-1
    base = pd.read_csv(PARENT / 'data' / 'production' / 'qb_games_base.csv')
    base['posteam'] = fmap(base['posteam'])
    grp = base.groupby(['season', 'posteam'])[['qb_epa_sum', 'qb_epa_count']].sum()
    off = grp['qb_epa_sum'] / grp['qb_epa_count']

    # L2: coaches from schedules
    sched = nfl.import_schedules(list(range(2006, 2026))).dropna(subset=['gameday'])
    home = sched[['season', 'gameday', 'home_team', 'home_coach']].rename(
        columns={'home_team': 'team', 'home_coach': 'coach'})
    away = sched[['season', 'gameday', 'away_team', 'away_coach']].rename(
        columns={'away_team': 'team', 'away_coach': 'coach'})
    tg = pd.concat([home, away], ignore_index=True)
    tg['team'] = fmap(tg['team'])
    tg = tg.dropna(subset=['coach']).sort_values(['season', 'team', 'gameday'])
    last_coach = tg.groupby(['season', 'team'])['coach'].last()
    first_coach = tg.groupby(['season', 'team'])['coach'].first()

    # L3: primary passer per team-season from weekly player stats
    frames = []
    for yr in range(2004, 2026):
        try:
            frames.append(nfl.import_weekly_data([yr]))
        except Exception as e:
            print(f'    [WARN] weekly {yr} unavailable ({type(e).__name__}); '
                  'churn windows touching it are truncated')
    wk = pd.concat(frames, ignore_index=True)
    pid_col = 'player_id' if 'player_id' in wk.columns else 'gsis_id'
    att = wk.dropna(subset=['attempts']).groupby(
        [pid_col, 'season', 'recent_team'])['attempts'].sum().reset_index()
    att['recent_team'] = fmap(att['recent_team'])
    primary = att.loc[att.groupby(['season', 'recent_team'])['attempts'].idxmax()]
    primary = primary.set_index(['season', 'recent_team'])[pid_col]

    rows = []
    for _, q in m.iterrows():
        team, dy = q['team'], int(q['draft_year'])
        rec = {'player_name': q['player_name'], 'draft_year': dy, 'nfl_team': team,
               'l1_team_pass_epa_prev': np.nan, 'l2_hc_new': np.nan,
               'l3_qb_churn3': np.nan}
        if pd.notna(team):
            rec['l1_team_pass_epa_prev'] = off.get((dy - 1, team), np.nan)
            c_prev = last_coach.get((dy - 1, team))
            c_new = first_coach.get((dy, team))
            if c_prev is not None and c_new is not None:
                rec['l2_hc_new'] = int(c_new != c_prev)
            starters = {primary.get((s, team)) for s in range(dy - 3, dy)}
            starters.discard(None)
            if starters:
                rec['l3_qb_churn3'] = len(starters)
        rows.append(rec)

    out = pd.DataFrame(rows)
    path = ROOT / 'data' / 'processed' / 'v3_landing_features.csv'
    out.to_csv(path, index=False)
    print(f'\n[OK] {len(out)} QBs -> {path.name}')
    for c in ['l1_team_pass_epa_prev', 'l2_hc_new', 'l3_qb_churn3']:
        print(f'  {c}: non-null {out[c].notna().mean():.3f}  mean {out[c].mean():+.3f}')


if __name__ == '__main__':
    main()
