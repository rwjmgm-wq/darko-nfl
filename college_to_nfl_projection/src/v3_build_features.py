"""
V3 Feature Builder (per SPEC_V3.md - written before evaluation).

Produces:
  data/processed/v3_features.csv       one row per QB (college block + draft block)
  data/processed/v3_person_periods.csv person-period rows for the hazard model
  data/processed/v3_params.json        shrinkage constant, game-level beta, diagnostics

College block (fixed, per spec): epa_adj (game-level where 2013+, else v2
season-level), success_rate, big_play_rate, log_attempts, rush_share, rush_ypg,
comp_pct, age_at_draft (+missing flag), seasons_played.

Shrinkage: epa_adj and success_rate shrunk toward the mean with w = n/(n+k),
k from the between/within variance decomposition of season-level EPA.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent

# reuse v2 team-name normalization
import sys
sys.path.insert(0, str(Path(__file__).parent))
from v2_build_adjusted_stats import TEAM_ALIASES, load_sp_plus, normalize_team


def norm_name(n):
    n = str(n).lower().strip()
    n = re.sub(r'\s+(jr|sr|ii|iii|iv|v)\.?$', '', n)
    n = re.sub(r"[^a-z\s]", '', n)
    return re.sub(r'\s+', ' ', n)


def fl_key(n):
    """F.Last key used by nflverse pbp passer names."""
    parts = str(n).replace('.', '').split()
    if len(parts) < 2:
        return str(n)
    last = parts[-1]
    if last.lower() in ('jr', 'sr', 'ii', 'iii', 'iv', 'v') and len(parts) >= 3:
        last = parts[-2]
    return f'{parts[0][0]}.{last}'.lower()


# ---------------------------------------------------------------- game-level adjustment

def build_game_level_adjustment(df_sp):
    """Per-QB-season adjusted EPA from game PPA (2013+, garbage time excluded)."""
    games = pd.read_csv(ROOT / 'data' / 'raw' / 'qb_game_ppa.csv')
    games = games.dropna(subset=['ppa_all', 'opponent'])

    sp_lookup = df_sp.set_index(['team', 'season'])['defense_rating']
    season_mean_def = df_sp.groupby('season')['defense_rating'].mean()
    valid = set(df_sp['team'].unique())

    opp_norm = {o: normalize_team(o, valid) for o in games['opponent'].unique()}
    games['opp_norm'] = games['opponent'].map(opp_norm)
    idx = pd.MultiIndex.from_frame(games[['opp_norm', 'season']])
    games['opp_def'] = sp_lookup.reindex(idx).values
    # FCS opponents: worst FBS defense that season (same convention as v2)
    worst = df_sp.groupby('season')['defense_rating'].max()
    fcs = games['opp_def'].isna()
    games.loc[fcs, 'opp_def'] = games.loc[fcs, 'season'].map(worst)
    games['opp_def_c'] = games['opp_def'] - games['season'].map(season_mean_def)

    # beta at game level, within QB-season (same estimator as v2, coarser unit)
    grp = games.groupby(['player_id', 'season'])
    games['ppa_dev'] = games['ppa_all'] - grp['ppa_all'].transform('mean')
    games['def_dev'] = games['opp_def_c'] - grp['opp_def_c'].transform('mean')
    denom = (games['def_dev'] ** 2).sum()
    beta_game = (games['ppa_dev'] * games['def_dev']).sum() / denom
    se = np.sqrt(((games['ppa_dev'] - beta_game * games['def_dev']) ** 2).sum()
                 / (len(games) - 2) / denom)
    print(f'    game-level beta = {beta_game:+.5f} (SE {se:.5f}, n={len(games):,} QB-games)')

    games['ppa_adj'] = games['ppa_all'] - beta_game * games['opp_def_c']
    per_season = games.groupby(['player_id', 'player', 'team', 'season']).agg(
        game_epa_adj=('ppa_adj', 'mean'),
        game_epa_raw=('ppa_all', 'mean'),
        n_games=('ppa_adj', 'size'),
        ppa_rush=('ppa_rush', 'mean'),
    ).reset_index()
    per_season['name_key'] = per_season['player'].map(norm_name)
    return per_season, beta_game, se


# ---------------------------------------------------------------- season stats (CFBD)

def build_season_stats():
    """Pivot CFBD season stats to per player-season passing/rushing columns."""
    raw = pd.read_csv(ROOT / 'data' / 'raw' / 'player_season_stats.csv')
    raw = raw[raw['position'] == 'QB'].copy()
    raw['stat'] = pd.to_numeric(raw['stat'], errors='coerce')
    raw['key'] = raw['category'].str.upper() + '_' + raw['stat_type']
    piv = raw.pivot_table(index=['player', 'team', 'season'], columns='key',
                          values='stat', aggfunc='first').reset_index()
    piv['name_key'] = piv['player'].map(norm_name)
    return piv


# ---------------------------------------------------------------- shrinkage constant

def estimate_shrinkage_k():
    """k such that w = n/(n+k), from season-level EPA variance decomposition."""
    career_dir = ROOT / 'data' / 'processed' / 'full_college_careers'
    summary = pd.read_csv(career_dir / 'career_summary.csv')
    means, ns, wvars = [], [], []
    for _, row in summary.iterrows():
        f = career_dir / row['file']
        if not f.exists():
            continue
        d = pd.read_csv(f)
        for season, g in d.groupby('season'):
            epa = g['epa'].dropna()
            if len(epa) >= 50:
                means.append(epa.mean())
                ns.append(len(epa))
                wvars.append(epa.var())
    means, ns, wvars = np.array(means), np.array(ns), np.array(wvars)
    s2_within = np.average(wvars, weights=ns)          # per-play noise variance
    var_obs = means.var()                               # observed spread of season means
    mean_noise = (wvars / ns).mean()                    # average sampling noise of a mean
    tau2 = max(var_obs - mean_noise, 1e-6)              # true between variance
    k = s2_within / tau2
    print(f'    shrinkage k = {k:.0f} plays (tau^2={tau2:.5f}, s2_within={s2_within:.3f}, '
          f'{len(means)} QB-seasons)')
    return float(k)


# ---------------------------------------------------------------- main assembly

def main():
    print('=' * 70)
    print('V3 FEATURE BUILDER')
    print('=' * 70)

    df_sp = load_sp_plus()

    print('\n[1] Game-level opponent adjustment (2013+)...')
    game_seasons, beta_game, beta_se = build_game_level_adjustment(df_sp)

    print('\n[2] CFBD season stats (comp%, rushing)...')
    sstats = build_season_stats()
    print(f'    {len(sstats)} QB-seasons of box stats')

    print('\n[3] Shrinkage constant...')
    k = estimate_shrinkage_k()

    print('\n[4] Assembling per-QB college block...')
    v2 = pd.read_csv(ROOT / 'data' / 'processed' / 'aggregated_stats_v2' / 'stats_career_average_v2.csv')
    v2['name_key'] = v2['player_name'].map(norm_name)

    # career play files give the season list per QB
    career_dir = ROOT / 'data' / 'processed' / 'full_college_careers'
    summary = pd.read_csv(career_dir / 'career_summary.csv')
    summary['name_key'] = summary['player_name'].map(norm_name)

    game_by_player = game_seasons.groupby('name_key')
    stats_by_player = sstats.groupby('name_key')

    rows = []
    for _, qb in v2.iterrows():
        nk = qb['name_key']
        row = {
            'player_name': qb['player_name'],
            'draft_year': qb['draft_year'],
            'college': qb['college'],
            'success_rate': qb['success_rate'],
            'big_play_rate': qb['big_play_rate'],
            'attempts': qb['attempts'],
            'log_attempts': np.log(max(qb['attempts'], 1)),
            'seasons_played': qb['seasons_played'],
            'is_fcs_team': qb.get('is_fcs_team', 0),
            'sos_percentile': qb.get('sos_percentile', np.nan),
        }

        # epa_adj: game-level (2013+) if this QB's college seasons are covered,
        # else v2 season-level schedule adjustment
        epa_adj, game_level = qb['epa_per_play_adj'], 0
        if nk in game_by_player.groups:
            first_season = qb['draft_year'] - qb['seasons_played']
            g = game_by_player.get_group(nk)
            # restrict to this QB's college window to avoid same-name collisions
            g = g[(g['season'] >= max(2013, first_season)) & (g['season'] < qb['draft_year'])]
            if g['n_games'].sum() >= 6 and first_season >= 2012:
                epa_adj = np.average(g['game_epa_adj'], weights=g['n_games'])
                game_level = 1
        row['epa_adj'] = epa_adj
        row['game_level_adj'] = game_level

        # EB shrinkage toward the sample means (weights = attempts/(attempts+k))
        row['epa_adj_shrunk'] = np.nan  # filled after loop (needs pool means)
        row['success_rate_shrunk'] = np.nan

        # rushing + accuracy from CFBD box stats
        rush_yds = rush_gp = pass_yds = comp = att = 0.0
        if nk in stats_by_player.groups:
            s = stats_by_player.get_group(nk)
            rush_yds = s.get('RUSHING_YDS', pd.Series(dtype=float)).sum()
            pass_yds = s.get('PASSING_YDS', pd.Series(dtype=float)).sum()
            comp = s.get('PASSING_COMPLETIONS', pd.Series(dtype=float)).sum()
            att = s.get('PASSING_ATT', pd.Series(dtype=float)).sum()
            rush_gp = len(s)
        # rushing production can't be negative (sack yardage nets below zero)
        rush_yds = max(rush_yds, 0)
        total_yds = rush_yds + pass_yds
        row['rush_share'] = rush_yds / total_yds if total_yds > 0 else np.nan
        row['rush_ypg'] = rush_yds / (rush_gp * 12) if rush_gp > 0 else np.nan  # ~12 games/season
        row['comp_pct'] = comp / att if att > 0 else np.nan
        row['has_box_stats'] = 1 if att > 0 else 0
        rows.append(row)

    feats = pd.DataFrame(rows)

    # shrinkage (pool = all QBs in the table)
    for col, out in [('epa_adj', 'epa_adj_shrunk'), ('success_rate', 'success_rate_shrunk')]:
        mu = feats[col].mean()
        w = feats['attempts'] / (feats['attempts'] + k)
        feats[out] = w * feats[col] + (1 - w) * mu

    # draft block
    picks = pd.read_csv(ROOT / 'data' / 'raw' / 'draft_picks_with_age.csv')
    picks['name_key'] = picks['player_name'].map(norm_name)
    picks = picks.drop_duplicates(subset=['name_key', 'draft_year'])
    feats['name_key'] = feats['player_name'].map(norm_name)
    feats = feats.merge(picks[['name_key', 'draft_year', 'pick']],
                        on=['name_key', 'draft_year'], how='left')

    # Stale prospect labels: the careers file tags some already-drafted QBs
    # with a future draft class (e.g., 2025 draftees listed as 2026 prospects).
    # If an "undrafted" QB matches a drafted QB within the prior year, adopt
    # the true draft year and pick.
    pick_by_name = picks.set_index(['name_key', 'draft_year'])['pick']
    relabels = 0
    for i, row in feats[feats['pick'].isna()].iterrows():
        for dy in (row['draft_year'] - 1, row['draft_year'] - 2):
            if (row['name_key'], dy) in pick_by_name.index:
                feats.loc[i, 'pick'] = pick_by_name.loc[(row['name_key'], dy)]
                feats.loc[i, 'draft_year'] = dy
                relabels += 1
                break
    if relabels:
        print(f'    [FIX] relabeled {relabels} stale prospects to their true draft class')

    # age: backfilled multi-source file (see v3_backfill_ages.py).
    # Exact for 100% of drafted QBs; class-year estimates for ~half of
    # pre-draft prospects; median + indicator for the rest.
    ages_path = ROOT / 'data' / 'raw' / 'qb_ages.csv'
    if ages_path.exists():
        ages = pd.read_csv(ages_path)
        feats = feats.merge(ages[['player_name', 'draft_year', 'age_at_draft', 'age_source']],
                            on=['player_name', 'draft_year'], how='left')
    else:
        feats['age_at_draft'] = np.nan
        feats['age_source'] = None
    feats['age_missing'] = feats['age_at_draft'].isna().astype(int)
    feats['age_at_draft'] = feats['age_at_draft'].fillna(feats['age_at_draft'].median())

    # A player can appear twice when a stale prospect row relabels onto his
    # real draft class (and the ages merge fans out). Keep the fullest record.
    feats = (feats.sort_values('attempts', ascending=False)
             .drop_duplicates(subset=['player_name', 'draft_year'])
             .sort_index())
    feats['pick_filled'] = feats['pick'].fillna(263)
    feats['log_pick'] = np.log(feats['pick_filled'])

    print(f'    {len(feats)} QBs | game-level adj: {feats.game_level_adj.mean()*100:.0f}% | '
          f'box stats: {feats.has_box_stats.mean()*100:.0f}% | age known: {(1-feats.age_missing.mean())*100:.0f}%')

    print('\n[5] Person-period table for the hazard model...')
    starts = pd.read_csv(ROOT / 'data' / 'processed' / 'nfl_starts_by_season.csv')
    starts['flk'] = starts['passer_player_name'].str.lower()
    feats['flk'] = feats['player_name'].map(fl_key)

    pp_rows = []
    collisions = []
    drafted = feats[feats['pick'].notna()]
    for _, qb in drafted.iterrows():
        dy = int(qb['draft_year'])
        s = starts[(starts['flk'] == qb['flk']) &
                   (starts['season'] >= dy) & (starts['season'] <= dy + 10)]
        # Guard against F.Last collisions: if several passer ids share the key
        # in this window, keep the id whose first season is closest to (and not
        # before) the draft year - a drafted QB debuts in his draft season.
        if s['passer_player_id'].nunique() > 1:
            firsts = s.groupby('passer_player_id')['season'].min()
            eligible = firsts[firsts >= dy]
            chosen = (eligible if len(eligible) else firsts).idxmin()
            collisions.append((qb['player_name'], dy, s['passer_player_id'].nunique()))
            s = s[s['passer_player_id'] == chosen]
        starter_seasons = set(s[s['games_10att'] >= 10]['season'])
        first_starter = min(starter_seasons) if starter_seasons else None
        for season in range(dy, min(dy + 5, 2026)):  # t = 1..5
            t = season - dy + 1
            if first_starter is not None and season > first_starter:
                break  # event already happened
            event = 1 if (first_starter is not None and season == first_starter) else 0
            pp_rows.append({'player_name': qb['player_name'], 'draft_year': dy,
                            'season': season, 't': t, 'event': event})
            if event:
                break
    pp = pd.DataFrame(pp_rows)
    n_events = pp.groupby('player_name')['event'].max()
    print(f'    {len(pp)} person-periods, {len(n_events)} QBs, '
          f'{n_events.sum()} became starters ({n_events.mean()*100:.0f}%)')
    if collisions:
        print(f'    [WARN] {len(collisions)} F.Last collisions resolved by debut season:')
        for name, dy, n in collisions[:10]:
            print(f'       {name} ({dy}): {n} candidate ids')

    print('\n[6] Saving...')
    out_dir = ROOT / 'data' / 'processed'
    feats.drop(columns=['flk']).to_csv(out_dir / 'v3_features.csv', index=False)
    pp.to_csv(out_dir / 'v3_person_periods.csv', index=False)
    with open(out_dir / 'v3_params.json', 'w') as f:
        json.dump({'shrinkage_k': k, 'beta_game': beta_game, 'beta_game_se': beta_se}, f, indent=2)
    print(f'    v3_features.csv ({len(feats)}), v3_person_periods.csv ({len(pp)})')


if __name__ == '__main__':
    main()
