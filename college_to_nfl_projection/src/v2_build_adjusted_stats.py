"""
V2 Adjusted Stats Builder.

Replaces the broken adjustment in apply_sp_plus_to_careers.py (which compared the
national-average defense to itself, so adjusted EPA always equaled raw EPA) and the
sign-inconsistent multiplicative adjustments in the older 50-QB pipeline.

Method
------
1. Estimate how much opponent defense quality moves QB EPA, from the one dataset
   with per-play opponents (50 QBs, 20.8k plays):
       within-QB-season centered:  ppa_dev ~ beta * opp_def_dev
   Centering within QB removes QB quality from the slope; centering defense within
   season removes rating-scale drift. SP+ defense rating: LOWER = better defense,
   so beta is expected POSITIVE (worse defenses allow more EPA).

2. For all 396 QBs (2007-2026 classes), compute each QB-season's actual schedule
   strength from the games table (FBS opponents' defense ratings; FCS opponents
   imputed at the worst FBS defense that season), then adjust additively:
       epa_adj = epa_raw - beta * (avg_opp_def - national_mean_def)
   Additive, not multiplicative: a multiplier on signed EPA punishes bad plays
   against good opponents, which is backwards.

3. Re-aggregate careers (career_average / recency_weighted / final_season) with a
   real sos_percentile (percentile of the team's schedule strength among all FBS
   teams that season).

Outputs: data/processed/aggregated_stats_v2/stats_{method}_v2.csv
         data/processed/aggregated_stats_v2/adjustment_params.json
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent

# Career-file college names that don't literally match CFBD team names.
TEAM_ALIASES = {
    'florida st.': 'Florida State',
    'colorado st.': 'Colorado State',
    'michigan st.': 'Michigan State',
    'mississippi st.': 'Mississippi State',
    'oklahoma st.': 'Oklahoma State',
    'oregon st.': 'Oregon State',
    'kansas st.': 'Kansas State',
    'arizona st.': 'Arizona State',
    'ohio st.': 'Ohio State',
    'penn st.': 'Penn State',
    'boise st.': 'Boise State',
    'fresno st.': 'Fresno State',
    'san diego st.': 'San Diego State',
    'san jose st.': 'San José State',
    'san josé st.': 'San José State',
    'san jose state': 'San José State',
    'appalachian st.': 'Appalachian State',
    'georgia st.': 'Georgia State',
    'utah st.': 'Utah State',
    'washington st.': 'Washington State',
    'iowa st.': 'Iowa State',
    'nc state': 'NC State',
    'north carolina st.': 'NC State',
    'ole miss': 'Ole Miss',
    'mississippi': 'Ole Miss',
    'pitt': 'Pittsburgh',
    'miami (fl)': 'Miami',
    'miami (fla.)': 'Miami',
    'miami (oh)': 'Miami (OH)',
    'miami (ohio)': 'Miami (OH)',
    'central florida': 'UCF',
    'texas christian': 'TCU',
    'southern california': 'USC',
    'southern methodist': 'SMU',
    'brigham young': 'BYU',
    'louisiana state': 'LSU',
    'texas-san antonio': 'UT San Antonio',
    'utsa': 'UT San Antonio',
    'texas a&m': 'Texas A&M',
    'ala-birmingham': 'UAB',
    'bowling green st.': 'Bowling Green',
    'la-monroe': 'Louisiana Monroe',
    'louisiana-monroe': 'Louisiana Monroe',
    'louisiana-lafayette': 'Louisiana',
    'w. michigan': 'Western Michigan',
    'western kentucky': 'Western Kentucky',
    'e. washington': 'Eastern Washington',
    'connecticut': 'UConn',
    'massachusetts': 'UMass',
    'hawaii': "Hawai'i",
    "hawai'i": "Hawai'i",
}


def normalize_team(name, valid_names):
    """Map a college name to its CFBD team name; return None if no match."""
    if pd.isna(name):
        return None
    name = str(name).strip()
    if name in valid_names:
        return name
    low = name.lower()
    if low in TEAM_ALIASES and TEAM_ALIASES[low] in valid_names:
        return TEAM_ALIASES[low]
    # "Xxx St." -> "Xxx State"
    expanded = re.sub(r'\bSt\.$', 'State', name)
    if expanded in valid_names:
        return expanded
    # case-insensitive exact
    for v in valid_names:
        if v.lower() == low:
            return v
    return None


def load_sp_plus():
    base = pd.read_csv(ROOT / 'data' / 'processed' / 'sp_plus_historical' / 'sp_plus_all_years.csv')
    extra_path = ROOT / 'data' / 'processed' / 'sp_plus_historical' / 'sp_plus_extra_years.csv'
    if extra_path.exists():
        extra = pd.read_csv(extra_path)
        base = pd.concat([base, extra[~extra.season.isin(base.season.unique())]], ignore_index=True)
    base = base.dropna(subset=['defense_rating'])
    return base


def estimate_beta(df_sp):
    """Within-QB-season regression of play EPA on opponent defense rating."""
    plays = pd.read_csv(ROOT / 'data' / 'processed' / 'play_by_play' / 'all_qb_plays_with_sp_plus.csv')

    sp_lookup = df_sp.set_index(['team', 'season'])['defense_rating']
    season_mean_def = df_sp.groupby('season')['defense_rating'].mean()

    valid = set(df_sp['team'].unique())
    plays['opp_norm'] = plays['opponent'].map(lambda n: normalize_team(n, valid))
    idx = pd.MultiIndex.from_frame(plays[['opp_norm', 'season']])
    plays['opp_def'] = sp_lookup.reindex(idx).values

    matched = plays['opp_def'].notna()
    print(f'    Play-level opponent SP+ match: {matched.mean() * 100:.1f}% of {len(plays):,} plays')

    d = plays[matched & plays['ppa'].notna()].copy()
    d['opp_def_c'] = d['opp_def'] - d['season'].map(season_mean_def)
    # Center within QB-season so the slope reflects schedule variation, not QB quality
    grp = d.groupby(['player_name', 'season'])
    d['ppa_dev'] = d['ppa'] - grp['ppa'].transform('mean')
    d['def_dev'] = d['opp_def_c'] - grp['opp_def_c'].transform('mean')

    beta = (d['ppa_dev'] * d['def_dev']).sum() / (d['def_dev'] ** 2).sum()
    resid = d['ppa_dev'] - beta * d['def_dev']
    se = np.sqrt((resid ** 2).sum() / (len(d) - 2) / (d['def_dev'] ** 2).sum())
    print(f'    beta = {beta:+.5f} EPA/play per defense-rating point (SE {se:.5f}, n={len(d):,})')
    print(f'    Interpretation: a 10-point-worse defense inflates EPA/play by ~{beta * 10:+.3f}')

    # Same slope for success (EPA > 0), so success_rate can be schedule-adjusted too
    d['succ'] = (d['ppa'] > 0).astype(float)
    d['succ_dev'] = d['succ'] - d.groupby(['player_name', 'season'])['succ'].transform('mean')
    gamma = (d['succ_dev'] * d['def_dev']).sum() / (d['def_dev'] ** 2).sum()
    resid_g = d['succ_dev'] - gamma * d['def_dev']
    se_g = np.sqrt((resid_g ** 2).sum() / (len(d) - 2) / (d['def_dev'] ** 2).sum())
    print(f'    gamma = {gamma:+.5f} success-rate points per defense-rating point (SE {se_g:.5f})')
    return beta, se, gamma, se_g, len(d)


def build_schedule_strength(df_sp):
    """Average opponent defense rating for every (team, season), from real schedules."""
    games = pd.read_csv(ROOT / 'data' / 'raw' / 'schedules_all_years.csv')
    games = games[games['completed'] != False]  # keep completed & unknown

    long = pd.concat([
        games.rename(columns={'home_team': 'team', 'away_team': 'opponent'})[['season', 'team', 'opponent']],
        games.rename(columns={'away_team': 'team', 'home_team': 'opponent'})[['season', 'team', 'opponent']],
    ], ignore_index=True)

    sp_lookup = df_sp.set_index(['team', 'season'])['defense_rating']
    idx = pd.MultiIndex.from_frame(long[['opponent', 'season']])
    long['opp_def'] = sp_lookup.reindex(idx).values

    # Only FBS teams have SP+ ratings; restrict rows to rated teams so we get
    # schedule strength for every FBS team, with FCS opponents imputed at the
    # worst FBS defense that season.
    team_idx = pd.MultiIndex.from_frame(long[['team', 'season']])
    long['team_rated'] = sp_lookup.reindex(team_idx).notna().values
    long = long[long['team_rated']]

    worst_def = df_sp.groupby('season')['defense_rating'].max()
    long['is_fcs'] = long['opp_def'].isna()
    long.loc[long['is_fcs'], 'opp_def'] = long.loc[long['is_fcs'], 'season'].map(worst_def)

    sched = long.groupby(['team', 'season']).agg(
        avg_opp_def=('opp_def', 'mean'),
        n_games=('opp_def', 'size'),
        n_fcs=('is_fcs', 'sum'),
    ).reset_index()

    # SOS percentile among all FBS teams that season (lower avg_opp_def = tougher)
    sched['sos_percentile'] = sched.groupby('season')['avg_opp_def'].rank(pct=True, ascending=False) * 100
    return sched


def season_stats(df_season):
    """Raw per-season stats (same metrics as the original pipeline).

    Career files come from two collectors and don't all share columns;
    missing inputs produce NaN rather than crashing.
    """
    epa = df_season['epa'].dropna()
    stats = {
        'attempts': len(df_season),
        'epa_per_play_raw': epa.mean() if len(epa) else np.nan,
        'success_rate': (epa > 0).mean() if len(epa) else np.nan,
    }
    if 'yards_gained' in df_season:
        stats['big_play_rate'] = (df_season['yards_gained'] >= 15).mean()
    else:
        stats['big_play_rate'] = np.nan

    has_down = 'down' in df_season
    has_ytg = 'yards_to_goal' in df_season

    third = df_season[df_season['down'] == 3]['epa'].dropna() if has_down else pd.Series(dtype=float)
    stats['third_down_epa'] = third.mean() if len(third) else np.nan

    rz = df_season[df_season['yards_to_goal'] <= 20]['epa'].dropna() if has_ytg else pd.Series(dtype=float)
    stats['red_zone_epa'] = rz.mean() if len(rz) else np.nan

    if has_down and has_ytg:
        hl_mask = (df_season['down'].isin([3, 4])) | (df_season['yards_to_goal'] <= 20)
    elif has_down:
        hl_mask = df_season['down'].isin([3, 4])
    else:
        hl_mask = pd.Series(False, index=df_season.index)
    hl = df_season[hl_mask]['epa'].dropna()
    stats['high_leverage_epa'] = hl.mean() if len(hl) else np.nan

    stats['scoring_play_rate'] = df_season['scoring'].mean() if 'scoring' in df_season else np.nan
    return stats


def recency_weights(n):
    if n == 1:
        return np.array([1.0])
    if n == 2:
        return np.array([0.3, 0.7])
    if n == 3:
        return np.array([0.2, 0.3, 0.5])
    if n == 4:
        return np.array([0.1, 0.2, 0.3, 0.4])
    w = [0.05] * (n - 3) + [0.2, 0.3, 0.5]
    return np.array(w) / sum(w)


def aggregate(season_rows, method):
    keys = [k for k in season_rows[0] if k != 'season']
    if method == 'final_season':
        return {k: season_rows[-1][k] for k in keys}
    if method == 'career_average':
        w = np.ones(len(season_rows))
    elif method == 'recency_weighted':
        w = recency_weights(len(season_rows))
    else:
        raise ValueError(method)
    out = {}
    for k in keys:
        vals = np.array([r.get(k, np.nan) for r in season_rows], dtype=float)
        m = ~np.isnan(vals)
        out[k] = float(np.sum(vals[m] * w[m]) / w[m].sum()) if m.any() else np.nan
    return out


def main():
    print('=' * 70)
    print('V2 ADJUSTED STATS BUILDER')
    print('=' * 70)

    print('\n[1] Loading SP+ (2003-2025)...')
    df_sp = load_sp_plus()
    print(f'    {len(df_sp)} team-seasons, {df_sp.season.min()}-{df_sp.season.max()}')
    season_mean_def = df_sp.groupby('season')['defense_rating'].mean()

    print('\n[2] Estimating EPA ~ opponent-defense slope (play level, within-QB)...')
    beta, beta_se, gamma, gamma_se, n_plays = estimate_beta(df_sp)

    print('\n[3] Building schedule strength for every FBS team-season...')
    sched = build_schedule_strength(df_sp)
    print(f'    {len(sched)} team-seasons of schedule strength')

    print('\n[4] Processing QB careers...')
    career_dir = ROOT / 'data' / 'processed' / 'full_college_careers'
    summary = pd.read_csv(career_dir / 'career_summary.csv')
    valid_teams = set(df_sp['team'].unique())
    sched_idx = sched.set_index(['team', 'season'])
    worst_def_by_season = df_sp.groupby('season')['defense_rating'].max()

    unmatched = []
    methods = ['career_average', 'recency_weighted', 'final_season']
    results = {m: [] for m in methods}

    for _, row in summary.iterrows():
        career_file = career_dir / row['file']
        if not career_file.exists():
            continue
        df_career = pd.read_csv(career_file)
        if len(df_career) == 0:
            continue

        team = normalize_team(row['college'], valid_teams)
        is_fcs_team = team is None
        if is_fcs_team:
            unmatched.append((row['player_name'], row['college']))

        season_rows = []
        for season in sorted(df_career['season'].dropna().unique()):
            df_s = df_career[df_career['season'] == season]
            s = season_stats(df_s)
            s['season'] = season

            avg_opp_def = np.nan
            sos_pct = np.nan
            n_fcs = np.nan
            if team is not None and (team, season) in sched_idx.index:
                sc = sched_idx.loc[(team, season)]
                avg_opp_def = sc['avg_opp_def']
                sos_pct = sc['sos_percentile']
                n_fcs = sc['n_fcs']
            elif is_fcs_team and season in season_mean_def.index:
                # Non-FBS program: no rated opponents at all. Impute the entire
                # schedule at the worst FBS defense that season (the same
                # imputation FBS teams get for their FCS games) - the maximum
                # penalty our calibration can honestly support.
                avg_opp_def = worst_def_by_season[season]
                sos_pct = 0.0

            s['avg_opp_def'] = avg_opp_def
            s['sos_percentile'] = sos_pct
            s['fcs_games'] = n_fcs
            if not np.isnan(avg_opp_def) and not np.isnan(s['epa_per_play_raw']):
                sched_delta = avg_opp_def - season_mean_def[season]
                s['epa_per_play_adj'] = s['epa_per_play_raw'] - beta * sched_delta
                s['success_rate_adj'] = s['success_rate'] - gamma * sched_delta
            else:
                s['epa_per_play_adj'] = s['epa_per_play_raw']
                s['success_rate_adj'] = s['success_rate']
            s['has_sos'] = 0.0 if np.isnan(avg_opp_def) else 1.0
            s['is_fcs_team'] = 1.0 if is_fcs_team else 0.0
            season_rows.append(s)

        if not season_rows:
            continue

        meta = {
            'player_name': row['player_name'],
            'draft_year': row['draft_year'],
            'college': row['college'],
            'seasons_played': row['seasons_played'],
        }
        for m in methods:
            agg = aggregate(season_rows, m)
            agg.update(meta)
            results[m].append(agg)

    if unmatched:
        print(f'    [WARN] {len(unmatched)} colleges not matched to CFBD teams:')
        for name, college in unmatched[:20]:
            print(f'       {name}: {college!r}')

    out_dir = ROOT / 'data' / 'processed' / 'aggregated_stats_v2'
    out_dir.mkdir(parents=True, exist_ok=True)
    print('\n[5] Saving...')
    for m in methods:
        df_m = pd.DataFrame(results[m])
        path = out_dir / f'stats_{m}_v2.csv'
        df_m.to_csv(path, index=False)
        with_sos = (df_m['has_sos'] > 0).mean() * 100
        moved = (df_m['epa_per_play_adj'] - df_m['epa_per_play_raw']).abs()
        print(f'    {m}: {len(df_m)} QBs -> {path.name}'
              f' | SOS coverage {with_sos:.0f}% | mean |adj-raw| = {moved.mean():.4f}')

    params = {
        'beta_epa_per_defense_point': beta,
        'beta_se': beta_se,
        'gamma_success_per_defense_point': gamma,
        'gamma_se': gamma_se,
        'beta_n_plays': n_plays,
        'unmatched_colleges': [list(u) for u in unmatched],
    }
    with open(out_dir / 'adjustment_params.json', 'w') as f:
        json.dump(params, f, indent=2)
    print(f'    params -> adjustment_params.json')


if __name__ == '__main__':
    main()
