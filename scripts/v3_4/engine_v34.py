"""
LEAF v3.4 walk-forward rating engine.

Corrects two leakage defects found auditing v3 (scripts/v3_honest/build_leaf_v3.py,
left untouched for reproducibility):

  A. FUSION LABEL LEAKAGE. v3 required only that the ORIGIN game be in the
     training era; the 16-game target could run into 2019+. Measured: 94 of 819
     v3 fusion pairs (11.5%) had target games in the frozen era. v3.4 requires
     EVERY target game to be <= TRAIN_MAX_SEASON, and logs each pair's target
     start/end season and calendar span.

     v3's "next 16 games" is really "next 16 APPEARANCES": measured spans reach
     4,375 days (12 years) for QBs with long inactive stretches. v3.4 therefore
     fits the primary fusion on a DENSE variant (target span <= MAX_TARGET_SPAN_DAYS)
     and reports the long-span variant separately, labeled, rather than silently
     treating a decade-spanning label as a short-horizon forecast.

  B. SAME-GAME OPPONENT-ADJUSTMENT LEAKAGE. v3 updated a defense's state after
     every passer row, so when an offense used 2+ passers, later passers in the
     SAME game read a defense state already containing that game. Measured: 2,360
     affected rows. v3 also advanced league aggregates mid-date, letting
     simultaneous games influence one another. v3.4 processes strictly by date:
     all pre-game states for a date are read first, each defense is then updated
     ONCE PER GAME with the play-weighted aggregate EPA allowed across all
     passers, and league aggregates advance only at date boundaries.

All tuning/fitting uses TRAIN_MAX_SEASON and earlier only.

Outputs: data/production/leaf_v34_ratings.csv, leaf_v34_params.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
TRAIN_MAX_SEASON = 2018
MAX_TARGET_SPAN_DAYS = 730          # dense-target eligibility (~2 seasons)
HORIZON = 16                        # target games
STRIDE = 8                          # non-overlapping-ish origins


# ------------------------------------------------------------------ data

def load_games(base_path=None):
    path = base_path or (ROOT / 'data' / 'production' / 'qb_games_base_v34.csv')
    df = pd.read_csv(path)
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values(['game_date', 'game_id', 'passer_player_id']).reset_index(drop=True)
    df['plays'] = df['qb_epa_count'].clip(lower=1)
    df['epa'] = df['qb_epa_mean']
    df['cpoe'] = df['cpoe_mean'].fillna(0)
    df['success'] = df['success_mean']

    meta = pd.read_csv(ROOT / 'data' / 'raw' / 'player_meta.csv')
    meta['birth_date'] = pd.to_datetime(meta['birth_date'], errors='coerce')
    df = df.merge(meta[['gsis_id', 'birth_date', 'draft_pick']],
                  left_on='passer_player_id', right_on='gsis_id', how='left')
    df['age'] = (df['game_date'] - df['birth_date']).dt.days / 365.25
    df['log_pick'] = np.log(df['draft_pick'].fillna(263))
    return df


# ------------------------------------------- L1: date-batched defense ratings

def walkforward_defense_batched(df, halflife_days, k_shrink):
    """Defense rating strictly before each game, with NO same-game and NO
    same-date contamination (audit finding B).

    Protocol per date, in order:
      1. read: every row on the date gets a rating from state as of the
         PREVIOUS date;
      2. aggregate: per (game_id, defteam), play-weighted mean EPA allowed
         across all that game's passers;
      3. update: each defense advances once per game; league aggregates
         advance only after the whole date is scored.
    """
    lam = np.log(2) / halflife_days
    state = {}                      # defteam -> [w, s, last_date]
    lg_w, lg_s, lg_date = 0.0, 0.0, None
    ratings = np.zeros(len(df))

    for date, day in df.groupby('game_date', sort=True):
        # ---- 1. decay league to today, then READ all ratings for this date ----
        if lg_date is not None:
            dec = np.exp(-lam * (date - lg_date) / np.timedelta64(1, 'D'))
            lg_w *= dec
            lg_s *= dec
        lg_date = date
        lg_mean = lg_s / lg_w if lg_w > 50 else 0.0

        decayed = {}
        for t in day['defteam'].unique():
            if t in state:
                w, s, last = state[t]
                dec = np.exp(-lam * (date - last) / np.timedelta64(1, 'D'))
                decayed[t] = (w * dec, s * dec)
            else:
                decayed[t] = (0.0, 0.0)
        for i, t in zip(day.index.values, day['defteam'].values):
            w, s = decayed[t]
            ratings[i] = s / (w + k_shrink)

        # ---- 2. aggregate this date's evidence (one entry per game-defense) ----
        # play-weighted EPA allowed across every passer the offense used.
        # Vectorized sums (portable across pandas versions; no include_groups).
        tmp = pd.DataFrame({'game_id': day['game_id'].values,
                            'defteam': day['defteam'].values,
                            'w': day['plays'].values,
                            'wx': (day['plays'] * day['epa']).values})
        agg = tmp.groupby(['game_id', 'defteam'], as_index=False)[['w', 'wx']].sum()

        # ---- 3. UPDATE after the entire date is scored ----
        for t, w_add, wx_add in zip(agg['defteam'].values, agg['w'].values,
                                    agg['wx'].values):
            epa_allowed = wx_add / w_add
            w, s = decayed[t]
            state[t] = (w + w_add, s + w_add * (epa_allowed - lg_mean), date)
            lg_w += w_add
            lg_s += wx_add

    return ratings


def tune_defense(df):
    train = (df['season'] <= TRAIN_MAX_SEASON).values
    best = None
    print('  tuning defense layer (train era only)...')
    for hl in [180, 365, 730]:
        for k in [100, 300, 600]:
            r = walkforward_defense_batched(df, hl, k)
            c = np.corrcoef(r[train], df.loc[train, 'epa'])[0, 1]
            print(f'    halflife={hl:4d}d k={k:3d}: r = {c:+.4f}')
            if best is None or c > best[0]:
                best = (c, hl, k, r)
    _, hl, k, ratings = best
    print(f'  -> chosen: halflife={hl}, k={k}')
    return hl, k, ratings


# ------------------------------------------------------------------ L2: Kalman

def kalman_pass(values, plays, q, r_play, p0, prior_mean, drift=None):
    n = len(values)
    pri = np.empty(n); post = np.empty(n); var = np.empty(n)
    m, P = prior_mean, p0
    for i in range(n):
        if drift is not None:
            m = m + drift[i]
        P = P + q
        pri[i] = m
        R = r_play / plays[i]
        K = P / (P + R)
        m = m + K * (values[i] - m)
        P = (1 - K) * P
        post[i] = m; var[i] = P
    return pri, post, var


def tune_kalman(df, col, prior_mean):
    train_mask = (df['season'] <= TRAIN_MAX_SEASON).values
    groups = df.groupby('passer_player_id', sort=False)
    best = None
    for q in [1e-5, 5e-5, 2e-4, 8e-4, 3e-3]:
        for r_play in [1.0, 2.0, 3.0]:
            for p0 in [0.005, 0.02, 0.06]:
                se, wsum = 0.0, 0.0
                for _, g in groups:
                    v = g[col].values; pl = g['plays'].values
                    tr = train_mask[g.index.values]
                    if not tr.any():
                        continue
                    pri, _, _ = kalman_pass(v, pl, q, r_play, p0, prior_mean)
                    w = pl[tr]
                    se += (w * (v[tr] - pri[tr]) ** 2).sum()
                    wsum += w.sum()
                mse = se / wsum
                if best is None or mse < best[0]:
                    best = (mse, q, r_play, p0)
    mse, q, r_play, p0 = best
    print(f'  {col}: q={q:g}, r_play={r_play}, p0={p0} (train 1-step wMSE={mse:.4f})')
    return q, r_play, p0


# ------------------------------------------------------------ L4: priors + age

def fit_rookie_prior(df):
    firsts = []
    for pid, g in df.groupby('passer_player_id'):
        g0 = g[g['season'] == g['season'].min()]
        if g0['season'].iloc[0] > TRAIN_MAX_SEASON or g0['plays'].sum() < 100:
            continue
        firsts.append({'log_pick': g0['log_pick'].iloc[0],
                       'epa': np.average(g0['adj_epa'], weights=g0['plays'])})
    f = pd.DataFrame(firsts)
    b, a = np.polyfit(f['log_pick'], f['epa'], 1)
    print(f'  rookie prior: adj_epa = {a:+.4f} + {b:+.4f}*log(pick) (n={len(f)})')
    return a, b


def fit_age_drift(df):
    rows = []
    for pid, g in df[df['season'] <= TRAIN_MAX_SEASON].groupby('passer_player_id'):
        seas = g[['season', 'adj_epa', 'plays', 'age']].groupby('season')[
            ['adj_epa', 'plays', 'age']].apply(
            lambda s: pd.Series({'epa': np.average(s['adj_epa'], weights=s['plays']),
                                 'plays': s['plays'].sum(), 'age': s['age'].mean()}))
        seas = seas[seas['plays'] >= 150]
        years = sorted(seas.index)
        for y0, y1 in zip(years, years[1:]):
            if y1 == y0 + 1:
                rows.append({'age': seas.loc[y0, 'age'],
                             'delta': seas.loc[y1, 'epa'] - seas.loc[y0, 'epa']})
    d = pd.DataFrame(rows)
    buckets = {'u25': d[d.age < 25], '25_32': d[(d.age >= 25) & (d.age < 32)],
               'o32': d[d.age >= 32]}
    drift = {k: float(v['delta'].mean()) if len(v) >= 10 else 0.0
             for k, v in buckets.items()}
    print(f'  age drift: {drift} (n={{k: len(v) for k, v in buckets.items()}})'
          .replace('{k: len(v) for k, v in buckets.items()}',
                   str({k: len(v) for k, v in buckets.items()})))
    return drift


def age_drift_per_game(age, drift):
    if np.isnan(age):
        return 0.0
    if age < 25:
        return drift['u25'] / 17
    if age < 32:
        return drift['25_32'] / 17
    return drift['o32'] / 17


# ------------------------------------------------------------------ baselines

def add_baselines(df, ewma_halflife):
    out = {}
    for pid, g in df.groupby('passer_player_id', sort=False):
        idx = g.index.values
        epa = g['epa'].values; pl = g['plays'].values
        n = len(g)
        expand = np.full(n, np.nan); ewma = np.full(n, np.nan); last12 = np.full(n, np.nan)
        csum = cw = 0.0
        alpha = 1 - 0.5 ** (1 / ewma_halflife)
        e = None
        for i in range(n):
            if cw > 0:
                expand[i] = csum / cw
            if e is not None:
                ewma[i] = e
            if i >= 1:
                lo = max(0, i - 12)
                last12[i] = np.average(epa[lo:i], weights=pl[lo:i])
            csum += epa[i] * pl[i]; cw += pl[i]
            e = epa[i] if e is None else (1 - alpha) * e + alpha * epa[i]
        out[pid] = (idx, expand, ewma, last12)
    for name, j in [('b1_expanding', 1), ('b3_ewma', 2), ('b4_last12', 3)]:
        col = np.full(len(df), np.nan)
        for pid, tup in out.items():
            col[tup[0]] = tup[j]
        df[name] = col
    seas = df.groupby(['passer_player_id', 'season'])[['epa', 'plays']].apply(
        lambda s: np.average(s['epa'], weights=s['plays'])).rename('season_epa').reset_index()
    seas['season'] += 1
    df = df.merge(seas, on=['passer_player_id', 'season'], how='left').rename(
        columns={'season_epa': 'b2_prev_season'})
    return df


def tune_ewma(df):
    train = df['season'] <= TRAIN_MAX_SEASON
    best = None
    for hl in [3, 5, 8, 12, 20]:
        d2 = add_baselines(df.copy(), hl)
        m = train & d2['b3_ewma'].notna()
        mse = np.average((d2.loc[m, 'epa'] - d2.loc[m, 'b3_ewma']) ** 2,
                         weights=d2.loc[m, 'plays'])
        if best is None or mse < best[0]:
            best = (mse, hl)
    print(f'  EWMA halflife tuned: {best[1]} games')
    return best[1]


# ------------------------------------------------------------------ L3: fusion

def build_fusion_pairs(df):
    """Every candidate (origin, 16-game target) pair with full provenance.

    Eligibility is decided by the CALLER from these logged columns, so the
    train-only rule is auditable rather than buried in a loop.
    """
    rows = []
    for pid, g in df.groupby('passer_player_id', sort=False):
        v = g.reset_index(drop=True)
        for i in range(0, len(v) - HORIZON, STRIDE):
            fut = v.iloc[i + 1:i + 1 + HORIZON]
            if len(fut) < HORIZON:
                continue
            rows.append({
                'passer_player_id': pid,
                'origin_season': int(v.loc[i, 'season']),
                'target_start_season': int(fut['season'].iloc[0]),
                'target_end_season': int(fut['season'].iloc[-1]),
                'target_span_days': int((fut['game_date'].iloc[-1]
                                         - fut['game_date'].iloc[0]).days),
                'k_epa': v.loc[i, 'k_adj_epa'], 'k_cpoe': v.loc[i, 'k_cpoe'],
                'k_success': v.loc[i, 'k_success'],
                'k_informed': v.loc[i, 'k_informed'],
                'y': np.average(fut['epa'], weights=fut['plays']),
            })
    return pd.DataFrame(rows)


def build_season_pairs(df, max_target_season):
    """T1-shaped (season Y -> Y+1) pairs whose TARGET season is <= the cap.

    Used only to calibrate forecast-origin intervals on the training era; the
    frozen era is never touched here.
    """
    rows = []
    for pid, g in df.groupby('passer_player_id'):
        seas = g.groupby('season')['plays'].sum()
        for y, p in seas.items():
            if p < 150 or (y + 1) not in seas or seas[y + 1] < 150:
                continue
            if y + 1 > max_target_season:
                continue
            gy, gy1 = g[g.season == y], g[g.season == y + 1]
            rows.append({'passer_player_id': pid, 'target_season': int(y + 1),
                         'target': np.average(gy1['epa'], weights=gy1['plays']),
                         'target_plays': float(gy1['plays'].sum()),
                         'state_var': float(gy['k_informed_var'].iloc[-1]),
                         'k_informed': float(gy['k_informed'].iloc[-1]),
                         'k_cpoe': float(gy['k_cpoe'].iloc[-1]),
                         'k_success': float(gy['k_success'].iloc[-1])})
    return pd.DataFrame(rows)


def calibrate_forecast_origin_intervals(df, fusion_pairs, skill_change_var,
                                        play_noise_var, first_cal_season=2012):
    """Calibrate FORECAST-ORIGIN predictive variance on the training era only.

    Two properties this must satisfy, both required by the Phase 1.1 review:

    1. It may not use realized future play volume. A forecast made at the
       origin does not know how many plays the QB will take next season, so the
       sampling-noise term uses the TRAIN-ERA expectation E[1/plays] over
       qualifying seasons rather than the realized value.

    2. For the FUSED rating it may not reuse k_informed_var as if it were the
       fused variance. leaf_v34 is a linear combination of three Kalman states
       whose errors are correlated, plus a fusion residual. Rather than assert
       an independence structure we do not model, we calibrate TOTAL forecast
       variance empirically against rolling out-of-fold training residuals:
       for each calibration season s, fusion weights are refit on pairs whose
       targets end at or before s-1 and used to predict season s. The resulting
       scale therefore absorbs fusion-coefficient uncertainty, CPOE/success
       state error, their covariance, and fusion residual error together.

    Returns a dict of per-mean calibration entries.
    """
    season_pairs = build_season_pairs(df, TRAIN_MAX_SEASON)
    # expectation of the sampling-noise multiplier at the origin
    expected_inv_plays = float((1.0 / season_pairs['target_plays']).mean())

    dense = fusion_pairs['target_span_days'] <= MAX_TARGET_SPAN_DAYS
    resid = {'leaf_v34': [], 'k_informed': []}
    base = {'leaf_v34': [], 'k_informed': []}

    cal_seasons = [s for s in sorted(season_pairs['target_season'].unique())
                   if s >= first_cal_season]
    for s in cal_seasons:
        hist = fusion_pairs[dense & (fusion_pairs['target_end_season'] <= s - 1)]
        if len(hist) < 50:
            continue
        X = np.column_stack([np.ones(len(hist)), hist['k_informed'],
                             hist['k_cpoe'], hist['k_success']])
        b = np.linalg.lstsq(X, hist['y'].values, rcond=None)[0]
        te = season_pairs[season_pairs['target_season'] == s]
        if not len(te):
            continue
        pred_fused = (b[0] + b[1] * te['k_informed'] + b[2] * te['k_cpoe']
                      + b[3] * te['k_success'])
        common = te['state_var'] + skill_change_var + play_noise_var * expected_inv_plays
        resid['leaf_v34'].extend((te['target'] - pred_fused).tolist())
        base['leaf_v34'].extend(common.tolist())
        resid['k_informed'].extend((te['target'] - te['k_informed']).tolist())
        base['k_informed'].extend(common.tolist())

    cal = {'mode': 'forecast_origin',
           'expected_inv_plays': expected_inv_plays,
           'implied_expected_plays': float(1.0 / expected_inv_plays),
           'calibration_seasons': [int(s) for s in cal_seasons],
           'note': ('variance = scale * (state_var + skill_change_var + '
                    'play_noise_var * E[1/plays]); scale is fit to rolling '
                    'out-of-fold TRAINING residuals and absorbs fusion '
                    'coefficient/covariance/residual error jointly rather than '
                    'assuming independence among the EPA/CPOE/success states')}
    for k in resid:
        r = np.asarray(resid[k]); bv = np.asarray(base[k])
        scale = float(np.mean(r ** 2 / bv)) if len(r) else 1.0
        cal[k] = {'scale': scale, 'n_oof_residuals': int(len(r)),
                  'oof_residual_sd': float(r.std()) if len(r) else float('nan')}
        print(f'  [{k}] OOF n={len(r)} residual sd={r.std():.4f} '
              f'variance scale={scale:.3f}')
    print(f'  E[1/plays]={expected_inv_plays:.6f} '
          f'(implied ~{1/expected_inv_plays:.0f} plays); NO realized volume used')
    return cal


def fit_fusion(pairs, label):
    X = np.column_stack([np.ones(len(pairs)), pairs['k_informed'],
                         pairs['k_cpoe'], pairs['k_success']])
    beta = np.linalg.lstsq(X, pairs['y'].values, rcond=None)[0]
    print(f'  [{label}] n={len(pairs)}  y = {beta[0]:+.4f} + {beta[1]:+.3f}*k_informed '
          f'+ {beta[2]:+.4f}*k_cpoe + {beta[3]:+.3f}*k_success')
    return beta


# ------------------------------------------------------------------ main

def main():
    print('=' * 72)
    print('LEAF v3.4 ENGINE — leak-corrected (audit findings A + B)')
    print('=' * 72)
    df = load_games()
    print(f'{len(df)} QB-games, {df.passer_player_id.nunique()} QBs, '
          f'{df.season.min()}-{df.season.max()}, season_type='
          f'{sorted(df.season_type.unique())} | train <= {TRAIN_MAX_SEASON}')

    print('\n[L1] Date-batched walk-forward defense ratings')
    hl_def, k_def, def_ratings = tune_defense(df)
    df['def_rating'] = def_ratings
    df['adj_epa'] = df['epa'] - df['def_rating']

    print('\n[baselines] EWMA halflife (train era)')
    hl_ewma = tune_ewma(df)
    df = add_baselines(df, hl_ewma)

    print('\n[L2] Kalman tuning (train era, one-step-ahead)')
    kp = {}
    for col, prior in [('adj_epa', -0.05), ('epa', -0.05), ('cpoe', -1.0), ('success', 0.42)]:
        kp[col] = tune_kalman(df, col, prior_mean=prior)

    print('\n[L2/L3] Running filters')
    for col, prior in [('adj_epa', -0.05), ('epa', -0.05), ('cpoe', -1.0), ('success', 0.42)]:
        q, r_play, p0 = kp[col]
        pri = np.empty(len(df)); post = np.empty(len(df)); var = np.empty(len(df))
        for pid, g in df.groupby('passer_player_id', sort=False):
            i = g.index.values
            a, b, c = kalman_pass(g[col].values, g['plays'].values, q, r_play, p0, prior)
            pri[i], post[i], var[i] = a, b, c
        df[f'k_{col}_pre'] = pri; df[f'k_{col}'] = post; df[f'k_{col}_var'] = var

    print('\n[L4] Rookie prior + age drift (train era)')
    a_pr, b_pr = fit_rookie_prior(df)
    drift = fit_age_drift(df)
    q, r_play, p0 = kp['adj_epa']
    pri = np.empty(len(df)); post = np.empty(len(df)); var = np.empty(len(df))
    for pid, g in df.groupby('passer_player_id', sort=False):
        i = g.index.values
        prior_mean = a_pr + b_pr * g['log_pick'].iloc[0]
        dr = np.array([age_drift_per_game(x, drift) for x in g['age'].values])
        a, b, c = kalman_pass(g['adj_epa'].values, g['plays'].values,
                              q, r_play, p0, prior_mean, drift=dr)
        pri[i], post[i], var[i] = a, b, c
    df['k_informed_pre'] = pri; df['k_informed'] = post; df['k_informed_var'] = var

    print('\n[L3] Fusion — TRAIN-ONLY targets (audit finding A)')
    pairs = build_fusion_pairs(df)
    train_ok = pairs['target_end_season'] <= TRAIN_MAX_SEASON
    dense = pairs['target_span_days'] <= MAX_TARGET_SPAN_DAYS
    n_v3_style = int((pairs['origin_season'] <= TRAIN_MAX_SEASON).sum())
    n_leaky = int(((pairs['origin_season'] <= TRAIN_MAX_SEASON) & ~train_ok).sum())
    print(f'  candidate pairs: {len(pairs)}')
    print(f'  v3-style eligible (origin only): {n_v3_style}, of which '
          f'{n_leaky} had targets in the frozen era -> EXCLUDED in v3.4')
    print(f'  v3.4 eligible (all target games <= {TRAIN_MAX_SEASON}): {int(train_ok.sum())}')
    long_span = int((train_ok & ~dense).sum())
    print(f'  of those, long-span "next-16-appearances" pairs '
          f'(> {MAX_TARGET_SPAN_DAYS}d): {long_span} -> fitted separately, labeled')

    primary = pairs[train_ok & dense]
    beta = fit_fusion(primary, 'PRIMARY dense-target')
    beta_all = fit_fusion(pairs[train_ok], 'labeled next-16-appearances (reported, not deployed)')

    pairs.to_csv(ROOT / 'data' / 'production' / 'leaf_v34_fusion_pairs.csv', index=False)

    df['leaf_v34'] = (beta[0] + beta[1] * df['k_informed']
                      + beta[2] * df['k_cpoe'] + beta[3] * df['k_success'])
    df['leaf_v34_pre'] = (beta[0] + beta[1] * df['k_informed_pre']
                          + beta[2] * df['k_cpoe_pre'] + beta[3] * df['k_success_pre'])
    # Clean UNRESTRICTED next-16-appearances fit, carried so the evaluator can
    # report both target definitions side by side. Both are leak-free; the
    # choice between them is NOT made using frozen-era outcomes.
    df['leaf_v34_unrestricted'] = (beta_all[0] + beta_all[1] * df['k_informed']
                                   + beta_all[2] * df['k_cpoe']
                                   + beta_all[3] * df['k_success'])

    # ---- predictive-interval components, fit on TRAIN ERA ONLY (finding E) ----
    # (calibration of the FORECAST-ORIGIN interval happens after this block;
    #  see calibrate_forecast_origin_intervals)
    tr = df[df['season'] <= TRAIN_MAX_SEASON]
    seas = tr.groupby(['passer_player_id', 'season']).apply(
        lambda g: pd.Series({'epa': np.average(g['epa'], weights=g['plays']),
                             'plays': g['plays'].sum()}),
        include_groups=False).reset_index()
    seas = seas[seas['plays'] >= 150].sort_values(['passer_player_id', 'season'])
    d = seas.groupby('passer_player_id')['epa'].diff()
    yr = seas.groupby('passer_player_id')['season'].diff()
    consec = d[(yr == 1)]
    # play-level noise: var of game epa around the QB-season mean, x plays
    gm = tr.merge(seas[['passer_player_id', 'season']], on=['passer_player_id', 'season'])
    resid = gm['epa'] - gm.groupby(['passer_player_id', 'season'])['epa'].transform('mean')
    play_noise_var = float(np.average(resid ** 2, weights=gm['plays']) * gm['plays'].mean())
    skill_change_var = float(max(consec.var() - 2 * play_noise_var / 500.0, 1e-6))
    print(f'\n[intervals] train-only variance components: '
          f'skill_change_var={skill_change_var:.5f}, play_noise_var={play_noise_var:.3f}')
    print('[intervals] forecast-origin calibration (rolling OOF, train era only)')
    interval_cal = calibrate_forecast_origin_intervals(
        df, pairs, skill_change_var, play_noise_var)

    out_cols = ['game_id', 'game_date', 'season', 'season_type', 'week',
                'passer_player_id', 'passer_player_name', 'posteam', 'defteam',
                'plays', 'epa', 'cpoe', 'success', 'age', 'log_pick',
                'def_rating', 'adj_epa', 'b1_expanding', 'b2_prev_season',
                'b3_ewma', 'b4_last12', 'k_epa', 'k_adj_epa', 'k_adj_epa_var',
                'k_cpoe', 'k_success',
                'k_informed', 'k_informed_pre', 'k_informed_var',
                'leaf_v34', 'leaf_v34_pre', 'leaf_v34_unrestricted']
    out = ROOT / 'data' / 'production' / 'leaf_v34_ratings.csv'
    df[out_cols].to_csv(out, index=False, float_format='%.6f')

    params = {
        'model_version': 'v3.4',
        'defense': {'halflife_days': hl_def, 'k_shrink': k_def, 'update': 'date-batched, one update per game-defense'},
        'ewma_halflife': hl_ewma,
        'kalman': {k: {'q': v[0], 'r_play': v[1], 'p0': v[2]} for k, v in kp.items()},
        'rookie_prior': {'intercept': a_pr, 'log_pick_slope': b_pr},
        'age_drift': drift,
        'fusion_beta': list(map(float, beta)),
        'fusion_beta_long_span_reported_only': list(map(float, beta_all)),
        'fusion': {'target': 'next-16-appearances, dense variant',
                   'max_target_span_days': MAX_TARGET_SPAN_DAYS,
                   'n_primary_pairs': int(len(primary)),
                   'n_excluded_for_target_leakage': n_leaky},
        'predictive_interval': {'skill_change_var': skill_change_var,
                                'play_noise_var': play_noise_var},
        'interval_calibration': interval_cal,
        'train_max_season': TRAIN_MAX_SEASON,
        'population': 'regular season only, QB passers only',
    }
    with open(ROOT / 'data' / 'production' / 'leaf_v34_params.json', 'w') as f:
        json.dump(params, f, indent=2)
    print(f'\n[OK] ratings -> {out.name}')


if __name__ == '__main__':
    main()
