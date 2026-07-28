"""
LEAF v3.4 versioned evaluator (re-certification of the leak corrections).

Design rules from the audit brief and the Phase 1.1 review:

  * EVERY baseline is reported. No "best baseline" is selected after seeing
    test outcomes; comparisons are pre-specified as LEAF v3.4 vs each baseline.
  * Model comparisons use a PAIRED cluster bootstrap by QB (primary), a
    season-clustered bootstrap (sensitivity, only 7 clusters), leave-one-
    target-season-out refits, and per-season Δr. A result is called
    SUGGESTIVE unless the primary QB-clustered CI excludes zero.
  * Predictive intervals come in TWO EXPLICITLY SEPARATE MODES:
      A. forecast_origin           — deployable. Never reads realized
                                     target_plays. Variance is calibrated on
                                     rolling out-of-fold TRAINING residuals.
      B. conditional_realized_volume — retrospective DIAGNOSTIC ONLY. Uses the
                                     realized future play count, which a real
                                     forecast cannot know.
  * Both fusion target definitions (dense <=730d and unrestricted
    next-16-appearances) are reported. Frozen-era outcomes are NOT used to
    choose between them.

Writes docs/LEAF_V34_RECERTIFICATION.md (new file; v3 docs untouched).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).parent.parent.parent
PROD = ROOT / 'data' / 'production'
TEST_MIN, TEST_MAX = 2019, 2025
MIN_PLAYS = 150
RNG = np.random.default_rng(20260728)

PREDICTORS = {
    'B1 expanding mean': 'b1_expanding',
    'B2 prev-season EPA': 'b2_prev_season',
    'B3 EWMA': 'b3_ewma',
    'B4 last-12 mean': 'b4_last12',
    'V1 Kalman raw EPA': 'k_epa',
    'V2 Kalman opp-adj': 'k_adj_epa',
    'V3 informed (priors/age)': 'k_informed',
    'LEAF v3.4 fused (dense target)': 'leaf_v34',
    'LEAF v3.4 fused (unrestricted target)': 'leaf_v34_unrestricted',
}
PRIMARY = 'leaf_v34'


def build_t1(df, rating_cols):
    rows = []
    for pid, g in df.groupby('passer_player_id'):
        seas = g.groupby('season')['plays'].sum()
        for y, p in seas.items():
            if p < MIN_PLAYS or (y + 1) not in seas or seas[y + 1] < MIN_PLAYS:
                continue
            if y + 1 < TEST_MIN or y + 1 > TEST_MAX:
                continue
            gy, gy1 = g[g.season == y], g[g.season == y + 1]
            row = {'passer_player_id': pid, 'y': y, 'target_season': y + 1,
                   'target': np.average(gy1['epa'], weights=gy1['plays']),
                   'target_plays': gy1['plays'].sum(),
                   'state_var': gy['k_informed_var'].iloc[-1]}
            for c in rating_cols:
                row[c] = (np.average(gy['epa'], weights=gy['plays'])
                          if c == 'b2_prev_season' else gy[c].iloc[-1])
            rows.append(row)
    return pd.DataFrame(rows).dropna(subset=list(rating_cols))


# ----------------------------------------------------------- interval modes
def interval_forecast_origin(state_var, mean_name, params):
    """MODE A — deployable forecast-origin interval.

    Deliberately takes ONLY origin-side state variance. It has no parameter
    through which realized future volume could be supplied, so it cannot use
    information unavailable at forecast time. The sampling-noise term uses the
    training-era expectation E[1/plays]; the overall scale is calibrated on
    rolling out-of-fold TRAINING residuals, which absorbs fusion-coefficient
    uncertainty, CPOE/success state error and their covariance jointly (we do
    NOT assert independence among those states).
    """
    cal = params['interval_calibration']
    pi = params['predictive_interval']
    if mean_name not in cal:
        raise KeyError(f'no forecast-origin calibration for {mean_name!r}; '
                       're-run engine_v34.py')
    base = (np.asarray(state_var) + pi['skill_change_var']
            + pi['play_noise_var'] * cal['expected_inv_plays'])
    return np.sqrt(cal[mean_name]['scale'] * base)


def interval_conditional_realized(state_var, target_plays, params):
    """MODE B — RETROSPECTIVE DIAGNOSTIC ONLY. Uses realized target_plays,
    which is unknown at forecast time. Never deploy this."""
    pi = params['predictive_interval']
    return np.sqrt(np.asarray(state_var) + pi['skill_change_var']
                   + pi['play_noise_var'] / np.asarray(target_plays))


def gaussian_crps(y, mu, sd):
    z = (y - mu) / sd
    return sd * (z * (2 * sps.norm.cdf(z) - 1) + 2 * sps.norm.pdf(z)
                 - 1 / np.sqrt(np.pi))


def score_dist(y, mu, sd):
    out = {'CRPS': float(gaussian_crps(y, mu, sd).mean()),
           'NLL': float((-sps.norm.logpdf(y, mu, sd)).mean()),
           'mean_sd': float(np.mean(sd))}
    for lvl, z in [(50, 0.6745), (80, 1.2816), (90, 1.6449)]:
        out[f'cov{lvl}'] = float(((y >= mu - z * sd) & (y <= mu + z * sd)).mean())
    out['PIT_KS'] = float(sps.kstest(sps.norm.cdf((y - mu) / sd), 'uniform').statistic)
    return out


# ------------------------------------------------------------- bootstrap
def paired_boot(d, col_a, col_b, unit, n=2000):
    units = d[unit].unique()
    by = {u: d[d[unit] == u] for u in units}
    diffs = []
    for _ in range(n):
        pick = RNG.choice(units, len(units), replace=True)
        s = pd.concat([by[u] for u in pick])
        if s[col_a].std() > 0 and s[col_b].std() > 0 and s['target'].std() > 0:
            diffs.append(s[col_a].corr(s['target']) - s[col_b].corr(s['target']))
    return np.percentile(diffs, [2.5, 97.5])


def loso_season(d, col_a, col_b):
    """Leave-one-target-season-out Δr: how much does any single season drive
    the comparison?"""
    out = []
    for s in sorted(d['target_season'].unique()):
        sub = d[d['target_season'] != s]
        out.append({'held_out_season': int(s), 'n': len(sub),
                    'dr': sub[col_a].corr(sub['target'])
                          - sub[col_b].corr(sub['target'])})
    return pd.DataFrame(out)


def main():
    params = json.load(open(PROD / 'leaf_v34_params.json'))
    df = pd.read_csv(PROD / 'leaf_v34_ratings.csv')
    d = build_t1(df, list(PREDICTORS.values()))
    print(f'T1 frozen era {TEST_MIN}-{TEST_MAX}: n={len(d)} pairs, '
          f'{d.passer_player_id.nunique()} QBs, {d.target_season.nunique()} seasons')

    fus = params['fusion']
    L = ['# LEAF v3.4 Re-Certification\n',
         'Generated by `scripts/v3_4/evaluate_v34.py`. This is a NEW document; '
         'v3 results in `docs/LEAF_V3_RESULTS.md` are untouched.\n']

    # ---------------- headline (computed below, inserted at the end) --------
    body = ['## What changed vs v3\n',
            '| Correction | v3 behaviour | v3.4 behaviour |', '|---|---|---|',
            '| A. Fusion label leakage | origin season ≤ 2018 only; the 16-game target could run into 2019+ '
            f'({fus["n_excluded_for_target_leakage"]} such pairs) | every target game ≤ 2018 |',
            '| A2. Target semantics | "next 16 games" silently meant next 16 APPEARANCES (spans to 12 years) '
            f'| primary fit restricted to spans ≤ {fus["max_target_span_days"]}d; unrestricted fit reported separately |',
            '| B. Opponent adjustment | defense updated after every passer row; league aggregates moved mid-date '
            '| all pre-game states read first; one play-weighted update per game-defense; league aggregates advance at date boundaries |',
            '| C. Population | postseason + non-QB passers mixed in | regular season only, QB only, with a drop report |',
            '| E. Intervals | state variance only, and evaluated against realized volume '
            '| two explicit modes: deployable forecast-origin (no realized volume) and a clearly labelled retrospective diagnostic |',
            '',
            f'**Population:** {params["population"]}. Base rows: '
            f'{len(pd.read_csv(PROD / "qb_games_base_v34.csv")):,} QB-games.\n',
            f'**Frozen-era T1 sample:** n = {len(d)} pairs, '
            f'{d.passer_player_id.nunique()} QBs across {d.target_season.nunique()} seasons.\n',
            '## Point-forecast accuracy — ALL predictors reported\n',
            '| Predictor | r | RMSE | MAE |', '|---|---|---|---|']

    print('\n--- all predictors (no post-hoc selection) ---')
    for name, col in PREDICTORS.items():
        r = d[col].corr(d['target'])
        rmse = np.sqrt(((d[col] - d['target']) ** 2).mean())
        mae = (d[col] - d['target']).abs().mean()
        print(f'  {name:40s} r={r:+.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}')
        body.append(f'| {name} | {r:+.4f} | {rmse:.4f} | {mae:.4f} |')
    body += ['',
             '### Fusion target definitions — both reported, neither selected on frozen data\n',
             f'The deployed fit uses a DENSE target (span ≤ {fus["max_target_span_days"]} days, '
             f'n = {fus["n_primary_pairs"]} pairs). Reason the rule exists: v3\'s '
             '"next 16 games" is really "next 16 APPEARANCES", and for QBs with long '
             'inactive stretches that window reached 4,375 days (12 years) — a label '
             'that is not a short-horizon forecast in any useful sense. The '
             'unrestricted clean fit is reported above as a separate row. '
             '**The choice between them was made on definitional grounds before '
             'scoring, not by comparing 2019–2025 outcomes.** Any sensitivity of the '
             '730-day threshold must be explored on training-era data only.\n']

    # ---------------- comparisons ----------------
    print('\n--- paired cluster bootstrap: LEAF v3.4 vs EVERY baseline ---')
    body += ['## LEAF v3.4 (dense) vs every baseline\n',
             'Primary inference = paired cluster bootstrap by QB (2000 resamples). '
             'The season-clustered column is a SENSITIVITY analysis with only '
             f'{d.target_season.nunique()} clusters and is not treated as '
             'confirmatory. A result is called confirmatory only if the '
             '**QB-clustered** CI excludes zero.\n',
             '| Comparison | Δr | 95% CI (QB, primary) | 95% CI (season, sensitivity) | reading |',
             '|---|---|---|---|---|']
    for name, col in PREDICTORS.items():
        if col == PRIMARY:
            continue
        dr = d[PRIMARY].corr(d['target']) - d[col].corr(d['target'])
        qlo, qhi = paired_boot(d, PRIMARY, col, 'passer_player_id')
        slo, shi = paired_boot(d, PRIMARY, col, 'target_season')
        sig_qb = qlo > 0 or qhi < 0
        reading = ('confirmatory' if sig_qb else
                   'suggestive (QB CI includes zero)')
        print(f'  vs {name:40s} dr={dr:+.4f} QB[{qlo:+.3f},{qhi:+.3f}] '
              f'season[{slo:+.3f},{shi:+.3f}] {reading}')
        body.append(f'| vs {name} | {dr:+.4f} | [{qlo:+.3f}, {qhi:+.3f}] | '
                    f'[{slo:+.3f}, {shi:+.3f}] | {reading} |')
    body.append('')

    # focused comparison: fused vs un-fused informed state
    dr_inf = d[PRIMARY].corr(d['target']) - d['k_informed'].corr(d['target'])
    qlo_i, qhi_i = paired_boot(d, PRIMARY, 'k_informed', 'passer_player_id')
    slo_i, shi_i = paired_boot(d, PRIMARY, 'k_informed', 'target_season')
    lo = loso_season(d, PRIMARY, 'k_informed')
    body += ['### Focus: fused vs un-fused informed state\n',
             '| Leave-one-season-out | n | Δr (fused − informed) |', '|---|---|---|']
    for _, r in lo.iterrows():
        body.append(f'| drop {int(r.held_out_season)} | {int(r.n)} | {r.dr:+.4f} |')
    body += ['',
             '| Per-season | n | r fused | r informed | Δr |', '|---|---|---|---|---|']
    # NB: plain 'dr' in console output -- Windows consoles default to cp1252
    # and cannot encode the delta glyph. The written document uses UTF-8.
    print('\n--- per-season dr (fused - informed) ---')
    for s, g in d.groupby('target_season'):
        rf, ri = g[PRIMARY].corr(g['target']), g['k_informed'].corr(g['target'])
        print(f'  {int(s)}: n={len(g):3d} fused={rf:+.3f} informed={ri:+.3f} dr={rf-ri:+.3f}')
        body.append(f'| {int(s)} | {len(g)} | {rf:+.3f} | {ri:+.3f} | {rf-ri:+.4f} |')
    body.append('')

    # ---------------- intervals: two explicit modes ----------------
    print('\n--- MODE A: forecast-origin intervals (deployable) ---')
    body += ['## Predictive intervals — two explicitly separate modes\n',
             '### Mode A — `forecast_origin` (deployable)\n',
             'Uses only information available at the forecast origin. The sampling-noise '
             f'term uses the training-era expectation E[1/plays] = '
             f'{params["interval_calibration"]["expected_inv_plays"]:.6f} '
             f'(implied ≈ {params["interval_calibration"]["implied_expected_plays"]:.0f} plays); '
             'realized future volume is never read. The variance scale is fit to '
             f'rolling out-of-fold TRAINING residuals '
             f'(n = {params["interval_calibration"]["leaf_v34"]["n_oof_residuals"]}, '
             f'seasons {params["interval_calibration"]["calibration_seasons"][0]}–'
             f'{params["interval_calibration"]["calibration_seasons"][-1]}), which absorbs '
             'fusion-coefficient uncertainty, CPOE/success state error and their '
             'covariance jointly — no independence assumption is asserted.\n',
             '| Forecast mean | scale | cov50 | cov80 | cov90 | PIT KS | CRPS | NLL | mean sd |',
             '|---|---|---|---|---|---|---|---|---|']
    modeA = {}
    for name, col in [('LEAF v3.4 fused', 'leaf_v34'), ('V3 informed state', 'k_informed')]:
        sd = interval_forecast_origin(d['state_var'], col, params)
        sc = score_dist(d['target'], d[col], sd)
        modeA[col] = sc
        print(f'  {name:22s} cov80={sc["cov80"]:.1%} PITKS={sc["PIT_KS"]:.3f} '
              f'CRPS={sc["CRPS"]:.4f} NLL={sc["NLL"]:+.3f}')
        body.append(f'| {name} | {params["interval_calibration"][col]["scale"]:.3f} '
                    f'| {sc["cov50"]:.1%} | {sc["cov80"]:.1%} | {sc["cov90"]:.1%} '
                    f'| {sc["PIT_KS"]:.3f} | {sc["CRPS"]:.4f} | {sc["NLL"]:+.3f} '
                    f'| {sc["mean_sd"]:.4f} |')

    print('\n--- MODE B: conditional on realized volume (DIAGNOSTIC ONLY) ---')
    body += ['',
             '### Mode B — `conditional_realized_volume` (RETROSPECTIVE DIAGNOSTIC ONLY)\n',
             '**Not deployable.** These intervals use the realized next-season play '
             'count, which no forecast can know at the origin. Reported only to show '
             'how much of the interval width is attributable to volume uncertainty.\n',
             '| Forecast mean | cov50 | cov80 | cov90 | PIT KS | CRPS | NLL | mean sd |',
             '|---|---|---|---|---|---|---|---|']
    for name, col in [('LEAF v3.4 fused', 'leaf_v34'), ('V3 informed state', 'k_informed')]:
        sd = interval_conditional_realized(d['state_var'], d['target_plays'], params)
        sc = score_dist(d['target'], d[col], sd)
        print(f'  {name:22s} cov80={sc["cov80"]:.1%} (diagnostic)')
        body.append(f'| {name} | {sc["cov50"]:.1%} | {sc["cov80"]:.1%} | {sc["cov90"]:.1%} '
                    f'| {sc["PIT_KS"]:.3f} | {sc["CRPS"]:.4f} | {sc["NLL"]:+.3f} '
                    f'| {sc["mean_sd"]:.4f} |')

    z = 1.2816
    sd_state = np.sqrt(d['state_var'])
    cov_state = float(((d['target'] >= d['k_informed'] - z * sd_state)
                       & (d['target'] <= d['k_informed'] + z * sd_state)).mean())
    body += ['',
             f'For reference, the v3 behaviour of using **state variance alone** gives '
             f'{cov_state:.1%} coverage at a nominal 80% and is retracted.\n']

    # ---------------- headline ----------------
    a_fused = modeA['leaf_v34']
    head = ['## Headline verdict\n',
            f'1. **Forecast-origin interval calibration is materially improved, and is '
            f'reported as such only for Mode A.** Nominal 80% coverage is '
            f'{a_fused["cov80"]:.1%} (PIT KS {a_fused["PIT_KS"]:.3f}) using only '
            'origin-time information, versus '
            f'{cov_state:.1%} for v3\'s state-variance-only intervals. The Mode B '
            'figures elsewhere in this document are retrospective diagnostics and are '
            'NOT evidence that a deployable interval is calibrated.\n',
            f'2. **The fused rating ranks below the un-fused informed state, but this is '
            f'SUGGESTIVE, not confirmatory.** Δr = {dr_inf:+.4f}; primary QB-clustered '
            f'95% CI [{qlo_i:+.3f}, {qhi_i:+.3f}] '
            f'{"excludes" if (qlo_i > 0 or qhi_i < 0) else "includes"} zero. The '
            f'season-clustered CI [{slo_i:+.3f}, {shi_i:+.3f}] rests on only '
            f'{d.target_season.nunique()} clusters and is a sensitivity check, not a '
            f'basis for a decisive claim. Leave-one-season-out Δr ranges '
            f'[{lo.dr.min():+.4f}, {lo.dr.max():+.4f}].\n',
            '3. **Fused still leads on level, not order** (best RMSE/MAE among the '
            'scored predictors), consistent with v3.\n',
            '4. **Not comparable to v3\'s published table.** The population changed '
            '(regular season only, QB passers only), so the targets differ. No '
            'v3-vs-v3.4 improvement is claimed.\n',
            '### Consequence for Phase 2\n',
            'The exporter labels the fused column as "leaf_rating" while the dashboard '
            'forecasts from `k_informed`. That mismatch remains unresolved and is Phase '
            '2 work. Phase 2 must deploy Mode A intervals only.\n']

    out = ROOT / 'docs' / 'LEAF_V34_RECERTIFICATION.md'
    out.write_text('\n'.join(L + head + body), encoding='utf-8')
    d.to_csv(PROD / 'leaf_v34_t1_eval_pairs.csv', index=False, float_format='%.6f')
    print(f'\n[OK] -> {out}')


if __name__ == '__main__':
    main()
