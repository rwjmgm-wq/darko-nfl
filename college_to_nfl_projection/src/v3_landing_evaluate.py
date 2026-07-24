"""
H3 landing-spot frozen evaluation (SPEC_V3.md, H3). ONE PASS.

M3 = draft block + landing (3 covariates); M4 = college + draft + landing.
Same hazard machinery as v3_evaluate (imported, not reimplemented).
Primary: DeLong M3 vs M0 on LODCO OOF 2007-2021. Holdout 2022-2024
directional. Output: results/v3_landing_report.md
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from v3_evaluate import (COLLEGE_BLOCK, FULL_WINDOW_MAX_CLASS, HOLDOUT_CLASSES,
                         auc_score, bootstrap_auc_ci, delong_test, fit_hazard,
                         oof_scores, prob_within)

ROOT = Path(__file__).parent.parent
LANDING = ['l1_team_pass_epa_prev', 'l2_hc_new', 'l3_qb_churn3']


def load_with_landing():
    feats = pd.read_csv(ROOT / 'data' / 'processed' / 'v3_features.csv')
    landing = pd.read_csv(ROOT / 'data' / 'processed' / 'v3_landing_features.csv')
    feats = feats.merge(landing[['player_name', 'draft_year'] + LANDING],
                        on=['player_name', 'draft_year'], how='left')
    pp = pd.read_csv(ROOT / 'data' / 'processed' / 'v3_person_periods.csv')
    pp = pp.merge(feats, on=['player_name', 'draft_year'], how='inner')
    return feats, pp


def main():
    print('=' * 70)
    print('H3 LANDING-SPOT EVALUATION (per SPEC_V3.md H3) - one frozen pass')
    print('=' * 70)
    feats, pp = load_with_landing()
    drafted = feats[feats['pick'].notna() & (feats['draft_year'] >= 2007) &
                    (feats['draft_year'] <= pp['draft_year'].max())]
    cov = drafted[LANDING].notna().all(axis=1).mean()
    print(f'landing coverage among drafted QBs: {cov:.3f}')

    spec_blocks = {
        'M0_draft': ([], True),
        'M3_landing': (LANDING, True),
        'M2_both': (COLLEGE_BLOCK, True),
        'M4_all': (COLLEGE_BLOCK + LANDING, True),
    }

    print(f'\n[1] LODCO OOF, classes 2007-{FULL_WINDOW_MAX_CLASS}...')
    scores = oof_scores(feats, pp, spec_blocks)
    print(f'    scored: {len(scores)} QBs, {scores.outcome.sum()} starters')

    lines = ['# H3 Landing-Spot Report\n',
             'Pre-registered in SPEC_V3.md (H3). One frozen pass.\n',
             f'Sample: {len(scores)} drafted QBs, classes 2007-{FULL_WINDOW_MAX_CLASS}, '
             f'{scores.outcome.sum()} starters. Landing coverage {cov:.1%}.\n',
             '## LODCO OOF AUC\n', '| Model | AUC | 95% CI |', '|---|---|---|']
    for name in spec_blocks:
        auc = auc_score(scores['outcome'], scores[name])
        lo, hi = bootstrap_auc_ci(scores, name)
        print(f'    {name:12s} AUC={auc:.3f}  [{lo:.3f}, {hi:.3f}]')
        lines.append(f'| {name} | {auc:.3f} | [{lo:.3f}, {hi:.3f}] |')
    lines.append('')

    print('\n[2] DeLong tests')
    verdicts = {}
    for a, b, tag in [('M3_landing', 'M0_draft', 'PRIMARY'),
                      ('M4_all', 'M2_both', 'secondary')]:
        auc_a, auc_b, pval = delong_test(scores['outcome'], scores[a], scores[b])
        verdicts[tag] = (auc_a - auc_b, pval)
        print(f'    [{tag}] {a} vs {b}: {auc_a:.3f} vs {auc_b:.3f}, p = {pval:.3f}')
        lines.append(f'- **{tag}** DeLong {a} vs {b}: AUC {auc_a:.3f} vs {auc_b:.3f}, '
                     f'p = {pval:.3f}')
    lines.append('')

    print('\n[3] Quasi-holdout (directional)')
    lines.append('## Quasi-holdout (classes 2022-2024, directional only)\n')
    train_pp = pp[pp['draft_year'] <= FULL_WINDOW_MAX_CLASS]
    models = {name: fit_hazard(train_pp, cc, ud) for name, (cc, ud) in spec_blocks.items()}
    hold = []
    for _, qb in feats[feats['draft_year'].isin(HOLDOUT_CLASSES) & feats['pick'].notna()].iterrows():
        qb_pp = pp[(pp['player_name'] == qb['player_name']) & (pp['draft_year'] == qb['draft_year'])]
        if len(qb_pp) == 0:
            continue
        rec = {'player_name': qb['player_name'], 'draft_year': qb['draft_year'],
               'outcome': int(qb_pp['event'].max())}
        for name, (fz, mm) in models.items():
            rec[name] = prob_within(fz, mm, qb, horizon=int(qb_pp['t'].max()))
        hold.append(rec)
    hold = pd.DataFrame(hold)
    hold_dir = None
    if len(hold) and hold['outcome'].nunique() == 2:
        lines.append(f'Sample: {len(hold)} QBs, {hold.outcome.sum()} starters so far.\n')
        aucs = {}
        for name in spec_blocks:
            aucs[name] = auc_score(hold['outcome'], hold[name])
            print(f'    {name:12s} AUC={aucs[name]:.3f}  (n={len(hold)})')
            lines.append(f'- {name}: AUC {aucs[name]:.3f}')
        hold_dir = aucs['M3_landing'] - aucs['M0_draft']
        lines.append('')

    # full-fit landing coefficients (no selection, disclosure only)
    fz, mdl = models['M3_landing']
    n_land = len(LANDING)
    coefs = mdl.coef_[0][:n_land]
    print('\n[4] M3 landing coefficients (full 2007-2021 fit, standardized):')
    lines.append('## Landing coefficients (M3, full dev fit, standardized)\n')
    for c, b in zip(LANDING, coefs):
        print(f'    {c:24s} {b:+.3f}')
        lines.append(f'- {c}: {b:+.3f}')
    lines.append('')

    d_auc, pval = verdicts['PRIMARY']
    if pval < 0.05 and d_auc > 0 and hold_dir is not None and hold_dir > 0:
        verdict = 'CONFIRMATORY'
    elif d_auc >= 0.01 and hold_dir is not None and hold_dir > 0:
        verdict = 'SUGGESTIVE'
    else:
        verdict = 'NULL'
    print(f'\nVERDICT (pre-registered criteria): {verdict} '
          f'(OOF dAUC={d_auc:+.3f}, p={pval:.3f}, holdout dAUC='
          f'{hold_dir if hold_dir is None else round(hold_dir, 3)})')
    lines.append(f'**Verdict (pre-registered criteria): {verdict}** — OOF dAUC '
                 f'{d_auc:+.3f} (DeLong p = {pval:.3f}), holdout directional dAUC '
                 f'{hold_dir if hold_dir is None else round(hold_dir, 3)}.\n')

    out = ROOT / 'results' / 'v3_landing_report.md'
    out.write_text('\n'.join(lines), encoding='utf-8')
    print(f'\n[OK] -> {out}')


if __name__ == '__main__':
    main()
