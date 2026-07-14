"""
V3 Projection Database Builder.

Deploys the SPEC_V3.md hazard models (trained on all available person-periods,
2007-2025 seasons) to produce P(primary starter within 5 seasons of draft) for
every QB in the dataset, including pre-draft prospects.

Replaces the v2 binary-logistic probabilities in the explorer JSON. Schema keys
are kept (starter_prob_college / starter_prob_combined) so the explorer needs
only text updates:
- starter definition changed: "a season with 10+ starts within 5 years of
  draft" (base rate ~47%) instead of "85+ career starts or 30+ streak" (~32%).
- CV metrics updated to v3 (see results/v3_evaluation_report.md).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from v3_evaluate import COLLEGE_BLOCK, fit_hazard, load, prob_within
from v2_build_projections import classify_outcomes, norm_name

CV_METRICS = {
    'starter_auc_college': 0.733,
    'starter_auc_college_ci': [0.666, 0.803],
    'starter_auc_combined': 0.894,
    'starter_auc_combined_ci': [0.829, 0.951],
    'starter_auc_pick_only': 0.887,
    'delong_combined_vs_pick_p': 0.718,
    'starter_base_rate': 0.48,
}
OUTCOME_MAX_CLASS = 2019  # for censoring-aware ACTUAL outcome labels (unchanged from v2)


def main():
    print('=' * 70)
    print('V3 PROJECTION DATABASE BUILDER')
    print('=' * 70)

    feats, pp = load()
    outcomes = pd.read_csv(ROOT / 'data' / 'processed' / 'nfl_outcomes_comprehensive.csv')
    feats = feats.merge(outcomes, on=['player_name', 'draft_year'], how='left')

    print(f'\n[1] Training final hazard models on {len(pp)} person-periods...')
    fz_c, m_c = fit_hazard(pp, COLLEGE_BLOCK, use_draft=False)
    fz_b, m_b = fit_hazard(pp, COLLEGE_BLOCK, use_draft=True)

    # Recalibration from OOF reliability (deployment-only; logged in SPEC_V3
    # Deviations). Slopes < 1 mean raw probabilities are overconfident.
    from v3_evaluate import calibration
    oof = pd.read_csv(ROOT / 'results' / 'v3_oof_scores.csv')
    cal = {}
    for key, col in [('college', 'M1_college'), ('combined', 'M2_both')]:
        ic, sl = calibration(oof['outcome'].values, oof[col].values)
        cal[key] = (ic, sl)
        print(f'    recalibration {key}: logit_cal = {ic:+.2f} + {sl:.2f} * logit_raw')

    def recal(p, key):
        ic, sl = cal[key]
        lo = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        out = 1 / (1 + np.exp(-(ic + sl * lo)))
        return float(np.clip(out, 0.01, 0.99))  # never display certainty

    print('[2] Generating projections...')
    feats['actual_outcome_label'] = classify_outcomes(feats)
    starts_by_qb = pp.groupby(['player_name', 'draft_year'])['event'].max()

    projections = []
    n_ineligible = 0
    for _, row in feats.iterrows():
        # Undrafted "prospects" include every college QB with attempts. A
        # class-year age estimate of <= 20.7 (year <= 2) means the player is
        # not draft-eligible for the listed draft year - the model's age
        # coefficient would reward an age at which he cannot actually declare.
        # Skip rather than publish an extrapolation.
        if (pd.isna(row['pick']) and row.get('age_source') == 'class_year'
                and row['age_at_draft'] <= 20.8):
            n_ineligible += 1
            continue
        p_college = recal(prob_within(fz_c, m_c, row), 'college')
        p_combined = recal(prob_within(fz_b, m_b, row), 'combined') if pd.notna(row['pick']) else None

        actual = row['actual_outcome_label']
        draft_year = int(row['draft_year'])
        if draft_year > OUTCOME_MAX_CLASS and actual not in ('Elite', 'Solid Starter'):
            actual = 'TBD' if actual is not None else None

        became_starter = starts_by_qb.get((row['player_name'], draft_year), None)

        projections.append({
            'player_name': row['player_name'],
            'draft_year': draft_year,
            'college': row.get('college') if pd.notna(row.get('college')) else 'Unknown',
            'draft_pick': int(row['pick']) if pd.notna(row['pick']) else None,
            'college_epa_adj': round(float(row['epa_adj']), 3) if pd.notna(row['epa_adj']) else None,
            'college_epa_shrunk': round(float(row['epa_adj_shrunk']), 3) if pd.notna(row['epa_adj_shrunk']) else None,
            'comp_pct': round(float(row['comp_pct']), 3) if pd.notna(row['comp_pct']) else None,
            'rush_share': round(float(row['rush_share']), 3) if pd.notna(row['rush_share']) else None,
            'age_at_draft': round(float(row['age_at_draft']), 1) if not row['age_missing'] else None,
            'sos_percentile': round(float(row['sos_percentile']), 1) if pd.notna(row['sos_percentile']) else None,
            'college_attempts': int(row['attempts']),
            'is_fcs': bool(row.get('is_fcs_team', 0) > 0),
            'game_level_adj': bool(row['game_level_adj']),
            'starter_prob_college': round(p_college, 3),
            'starter_prob_combined': round(p_combined, 3) if p_combined is not None else None,
            'became_starter_5yr': int(became_starter) if became_starter is not None else None,
            'actual_outcome': actual,
            'actual_rookie_leaf': round(float(row['rookie_leaf']), 3) if pd.notna(row.get('rookie_leaf')) else None,
            'actual_total_starts': int(row['total_starts']) if pd.notna(row.get('total_starts')) else None,
        })

    out = {
        'meta': {
            'model': 'v3 discrete-time hazard (SPEC_V3.md), trained on classes 2007-2025',
            'features': COLLEGE_BLOCK,
            'cv': CV_METRICS,
            'notes': [
                'Starter = a season with 10+ games of 10+ pass attempts, within 5 years of draft. '
                'Base rate ~48% of drafted QBs.',
                'Out-of-sample (leave-one-draft-class-out): college-only AUC 0.733 [0.666-0.803], '
                'college+draft 0.894, draft position alone 0.887. Combined vs draft-only: '
                'statistical tie (DeLong p = 0.72).',
                'Strongest college signals: age at draft, rushing share, completion %. '
                'Box-score EPA adds little beyond these.',
                'Rookie-season performance is not predictable out-of-sample and is not projected.',
            ],
        },
        'projections': projections,
    }

    out_path = ROOT / 'data' / 'projections' / 'all_projections.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    probs = [p['starter_prob_college'] for p in projections]
    print(f'[OK] {len(projections)} projections ({n_ineligible} non-draft-eligible skipped) -> {out_path}')
    print(f'    P(starter|college): mean {np.mean(probs):.2f}, range {np.min(probs):.2f}-{np.max(probs):.2f}')


if __name__ == '__main__':
    main()
