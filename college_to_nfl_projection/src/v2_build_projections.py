"""
V2 Projection Database Builder.

Replaces build_projection_database.py. Key changes, all driven by the honest
evaluation in v2_evaluate.py (results/v2_evaluation_report.md):

- The 5-class outcome probabilities are gone: out-of-sample they were WORSE
  than just quoting base rates (CV log-loss 1.334 vs 1.244). What survives
  honest validation is a binary "becomes an NFL starter" probability:
    * college stats only          (CV AUC 0.660, beats base-rate log-loss)
    * college stats + log(pick)   (CV AUC 0.796; draft capital alone is 0.828)
- Trained only on draft classes <= 2019: newer classes haven't had enough
  seasons to reach 85 starts, so their labels are right-censored.
- Rookie LEAF is NOT predictable out-of-sample (CV R^2 <= 0 for every model,
  including draft capital), so no rookie LEAF projection is produced.
- Fabricated career trajectories (hardcoded career lengths / peak LEAFs) removed.
- Actual outcomes for classes >= 2020 are shown as TBD unless already locked in
  (Elite / Solid Starter achievements are irreversible; Bust / Journeyman are not).

Output: data/projections/all_projections.json (consumed by college_qb_explorer.html)
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent

COLLEGE_FEATURES = ['epa_per_play_adj', 'success_rate', 'big_play_rate', 'attempts']
OUTCOME_MAX_CLASS = 2019

# From results/v2_evaluation_report.md (leave-one-draft-class-out)
CV_METRICS = {
    'starter_auc_college': 0.680,
    'starter_auc_combined': 0.813,
    'starter_auc_pick_only': 0.828,
    'starter_base_rate': 0.32,
}


def norm_name(n):
    n = str(n).lower().strip()
    n = re.sub(r'\s+(jr|sr|ii|iii|iv|v)\.?$', '', n)
    n = re.sub(r"[^a-z\s]", '', n)
    return re.sub(r'\s+', ' ', n)


def classify_outcomes(df):
    """Mutually exclusive outcome labels (Backup folded into Journeyman)."""
    out = []
    for _, r in df.iterrows():
        if pd.isna(r.get('is_elite')) or pd.isna(r.get('is_bust')):
            out.append(None)
            continue
        starts = r.get('total_starts', 0) or 0
        if r.get('is_elite') == 1 and starts >= 85 and (r.get('rookie_leaf') or 0) > 0.05:
            out.append('Elite')
        elif r.get('is_sustained_starter') == 1 or r.get('reached_85_starts') == 1:
            out.append('Solid Starter')
        elif starts >= 16:
            out.append('Journeyman')
        else:
            out.append('Bust')
    return out


def main():
    print('=' * 70)
    print('V2 PROJECTION DATABASE BUILDER')
    print('=' * 70)

    print('\n[1] Loading data...')
    college = pd.read_csv(ROOT / 'data' / 'processed' / 'aggregated_stats_v2' / 'stats_career_average_v2.csv')
    outcomes = pd.read_csv(ROOT / 'data' / 'processed' / 'nfl_outcomes_comprehensive.csv')
    picks = pd.read_csv(ROOT / 'data' / 'raw' / 'draft_picks_all.csv')
    picks['name_key'] = picks['player_name'].map(norm_name)
    picks = picks.drop_duplicates(subset=['name_key', 'draft_year'])

    df = college.merge(outcomes, on=['player_name', 'draft_year'], how='left')
    df['name_key'] = df['player_name'].map(norm_name)
    df = df.merge(picks[['name_key', 'draft_year', 'pick']], on=['name_key', 'draft_year'], how='left')
    df['log_pick'] = np.log(df['pick'])
    print(f'    {len(df)} QBs | {df.pick.notna().sum()} with draft pick | '
          f'{df.is_elite.notna().sum()} with NFL outcome data')

    print('\n[2] Training starter models (classes <= %d)...' % OUTCOME_MAX_CLASS)
    train = df[(df['draft_year'] <= OUTCOME_MAX_CLASS)].copy()
    train['starter'] = ((train['reached_85_starts'] == 1) | (train['is_sustained_starter'] == 1)).astype(int)
    train = train.dropna(subset=COLLEGE_FEATURES + ['starter'])
    print(f'    Training sample: {len(train)} QBs, {train.starter.sum()} starters')

    model_college = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    model_college.fit(train[COLLEGE_FEATURES], train['starter'])

    train_p = train.dropna(subset=['log_pick'])
    model_combined = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    model_combined.fit(train_p[COLLEGE_FEATURES + ['log_pick']], train_p['starter'])

    print('\n[3] Generating projections...')
    df['actual_outcome_label'] = classify_outcomes(df)

    projections = []
    skipped = 0
    for _, row in df.iterrows():
        if row[COLLEGE_FEATURES].isna().any():
            skipped += 1
            continue

        X_c = row[COLLEGE_FEATURES].to_frame().T.astype(float)
        p_college = float(model_college.predict_proba(X_c)[0, 1])

        p_combined = None
        if pd.notna(row['log_pick']):
            X_b = row[COLLEGE_FEATURES + ['log_pick']].to_frame().T.astype(float)
            p_combined = float(model_combined.predict_proba(X_b)[0, 1])

        # Censoring-aware actual outcome: for recent classes only irreversible
        # achievements (Elite / Solid Starter) are final; anything else is TBD.
        actual = row['actual_outcome_label']
        draft_year = int(row['draft_year'])
        if draft_year > OUTCOME_MAX_CLASS and actual not in ('Elite', 'Solid Starter'):
            actual = 'TBD' if actual is not None else None

        projections.append({
            'player_name': row['player_name'],
            'draft_year': draft_year,
            'college': row.get('college') if pd.notna(row.get('college')) else 'Unknown',
            'draft_pick': int(row['pick']) if pd.notna(row['pick']) else None,
            'college_epa_raw': round(float(row['epa_per_play_raw']), 3) if pd.notna(row['epa_per_play_raw']) else None,
            'college_epa_adj': round(float(row['epa_per_play_adj']), 3),
            'sos_percentile': round(float(row['sos_percentile']), 1) if pd.notna(row['sos_percentile']) else None,
            'college_attempts': int(row['attempts']),
            'is_fcs': bool(row.get('is_fcs_team', 0) > 0),
            'starter_prob_college': round(p_college, 3),
            'starter_prob_combined': round(p_combined, 3) if p_combined is not None else None,
            'actual_outcome': actual,
            'actual_rookie_leaf': round(float(row['rookie_leaf']), 3) if pd.notna(row.get('rookie_leaf')) else None,
            'actual_total_starts': int(row['total_starts']) if pd.notna(row.get('total_starts')) else None,
        })

    print(f'    {len(projections)} projections ({skipped} skipped for missing features)')

    out = {
        'meta': {
            'model': 'v2 logistic (L2), trained on draft classes 2007-%d' % OUTCOME_MAX_CLASS,
            'features': COLLEGE_FEATURES,
            'cv': CV_METRICS,
            'notes': [
                'Starter = 85+ career starts or a 30+ consecutive-start streak.',
                'Probabilities are out-of-sample honest: college-only CV AUC 0.680, '
                'college+draft CV AUC 0.813, draft capital alone 0.828.',
                'Rookie-season performance is not predictable out-of-sample and is not projected.',
            ],
        },
        'projections': projections,
    }

    out_path = ROOT / 'data' / 'projections' / 'all_projections.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\n[OK] Saved -> {out_path}')

    probs = [p['starter_prob_college'] for p in projections]
    print(f'    starter_prob_college: mean {np.mean(probs):.2f}, '
          f'min {np.min(probs):.2f}, max {np.max(probs):.2f}')


if __name__ == '__main__':
    main()
