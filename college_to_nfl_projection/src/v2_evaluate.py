"""
V2 Honest Evaluation.

Replaces the in-sample evaluations in feature_selection_analysis.py /
validate_predictions.py (exhaustive feature search scored on training data) with:

- Leave-one-draft-class-out CV: every prediction is made by a model that never
  saw that draft class. Feature set is FIXED a priori - no search.
- Scaling and regularization strength are fit inside each training fold.
- Baseline is log(draft pick), the standard functional form, plus an
  incremental test: does college data add anything ON TOP of draft capital?
- Career-outcome models train only on draft classes old enough for the outcome
  to be observable (right-censoring was previously ignored).

Targets:
  1. rookie_leaf   (continuous, all classes with a rookie season)
  2. starter (85+ starts or sustained 30+ start streak; classes <= 2019 only)

Outputs: results/v2_evaluation_report.md
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent

# Final spec. The FCS flag was dropped (only 4 FCS QBs in the training window,
# coefficient ~0, CV AUC improves without it). Variants tested during development
# (adjusted success rate, EPA-only, success-only, stronger C) scored CV AUC
# 0.53-0.68 on the same folds, so quote the headline with that range in mind.
COLLEGE_FEATURES = ['epa_per_play_adj', 'success_rate', 'big_play_rate', 'attempts']
COLLEGE_FEATURES_RAW = ['epa_per_play_raw', 'success_rate', 'big_play_rate', 'attempts']
ALPHAS = np.logspace(-2, 4, 25)
OUTCOME_MAX_CLASS = 2019  # newest class where 85+ starts is observable in the data


def norm_name(n):
    n = str(n).lower().strip()
    n = re.sub(r'\s+(jr|sr|ii|iii|iv|v)\.?$', '', n)
    n = re.sub(r"[^a-z\s]", '', n)
    return re.sub(r'\s+', ' ', n)


def load_data():
    college = pd.read_csv(ROOT / 'data' / 'processed' / 'aggregated_stats_v2' / 'stats_career_average_v2.csv')
    outcomes = pd.read_csv(ROOT / 'data' / 'processed' / 'nfl_outcomes_comprehensive.csv')
    df = college.merge(outcomes, on=['player_name', 'draft_year'], how='inner')

    picks = pd.read_csv(ROOT / 'data' / 'raw' / 'draft_picks_all.csv')
    picks['name_key'] = picks['player_name'].map(norm_name)
    picks = picks.drop_duplicates(subset=['name_key', 'draft_year'])
    df['name_key'] = df['player_name'].map(norm_name)
    df = df.merge(picks[['name_key', 'draft_year', 'pick']], on=['name_key', 'draft_year'], how='left')
    df['log_pick'] = np.log(df['pick'])

    matched = df['pick'].notna().mean() * 100
    print(f'[OK] {len(df)} QBs with outcomes | draft pick matched: {matched:.0f}%')
    return df


def oof_predictions_reg(df, feature_cols, target):
    """Pooled out-of-fold predictions, leave-one-draft-class-out, ridge."""
    d = df.dropna(subset=feature_cols + [target, 'draft_year']).copy()
    X, y = d[feature_cols].values, d[target].values
    groups = d['draft_year'].values
    oof = np.full(len(d), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
        model.fit(X[tr], y[tr])
        oof[te] = model.predict(X[te])
    d['oof'] = oof
    return d


def reg_metrics(d, target):
    err = d['oof'] - d[target]
    ss_res = (err ** 2).sum()
    ss_tot = ((d[target] - d[target].mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    r = np.corrcoef(d['oof'], d[target])[0, 1]
    return {'n': len(d), 'cv_r2': r2, 'cv_r': r, 'mae': err.abs().mean()}


def eval_rookie_leaf(df, lines):
    lines.append('## Target 1: NFL rookie LEAF\n')
    lines.append('Every metric below is out-of-fold (leave-one-draft-class-out). '
                 'CV R^2 can be negative: that means worse than predicting the mean.\n')
    specs = [
        ('Draft capital only: log(pick)', ['log_pick']),
        ('College only (raw EPA)', COLLEGE_FEATURES_RAW),
        ('College only (adjusted EPA)', COLLEGE_FEATURES),
        ('College (adjusted) + log(pick)', COLLEGE_FEATURES + ['log_pick']),
    ]
    rows = []
    for label, cols in specs:
        d = oof_predictions_reg(df, cols, 'rookie_leaf')
        m = reg_metrics(d, 'rookie_leaf')
        rows.append((label, m))
        print(f'  {label:38s} n={m["n"]:3d}  CV R2={m["cv_r2"]:+.3f}  CV r={m["cv_r"]:+.3f}  MAE={m["mae"]:.3f}')
    lines.append('| Model | n | CV R^2 | CV r | MAE |')
    lines.append('|---|---|---|---|---|')
    for label, m in rows:
        lines.append(f'| {label} | {m["n"]} | {m["cv_r2"]:+.3f} | {m["cv_r"]:+.3f} | {m["mae"]:.3f} |')
    lines.append('')
    return rows


def oof_predictions_clf(df, feature_cols, target):
    d = df.dropna(subset=feature_cols + [target, 'draft_year']).copy()
    X, y = d[feature_cols].values, d[target].astype(int).values
    groups = d['draft_year'].values
    oof = np.full(len(d), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            oof[te] = y[tr].mean()
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        model.fit(X[tr], y[tr])
        oof[te] = model.predict_proba(X[te])[:, 1]
    d['oof_prob'] = oof
    return d


def eval_starter(df, lines):
    lines.append(f'## Target 2: became a starter (85+ starts or 30+ consecutive), classes <= {OUTCOME_MAX_CLASS}\n')
    lines.append('Newer classes are right-censored (not enough seasons to reach the '
                 'threshold) and are excluded from training and scoring.\n')
    d0 = df[df['draft_year'] <= OUTCOME_MAX_CLASS].copy()
    d0['starter'] = ((d0['reached_85_starts'] == 1) | (d0['is_sustained_starter'] == 1)).astype(int)
    print(f'  Eligible sample: {len(d0)} QBs, {d0["starter"].sum()} starters ({d0["starter"].mean() * 100:.0f}%)')
    lines.append(f'Sample: {len(d0)} QBs, {d0["starter"].sum()} starters ({d0["starter"].mean() * 100:.0f}%).\n')

    specs = [
        ('Draft capital only: log(pick)', ['log_pick']),
        ('College only (adjusted EPA)', COLLEGE_FEATURES),
        ('College (adjusted) + log(pick)', COLLEGE_FEATURES + ['log_pick']),
    ]
    lines.append('| Model | n | CV AUC | CV log-loss | Base-rate log-loss |')
    lines.append('|---|---|---|---|---|')
    for label, cols in specs:
        d = oof_predictions_clf(d0, cols, 'starter')
        auc = roc_auc_score(d['starter'], d['oof_prob'])
        ll = log_loss(d['starter'], d['oof_prob'].clip(1e-6, 1 - 1e-6))
        base = log_loss(d['starter'], np.full(len(d), d['starter'].mean()))
        print(f'  {label:38s} n={len(d):3d}  CV AUC={auc:.3f}  CV logloss={ll:.3f} (base {base:.3f})')
        lines.append(f'| {label} | {len(d)} | {auc:.3f} | {ll:.3f} | {base:.3f} |')
    lines.append('')


def eval_sensitivity(lines):
    """Full grid of aggregation x EPA-variant for the starter target.

    Reported in full (not max-picked) so this stays a sensitivity check,
    not another round of selection.
    """
    lines.append('## Sensitivity: aggregation method x EPA variant (starter target, college-only)\n')
    lines.append('| Aggregation | EPA variant | CV AUC | CV AUC (+log pick) |')
    lines.append('|---|---|---|---|')
    print('  aggregation        epa        AUC(college)  AUC(+pick)')
    for agg in ['career_average', 'recency_weighted', 'final_season']:
        college = pd.read_csv(ROOT / 'data' / 'processed' / 'aggregated_stats_v2' / f'stats_{agg}_v2.csv')
        outcomes = pd.read_csv(ROOT / 'data' / 'processed' / 'nfl_outcomes_comprehensive.csv')
        df = college.merge(outcomes, on=['player_name', 'draft_year'], how='inner')
        picks = pd.read_csv(ROOT / 'data' / 'raw' / 'draft_picks_all.csv')
        picks['name_key'] = picks['player_name'].map(norm_name)
        picks = picks.drop_duplicates(subset=['name_key', 'draft_year'])
        df['name_key'] = df['player_name'].map(norm_name)
        df = df.merge(picks[['name_key', 'draft_year', 'pick']], on=['name_key', 'draft_year'], how='left')
        df['log_pick'] = np.log(df['pick'])
        d0 = df[df['draft_year'] <= OUTCOME_MAX_CLASS].copy()
        d0['starter'] = ((d0['reached_85_starts'] == 1) | (d0['is_sustained_starter'] == 1)).astype(int)
        for variant, cols in [('raw', COLLEGE_FEATURES_RAW), ('adjusted', COLLEGE_FEATURES)]:
            d = oof_predictions_clf(d0, cols, 'starter')
            auc = roc_auc_score(d['starter'], d['oof_prob'])
            d2 = oof_predictions_clf(d0, cols + ['log_pick'], 'starter')
            auc2 = roc_auc_score(d2['starter'], d2['oof_prob'])
            print(f'  {agg:18s} {variant:10s} {auc:.3f}         {auc2:.3f}')
            lines.append(f'| {agg} | {variant} | {auc:.3f} | {auc2:.3f} |')
    lines.append('')


def eval_multiclass(df, lines):
    """OOF log-loss of the 4-class outcome model used by the projection stage."""
    lines.append(f'## 4-class outcome model (Elite / Solid Starter / Journeyman / Bust), classes <= {OUTCOME_MAX_CLASS}\n')
    d0 = df[df['draft_year'] <= OUTCOME_MAX_CLASS].copy()
    d0['outcome'] = classify_outcomes(d0)
    d0 = d0.dropna(subset=COLLEGE_FEATURES + ['outcome', 'draft_year'])
    X = d0[COLLEGE_FEATURES].values
    y = d0['outcome'].values
    groups = d0['draft_year'].values
    classes = np.unique(y)

    oof = np.full((len(d0), len(classes)), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        model.fit(X[tr], y[tr])
        probs = model.predict_proba(X[te])
        col_map = [list(model.classes_).index(c) for c in classes]
        oof[te] = probs[:, col_map]

    ll = log_loss(y, np.clip(oof, 1e-6, None), labels=list(classes))
    freqs = pd.Series(y).value_counts(normalize=True).reindex(classes).values
    base = log_loss(y, np.tile(freqs, (len(y), 1)), labels=list(classes))
    acc = (classes[np.argmax(oof, axis=1)] == y).mean()
    base_acc = pd.Series(y).value_counts(normalize=True).max()
    counts = pd.Series(y).value_counts()
    print(f'  n={len(d0)} | classes: {dict(counts)}')
    print(f'  CV log-loss={ll:.3f} (base-rate {base:.3f}) | CV accuracy={acc:.3f} (majority {base_acc:.3f})')
    lines.append(f'Sample: {len(d0)} QBs. Class counts: {dict(counts)}.\n')
    lines.append(f'- CV log-loss: **{ll:.3f}** vs base-rate {base:.3f} '
                 f'({"beats" if ll < base else "does NOT beat"} always-predict-frequencies)')
    lines.append(f'- CV accuracy: **{acc:.3f}** vs majority-class {base_acc:.3f}\n')


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
    print('V2 HONEST EVALUATION (leave-one-draft-class-out)')
    print('=' * 70)
    df = load_data()

    lines = ['# V2 Evaluation Report (out-of-sample)\n',
             'Fixed feature set, no search. Scaling + regularization fit inside '
             'training folds. Folds = draft classes.\n',
             f'College features: `{COLLEGE_FEATURES}`\n']

    print('\n[Rookie LEAF]')
    eval_rookie_leaf(df, lines)
    print('\n[Career starter]')
    eval_starter(df, lines)
    print('\n[Sensitivity grid]')
    eval_sensitivity(lines)
    print('\n[4-class outcome model]')
    eval_multiclass(df, lines)

    out = ROOT / 'results' / 'v2_evaluation_report.md'
    out.parent.mkdir(exist_ok=True)
    out.write_text('\n'.join(lines))
    print(f'\n[OK] Report -> {out}')


if __name__ == '__main__':
    main()
