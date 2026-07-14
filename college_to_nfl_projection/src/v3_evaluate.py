"""
V3 Evaluation - implements SPEC_V3.md exactly. Read the spec first.

Discrete-time logistic hazard on person-periods, leave-one-draft-class-out.
M0 = draft block, M1 = college block, M2 = both.
Primary: AUC of P(starter within 5 seasons), classes 2007-2021,
with cluster bootstrap (draft classes) CIs and DeLong tests.
Quasi-holdout: classes 2022-2024 at their observed horizons.
T2: veteran contract (APY >= 4% of cap, sensitivity 3%/5%) within 6 years.

Output: results/v3_evaluation_report.md
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).parent.parent
RNG = np.random.default_rng(42)

COLLEGE_BLOCK = ['epa_adj_shrunk', 'success_rate_shrunk', 'big_play_rate', 'log_attempts',
                 'rush_share', 'rush_ypg', 'comp_pct', 'age_at_draft', 'age_missing',
                 'seasons_played', 'game_level_adj', 'has_box_stats']
FULL_WINDOW_MAX_CLASS = 2021   # classes fully observed for the 5-season window
HOLDOUT_CLASSES = [2022, 2023, 2024]


def norm_name(n):
    n = str(n).lower().strip()
    n = re.sub(r'\s+(jr|sr|ii|iii|iv|v)\.?$', '', n)
    n = re.sub(r"[^a-z\s]", '', n)
    return re.sub(r'\s+', ' ', n)


# ---------------------------------------------------------------- design matrix

def natural_spline_basis(x, knots):
    """Natural cubic spline basis (df = len(knots) - 1 columns incl. linear)."""
    x = np.asarray(x, dtype=float)
    k = np.asarray(knots, dtype=float)
    K = len(k)

    def d(j):
        return (np.maximum(x - k[j], 0) ** 3 - np.maximum(x - k[-1], 0) ** 3) / (k[-1] - k[j])

    cols = [x]
    for j in range(K - 2):
        cols.append(d(j) - d(K - 2))
    return np.column_stack(cols)


class Featurizer:
    """Fold-local preprocessing: median imputation, scaling, spline knots."""

    def __init__(self, college_cols, use_draft):
        self.college_cols = college_cols
        self.use_draft = use_draft

    def fit(self, df):
        self.medians = df[self.college_cols].median() if self.college_cols else None
        if self.college_cols:
            X = df[self.college_cols].fillna(self.medians)
            self.mu, self.sd = X.mean(), X.std().replace(0, 1)
        if self.use_draft:
            self.knots = np.quantile(df['log_pick'], [0.05, 0.35, 0.65, 0.95])
            S = natural_spline_basis(df['log_pick'], self.knots)
            self.smu, self.ssd = S.mean(0), np.where(S.std(0) == 0, 1, S.std(0))
        return self

    def transform(self, df):
        parts = []
        if self.college_cols:
            X = df[self.college_cols].fillna(self.medians)
            parts.append(((X - self.mu) / self.sd).values)
        if self.use_draft:
            S = natural_spline_basis(df['log_pick'], self.knots)
            parts.append((S - self.smu) / self.ssd)
        # season-index dummies (t is small and bounded; never standardized)
        T = pd.get_dummies(df['t'].clip(1, 5)).reindex(columns=range(1, 6), fill_value=0).values
        parts.append(T.astype(float))
        return np.hstack(parts)


def fit_hazard(train_pp, college_cols, use_draft):
    fz = Featurizer(college_cols, use_draft).fit(train_pp)
    X = fz.transform(train_pp)
    m = LogisticRegression(C=1.0, max_iter=4000)
    m.fit(X, train_pp['event'].values)
    return fz, m


def prob_within(fz, model, qb_row, horizon=5):
    """P(first starter season within `horizon`) for one QB."""
    reps = pd.DataFrame([qb_row.to_dict()] * horizon)
    reps['t'] = range(1, horizon + 1)
    h = model.predict_proba(fz.transform(reps))[:, 1]
    return 1 - np.prod(1 - h)


# ---------------------------------------------------------------- inference helpers

def delong_test(y, p1, p2):
    """DeLong test for correlated AUCs (two models, same subjects)."""
    y = np.asarray(y)
    pos1, neg1 = np.where(y == 1)[0], np.where(y == 0)[0]

    def structural(p):
        pp, nn = p[pos1], p[neg1]
        v10 = np.array([(np.sum(x > nn) + 0.5 * np.sum(x == nn)) / len(nn) for x in pp])
        v01 = np.array([(np.sum(pp > x) + 0.5 * np.sum(pp == x)) / len(pp) for x in nn])
        return v10, v01, v10.mean()

    v10a, v01a, auc_a = structural(np.asarray(p1))
    v10b, v01b, auc_b = structural(np.asarray(p2))
    m, n = len(pos1), len(neg1)
    s10 = np.cov(np.vstack([v10a, v10b]))
    s01 = np.cov(np.vstack([v01a, v01b]))
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n
    if var <= 0:
        return auc_a, auc_b, np.nan
    z = (auc_a - auc_b) / np.sqrt(var)
    return auc_a, auc_b, 2 * (1 - sps.norm.cdf(abs(z)))


def auc_score(y, p):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, p)


def bootstrap_auc_ci(df_scores, col, n=2000):
    """Cluster bootstrap over draft classes."""
    classes = df_scores['draft_year'].unique()
    by_class = {c: df_scores[df_scores['draft_year'] == c] for c in classes}
    aucs = []
    for _ in range(n):
        pick = RNG.choice(classes, size=len(classes), replace=True)
        sample = pd.concat([by_class[c] for c in pick])
        if sample['outcome'].nunique() == 2:
            aucs.append(auc_score(sample['outcome'], sample[col]))
    return np.percentile(aucs, [2.5, 97.5])


def calibration(y, p):
    """Reliability intercept/slope: logistic regression of y on logit(p)."""
    lo = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    m = LogisticRegression(C=1e6, max_iter=4000)
    m.fit(lo.reshape(-1, 1), y)
    return m.intercept_[0], m.coef_[0][0]


# ---------------------------------------------------------------- main

def load():
    feats = pd.read_csv(ROOT / 'data' / 'processed' / 'v3_features.csv')
    pp = pd.read_csv(ROOT / 'data' / 'processed' / 'v3_person_periods.csv')
    pp = pp.merge(feats, on=['player_name', 'draft_year'], how='inner')
    return feats, pp


def oof_scores(feats, pp, spec_blocks):
    """LODCO out-of-fold P(starter within 5) for classes <= FULL_WINDOW_MAX_CLASS."""
    scored_qbs = pp.groupby(['player_name', 'draft_year'])['event'].max().reset_index()
    scored_qbs = scored_qbs[scored_qbs['draft_year'] <= FULL_WINDOW_MAX_CLASS]
    out = []
    for held in sorted(scored_qbs['draft_year'].unique()):
        train_pp = pp[pp['draft_year'] != held]
        models = {name: fit_hazard(train_pp, cc, ud) for name, (cc, ud) in spec_blocks.items()}
        test_qbs = feats[(feats['draft_year'] == held) & feats['pick'].notna()]
        for _, qb in test_qbs.iterrows():
            obs = scored_qbs[(scored_qbs['player_name'] == qb['player_name']) &
                             (scored_qbs['draft_year'] == held)]
            if len(obs) == 0:
                continue
            rec = {'player_name': qb['player_name'], 'draft_year': held,
                   'outcome': int(obs['event'].iloc[0])}
            for name, (fz, m) in models.items():
                rec[name] = prob_within(fz, m, qb)
            out.append(rec)
    return pd.DataFrame(out)


def main():
    print('=' * 70)
    print('V3 EVALUATION (per SPEC_V3.md)')
    print('=' * 70)
    feats, pp = load()
    n_qb = pp.groupby(['player_name', 'draft_year']).ngroups
    print(f'\nPerson-periods: {len(pp)} rows, {n_qb} drafted QBs, '
          f'{pp.event.sum()} starter events')

    spec_blocks = {
        'M0_draft': ([], True),
        'M1_college': (COLLEGE_BLOCK, False),
        'M2_both': (COLLEGE_BLOCK, True),
    }

    print('\n[1] LODCO out-of-fold predictions (classes 2007-%d)...' % FULL_WINDOW_MAX_CLASS)
    scores = oof_scores(feats, pp, spec_blocks)
    print(f'    scored QBs: {len(scores)}, starters: {scores.outcome.sum()} '
          f'({scores.outcome.mean()*100:.0f}%)')

    lines = ['# V3 Evaluation Report\n',
             'Implements SPEC_V3.md (pre-registered). All numbers out-of-fold, '
             'leave-one-draft-class-out.\n',
             f'Sample: {len(scores)} drafted QBs, classes 2007-{FULL_WINDOW_MAX_CLASS}, '
             f'{scores.outcome.sum()} became primary starters within 5 seasons '
             f'({scores.outcome.mean()*100:.0f}%).\n',
             '## T1: primary starter within 5 seasons\n',
             '| Model | CV AUC | 95% CI (class bootstrap) |', '|---|---|---|']

    print('\n[2] Primary metric: AUC with bootstrap CIs')
    for name in spec_blocks:
        auc = auc_score(scores['outcome'], scores[name])
        lo, hi = bootstrap_auc_ci(scores, name)
        print(f'    {name:12s} AUC={auc:.3f}  95% CI [{lo:.3f}, {hi:.3f}]')
        lines.append(f'| {name} | {auc:.3f} | [{lo:.3f}, {hi:.3f}] |')
    lines.append('')

    print('\n[3] DeLong tests')
    for a, b in [('M2_both', 'M0_draft'), ('M1_college', 'M0_draft')]:
        auc_a, auc_b, pval = delong_test(scores['outcome'], scores[a], scores[b])
        print(f'    {a} vs {b}: {auc_a:.3f} vs {auc_b:.3f}, p = {pval:.3f}')
        lines.append(f'- DeLong {a} vs {b}: AUC {auc_a:.3f} vs {auc_b:.3f}, **p = {pval:.3f}**')
    lines.append('')

    print('\n[4] Calibration (OOF)')
    lines.append('## Calibration (reliability intercept / slope; ideal 0 / 1)\n')
    for name in spec_blocks:
        ic, sl = calibration(scores['outcome'].values, scores[name].values)
        print(f'    {name:12s} intercept={ic:+.2f} slope={sl:.2f}')
        lines.append(f'- {name}: intercept {ic:+.2f}, slope {sl:.2f}')
    lines.append('')

    print('\n[5] Quasi-holdout: classes %s at observed horizons' % HOLDOUT_CLASSES)
    lines.append('## Quasi-holdout (classes 2022-2024, observed horizons)\n')
    train_pp = pp[pp['draft_year'] <= FULL_WINDOW_MAX_CLASS]
    models = {name: fit_hazard(train_pp, cc, ud) for name, (cc, ud) in spec_blocks.items()}
    hold = []
    for _, qb in feats[feats['draft_year'].isin(HOLDOUT_CLASSES) & feats['pick'].notna()].iterrows():
        qb_pp = pp[(pp['player_name'] == qb['player_name']) & (pp['draft_year'] == qb['draft_year'])]
        if len(qb_pp) == 0:
            continue
        horizon = int(qb_pp['t'].max())
        rec = {'player_name': qb['player_name'], 'draft_year': qb['draft_year'],
               'outcome': int(qb_pp['event'].max())}
        for name, (fz, m) in models.items():
            rec[name] = prob_within(fz, m, qb, horizon=horizon)
        hold.append(rec)
    hold = pd.DataFrame(hold)
    if len(hold) and hold['outcome'].nunique() == 2:
        lines.append(f'Sample: {len(hold)} QBs, {hold.outcome.sum()} starters so far.\n')
        for name in spec_blocks:
            auc = auc_score(hold['outcome'], hold[name])
            print(f'    {name:12s} AUC={auc:.3f}  (n={len(hold)}, events={hold.outcome.sum()})')
            lines.append(f'- {name}: AUC {auc:.3f}')
    lines.append('')

    print('\n[6] T2: veteran contract within 6 years')
    lines.append('## T2: veteran contract (APY % of cap) within 6 years, classes <= 2020\n')
    contracts = pd.read_csv(ROOT / 'data' / 'raw' / 'contracts_qb.csv')
    contracts['name_key'] = contracts['player'].map(norm_name)
    feats['name_key2'] = feats['player_name'].map(norm_name)
    vet = contracts[contracts['is_active'].notna()] if 'is_active' in contracts else contracts
    lines.append('| Threshold | M0 AUC | M1 AUC | M2 AUC | base rate |')
    lines.append('|---|---|---|---|---|')
    for thresh in [0.03, 0.04, 0.05]:
        rows = []
        for _, qb in feats[feats['pick'].notna() & (feats['draft_year'] <= 2020)].iterrows():
            c = contracts[(contracts['name_key'] == qb['name_key2']) &
                          (contracts['year_signed'] > qb['draft_year']) &
                          (contracts['year_signed'] <= qb['draft_year'] + 6)]
            hit = int((c['apy_cap_pct'] >= thresh).any()) if len(c) else 0
            rows.append({'player_name': qb['player_name'], 'draft_year': qb['draft_year'],
                         'outcome': hit})
        t2 = pd.DataFrame(rows).merge(feats, on=['player_name', 'draft_year']).reset_index(drop=True)
        aucs = {}
        for name, (cc, ud) in spec_blocks.items():
            oof = np.full(len(t2), np.nan)
            for held in t2['draft_year'].unique():
                tr, te = t2[t2.draft_year != held], t2[t2.draft_year == held]
                if tr['outcome'].nunique() < 2:
                    continue
                fz = Featurizer(cc, ud).fit(tr.assign(t=1))
                m = LogisticRegression(C=1.0, max_iter=4000)
                m.fit(fz.transform(tr.assign(t=1)), tr['outcome'])
                oof[te.index.values] = m.predict_proba(fz.transform(te.assign(t=1)))[:, 1]
            mask = ~np.isnan(oof)
            aucs[name] = auc_score(t2['outcome'][mask], oof[mask])
        base = t2['outcome'].mean()
        print(f'    APY>={thresh:.0%}: M0={aucs["M0_draft"]:.3f} M1={aucs["M1_college"]:.3f} '
              f'M2={aucs["M2_both"]:.3f} (base rate {base:.0%})')
        lines.append(f'| {thresh:.0%} | {aucs["M0_draft"]:.3f} | {aucs["M1_college"]:.3f} '
                     f'| {aucs["M2_both"]:.3f} | {base:.0%} |')
    lines.append('')

    out = ROOT / 'results' / 'v3_evaluation_report.md'
    out.write_text('\n'.join(lines))
    scores.to_csv(ROOT / 'results' / 'v3_oof_scores.csv', index=False)
    print(f'\n[OK] Report -> {out}')


if __name__ == '__main__':
    main()
