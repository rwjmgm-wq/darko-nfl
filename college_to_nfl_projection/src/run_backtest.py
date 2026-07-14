"""
Comprehensive Backtest of College-to-NFL Projection Models

Evaluates model performance across multiple metrics and identifies weaknesses.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


def get_project_root():
    return Path(__file__).parent.parent


def get_outcome(row):
    """Create mutually exclusive career outcomes"""
    is_elite = row.get('is_elite', 0)
    is_sustained = row.get('is_sustained_starter', 0)
    reached_85 = row.get('reached_85_starts', 0)
    is_bust = row.get('is_bust', 0)
    total_starts = row.get('total_starts', 0)

    if pd.isna(is_elite) or pd.isna(is_bust):
        return None

    if is_elite == 1 and total_starts >= 85:
        return 'Elite'
    elif is_sustained == 1 or reached_85 == 1:
        return 'Solid Starter'
    elif total_starts >= 16:
        return 'Journeyman'
    else:
        return 'Bust'


def main():
    root = get_project_root()

    print('=' * 80)
    print('COMPREHENSIVE BACKTEST OF COLLEGE-TO-NFL PROJECTION MODELS')
    print('=' * 80)

    # Load data
    stats_path = root / 'data' / 'processed' / 'aggregated_stats' / 'stats_career_average_sp_adjusted.csv'
    df_college = pd.read_csv(stats_path)
    print(f'\nLoaded {len(df_college)} QBs with college stats')

    outcomes_path = root / 'data' / 'processed' / 'nfl_outcomes_comprehensive.csv'
    df_nfl = pd.read_csv(outcomes_path)
    print(f'Loaded {len(df_nfl)} QBs with NFL outcomes')

    # Merge
    df = pd.merge(df_college, df_nfl, on=['player_name', 'draft_year'], how='inner')
    print(f'Merged: {len(df)} QBs with both college and NFL data')

    # Filter to QBs with sufficient NFL data
    df_backtest = df[df['draft_year'] <= 2022].copy()
    print(f'Backtest sample (2007-2022 drafts): {len(df_backtest)} QBs')

    # Features
    features = ['epa_per_play_adj', 'attempts', 'big_play_rate', 'high_leverage_epa', 'success_rate_raw']

    # =========================================================================
    # 1. ROOKIE LEAF PREDICTION
    # =========================================================================
    print('\n' + '=' * 80)
    print('1. ROOKIE LEAF PREDICTION (Regression)')
    print('=' * 80)

    df_rookie = df[df['rookie_leaf'].notna() & df[features].notna().all(axis=1)].copy()
    print(f'\nSample size: {len(df_rookie)} QBs with rookie LEAF')

    X = df_rookie[features]
    y = df_rookie['rookie_leaf']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    y_pred = model.predict(X_scaled)

    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    corr = np.corrcoef(y, y_pred)[0, 1]

    print(f'\nIn-sample performance:')
    print(f'  R-squared: {r2:.3f} ({r2*100:.1f}% variance explained)')
    print(f'  RMSE: {rmse:.3f}')
    print(f'  Correlation: {corr:.3f}')

    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
    print(f'\n5-Fold CV R-squared: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})')

    print(f'\nFeature coefficients:')
    for feat, coef in sorted(zip(features, model.coef_), key=lambda x: abs(x[1]), reverse=True):
        print(f'  {feat:25s}: {coef:+.3f}')

    # Big misses
    df_rookie['predicted_leaf'] = y_pred
    df_rookie['error'] = df_rookie['rookie_leaf'] - df_rookie['predicted_leaf']
    df_rookie['abs_error'] = df_rookie['error'].abs()

    print(f'\nBiggest prediction errors (misses):')
    worst = df_rookie.nlargest(10, 'abs_error')[['player_name', 'draft_year', 'rookie_leaf', 'predicted_leaf', 'error']]
    for _, row in worst.iterrows():
        direction = 'overperformed' if row['error'] > 0 else 'underperformed'
        print(f"  {row['player_name']} ({int(row['draft_year'])}): Pred {row['predicted_leaf']:+.2f}, Actual {row['rookie_leaf']:+.2f} ({direction})")

    # =========================================================================
    # 2. CAREER OUTCOME CLASSIFICATION
    # =========================================================================
    print('\n' + '=' * 80)
    print('2. CAREER OUTCOME CLASSIFICATION')
    print('=' * 80)

    df_backtest['outcome'] = df_backtest.apply(get_outcome, axis=1)
    df_career = df_backtest[df_backtest['outcome'].notna() & df_backtest[features].notna().all(axis=1)].copy()
    print(f'\nSample size: {len(df_career)} QBs with career outcomes')

    print(f'\nOutcome distribution:')
    for outcome, count in df_career['outcome'].value_counts().items():
        pct = count / len(df_career) * 100
        print(f'  {outcome:15s}: {count:3d} ({pct:5.1f}%)')

    X = df_career[features]
    y = df_career['outcome']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model_multi = LogisticRegression(random_state=42, max_iter=1000, multi_class='multinomial')
    model_multi.fit(X_scaled, y)

    y_pred = model_multi.predict(X_scaled)

    accuracy = accuracy_score(y, y_pred)
    print(f'\nIn-sample accuracy: {accuracy:.1%}')

    print(f'\nPer-class accuracy:')
    for outcome in model_multi.classes_:
        mask = y == outcome
        if mask.sum() > 0:
            class_acc = (y_pred[mask] == outcome).mean()
            print(f'  {outcome:15s}: {class_acc:.1%} ({mask.sum()} samples)')

    cv_scores = cross_val_score(model_multi, X_scaled, y, cv=5, scoring='accuracy')
    print(f'\n5-Fold CV Accuracy: {cv_scores.mean():.1%} (+/- {cv_scores.std()*2:.1%})')

    # Confusion matrix
    print(f'\nConfusion Matrix:')
    cm = confusion_matrix(y, y_pred, labels=['Elite', 'Solid Starter', 'Journeyman', 'Bust'])
    print(f'                    Predicted')
    print(f'                    Elite  Solid  Journ  Bust')
    for i, outcome in enumerate(['Elite', 'Solid Starter', 'Journeyman', 'Bust']):
        row = cm[i]
        print(f'  Actual {outcome:12s}: {row[0]:5d} {row[1]:5d} {row[2]:5d} {row[3]:5d}')

    # Binary classifications
    print('\n--- Elite Detection ---')
    y_elite = (y == 'Elite').astype(int)
    if y_elite.sum() > 0:
        model_elite = LogisticRegression(random_state=42, max_iter=1000)
        model_elite.fit(X_scaled, y_elite)
        y_elite_proba = model_elite.predict_proba(X_scaled)[:, 1]
        auc_elite = roc_auc_score(y_elite, y_elite_proba)
        print(f'  AUC: {auc_elite:.3f}')
        print(f'  Elite QBs: {y_elite.sum()} / {len(y_elite)}')

    print('\n--- Bust Detection ---')
    y_bust = (y == 'Bust').astype(int)
    model_bust = LogisticRegression(random_state=42, max_iter=1000)
    model_bust.fit(X_scaled, y_bust)
    y_bust_proba = model_bust.predict_proba(X_scaled)[:, 1]
    auc_bust = roc_auc_score(y_bust, y_bust_proba)
    print(f'  AUC: {auc_bust:.3f}')
    print(f'  Busts: {y_bust.sum()} / {len(y_bust)}')

    # =========================================================================
    # 3. YEAR-BY-YEAR OUT-OF-SAMPLE BACKTEST
    # =========================================================================
    print('\n' + '=' * 80)
    print('3. YEAR-BY-YEAR BACKTEST (Out-of-Sample)')
    print('=' * 80)

    results_by_year = []

    for test_year in range(2015, 2023):
        train = df_career[df_career['draft_year'] < test_year]
        test = df_career[df_career['draft_year'] == test_year]

        if len(train) < 20 or len(test) < 3:
            continue

        X_train = train[features]
        y_train = train['outcome']
        X_test = test[features]
        y_test = test['outcome']

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(random_state=42, max_iter=1000, multi_class='multinomial')
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)

        results_by_year.append({
            'year': test_year,
            'n_test': len(test),
            'accuracy': acc,
            'correct': (y_pred == y_test).sum()
        })

    print(f'\nOut-of-sample accuracy by draft year:')
    for r in results_by_year:
        print(f"  {r['year']}: {r['accuracy']:.1%} ({r['correct']}/{r['n_test']} correct)")

    if results_by_year:
        avg_oos = np.mean([r['accuracy'] for r in results_by_year])
        print(f'\nAverage out-of-sample accuracy: {avg_oos:.1%}')

    # =========================================================================
    # 4. COMPARISON VS DRAFT PICK BASELINE
    # =========================================================================
    print('\n' + '=' * 80)
    print('4. COMPARISON VS BASELINE (Draft Pick)')
    print('=' * 80)

    draft_path = root / 'data' / 'raw' / 'drafted_qbs.csv'
    if draft_path.exists():
        df_draft = pd.read_csv(draft_path)
        df_with_pick = pd.merge(df_career, df_draft[['player_name', 'draft_year', 'pick']],
                                on=['player_name', 'draft_year'], how='left')
        df_with_pick = df_with_pick[df_with_pick['pick'].notna()]

        if len(df_with_pick) > 20:
            outcome_numeric = df_with_pick['outcome'].map({'Elite': 4, 'Solid Starter': 3, 'Journeyman': 2, 'Bust': 1})
            pick_corr = df_with_pick['pick'].corr(outcome_numeric)
            print(f'\nDraft pick correlation with outcome: {pick_corr:.3f}')
            print('  (Negative = higher picks become better players)')

            y_elite = (df_with_pick['outcome'] == 'Elite').astype(int)

            if y_elite.sum() > 0:
                X_model = df_with_pick[features]
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_model)
                model = LogisticRegression(random_state=42, max_iter=1000)
                model.fit(X_scaled, y_elite)
                model_proba = model.predict_proba(X_scaled)[:, 1]
                model_auc = roc_auc_score(y_elite, model_proba)

                pick_proba = 1 - (df_with_pick['pick'] / df_with_pick['pick'].max())
                pick_auc = roc_auc_score(y_elite, pick_proba)

                print(f'\nElite prediction AUC:')
                print(f'  College stats model: {model_auc:.3f}')
                print(f'  Draft pick alone:    {pick_auc:.3f}')
                print(f'  Improvement:         {model_auc - pick_auc:+.3f}')

                # Combined model
                df_with_pick['pick_normalized'] = 1 - (df_with_pick['pick'] / df_with_pick['pick'].max())
                X_combined = np.column_stack([X_scaled, df_with_pick['pick_normalized'].values])
                model_combined = LogisticRegression(random_state=42, max_iter=1000)
                model_combined.fit(X_combined, y_elite)
                combined_proba = model_combined.predict_proba(X_combined)[:, 1]
                combined_auc = roc_auc_score(y_elite, combined_proba)
                print(f'  Combined (stats+pick): {combined_auc:.3f}')

    # =========================================================================
    # 5. IDENTIFY GLARING WEAKNESSES
    # =========================================================================
    print('\n' + '=' * 80)
    print('5. IDENTIFIED WEAKNESSES & ISSUES')
    print('=' * 80)

    # Check for systematic biases
    print('\n--- Systematic Biases ---')

    # By conference (if available)
    if 'college' in df_career.columns:
        # High vs low profile schools
        power_schools = ['Alabama', 'Ohio State', 'Clemson', 'Georgia', 'LSU', 'Oklahoma',
                        'USC', 'Michigan', 'Notre Dame', 'Texas', 'Florida', 'Oregon']
        df_career['power_school'] = df_career['college'].isin(power_schools)

        power = df_career[df_career['power_school']]
        non_power = df_career[~df_career['power_school']]

        power_elite_rate = (power['outcome'] == 'Elite').mean() if len(power) > 0 else 0
        non_power_elite_rate = (non_power['outcome'] == 'Elite').mean() if len(non_power) > 0 else 0

        print(f'\n  Power school Elite rate: {power_elite_rate:.1%} ({len(power)} QBs)')
        print(f'  Non-power Elite rate:    {non_power_elite_rate:.1%} ({len(non_power)} QBs)')

    # Check prediction errors by draft round
    print('\n--- Model Limitations ---')
    print(f'\n  1. Rookie LEAF R-squared: {r2:.3f}')
    print(f'     -> Only {r2*100:.1f}% of rookie performance explained by college stats')
    print(f'     -> Team situation, coaching, weapons not captured')

    print(f'\n  2. Sample size issues:')
    print(f'     -> Elite QBs: only {y_elite.sum()} in sample (rare event)')
    print(f'     -> Model may overfit to small elite sample')

    print(f'\n  3. Out-of-sample degradation:')
    if results_by_year:
        in_sample_acc = accuracy
        oos_acc = avg_oos
        degradation = in_sample_acc - oos_acc
        print(f'     -> In-sample: {in_sample_acc:.1%}, Out-of-sample: {oos_acc:.1%}')
        print(f'     -> Degradation: {degradation:.1%}')

    # Missing predictors
    print(f'\n  4. Missing predictors (not in model):')
    print(f'     -> Athleticism / mobility metrics')
    print(f'     -> Arm strength / ball velocity')
    print(f'     -> NFL team situation (weapons, O-line, coaching)')
    print(f'     -> Injury history')
    print(f'     -> Mental / leadership intangibles')

    print('\n' + '=' * 80)
    print('BACKTEST COMPLETE')
    print('=' * 80)


if __name__ == '__main__':
    main()
