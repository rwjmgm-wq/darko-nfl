"""
Comprehensive Model Testing Framework

Tests all combinations:
- 6 aggregation methods (final, average, recency, cumulative, best, volume)
- Multiple stat combinations (7-feature ensemble + exhaustive search)
- 5 NFL outcomes (rookie LEAF, 85+ starts, bust, elite, sustained starter)

Total tests: 6 methods × ~50 stat combos × 5 outcomes = ~1,500 models
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr, pointbiserialr
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent


def load_aggregated_stats(method_name):
    """Load stats for a specific aggregation method"""
    root = get_project_root()
    stats_path = root / 'data' / 'processed' / 'aggregated_stats' / f'stats_{method_name}.csv'

    if stats_path.exists():
        return pd.read_csv(stats_path)
    else:
        return None


def load_nfl_outcomes():
    """Load all NFL outcome metrics"""
    root = get_project_root()
    outcomes_path = root / 'data' / 'processed' / 'nfl_outcomes_comprehensive.csv'

    if outcomes_path.exists():
        return pd.read_csv(outcomes_path)
    else:
        return None


def merge_college_and_nfl(df_college, df_nfl):
    """Merge college stats with NFL outcomes"""
    df_merged = pd.merge(
        df_college,
        df_nfl,
        on=['player_name', 'draft_year'],
        how='inner'
    )

    return df_merged


def test_regression_model(X, y, features):
    """
    Test linear regression model (for continuous outcomes like rookie LEAF)

    Returns: R-squared, correlation coefficient, RMSE
    """
    # Remove any NaN values
    mask = ~(X[features].isna().any(axis=1) | y.isna())
    X_clean = X[features][mask]
    y_clean = y[mask]

    if len(X_clean) < 10:  # Need minimum sample size
        return None

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    # Fit model
    model = LinearRegression()
    model.fit(X_scaled, y_clean)

    # Predictions
    y_pred = model.predict(X_scaled)

    # Metrics
    r = pearsonr(y_clean, y_pred)[0]
    r_squared = r ** 2
    rmse = np.sqrt(np.mean((y_clean - y_pred) ** 2))

    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y_clean, cv=5, scoring='r2')
    cv_r2 = cv_scores.mean()

    return {
        'n': len(X_clean),
        'r': r,
        'r_squared': r_squared,
        'rmse': rmse,
        'cv_r2': cv_r2,
        'coefficients': dict(zip(features, model.coef_))
    }


def test_classification_model(X, y, features):
    """
    Test logistic regression model (for binary outcomes)

    Returns: AUC, accuracy, point-biserial correlation
    """
    # Remove any NaN values
    mask = ~(X[features].isna().any(axis=1) | y.isna())
    X_clean = X[features][mask]
    y_clean = y[mask]

    if len(X_clean) < 10 or y_clean.nunique() < 2:  # Need minimum sample and both classes
        return None

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    # Fit model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, y_clean)

    # Predictions
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    y_pred = model.predict(X_scaled)

    # Metrics
    from sklearn.metrics import roc_auc_score, accuracy_score

    try:
        auc = roc_auc_score(y_clean, y_pred_proba)
    except:
        auc = np.nan

    accuracy = accuracy_score(y_clean, y_pred)

    # Point-biserial correlation (correlation between binary outcome and predicted probability)
    r_pb = pointbiserialr(y_clean, y_pred_proba)[0]

    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y_clean, cv=5, scoring='roc_auc')
    cv_auc = cv_scores.mean()

    # Class distribution
    pos_rate = y_clean.mean()

    return {
        'n': len(X_clean),
        'auc': auc,
        'cv_auc': cv_auc,
        'accuracy': accuracy,
        'r_pointbiserial': r_pb,
        'positive_rate': pos_rate,
        'coefficients': dict(zip(features, model.coef_[0]))
    }


def get_feature_combinations():
    """
    Generate feature combinations to test

    Start with 7-feature ensemble, then explore subsets
    """
    # Core 7-feature ensemble (baseline)
    core_features = [
        'epa_per_play',
        'attempts',
        'big_play_rate',
        'third_down_epa',
        'red_zone_epa',
        'high_leverage_epa',
        'success_rate'
    ]

    combinations_to_test = [
        ('7_feature_ensemble', core_features)
    ]

    # Individual features
    for feat in core_features:
        combinations_to_test.append((f'solo_{feat}', [feat]))

    # Pairs of key features
    key_pairs = [
        (['epa_per_play', 'attempts'], 'epa_volume'),
        (['epa_per_play', 'big_play_rate'], 'epa_bigplay'),
        (['epa_per_play', 'high_leverage_epa'], 'epa_leverage'),
        (['attempts', 'big_play_rate'], 'volume_bigplay'),
        (['third_down_epa', 'red_zone_epa'], 'situational'),
    ]

    for features, name in key_pairs:
        combinations_to_test.append((f'pair_{name}', features))

    # Triplets
    triplets = [
        (['epa_per_play', 'attempts', 'big_play_rate'], 'epa_vol_bp'),
        (['epa_per_play', 'high_leverage_epa', 'success_rate'], 'epa_lev_sr'),
        (['third_down_epa', 'red_zone_epa', 'high_leverage_epa'], 'situational_full'),
    ]

    for features, name in triplets:
        combinations_to_test.append((f'triple_{name}', features))

    # 5-feature combinations (remove 2 from ensemble)
    important_5 = [
        (['epa_per_play', 'attempts', 'big_play_rate', 'high_leverage_epa', 'success_rate'], 'no_situational'),
        (['epa_per_play', 'attempts', 'third_down_epa', 'red_zone_epa', 'high_leverage_epa'], 'no_basic'),
    ]

    for features, name in important_5:
        combinations_to_test.append((f'five_{name}', features))

    return combinations_to_test


def main():
    """Main execution"""
    print("="*80)
    print("COMPREHENSIVE MODEL TESTING FRAMEWORK")
    print("="*80)
    print("\nTesting: 6 aggregations × ~20 stat combos × 5 outcomes")

    root = get_project_root()

    # Load NFL outcomes
    print("\n[1] Loading NFL outcomes...")
    df_outcomes = load_nfl_outcomes()

    if df_outcomes is None:
        print("    [ERROR] NFL outcomes not found. Run extract_nfl_outcomes_comprehensive.py first")
        return

    print(f"    [OK] {len(df_outcomes)} QBs with NFL outcomes")

    # Define aggregation methods to test
    aggregation_methods = [
        'final_season',
        'career_average',
        'recency_weighted',
        'cumulative',
        'best_season',
        'volume_weighted'
    ]

    # Define outcomes to predict
    outcomes = [
        ('rookie_leaf', 'regression', 'Rookie LEAF'),
        ('reached_85_starts', 'classification', '85+ Starts'),
        ('is_bust', 'classification', 'Bust'),
        ('is_elite', 'classification', 'Elite Peak'),
        ('is_sustained_starter', 'classification', 'Sustained Starter')
    ]

    # Get feature combinations to test
    feature_combinations = get_feature_combinations()
    print(f"\n[2] Testing {len(feature_combinations)} feature combinations")

    # Run comprehensive tests
    print(f"\n[3] Running comprehensive model tests...")
    print(f"    Total models: {len(aggregation_methods)} methods × {len(feature_combinations)} features × {len(outcomes)} outcomes")
    print(f"                  = {len(aggregation_methods) * len(feature_combinations) * len(outcomes)} models")
    print("-"*80)

    all_results = []
    test_count = 0
    total_tests = len(aggregation_methods) * len(feature_combinations) * len(outcomes)

    for method_name in aggregation_methods:
        # Load college stats for this aggregation method
        df_college = load_aggregated_stats(method_name)

        if df_college is None:
            print(f"\n  [SKIP] {method_name} - no data")
            continue

        # Merge with NFL outcomes
        df_merged = merge_college_and_nfl(df_college, df_outcomes)

        print(f"\n  {method_name.upper()} ({len(df_merged)} QBs)")
        print("  " + "-"*76)

        for outcome_var, outcome_type, outcome_name in outcomes:
            print(f"\n    Testing: {outcome_name}")

            for combo_name, features in feature_combinations:
                test_count += 1

                # Check if all features exist
                missing_features = [f for f in features if f not in df_merged.columns]
                if missing_features:
                    continue

                # Run test
                X = df_merged
                y = df_merged[outcome_var]

                if outcome_type == 'regression':
                    result = test_regression_model(X, y, features)
                    if result:
                        print(f"      {combo_name:30s} r={result['r']:+.3f} R²={result['r_squared']:.3f} (n={result['n']})")
                        all_results.append({
                            'aggregation_method': method_name,
                            'outcome': outcome_name,
                            'outcome_type': outcome_type,
                            'feature_combo': combo_name,
                            'n_features': len(features),
                            'features': ', '.join(features),
                            'n_samples': result['n'],
                            'r': result['r'],
                            'r_squared': result['r_squared'],
                            'cv_r2': result['cv_r2'],
                            'rmse': result['rmse'],
                            'auc': np.nan,
                            'accuracy': np.nan,
                        })

                else:  # classification
                    result = test_classification_model(X, y, features)
                    if result:
                        print(f"      {combo_name:30s} AUC={result['auc']:.3f} r={result['r_pointbiserial']:+.3f} (n={result['n']})")
                        all_results.append({
                            'aggregation_method': method_name,
                            'outcome': outcome_name,
                            'outcome_type': outcome_type,
                            'feature_combo': combo_name,
                            'n_features': len(features),
                            'features': ', '.join(features),
                            'n_samples': result['n'],
                            'r': result['r_pointbiserial'],
                            'r_squared': np.nan,
                            'cv_r2': np.nan,
                            'rmse': np.nan,
                            'auc': result['auc'],
                            'accuracy': result['accuracy'],
                        })

    # Save all results
    print("\n[4] Saving comprehensive results...")
    df_results = pd.DataFrame(all_results)
    output_path = root / 'data' / 'processed' / 'comprehensive_model_results.csv'
    df_results.to_csv(output_path, index=False)
    print(f"    [OK] {len(df_results)} model results saved to: {output_path}")

    # Generate summary
    print("\n" + "="*80)
    print("TOP MODELS BY OUTCOME")
    print("="*80)

    for outcome_name in df_results['outcome'].unique():
        df_outcome = df_results[df_results['outcome'] == outcome_name]

        print(f"\n  {outcome_name}:")

        if df_outcome['outcome_type'].iloc[0] == 'regression':
            # Sort by R²
            top_models = df_outcome.nlargest(5, 'r_squared')
            print(f"    Metric: R² (top 5)")
            for _, row in top_models.iterrows():
                print(f"      {row['aggregation_method']:20s} | {row['feature_combo']:30s} | R²={row['r_squared']:.3f} r={row['r']:+.3f}")
        else:
            # Sort by AUC
            top_models = df_outcome.nlargest(5, 'auc')
            print(f"    Metric: AUC (top 5)")
            for _, row in top_models.iterrows():
                print(f"      {row['aggregation_method']:20s} | {row['feature_combo']:30s} | AUC={row['auc']:.3f} r={row['r']:+.3f}")

    # Best aggregation method per outcome
    print("\n" + "="*80)
    print("BEST AGGREGATION METHOD PER OUTCOME")
    print("="*80)

    for outcome_name in df_results['outcome'].unique():
        df_outcome = df_results[df_outcome['outcome'] == outcome_name]

        # Average performance by aggregation method (using 7-feature ensemble only)
        df_ensemble = df_outcome[df_outcome['feature_combo'] == '7_feature_ensemble']

        if len(df_ensemble) > 0:
            print(f"\n  {outcome_name} (7-feature ensemble):")

            if df_outcome['outcome_type'].iloc[0] == 'regression':
                best = df_ensemble.nlargest(1, 'r_squared').iloc[0]
                print(f"    Best: {best['aggregation_method']} (R²={best['r_squared']:.3f})")

                print(f"    All methods:")
                for _, row in df_ensemble.iterrows():
                    print(f"      {row['aggregation_method']:20s} R²={row['r_squared']:.3f} r={row['r']:+.3f}")
            else:
                best = df_ensemble.nlargest(1, 'auc').iloc[0]
                print(f"    Best: {best['aggregation_method']} (AUC={best['auc']:.3f})")

                print(f"    All methods:")
                for _, row in df_ensemble.iterrows():
                    print(f"      {row['aggregation_method']:20s} AUC={row['auc']:.3f} r={row['r']:+.3f}")

    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)
    print(f"\n  Total models tested: {len(df_results)}")
    print(f"  Results saved to: {output_path}")
    print("\n  Next: Generate detailed comparison report")


if __name__ == "__main__":
    main()
