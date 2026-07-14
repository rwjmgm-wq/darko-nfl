"""
DEPRECATED - do not use.

This script selects the TARGET variable (which NFL-performance weighting to
predict) by maximizing in-sample agreement with the college model - selecting
on the dependent variable. The p-values it prints are computed on the searched
maximum and are meaningless. The v2 pipeline fixes the target a priori.

---
Optimize NFL Performance Weighting Scheme

Tests which weighting scheme for first 17 career games correlates
best with the 7-feature college model predictions.

This determines the optimal way to measure NFL success from early career data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent


def main():
    """Main execution"""
    print("="*80)
    print("WEIGHTING SCHEME OPTIMIZATION")
    print("="*80)
    print("\nFinding which NFL performance metric is most predictable from college")

    root = get_project_root()

    # Load first 17 games NFL data
    print("\n[1] Loading NFL first 17 games data...")
    nfl_path = root / 'data' / 'processed' / 'nfl_first_17_games.csv'
    df_nfl = pd.read_csv(nfl_path)
    print(f"    [OK] {len(df_nfl)} QBs with 17+ career games")

    # Load college comprehensive stats
    print("\n[2] Loading college comprehensive stats...")
    college_path = root / 'data' / 'processed' / 'qb_comprehensive_stats.csv'
    df_college = pd.read_csv(college_path)
    print(f"    [OK] {len(df_college)} QBs with college stats")

    # Merge datasets
    print("\n[3] Merging college and NFL data...")
    df_merged = df_college.merge(
        df_nfl,
        on='player_name',
        how='inner',
        suffixes=('_college', '_nfl')
    )
    print(f"    [OK] {len(df_merged)} QBs with both college and NFL data")

    # 7-feature model
    features = [
        'attempts',
        'big_play_rate',
        'explosive_rate',
        'high_leverage_epa',
        'red_zone_epa',
        'third_down_epa',
        'opp_adj_epa'
    ]

    # Prepare college feature data
    X = df_merged[features]
    valid_mask = X.notna().all(axis=1)
    df_valid = df_merged[valid_mask].copy()
    X_valid = X[valid_mask]

    print(f"\n    Analysis sample: {len(df_valid)} QBs")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)

    # Test each weighting scheme
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)

    weighting_schemes = [
        'uniform',
        'linear_2x',
        'linear_3x',
        'exponential',
        'back_half',
        'final_third'
    ]

    results = []

    for weighting in weighting_schemes:
        col = f'leaf_{weighting}'

        if col not in df_valid.columns:
            continue

        y = df_valid[col]

        if y.notna().sum() < 10:
            continue

        # Train linear model
        model = LinearRegression()
        model.fit(X_scaled, y)

        # Predictions
        y_pred = model.predict(X_scaled)

        # Calculate correlation
        corr = np.corrcoef(y, y_pred)[0, 1]
        r2 = model.score(X_scaled, y)

        # Calculate p-value
        t_stat = corr * np.sqrt(len(y) - 2) / np.sqrt(1 - corr**2)
        p_value = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), len(y) - 2))

        results.append({
            'weighting': weighting,
            'correlation': corr,
            'r_squared': r2,
            'p_value': p_value,
            'n': len(y)
        })

        print(f"\n{weighting.upper():20s}")
        print(f"  Correlation: r = {corr:+.3f}")
        print(f"  R-squared:   {r2:.3f} ({r2*100:.1f}%)")
        print(f"  P-value:     {p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''}")
        print(f"  Sample size: n = {len(y)}")

    df_results = pd.DataFrame(results)

    # Rank by correlation
    df_results = df_results.sort_values('r_squared', ascending=False)

    # Summary
    print("\n" + "="*80)
    print("OPTIMAL WEIGHTING SCHEME")
    print("="*80)

    best = df_results.iloc[0]
    print(f"\nBest predictor of college model: {best['weighting'].upper()}")
    print(f"  Correlation: r = {best['correlation']:+.3f}")
    print(f"  R-squared:   {best['r_squared']:.3f} ({best['r_squared']*100:.1f}%)")
    print(f"  P-value:     {best['p_value']:.4f}")

    # Comparison table
    print("\n" + "="*80)
    print("FULL COMPARISON (sorted by R²)")
    print("="*80)
    print(f"\n{'Weighting':<20} {'Correlation':>12} {'R²':>8} {'% Variance':>12} {'P-value':>10}")
    print("-"*80)

    for _, row in df_results.iterrows():
        significance = ''
        if row['p_value'] < 0.001:
            significance = '***'
        elif row['p_value'] < 0.01:
            significance = '**'
        elif row['p_value'] < 0.05:
            significance = '*'

        print(f"{row['weighting']:<20} {row['correlation']:>+12.3f} {row['r_squared']:>8.3f} {row['r_squared']*100:>11.1f}% {row['p_value']:>9.4f} {significance}")

    # Improvement over baseline
    print("\n" + "="*80)
    print("IMPROVEMENT OVER ROOKIE SEASON BASELINE")
    print("="*80)

    # Need to check correlation with original rookie_leaf_mean
    if 'rookie_leaf_mean' in df_valid.columns:
        y_rookie = df_valid['rookie_leaf_mean']
        model_rookie = LinearRegression()
        model_rookie.fit(X_scaled, y_rookie)
        y_pred_rookie = model_rookie.predict(X_scaled)
        corr_rookie = np.corrcoef(y_rookie, y_pred_rookie)[0, 1]
        r2_rookie = model_rookie.score(X_scaled, y_rookie)

        print(f"\nRookie Season LEAF (original target):")
        print(f"  Correlation: r = {corr_rookie:+.3f}")
        print(f"  R-squared:   {r2_rookie:.3f} ({r2_rookie*100:.1f}%)")

        print(f"\nBest 17-game weighting ({best['weighting']}):")
        print(f"  Correlation: r = {best['correlation']:+.3f}")
        print(f"  R-squared:   {best['r_squared']:.3f} ({best['r_squared']*100:.1f}%)")

        if best['r_squared'] > r2_rookie:
            improvement = (best['r_squared'] - r2_rookie) / r2_rookie * 100
            print(f"\n  [SUCCESS] {improvement:+.1f}% improvement over rookie season!")
        else:
            decline = (r2_rookie - best['r_squared']) / r2_rookie * 100
            print(f"\n  [--] {decline:.1f}% decline vs rookie season")

    # Save results
    print("\n[4] Saving optimization results...")
    output_path = root / 'data' / 'processed' / 'weighting_optimization_results.csv'
    df_results.to_csv(output_path, index=False)
    print(f"    [OK] Saved to: {output_path}")

    # Specific examples
    print("\n" + "="*80)
    print("SPECIFIC EXAMPLES")
    print("="*80)

    examples = ['Drew Lock', 'Patrick Mahomes', 'Dak Prescott', 'Baker Mayfield']

    for example_name in examples:
        example = df_valid[df_valid['player_name'] == example_name]
        if len(example) > 0:
            print(f"\n{example_name}:")
            for weighting in weighting_schemes:
                col = f'leaf_{weighting}'
                if col in example.columns:
                    value = example[col].iloc[0]
                    print(f"  {weighting:20s}: {value:+.3f}")
        else:
            print(f"\n{example_name}: Not found in dataset")

    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"\nRecommendation: Use {best['weighting'].upper()} weighting for NFL evaluation")
    print(f"This maximizes correlation with college predictive model (r = {best['correlation']:+.3f})")


if __name__ == "__main__":
    main()
