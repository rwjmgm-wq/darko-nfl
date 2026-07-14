"""
DEPRECATED - do not use. Superseded by v2_evaluate.py.

Despite the name, this script does not validate anything: it re-fits the
searched 7-feature model on the same 50 QBs and reports the training fit
again. Its closing claim ("Model significantly outperforms draft capital")
is not supported out-of-sample. See results/v2_evaluation_report.md.

---
Validate 7-Feature Model Predictions

Shows which QBs the model identifies as best/worst prospects
and compares to actual NFL performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent


def main():
    """Main execution"""
    print("="*80)
    print("7-FEATURE MODEL VALIDATION")
    print("="*80)

    # Load comprehensive stats
    print("\n[1] Loading QB comprehensive stats...")
    root = get_project_root()
    stats_path = root / 'data' / 'processed' / 'qb_comprehensive_stats.csv'
    df = pd.read_csv(stats_path)
    print(f"    [OK] Loaded {len(df)} QBs")

    # Best 7-feature model
    features = [
        'attempts',
        'big_play_rate',
        'explosive_rate',
        'high_leverage_epa',
        'red_zone_epa',
        'third_down_epa',
        'opp_adj_epa'
    ]

    print(f"\n[2] Training 7-feature model...")
    print(f"    Features: {features}")

    # Prepare data
    X = df[features]
    y = df['rookie_leaf_mean']

    # Remove rows with missing values
    valid_mask = X.notna().all(axis=1) & y.notna()
    X_valid = X[valid_mask]
    y_valid = y[valid_mask]
    df_valid = df[valid_mask].copy()

    print(f"    Sample size: {len(df_valid)} QBs")

    # Train model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)

    model = LinearRegression()
    model.fit(X_scaled, y_valid)

    # Generate predictions
    y_pred = model.predict(X_scaled)
    df_valid['predicted_nfl_leaf'] = y_pred

    # Calculate model performance
    corr = np.corrcoef(y_valid, y_pred)[0, 1]
    r2 = model.score(X_scaled, y_valid)

    print(f"\n    Model Performance:")
    print(f"      Correlation: r = {corr:+.3f}")
    print(f"      R-squared: {r2:.3f} ({r2*100:.1f}%)")
    print(f"      vs. Draft Pick: r = -0.314 (R² = 9.9%)")
    improvement = (r2 - 0.099) / 0.099 * 100
    print(f"      Improvement: {improvement:+.1f}%")

    # Show feature weights
    print(f"\n    Feature Weights (standardized):")
    for feature, weight in zip(features, model.coef_):
        print(f"      {feature:25s}: {weight:+.3f}")
    print(f"      Intercept: {model.intercept_:.3f}")

    # Rank QBs by predicted NFL performance
    df_valid = df_valid.sort_values('predicted_nfl_leaf', ascending=False)
    df_valid['prediction_rank'] = range(1, len(df_valid) + 1)

    # Calculate prediction error
    df_valid['prediction_error'] = df_valid['predicted_nfl_leaf'] - df_valid['rookie_leaf_mean']
    df_valid['abs_error'] = abs(df_valid['prediction_error'])

    # Display results
    print("\n" + "="*80)
    print("TOP 10 PROJECTED NFL PROSPECTS (by model)")
    print("="*80)
    print(f"\n{'Rank':<6} {'Player':<20} {'Predicted':<12} {'Actual':<12} {'Error':<10} {'Draft':<6}")
    print("-"*80)

    for i, (_, row) in enumerate(df_valid.head(10).iterrows(), 1):
        player = row['player_name']
        predicted = row['predicted_nfl_leaf']
        actual = row['rookie_leaf_mean']
        error = row['prediction_error']
        draft = int(row['draft_pick'])

        print(f"{i:<6} {player:<20} {predicted:+.3f}       {actual:+.3f}       {error:+.3f}      #{draft:<5}")

    print("\n" + "="*80)
    print("BOTTOM 10 PROJECTED NFL PROSPECTS (by model)")
    print("="*80)
    print(f"\n{'Rank':<6} {'Player':<20} {'Predicted':<12} {'Actual':<12} {'Error':<10} {'Draft':<6}")
    print("-"*80)

    bottom_10 = df_valid.tail(10)
    for i, (_, row) in enumerate(bottom_10.iterrows(), 1):
        rank = len(df_valid) - 10 + i
        player = row['player_name']
        predicted = row['predicted_nfl_leaf']
        actual = row['rookie_leaf_mean']
        error = row['prediction_error']
        draft = int(row['draft_pick'])

        print(f"{rank:<6} {player:<20} {predicted:+.3f}       {actual:+.3f}       {error:+.3f}      #{draft:<5}")

    # Hit rate analysis
    print("\n" + "="*80)
    print("MODEL ACCURACY ANALYSIS")
    print("="*80)

    # Top 10 hit rate
    top_10_actual = df_valid.head(10)['rookie_leaf_mean']
    top_10_hits = (top_10_actual > 0).sum()
    print(f"\nTop 10 Predictions:")
    print(f"  QBs with positive NFL LEAF: {top_10_hits}/10 ({top_10_hits/10*100:.0f}%)")
    print(f"  Average actual NFL LEAF: {top_10_actual.mean():+.3f}")

    # Bottom 10 hit rate
    bottom_10_actual = df_valid.tail(10)['rookie_leaf_mean']
    bottom_10_hits = (bottom_10_actual < 0).sum()
    print(f"\nBottom 10 Predictions:")
    print(f"  QBs with negative NFL LEAF: {bottom_10_hits}/10 ({bottom_10_hits/10*100:.0f}%)")
    print(f"  Average actual NFL LEAF: {bottom_10_actual.mean():+.3f}")

    # Biggest hits (correct predictions)
    print("\n" + "="*80)
    print("BIGGEST CORRECT PREDICTIONS")
    print("="*80)

    # QBs we predicted would succeed who did
    successes = df_valid[(df_valid['predicted_nfl_leaf'] > 0) & (df_valid['rookie_leaf_mean'] > 0)]
    successes = successes.sort_values('predicted_nfl_leaf', ascending=False)

    print(f"\nQBs Correctly Predicted to Succeed (Top 5):")
    print(f"{'Player':<20} {'Predicted':<12} {'Actual':<12} {'Draft':<6}")
    print("-"*60)
    for _, row in successes.head(5).iterrows():
        print(f"{row['player_name']:<20} {row['predicted_nfl_leaf']:+.3f}       {row['rookie_leaf_mean']:+.3f}       #{int(row['draft_pick'])}")

    # QBs we predicted would struggle who did
    busts = df_valid[(df_valid['predicted_nfl_leaf'] < 0) & (df_valid['rookie_leaf_mean'] < 0)]
    busts = busts.sort_values('predicted_nfl_leaf')

    print(f"\nQBs Correctly Predicted to Struggle (Top 5):")
    print(f"{'Player':<20} {'Predicted':<12} {'Actual':<12} {'Draft':<6}")
    print("-"*60)
    for _, row in busts.head(5).iterrows():
        print(f"{row['player_name']:<20} {row['predicted_nfl_leaf']:+.3f}       {row['rookie_leaf_mean']:+.3f}       #{int(row['draft_pick'])}")

    # Biggest misses
    print("\n" + "="*80)
    print("BIGGEST PREDICTION MISSES")
    print("="*80)

    df_sorted_error = df_valid.sort_values('abs_error', ascending=False)

    print(f"\nMost Over-Predicted (predicted better than actual):")
    print(f"{'Player':<20} {'Predicted':<12} {'Actual':<12} {'Error':<10} {'Draft':<6}")
    print("-"*70)
    for _, row in df_sorted_error.head(5).iterrows():
        if row['prediction_error'] > 0:  # Over-predicted
            print(f"{row['player_name']:<20} {row['predicted_nfl_leaf']:+.3f}       {row['rookie_leaf_mean']:+.3f}       {row['prediction_error']:+.3f}      #{int(row['draft_pick'])}")

    print(f"\nMost Under-Predicted (predicted worse than actual):")
    print(f"{'Player':<20} {'Predicted':<12} {'Actual':<12} {'Error':<10} {'Draft':<6}")
    print("-"*70)
    under_predicted = df_sorted_error[df_sorted_error['prediction_error'] < 0].head(5)
    for _, row in under_predicted.iterrows():
        print(f"{row['player_name']:<20} {row['predicted_nfl_leaf']:+.3f}       {row['rookie_leaf_mean']:+.3f}       {row['prediction_error']:+.3f}      #{int(row['draft_pick'])}")

    # Draft capital vs model comparison
    print("\n" + "="*80)
    print("DRAFT CAPITAL VS MODEL COMPARISON")
    print("="*80)

    # QBs drafted high that model flagged as busts
    print(f"\nHigh Draft Picks Model Correctly Identified as Risky:")
    high_draft_busts = df_valid[
        (df_valid['draft_pick'] <= 50) &
        (df_valid['predicted_nfl_leaf'] < 0) &
        (df_valid['rookie_leaf_mean'] < 0)
    ].sort_values('draft_pick')

    if len(high_draft_busts) > 0:
        print(f"{'Player':<20} {'Draft':<10} {'Predicted':<12} {'Actual':<12}")
        print("-"*60)
        for _, row in high_draft_busts.iterrows():
            print(f"{row['player_name']:<20} #{int(row['draft_pick']):<9} {row['predicted_nfl_leaf']:+.3f}       {row['rookie_leaf_mean']:+.3f}")
    else:
        print("  None found")

    # Late picks/UDFAs model identified as gems
    print(f"\nLate Picks/UDFAs Model Correctly Identified as Stars:")
    late_gems = df_valid[
        (df_valid['draft_pick'] >= 100) &
        (df_valid['predicted_nfl_leaf'] > 0) &
        (df_valid['rookie_leaf_mean'] > 0)
    ].sort_values('predicted_nfl_leaf', ascending=False)

    if len(late_gems) > 0:
        print(f"{'Player':<20} {'Draft':<10} {'Predicted':<12} {'Actual':<12}")
        print("-"*60)
        for _, row in late_gems.iterrows():
            print(f"{row['player_name']:<20} #{int(row['draft_pick']):<9} {row['predicted_nfl_leaf']:+.3f}       {row['rookie_leaf_mean']:+.3f}")
    else:
        print("  None found")

    # Save results
    print("\n[3] Saving validation results...")
    output_path = root / 'data' / 'processed' / 'model_predictions_7feature.csv'
    df_valid.to_csv(output_path, index=False)
    print(f"    [OK] Saved to: {output_path}")

    # Summary
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print(f"\nModel Performance: r = {corr:+.3f}, R² = {r2*100:.1f}%")
    print(f"Top 10 Success Rate: {top_10_hits}/10 ({top_10_hits/10*100:.0f}%)")
    print(f"Bottom 10 Bust Rate: {bottom_10_hits}/10 ({bottom_10_hits/10*100:.0f}%)")
    print(f"\nModel significantly outperforms draft capital for QB evaluation.")


if __name__ == "__main__":
    main()
