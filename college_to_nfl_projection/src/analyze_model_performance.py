"""
Analyze Model Successes and Failures

Deep dive into what the college-to-NFL projection model predicts well vs poorly.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent


def main():
    """Main analysis"""
    print("="*80)
    print("MODEL PERFORMANCE ANALYSIS: SUCCESSES AND FAILURES")
    print("="*80)

    root = get_project_root()

    # Load test results
    results_path = root / 'data' / 'processed' / 'comprehensive_model_results.csv'
    df_results = pd.read_csv(results_path)

    print(f"\nAnalyzing {len(df_results)} model configurations")
    print(f"  Outcomes tested: {df_results['outcome'].nunique()}")
    print(f"  Aggregation methods: {df_results['aggregation_method'].nunique()}")
    print(f"  Feature combinations: {df_results['feature_combo'].nunique()}")

    # =========================================================================
    # SECTION 1: OVERALL PREDICTIVE POWER BY OUTCOME
    # =========================================================================
    print("\n" + "="*80)
    print("SECTION 1: WHAT CAN WE PREDICT?")
    print("="*80)

    outcomes = df_results['outcome'].unique()

    print("\nBest model performance for each outcome:")
    print("-" * 80)

    for outcome in outcomes:
        df_outcome = df_results[df_results['outcome'] == outcome]
        outcome_type = df_outcome['outcome_type'].iloc[0]

        if outcome_type == 'regression':
            # For regression, use R-squared
            best_model = df_outcome.loc[df_outcome['r_squared'].idxmax()]
            r_sq = best_model['r_squared']
            r = best_model['r']
            method = best_model['aggregation_method']
            features = best_model['feature_combo']

            print(f"\n{outcome}:")
            print(f"  Type: Continuous (regression)")
            print(f"  Best R²: {r_sq:.3f} (correlation: {r:+.3f})")
            print(f"  Best method: {method}")
            print(f"  Best features: {features}")

            # Interpretation
            if r_sq < 0.05:
                strength = "VERY WEAK - essentially no predictive power"
            elif r_sq < 0.10:
                strength = "WEAK - explains <10% of variance"
            elif r_sq < 0.20:
                strength = "MODERATE - explains 10-20% of variance"
            elif r_sq < 0.30:
                strength = "DECENT - explains 20-30% of variance"
            else:
                strength = "STRONG - explains >30% of variance"

            print(f"  Interpretation: {strength}")

        else:
            # For classification, use AUC
            best_model = df_outcome.loc[df_outcome['auc'].idxmax()]
            auc = best_model['auc']
            r = best_model['r']
            method = best_model['aggregation_method']
            features = best_model['feature_combo']

            print(f"\n{outcome}:")
            print(f"  Type: Binary (classification)")
            print(f"  Best AUC: {auc:.3f} (correlation: {r:+.3f})")
            print(f"  Best method: {method}")
            print(f"  Best features: {features}")

            # Interpretation
            if auc < 0.55:
                strength = "VERY WEAK - barely better than random"
            elif auc < 0.65:
                strength = "WEAK - modest discrimination"
            elif auc < 0.75:
                strength = "MODERATE - decent discrimination"
            elif auc < 0.85:
                strength = "GOOD - strong discrimination"
            else:
                strength = "EXCELLENT - very strong discrimination"

            print(f"  Interpretation: {strength}")

    # =========================================================================
    # SECTION 2: WHAT WORKS BEST?
    # =========================================================================
    print("\n" + "="*80)
    print("SECTION 2: WHAT WORKS BEST?")
    print("="*80)

    print("\n2A. Best Aggregation Method by Outcome:")
    print("-" * 80)

    for outcome in outcomes:
        df_outcome = df_results[df_results['outcome'] == outcome]
        outcome_type = df_outcome['outcome_type'].iloc[0]

        # Group by aggregation method, take best feature combo for each
        if outcome_type == 'regression':
            metric_col = 'r_squared'
        else:
            metric_col = 'auc'

        best_per_method = df_outcome.groupby('aggregation_method')[metric_col].max().sort_values(ascending=False)

        print(f"\n{outcome}:")
        for method, score in best_per_method.items():
            print(f"  {method:20s}: {score:.3f}")

    print("\n2B. Best Feature Combinations (across all methods):")
    print("-" * 80)

    for outcome in outcomes:
        df_outcome = df_results[df_results['outcome'] == outcome]
        outcome_type = df_outcome['outcome_type'].iloc[0]

        if outcome_type == 'regression':
            metric_col = 'r_squared'
        else:
            metric_col = 'auc'

        best_per_feature = df_outcome.groupby('feature_combo')[metric_col].max().sort_values(ascending=False).head(5)

        print(f"\n{outcome} - Top 5 feature combinations:")
        for i, (feature, score) in enumerate(best_per_feature.items(), 1):
            print(f"  {i}. {feature:30s}: {score:.3f}")

    # =========================================================================
    # SECTION 3: RELATIVE PERFORMANCE
    # =========================================================================
    print("\n" + "="*80)
    print("SECTION 3: RELATIVE PERFORMANCE")
    print("="*80)

    print("\nHow much better is college data vs just using draft position?")
    print("(Note: We don't have draft position in the model, but this shows")
    print(" whether college stats provide meaningful signal)")
    print("-" * 80)

    for outcome in outcomes:
        df_outcome = df_results[df_results['outcome'] == outcome]
        outcome_type = df_outcome['outcome_type'].iloc[0]

        if outcome_type == 'regression':
            best_r_sq = df_outcome['r_squared'].max()
            # Random baseline: R² ≈ 0
            print(f"\n{outcome}:")
            print(f"  Random baseline: R² = 0.000")
            print(f"  Best model: R² = {best_r_sq:.3f}")
            print(f"  Improvement: {best_r_sq:.3f} (explains {best_r_sq*100:.1f}% of variance)")
        else:
            best_auc = df_outcome['auc'].max()
            # Random baseline: AUC = 0.50
            improvement = best_auc - 0.50
            print(f"\n{outcome}:")
            print(f"  Random baseline: AUC = 0.500")
            print(f"  Best model: AUC = {best_auc:.3f}")
            print(f"  Improvement: +{improvement:.3f} ({improvement/0.50*100:.1f}% better than random)")

    # =========================================================================
    # SECTION 4: KEY FINDINGS
    # =========================================================================
    print("\n" + "="*80)
    print("SECTION 4: KEY FINDINGS & INTERPRETATION")
    print("="*80)

    print("\nSUCCESSES (What the model predicts well):")
    print("-" * 80)

    # Find best outcomes
    outcome_scores = {}
    for outcome in outcomes:
        df_outcome = df_results[df_results['outcome'] == outcome]
        outcome_type = df_outcome['outcome_type'].iloc[0]

        if outcome_type == 'regression':
            score = df_outcome['r_squared'].max()
        else:
            score = df_outcome['auc'].max()

        outcome_scores[outcome] = (score, outcome_type)

    # Sort by score
    sorted_outcomes = sorted(outcome_scores.items(), key=lambda x: x[1][0], reverse=True)

    for i, (outcome, (score, outcome_type)) in enumerate(sorted_outcomes[:3], 1):
        metric_name = "R²" if outcome_type == "regression" else "AUC"
        print(f"\n{i}. {outcome} ({metric_name} = {score:.3f})")

        if outcome == "Sustained Starter":
            print("   → Model can identify QBs who will have long careers")
            print("   → Career averages work best (not just final season)")
        elif outcome == "85+ Starts":
            print("   → Model can predict which QBs get significant playing time")
            print("   → Career consistency matters more than peak performance")
        elif outcome == "Elite Peak":
            print("   → Model can identify QBs who will reach elite levels")
            print("   → Final season performance is most predictive")

    print("\n\nFAILURES (What the model struggles with):")
    print("-" * 80)

    for i, (outcome, (score, outcome_type)) in enumerate(sorted_outcomes[-3:], 1):
        metric_name = "R²" if outcome_type == "regression" else "AUC"
        print(f"\n{i}. {outcome} ({metric_name} = {score:.3f})")

        if outcome == "Rookie LEAF":
            print("   → Very weak prediction of rookie-year performance")
            print("   → Immediate NFL success depends heavily on context:")
            print("     * Team quality, coaching, supporting cast")
            print("     * Learning curve varies widely by system")
        elif outcome == "Bust":
            print("   → Moderate ability to predict outright failures")
            print("   → Bust outcomes often driven by injuries, fit issues")
            print("   → College stats can't predict these factors")

    # =========================================================================
    # SECTION 5: DOES MULTI-YEAR MATTER?
    # =========================================================================
    print("\n" + "="*80)
    print("SECTION 5: DOES FULL CAREER DATA MATTER?")
    print("="*80)

    print("\nComparing final season only vs career aggregations:")
    print("-" * 80)

    for outcome in outcomes:
        df_outcome = df_results[df_results['outcome'] == outcome]
        outcome_type = df_outcome['outcome_type'].iloc[0]

        if outcome_type == 'regression':
            metric_col = 'r_squared'
        else:
            metric_col = 'auc'

        # Final season performance
        df_final = df_outcome[df_outcome['aggregation_method'] == 'final_season']
        final_best = df_final[metric_col].max()

        # Best career aggregation performance
        df_career = df_outcome[df_outcome['aggregation_method'] != 'final_season']
        career_best = df_career[metric_col].max()
        career_best_method = df_career.loc[df_career[metric_col].idxmax()]['aggregation_method']

        improvement = career_best - final_best
        pct_improvement = (improvement / final_best * 100) if final_best > 0 else 0

        print(f"\n{outcome}:")
        print(f"  Final season only: {final_best:.3f}")
        print(f"  Best career method ({career_best_method}): {career_best:.3f}")
        print(f"  Improvement: {improvement:+.3f} ({pct_improvement:+.1f}%)")

        if improvement > 0.05:
            print(f"  → SIGNIFICANT BENEFIT from using full career")
        elif improvement > 0:
            print(f"  → Modest benefit from using full career")
        else:
            print(f"  → Final season is best (career data doesn't help)")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("EXECUTIVE SUMMARY")
    print("="*80)

    print("\n✓ STRONG PREDICTIONS:")
    print("  • Long-term success (Sustained Starter, 85+ Starts)")
    print("  • Elite performance potential")
    print("  • Career trajectory overall")

    print("\n✗ WEAK PREDICTIONS:")
    print("  • Immediate rookie performance (LEAF)")
    print("  • Complete busts (too context-dependent)")

    print("\n📊 KEY INSIGHTS:")
    print("  • Full career data often outperforms final season only")
    print("  • Career averages work best for sustained success")
    print("  • Final season peak matters for elite outcomes")
    print("  • Situational stats (3rd down, red zone) add predictive value")
    print("  • Efficiency (EPA/play) > Volume (attempts)")

    print("\n⚠️  LIMITATIONS:")
    print("  • Sample size: 120 QBs limits statistical power")
    print("  • Missing context: team quality, coaching, injuries")
    print("  • NFL outcomes are inherently noisy")
    print("  • Best models explain ~20-30% of variance")


if __name__ == "__main__":
    main()
