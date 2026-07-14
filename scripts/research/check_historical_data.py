"""Check nflfastR data availability for earlier years."""

import nfl_data_py as nfl
import pandas as pd

print("="*80)
print("nflfastR Historical Data Availability Check")
print("="*80)

# Test years from 1999 to 2016
test_years = [1999, 2000, 2006, 2009, 2010, 2012, 2015, 2016]

print("\nChecking which years have the features we need for QB LEAF...\n")
print(f"{'Year':<6} {'Pass Plays':>12} {'EPA':>6} {'QB_EPA':>8} {'CPOE':>6} {'Success':>8}")
print("-"*60)

results = []

for year in test_years:
    try:
        # Load play-by-play data for this year
        df = nfl.import_pbp_data([year], downcast=False)

        # Check key columns we need
        has_epa = 'epa' in df.columns
        has_qb_epa = 'qb_epa' in df.columns
        has_cpoe = 'cpoe' in df.columns
        has_success = 'success' in df.columns

        # Count pass plays
        qb_plays = len(df[df['pass'] == 1])

        # Check if EPA values are actually populated (not all NaN)
        if has_epa:
            epa_populated = df['epa'].notna().sum() / len(df)
        else:
            epa_populated = 0.0

        if has_cpoe:
            cpoe_populated = df['cpoe'].notna().sum() / len(df[df['pass'] == 1])
        else:
            cpoe_populated = 0.0

        results.append({
            'year': year,
            'pass_plays': qb_plays,
            'has_epa': has_epa,
            'has_qb_epa': has_qb_epa,
            'has_cpoe': has_cpoe,
            'has_success': has_success,
            'epa_populated': epa_populated,
            'cpoe_populated': cpoe_populated
        })

        print(f"{year:<6} {qb_plays:>12} {str(has_epa):>6} {str(has_qb_epa):>8} {str(has_cpoe):>6} {str(has_success):>8}")

    except Exception as e:
        print(f"{year:<6} ERROR: {str(e)[:40]}")
        results.append({
            'year': year,
            'error': str(e)
        })

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

results_df = pd.DataFrame([r for r in results if 'error' not in r])

if len(results_df) > 0:
    print(f"\nData quality by year:")
    print(f"\n{'Year':<6} {'EPA Populated':>15} {'CPOE Populated':>16}")
    print("-"*40)
    for _, row in results_df.iterrows():
        print(f"{row['year']:<6} {row['epa_populated']:>14.1%} {row['cpoe_populated']:>15.1%}")

    # Determine earliest usable year
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    # For QB LEAF, we need: EPA, QB_EPA, Success (minimum)
    # CPOE is nice to have but not critical

    usable_years = results_df[
        (results_df['has_epa']) &
        (results_df['has_qb_epa']) &
        (results_df['has_success']) &
        (results_df['epa_populated'] > 0.5)
    ]

    if len(usable_years) > 0:
        earliest_year = usable_years['year'].min()
        print(f"\nEarliest fully usable year: {earliest_year}")
        print(f"  Has EPA: Yes")
        print(f"  Has QB_EPA: Yes")
        print(f"  Has Success: Yes")
        print(f"  Data quality: Good (>50% populated)")

        current_start = 2016
        years_gained = current_start - earliest_year

        print(f"\nCurrent dataset: {current_start}-2025 (10 seasons)")
        print(f"Potential dataset: {earliest_year}-2025 ({2025-earliest_year+1} seasons)")
        print(f"Additional years: +{years_gained} years")

        # Estimate additional training samples
        avg_samples_per_year = 6410 / 10  # Current dataset
        estimated_new_samples = years_gained * avg_samples_per_year

        print(f"\nEstimated impact:")
        print(f"  Current training samples: ~3,052")
        print(f"  Additional samples: ~{int(estimated_new_samples * 0.5):,} (rough estimate)")
        print(f"  New total: ~{int(3052 + estimated_new_samples * 0.5):,}")

        print(f"\nBenefit for ML models:")
        print(f"  More training data generally improves predictive power")
        print(f"  Especially valuable for capturing different QB archetypes")
        print(f"  Would include era effects (rule changes, offensive evolution)")

    else:
        print("\nNo fully usable years found with complete feature set.")
        print("Current dataset (2016-2025) may already be optimal.")
else:
    print("\nNo valid results to analyze.")

print("\n" + "="*80)
