"""
Check Historical Data Availability

Tests how far back we can get college and NFL data for the projection model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import cfbd
from cfbd.rest import ApiException
import nfl_data_py as nfl


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent


def check_cfbd_availability():
    """Check which years have college play-by-play data available"""
    print("\n" + "="*80)
    print("COLLEGE FOOTBALL DATABASE (CFBD) API AVAILABILITY")
    print("="*80)

    # Load API key
    root = get_project_root()
    api_key_file = root / '.env'

    if not api_key_file.exists():
        print("\n[ERROR] No .env file found")
        return []

    with open(api_key_file) as f:
        for line in f:
            if line.startswith('CFBD_API_KEY='):
                api_key = line.strip().split('=')[1]
                break

    # Configure API
    configuration = cfbd.Configuration()
    configuration.api_key['Authorization'] = api_key
    configuration.api_key_prefix['Authorization'] = 'Bearer'

    # Test years from 1999 to 2023
    test_years = range(1999, 2024)
    available_years = []

    print("\nTesting play-by-play data availability by year:")
    print("-"*80)

    api_instance = cfbd.PlaysApi(cfbd.ApiClient(configuration))

    for year in test_years:
        try:
            # Try to get a small sample of plays
            plays = api_instance.get_plays(
                year=year,
                week=1,
                team='Alabama'  # Use consistent team
            )

            if plays and len(plays) > 0:
                available_years.append(year)
                print(f"  {year}: ✓ Available ({len(plays)} plays in week 1)")
            else:
                print(f"  {year}: ✗ No data")

        except ApiException as e:
            print(f"  {year}: ✗ API error")
        except Exception as e:
            print(f"  {year}: ✗ Error: {str(e)[:50]}")

    print(f"\n[SUMMARY] College PBP available: {min(available_years)} - {max(available_years)}")
    print(f"          Total years: {len(available_years)}")

    return available_years


def check_sp_plus_availability():
    """Check SP+ ratings availability from cached data"""
    print("\n" + "="*80)
    print("SP+ RATINGS AVAILABILITY (from cached betting model)")
    print("="*80)

    betting_model_path = Path("C:/Users/rwjmg/OneDrive/Pictures/Writing/new_cfb_betting_model")

    if not betting_model_path.exists():
        print("\n[ERROR] Betting model path not found")
        return []

    import glob

    # Find all cached SP+ files
    pattern = f"{betting_model_path}/data/raw/sp_plus_*_week*.csv"
    files = glob.glob(pattern)

    # Extract years
    years = set()
    for file in files:
        # Extract year from filename like "sp_plus_2020_week12.csv"
        import re
        match = re.search(r'sp_plus_(\d{4})_week', file)
        if match:
            years.add(int(match.group(1)))

    years = sorted(years)

    print(f"\nSP+ cached data available for years: {years}")
    print(f"Range: {min(years)} - {max(years)}" if years else "No data found")

    # Check official SP+ history
    print("\n[NOTE] Bill Connelly's SP+ ratings:")
    print("  - Official SP+ calculated back to 2005")
    print("  - FEI (predecessor) started in 2007")
    print("  - Public API access may be limited")

    return years


def check_nfl_data_availability():
    """Check NFL play-by-play data availability"""
    print("\n" + "="*80)
    print("NFL PLAY-BY-PLAY DATA AVAILABILITY (nfl_data_py)")
    print("="*80)

    print("\nTesting NFL data by year (this may take a moment)...")
    print("-"*80)

    # Test sample years
    test_years = [1999, 2000, 2005, 2010, 2015, 2020, 2023]
    available_years = []

    for year in test_years:
        try:
            print(f"  Testing {year}...", end=' ', flush=True)
            df = nfl.import_pbp_data([year], downcast=False)

            if df is not None and len(df) > 0:
                available_years.append(year)
                qb_plays = df[df['passer_player_name'].notna()]
                print(f"✓ ({len(qb_plays):,} QB plays)")
            else:
                print(f"✗ No data")

        except Exception as e:
            print(f"✗ Error: {str(e)[:50]}")

    print(f"\n[SUMMARY] nfl_data_py (nflfastR) provides data from 1999-present")
    print(f"          EPA calculations available for all years")
    print(f"          Sample years tested: {available_years}")

    return available_years


def check_draft_data_availability():
    """Check NFL draft data availability"""
    print("\n" + "="*80)
    print("NFL DRAFT DATA AVAILABILITY")
    print("="*80)

    # NFL draft data is widely available
    print("\n[INFO] NFL Draft data is available from:")
    print("  - Pro Football Reference: 1936-present")
    print("  - Sports Reference API: 1936-present")
    print("  - Our current dataset: 2015-2023")

    root = get_project_root()
    current_data = root / 'data' / 'processed' / 'merged_with_sp_plus.csv'

    if current_data.exists():
        df = pd.read_csv(current_data)
        years = sorted(df['draft_year'].unique())
        print(f"\n  Current QBs in dataset: {len(df)}")
        print(f"  Draft years covered: {min(years)} - {max(years)}")


def main():
    """Main execution"""
    print("="*80)
    print("HISTORICAL DATA AVAILABILITY CHECK")
    print("="*80)
    print("\nChecking how far back we can extend the college-to-NFL projection model")

    # Check each data source
    cfbd_years = check_cfbd_availability()
    sp_plus_years = check_sp_plus_availability()
    nfl_years = check_nfl_data_availability()
    check_draft_data_availability()

    # Summary and recommendations
    print("\n" + "="*80)
    print("PRACTICAL LIMITS FOR MODEL EXTENSION")
    print("="*80)

    print("\n[1] OPTIMAL RANGE (all data sources available):")
    if cfbd_years and sp_plus_years:
        optimal_start = max(min(cfbd_years), min(sp_plus_years))
        optimal_end = min(max(cfbd_years), max(sp_plus_years))
        print(f"    {optimal_start} - {optimal_end}")
        print(f"    • College PBP: ✓")
        print(f"    • SP+ ratings: ✓")
        print(f"    • NFL outcomes: ✓ (1999-present)")

    print("\n[2] WITHOUT SP+ ADJUSTMENT:")
    if cfbd_years:
        print(f"    {min(cfbd_years)} - {max(cfbd_years)}")
        print(f"    • Can use raw EPA without opponent adjustment")
        print(f"    • Model will be less accurate without competition context")

    print("\n[3] CURRENT DATASET:")
    print(f"    2015-2023 (QBs drafted in these years)")
    print(f"    • College data: 2011-2022 (their final college seasons)")
    print(f"    • Sample size: 62 QBs")

    print("\n[4] RECOMMENDED EXPANSION:")
    if cfbd_years and sp_plus_years:
        optimal_start = max(min(cfbd_years), min(sp_plus_years))
        expansion_years = optimal_end - 2015 + 1  # Current starts at 2015
        backward_years = 2015 - optimal_start

        print(f"    Expand backward to {optimal_start} (add ~{backward_years * 8} more QBs)")
        print(f"    • Significantly larger sample size")
        print(f"    • More robust statistical validation")
        print(f"    • Better coverage of different eras")

    print("\n[5] DATA QUALITY NOTES:")
    print(f"    • Earlier years (2001-2010) may have:")
    print(f"      - Less complete play-by-play coverage")
    print(f"      - Fewer tracked teams")
    print(f"      - Different EPA calculation methods")
    print(f"    • Recommend starting no earlier than 2005 for consistency")

    print("\n" + "="*80)
    print("EXPANSION RECOMMENDATION")
    print("="*80)

    if cfbd_years and sp_plus_years:
        recommended_start = max(min(cfbd_years), min(sp_plus_years), 2005)
        print(f"\n✓ RECOMMENDED: Expand dataset to {recommended_start}-2023")
        print(f"  • QBs drafted {recommended_start}-2023")
        print(f"  • College data from ~{recommended_start-4} onwards")
        print(f"  • Estimated {(2023 - recommended_start + 1) * 8} total QBs")
        print(f"  • {(2023 - recommended_start + 1) * 8 - 62} additional QBs beyond current dataset")


if __name__ == "__main__":
    main()
