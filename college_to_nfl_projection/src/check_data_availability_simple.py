"""
Check Historical Data Availability (Simple Version)

Documents how far back we can extend the college-to-NFL projection model
based on known data source limitations.
"""

import pandas as pd
from pathlib import Path
import glob


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent


def main():
    """Main execution"""
    print("="*80)
    print("HISTORICAL DATA AVAILABILITY ASSESSMENT")
    print("="*80)
    print("\nHow far back can we extend the college-to-NFL projection model?")

    # College play-by-play data (CFBD)
    print("\n" + "="*80)
    print("1. COLLEGE PLAY-BY-PLAY DATA (CFBD API)")
    print("="*80)
    print("\nCollege Football Database API coverage:")
    print("  - Play-by-play data: 2001-present")
    print("  - Best coverage: 2004-present (more complete)")
    print("  - Optimal start: 2007-present (most reliable)")
    print("\n  Early years (2001-2006) have:")
    print("    - Fewer teams tracked")
    print("    - Less complete play-by-play")
    print("    - Missing situational data")

    # SP+ ratings
    print("\n" + "="*80)
    print("2. SP+ OPPONENT STRENGTH RATINGS")
    print("="*80)

    betting_model_path = Path("C:/Users/rwjmg/OneDrive/Pictures/Writing/new_cfb_betting_model")

    if betting_model_path.exists():
        pattern = f"{betting_model_path}/data/raw/sp_plus_*_week*.csv"
        files = glob.glob(pattern)

        # Extract years
        years = set()
        for file in files:
            import re
            match = re.search(r'sp_plus_(\d{4})_week', file)
            if match:
                years.add(int(match.group(1)))

        years = sorted(years)

        print(f"\nCached SP+ data (from betting model):")
        print(f"  - Available years: {years}")
        print(f"  - Range: {min(years)}-{max(years)}")
    else:
        print("\n[NOTE] Betting model path not found")

    print("\nOfficial SP+ history:")
    print("  - SP+ ratings: 2005-present (Bill Connelly)")
    print("  - FEI (predecessor): 2007-present")
    print("  - Public availability: varies by source")
    print("\n  Recommendation: 2014+ for reliable cached data")

    # NFL outcomes
    print("\n" + "="*80)
    print("3. NFL OUTCOME DATA")
    print("="*80)
    print("\nnfl_data_py (nflfastR) coverage:")
    print("  - Play-by-play data: 1999-present")
    print("  - EPA calculations: 1999-present")
    print("  - Reliable for all modern QBs")
    print("\n  No limitation here - any QB from 1999+ works")

    # Draft data
    print("\n" + "="*80)
    print("4. NFL DRAFT DATA")
    print("="*80)
    print("\nDraft data coverage:")
    print("  - Available: 1936-present")
    print("  - Widely accessible")
    print("\n  No limitation - can get any draft pick")

    # Current dataset
    print("\n" + "="*80)
    print("5. CURRENT DATASET")
    print("="*80)

    root = get_project_root()
    current_data = root / 'data' / 'processed' / 'merged_with_sp_plus.csv'

    if current_data.exists():
        df = pd.read_csv(current_data)
        years = sorted(df['draft_year'].unique())
        print(f"\nCurrent QBs in model:")
        print(f"  - Total QBs: {len(df)}")
        print(f"  - Draft years: {min(years)}-{max(years)}")
        print(f"  - College seasons: approximately {min(years)-4}-{max(years)-1}")

        # Show distribution by year
        print(f"\n  QBs per draft year:")
        year_counts = df['draft_year'].value_counts().sort_index()
        for year in sorted(years):
            count = year_counts.get(year, 0)
            print(f"    {year}: {count} QBs")

    # Bottleneck analysis
    print("\n" + "="*80)
    print("BOTTLENECK ANALYSIS")
    print("="*80)

    print("\nData source constraints (from most to least limiting):")
    print("\n  1. SP+ ratings (cached): 2014-2021")
    print("     -> MOST LIMITING with current cache")
    print("\n  2. College PBP quality: 2007+ recommended")
    print("     -> Some constraint on early years")
    print("\n  3. College PBP availability: 2001+")
    print("     -> Minor constraint")
    print("\n  4. NFL outcomes: 1999+")
    print("     -> No constraint")
    print("\n  5. Draft data: 1936+")
    print("     -> No constraint")

    # Recommendations
    print("\n" + "="*80)
    print("EXPANSION RECOMMENDATIONS")
    print("="*80)

    print("\nOPTION 1: Maximum backward expansion (with current cache)")
    print("  - Extend to 2007 draft class")
    print("  - Uses cached SP+ (2014-2021 seasons)")
    print("  - College data: 2003-2021 (QB final college seasons)")
    print("  - Estimated QBs: ~8 per year x 17 years = 136 QBs")
    print("  - Gain: +74 QBs (136 - 62 current)")
    print("\n  PROS:")
    print("    + Significantly larger sample")
    print("    + Includes more historical cases")
    print("    + Better statistical power")
    print("\n  CONS:")
    print("    - Early college years (2003-2013) lack SP+ adjustment")
    print("    - Would need raw EPA for those years")
    print("    - Mixed quality data")

    print("\n\nOPTION 2: Conservative expansion (SP+ coverage only)")
    print("  - Extend to 2010 draft class")
    print("  - Full SP+ coverage (2006-2021 for college seasons)")
    print("  - Estimated QBs: ~8 per year x 14 years = 112 QBs")
    print("  - Gain: +50 QBs")
    print("\n  PROS:")
    print("    + All QBs have SP+ adjustment")
    print("    + Consistent data quality")
    print("    + Easier to implement")
    print("\n  CONS:")
    print("    - Doesn't use oldest available data")
    print("    - Moderate sample size gain")

    print("\n\nOPTION 3: Fetch additional SP+ data")
    print("  - Get SP+ for 2005-2013 from other sources")
    print("  - Options:")
    print("    a) CFBD API (has SP+ endpoint)")
    print("    b) Sports Reference")
    print("    c) Archived Bill Connelly data")
    print("  - Then extend to 2007+ draft classes")
    print("\n  PROS:")
    print("    + Maximum historical coverage")
    print("    + Consistent SP+ adjustment throughout")
    print("    + Largest possible sample")
    print("\n  CONS:")
    print("    - Requires additional data collection")
    print("    - Time investment")

    print("\n\nOPTION 4: No SP+ for early years")
    print("  - Extend to 2007 draft class")
    print("  - Use raw EPA (no opponent adjustment) for 2003-2013 college seasons")
    print("  - Use SP+ adjusted for 2014-2021 college seasons")
    print("  - Mixed approach")
    print("\n  PROS:")
    print("    + Maximum sample size without extra data collection")
    print("    + Some historical depth")
    print("\n  CONS:")
    print("    - Inconsistent methodology")
    print("    - May introduce bias")

    # Final recommendation
    print("\n" + "="*80)
    print("RECOMMENDED APPROACH")
    print("="*80)

    print("\nBest option: OPTION 3 (Fetch additional SP+ data)")
    print("\nImplementation:")
    print("  1. Use CFBD API to get SP+ ratings for 2005-2013")
    print("  2. Extend dataset to 2007 draft class")
    print("  3. Full opponent adjustment for all years")
    print("\n  Final dataset:")
    print("    - Draft years: 2007-2023 (17 years)")
    print("    - College seasons: 2003-2022")
    print("    - Estimated ~136 QBs (vs 62 currently)")
    print("    - +119% increase in sample size")
    print("\n  Impact on model:")
    print("    - More robust statistical validation")
    print("    - Better confidence intervals")
    print("    - Cross-era validation")
    print("    - Stronger conclusions")

    print("\nIf SP+ fetch is blocked by API limits:")
    print("  -> Fall back to OPTION 2 (2010+ only)")
    print("  -> Still gain 50 QBs (+81% sample)")

    print("\n" + "="*80)
    print("ASSESSMENT COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
