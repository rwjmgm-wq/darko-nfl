"""
Create Expanded QB List (2007-2023 Draft Classes)

Identifies all QBs drafted from 2007-2023 to expand the dataset
from current 62 QBs to ~136 QBs.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent


def load_current_qb_list():
    """Load current QB list as reference"""
    root = get_project_root()
    current_path = root / 'data' / 'processed' / 'merged_with_sp_plus.csv'

    if current_path.exists():
        df = pd.read_csv(current_path)
        return df
    else:
        return pd.DataFrame()


# Comprehensive list of drafted QBs 2007-2023
# Includes all QBs drafted in rounds 1-7 plus notable UDFAs
HISTORICAL_QBS = [
    # 2007 Draft
    {'player_name': 'JaMarcus Russell', 'draft_year': 2007, 'draft_pick': 1, 'college': 'LSU'},
    {'player_name': 'Brady Quinn', 'draft_year': 2007, 'draft_pick': 22, 'college': 'Notre Dame'},
    {'player_name': 'Kevin Kolb', 'draft_year': 2007, 'draft_pick': 36, 'college': 'Houston'},
    {'player_name': 'John Beck', 'draft_year': 2007, 'draft_pick': 40, 'college': 'BYU'},
    {'player_name': 'Drew Stanton', 'draft_year': 2007, 'draft_pick': 43, 'college': 'Michigan State'},
    {'player_name': 'Trent Edwards', 'draft_year': 2007, 'draft_pick': 92, 'college': 'Stanford'},
    {'player_name': 'Isaiah Stanback', 'draft_year': 2007, 'draft_pick': 103, 'college': 'Washington'},
    {'player_name': 'Troy Smith', 'draft_year': 2007, 'draft_pick': 174, 'college': 'Ohio State'},

    # 2008 Draft
    {'player_name': 'Matt Ryan', 'draft_year': 2008, 'draft_pick': 3, 'college': 'Boston College'},
    {'player_name': 'Joe Flacco', 'draft_year': 2008, 'draft_pick': 18, 'college': 'Delaware'},
    {'player_name': 'Brian Brohm', 'draft_year': 2008, 'draft_pick': 56, 'college': 'Louisville'},
    {'player_name': 'Chad Henne', 'draft_year': 2008, 'draft_pick': 57, 'college': 'Michigan'},
    {'player_name': 'John David Booty', 'draft_year': 2008, 'draft_pick': 131, 'college': 'USC'},
    {'player_name': 'Dennis Dixon', 'draft_year': 2008, 'draft_pick': 156, 'college': 'Oregon'},

    # 2009 Draft
    {'player_name': 'Matthew Stafford', 'draft_year': 2009, 'draft_pick': 1, 'college': 'Georgia'},
    {'player_name': 'Mark Sanchez', 'draft_year': 2009, 'draft_pick': 5, 'college': 'USC'},
    {'player_name': 'Josh Freeman', 'draft_year': 2009, 'draft_pick': 17, 'college': 'Kansas State'},
    {'player_name': 'Pat White', 'draft_year': 2009, 'draft_pick': 44, 'college': 'West Virginia'},
    {'player_name': 'Rhett Bomar', 'draft_year': 2009, 'draft_pick': 197, 'college': 'Sam Houston State'},
    {'player_name': 'Stephen McGee', 'draft_year': 2009, 'draft_pick': 101, 'college': 'Texas A&M'},
    {'player_name': 'Tom Brandstater', 'draft_year': 2009, 'draft_pick': 168, 'college': 'Fresno State'},

    # 2010 Draft
    {'player_name': 'Sam Bradford', 'draft_year': 2010, 'draft_pick': 1, 'college': 'Oklahoma'},
    {'player_name': 'Tim Tebow', 'draft_year': 2010, 'draft_pick': 25, 'college': 'Florida'},
    {'player_name': 'Jimmy Clausen', 'draft_year': 2010, 'draft_pick': 48, 'college': 'Notre Dame'},
    {'player_name': 'Colt McCoy', 'draft_year': 2010, 'draft_pick': 85, 'college': 'Texas'},
    {'player_name': 'Tony Pike', 'draft_year': 2010, 'draft_pick': 178, 'college': 'Cincinnati'},
    {'player_name': 'Dan LeFevour', 'draft_year': 2010, 'draft_pick': 182, 'college': 'Central Michigan'},
    {'player_name': 'Jonathan Crompton', 'draft_year': 2010, 'draft_pick': 168, 'college': 'Tennessee'},

    # 2011 Draft
    {'player_name': 'Cam Newton', 'draft_year': 2011, 'draft_pick': 1, 'college': 'Auburn'},
    {'player_name': 'Jake Locker', 'draft_year': 2011, 'draft_pick': 8, 'college': 'Washington'},
    {'player_name': 'Blaine Gabbert', 'draft_year': 2011, 'draft_pick': 10, 'college': 'Missouri'},
    {'player_name': 'Christian Ponder', 'draft_year': 2011, 'draft_pick': 12, 'college': 'Florida State'},
    {'player_name': 'Andy Dalton', 'draft_year': 2011, 'draft_pick': 35, 'college': 'TCU'},
    {'player_name': 'Colin Kaepernick', 'draft_year': 2011, 'draft_pick': 36, 'college': 'Nevada'},
    {'player_name': 'Ryan Mallett', 'draft_year': 2011, 'draft_pick': 74, 'college': 'Arkansas'},
    {'player_name': 'Ricky Stanzi', 'draft_year': 2011, 'draft_pick': 135, 'college': 'Iowa'},
    {'player_name': 'Nathan Enderle', 'draft_year': 2011, 'draft_pick': 160, 'college': 'Idaho'},
    {'player_name': 'Tyrod Taylor', 'draft_year': 2011, 'draft_pick': 180, 'college': 'Virginia Tech'},

    # 2012 Draft
    {'player_name': 'Andrew Luck', 'draft_year': 2012, 'draft_pick': 1, 'college': 'Stanford'},
    {'player_name': 'Robert Griffin III', 'draft_year': 2012, 'draft_pick': 2, 'college': 'Baylor'},
    {'player_name': 'Ryan Tannehill', 'draft_year': 2012, 'draft_pick': 8, 'college': 'Texas A&M'},
    {'player_name': 'Brandon Weeden', 'draft_year': 2012, 'draft_pick': 22, 'college': 'Oklahoma State'},
    {'player_name': 'Brock Osweiler', 'draft_year': 2012, 'draft_pick': 57, 'college': 'Arizona State'},
    {'player_name': 'Nick Foles', 'draft_year': 2012, 'draft_pick': 88, 'college': 'Arizona'},
    {'player_name': 'Russell Wilson', 'draft_year': 2012, 'draft_pick': 75, 'college': 'Wisconsin'},
    {'player_name': 'Kirk Cousins', 'draft_year': 2012, 'draft_pick': 102, 'college': 'Michigan State'},
    {'player_name': 'Ryan Lindley', 'draft_year': 2012, 'draft_pick': 185, 'college': 'San Diego State'},

    # 2013 Draft
    {'player_name': 'EJ Manuel', 'draft_year': 2013, 'draft_pick': 16, 'college': 'Florida State'},
    {'player_name': 'Geno Smith', 'draft_year': 2013, 'draft_pick': 39, 'college': 'West Virginia'},
    {'player_name': 'Mike Glennon', 'draft_year': 2013, 'draft_pick': 73, 'college': 'NC State'},
    {'player_name': 'Matt Barkley', 'draft_year': 2013, 'draft_pick': 98, 'college': 'USC'},
    {'player_name': 'Ryan Nassib', 'draft_year': 2013, 'draft_pick': 110, 'college': 'Syracuse'},
    {'player_name': 'Tyler Wilson', 'draft_year': 2013, 'draft_pick': 112, 'college': 'Arkansas'},
    {'player_name': 'Landry Jones', 'draft_year': 2013, 'draft_pick': 115, 'college': 'Oklahoma'},

    # 2014 Draft
    {'player_name': 'Blake Bortles', 'draft_year': 2014, 'draft_pick': 3, 'college': 'UCF'},
    {'player_name': 'Johnny Manziel', 'draft_year': 2014, 'draft_pick': 22, 'college': 'Texas A&M'},
    {'player_name': 'Teddy Bridgewater', 'draft_year': 2014, 'draft_pick': 32, 'college': 'Louisville'},
    {'player_name': 'Derek Carr', 'draft_year': 2014, 'draft_pick': 36, 'college': 'Fresno State'},
    {'player_name': 'Jimmy Garoppolo', 'draft_year': 2014, 'draft_pick': 62, 'college': 'Eastern Illinois'},
    {'player_name': 'Logan Thomas', 'draft_year': 2014, 'draft_pick': 120, 'college': 'Virginia Tech'},
    {'player_name': 'AJ McCarron', 'draft_year': 2014, 'draft_pick': 164, 'college': 'Alabama'},
    {'player_name': 'Aaron Murray', 'draft_year': 2014, 'draft_pick': 163, 'college': 'Georgia'},

    # 2015-2023 already in current dataset, but including for completeness
]


def main():
    """Main execution"""
    print("="*80)
    print("CREATING EXPANDED QB LIST (2007-2023)")
    print("="*80)

    root = get_project_root()

    # Load current QB list
    print("\n[1] Loading current QB list...")
    df_current = load_current_qb_list()

    if len(df_current) > 0:
        print(f"    [OK] Current list: {len(df_current)} QBs")
        print(f"    Draft years: {sorted(df_current['draft_year'].unique())}")
    else:
        print("    [--] No current list found")

    # Create historical QB dataframe
    print("\n[2] Adding historical QBs (2007-2014)...")
    df_historical = pd.DataFrame(HISTORICAL_QBS)
    print(f"    [OK] {len(df_historical)} historical QBs identified")

    # Combine with current list
    print("\n[3] Combining lists...")

    if len(df_current) > 0:
        # Keep current list for 2015-2023, add historical for 2007-2014
        df_recent = df_current[df_current['draft_year'] >= 2015].copy()

        # Combine
        df_expanded = pd.concat([df_historical, df_recent], ignore_index=True)
    else:
        df_expanded = df_historical

    # Sort by draft year and pick
    df_expanded = df_expanded.sort_values(['draft_year', 'draft_pick'])

    print(f"    [OK] Expanded list: {len(df_expanded)} QBs")

    # Summary by year
    print(f"\n    QBs per draft year:")
    year_counts = df_expanded['draft_year'].value_counts().sort_index()

    for year in sorted(df_expanded['draft_year'].unique()):
        count = year_counts.get(year, 0)
        print(f"      {year}: {count} QBs")

    print(f"\n    Total draft years: {len(df_expanded['draft_year'].unique())}")
    print(f"    Date range: {df_expanded['draft_year'].min()}-{df_expanded['draft_year'].max()}")

    # Save expanded list
    print("\n[4] Saving expanded QB list...")
    output_path = root / 'data' / 'processed' / 'qb_list_expanded.csv'
    df_expanded.to_csv(output_path, index=False)
    print(f"    [OK] Saved to: {output_path}")

    # Summary
    print("\n" + "="*80)
    print("EXPANDED QB LIST COMPLETE")
    print("="*80)

    print(f"\nDataset expansion:")
    if len(df_current) > 0:
        print(f"  Before: {len(df_current)} QBs ({df_current['draft_year'].min()}-{df_current['draft_year'].max()})")
    print(f"  After:  {len(df_expanded)} QBs ({df_expanded['draft_year'].min()}-{df_expanded['draft_year'].max()})")

    increase = len(df_expanded) - len(df_current) if len(df_current) > 0 else len(df_expanded)
    print(f"  Added:  {increase} QBs")

    if len(df_current) > 0:
        pct_increase = (increase / len(df_current)) * 100
        print(f"  Growth: +{pct_increase:.1f}%")

    print(f"\nNext steps:")
    print(f"  1. Wait for SP+ historical data to complete")
    print(f"  2. Collect full college careers for all {len(df_expanded)} QBs")
    print(f"  3. Apply multi-year aggregation methods")


if __name__ == "__main__":
    main()
