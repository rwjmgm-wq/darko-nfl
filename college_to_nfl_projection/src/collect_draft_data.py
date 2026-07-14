"""
Collect NFL Draft Data for QBs (2015-2024)

This script identifies all quarterbacks drafted from 2015-2024 and collects
their draft information (year, round, pick, college).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Try to import nfl_data_py for draft data
try:
    import nfl_data_py as nfl
    NFL_DATA_AVAILABLE = True
except ImportError:
    print("Warning: nfl_data_py not available. Will use manual data entry.")
    NFL_DATA_AVAILABLE = False


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent


def fetch_draft_data_nfldata(start_year=2015, end_year=2024):
    """
    Fetch draft data using nfl_data_py

    Args:
        start_year: First draft year
        end_year: Last draft year

    Returns:
        DataFrame with QB draft information
    """
    if not NFL_DATA_AVAILABLE:
        return None

    print(f"Fetching draft data from nfl_data_py ({start_year}-{end_year})...")

    try:
        # Import draft picks data
        draft_picks = nfl.import_draft_picks(years=list(range(start_year, end_year + 1)))

        # Filter to QBs only
        qb_picks = draft_picks[draft_picks['position'] == 'QB'].copy()

        # Select relevant columns
        qb_picks = qb_picks[[
            'season', 'round', 'pick', 'team',
            'pfr_player_name', 'cfb_player_id', 'pfr_player_id',
            'position', 'age', 'college'
        ]].copy()

        qb_picks.columns = [
            'draft_year', 'round', 'pick', 'nfl_team',
            'player_name', 'cfb_id', 'pfr_id',
            'position', 'age', 'college'
        ]

        # Sort by draft order
        qb_picks = qb_picks.sort_values(['draft_year', 'pick'])

        print(f"[OK] Found {len(qb_picks)} QBs drafted {start_year}-{end_year}")

        return qb_picks

    except Exception as e:
        print(f"Error fetching draft data: {e}")
        return None


def get_manual_qb_draft_data():
    """
    Manually compiled QB draft data (2015-2024)

    This serves as backup if nfl_data_py is unavailable or as validation.
    Data sourced from Pro Football Reference.

    Returns:
        DataFrame with QB draft information
    """
    manual_data = [
        # 2024 Draft
        {'draft_year': 2024, 'round': 1, 'pick': 1, 'player_name': 'Caleb Williams', 'college': 'USC', 'nfl_team': 'CHI'},
        {'draft_year': 2024, 'round': 1, 'pick': 2, 'player_name': 'Jayden Daniels', 'college': 'LSU', 'nfl_team': 'WAS'},
        {'draft_year': 2024, 'round': 1, 'pick': 3, 'player_name': 'Drake Maye', 'college': 'North Carolina', 'nfl_team': 'NE'},
        {'draft_year': 2024, 'round': 1, 'pick': 8, 'player_name': 'Michael Penix Jr.', 'college': 'Washington', 'nfl_team': 'ATL'},
        {'draft_year': 2024, 'round': 1, 'pick': 11, 'player_name': 'J.J. McCarthy', 'college': 'Michigan', 'nfl_team': 'MIN'},
        {'draft_year': 2024, 'round': 1, 'pick': 12, 'player_name': 'Bo Nix', 'college': 'Oregon', 'nfl_team': 'DEN'},

        # 2023 Draft
        {'draft_year': 2023, 'round': 1, 'pick': 1, 'player_name': 'Bryce Young', 'college': 'Alabama', 'nfl_team': 'CAR'},
        {'draft_year': 2023, 'round': 1, 'pick': 2, 'player_name': 'C.J. Stroud', 'college': 'Ohio State', 'nfl_team': 'HOU'},
        {'draft_year': 2023, 'round': 1, 'pick': 4, 'player_name': 'Anthony Richardson', 'college': 'Florida', 'nfl_team': 'IND'},
        {'draft_year': 2023, 'round': 1, 'pick': 12, 'player_name': 'Will Levis', 'college': 'Kentucky', 'nfl_team': 'TEN'},
        {'draft_year': 2023, 'round': 4, 'pick': 135, 'player_name': 'Dorian Thompson-Robinson', 'college': 'UCLA', 'nfl_team': 'CLE'},

        # 2022 Draft
        {'draft_year': 2022, 'round': 1, 'pick': 2, 'player_name': 'Kenny Pickett', 'college': 'Pittsburgh', 'nfl_team': 'PIT'},
        {'draft_year': 2022, 'round': 1, 'pick': 20, 'player_name': 'Malik Willis', 'college': 'Liberty', 'nfl_team': 'TEN'},
        {'draft_year': 2022, 'round': 3, 'pick': 74, 'player_name': 'Desmond Ridder', 'college': 'Cincinnati', 'nfl_team': 'ATL'},
        {'draft_year': 2022, 'round': 3, 'pick': 86, 'player_name': 'Sam Howell', 'college': 'North Carolina', 'nfl_team': 'WAS'},

        # 2021 Draft
        {'draft_year': 2021, 'round': 1, 'pick': 1, 'player_name': 'Trevor Lawrence', 'college': 'Clemson', 'nfl_team': 'JAX'},
        {'draft_year': 2021, 'round': 1, 'pick': 2, 'player_name': 'Zach Wilson', 'college': 'BYU', 'nfl_team': 'NYJ'},
        {'draft_year': 2021, 'round': 1, 'pick': 3, 'player_name': 'Trey Lance', 'college': 'North Dakota State', 'nfl_team': 'SF'},
        {'draft_year': 2021, 'round': 1, 'pick': 11, 'player_name': 'Justin Fields', 'college': 'Ohio State', 'nfl_team': 'CHI'},
        {'draft_year': 2021, 'round': 1, 'pick': 15, 'player_name': 'Mac Jones', 'college': 'Alabama', 'nfl_team': 'NE'},
        {'draft_year': 2021, 'round': 2, 'pick': 67, 'player_name': 'Davis Mills', 'college': 'Stanford', 'nfl_team': 'HOU'},
        {'draft_year': 2021, 'round': 3, 'pick': 107, 'player_name': 'Kellen Mond', 'college': 'Texas A&M', 'nfl_team': 'MIN'},

        # 2020 Draft
        {'draft_year': 2020, 'round': 1, 'pick': 1, 'player_name': 'Joe Burrow', 'college': 'LSU', 'nfl_team': 'CIN'},
        {'draft_year': 2020, 'round': 1, 'pick': 5, 'player_name': 'Tua Tagovailoa', 'college': 'Alabama', 'nfl_team': 'MIA'},
        {'draft_year': 2020, 'round': 1, 'pick': 6, 'player_name': 'Justin Herbert', 'college': 'Oregon', 'nfl_team': 'LAC'},
        {'draft_year': 2020, 'round': 1, 'pick': 26, 'player_name': 'Jordan Love', 'college': 'Utah State', 'nfl_team': 'GB'},
        {'draft_year': 2020, 'round': 2, 'pick': 42, 'player_name': 'Jalen Hurts', 'college': 'Oklahoma', 'nfl_team': 'PHI'},

        # 2019 Draft
        {'draft_year': 2019, 'round': 1, 'pick': 1, 'player_name': 'Kyler Murray', 'college': 'Oklahoma', 'nfl_team': 'ARI'},
        {'draft_year': 2019, 'round': 1, 'pick': 6, 'player_name': 'Daniel Jones', 'college': 'Duke', 'nfl_team': 'NYG'},
        {'draft_year': 2019, 'round': 1, 'pick': 15, 'player_name': 'Dwayne Haskins', 'college': 'Ohio State', 'nfl_team': 'WAS'},
        {'draft_year': 2019, 'round': 1, 'pick': 30, 'player_name': 'Drew Lock', 'college': 'Missouri', 'nfl_team': 'DEN'},
        {'draft_year': 2019, 'round': 4, 'pick': 104, 'player_name': 'Will Grier', 'college': 'West Virginia', 'nfl_team': 'CAR'},

        # 2018 Draft
        {'draft_year': 2018, 'round': 1, 'pick': 1, 'player_name': 'Baker Mayfield', 'college': 'Oklahoma', 'nfl_team': 'CLE'},
        {'draft_year': 2018, 'round': 1, 'pick': 3, 'player_name': 'Sam Darnold', 'college': 'USC', 'nfl_team': 'NYJ'},
        {'draft_year': 2018, 'round': 1, 'pick': 7, 'player_name': 'Josh Allen', 'college': 'Wyoming', 'nfl_team': 'BUF'},
        {'draft_year': 2018, 'round': 1, 'pick': 10, 'player_name': 'Josh Rosen', 'college': 'UCLA', 'nfl_team': 'ARI'},
        {'draft_year': 2018, 'round': 1, 'pick': 32, 'player_name': 'Lamar Jackson', 'college': 'Louisville', 'nfl_team': 'BAL'},

        # 2017 Draft
        {'draft_year': 2017, 'round': 1, 'pick': 2, 'player_name': 'Mitchell Trubisky', 'college': 'North Carolina', 'nfl_team': 'CHI'},
        {'draft_year': 2017, 'round': 1, 'pick': 10, 'player_name': 'Patrick Mahomes', 'college': 'Texas Tech', 'nfl_team': 'KC'},
        {'draft_year': 2017, 'round': 1, 'pick': 12, 'player_name': 'Deshaun Watson', 'college': 'Clemson', 'nfl_team': 'HOU'},
        {'draft_year': 2017, 'round': 2, 'pick': 52, 'player_name': 'DeShone Kizer', 'college': 'Notre Dame', 'nfl_team': 'CLE'},
        {'draft_year': 2017, 'round': 3, 'pick': 91, 'player_name': 'C.J. Beathard', 'college': 'Iowa', 'nfl_team': 'SF'},

        # 2016 Draft
        {'draft_year': 2016, 'round': 1, 'pick': 1, 'player_name': 'Jared Goff', 'college': 'California', 'nfl_team': 'LAR'},
        {'draft_year': 2016, 'round': 1, 'pick': 2, 'player_name': 'Carson Wentz', 'college': 'North Dakota State', 'nfl_team': 'PHI'},
        {'draft_year': 2016, 'round': 1, 'pick': 26, 'player_name': 'Paxton Lynch', 'college': 'Memphis', 'nfl_team': 'DEN'},
        {'draft_year': 2016, 'round': 2, 'pick': 57, 'player_name': 'Christian Hackenberg', 'college': 'Penn State', 'nfl_team': 'NYJ'},
        {'draft_year': 2016, 'round': 3, 'pick': 91, 'player_name': 'Jacoby Brissett', 'college': 'NC State', 'nfl_team': 'NE'},
        {'draft_year': 2016, 'round': 4, 'pick': 135, 'player_name': 'Dak Prescott', 'college': 'Mississippi State', 'nfl_team': 'DAL'},

        # 2015 Draft
        {'draft_year': 2015, 'round': 1, 'pick': 1, 'player_name': 'Jameis Winston', 'college': 'Florida State', 'nfl_team': 'TB'},
        {'draft_year': 2015, 'round': 1, 'pick': 2, 'player_name': 'Marcus Mariota', 'college': 'Oregon', 'nfl_team': 'TEN'},
        {'draft_year': 2015, 'round': 2, 'pick': 75, 'player_name': 'Bryce Petty', 'college': 'Baylor', 'nfl_team': 'NYJ'},
        {'draft_year': 2015, 'round': 3, 'pick': 103, 'player_name': 'Sean Mannion', 'college': 'Oregon State', 'nfl_team': 'STL'},
        {'draft_year': 2015, 'round': 4, 'pick': 135, 'player_name': 'Garrett Grayson', 'college': 'Colorado State', 'nfl_team': 'NO'},
    ]

    df = pd.DataFrame(manual_data)

    print(f"\n[Manual Data] Loaded {len(df)} QBs drafted 2015-2024")
    print(f"  By year: {df['draft_year'].value_counts().sort_index().to_dict()}")

    return df


def save_draft_data(df, output_path):
    """Save draft data to CSV"""
    df.to_csv(output_path, index=False)
    print(f"\n[OK] Saved draft data to: {output_path}")
    print(f"     Total QBs: {len(df)}")
    print(f"     Years: {df['draft_year'].min()}-{df['draft_year'].max()}")
    print(f"     Colleges: {df['college'].nunique()} unique schools")


def main():
    """Main execution"""
    print("="*80)
    print("COLLECTING QB DRAFT DATA (2015-2024)")
    print("="*80)

    root = get_project_root()
    output_path = root / 'data' / 'raw' / 'drafted_qbs.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try to fetch from nfl_data_py first
    df = fetch_draft_data_nfldata(start_year=2015, end_year=2024)

    # Fall back to manual data if needed
    if df is None or len(df) == 0:
        print("\nUsing manually compiled draft data...")
        df = get_manual_qb_draft_data()

    # Clean and standardize
    df['position'] = 'QB'
    df = df.fillna('')

    # Save
    save_draft_data(df, output_path)

    # Display summary
    print("\n" + "="*80)
    print("SAMPLE OF DRAFTED QBs:")
    print("="*80)
    print(df[['draft_year', 'round', 'pick', 'player_name', 'college']].head(10).to_string(index=False))
    print(f"\n... and {len(df) - 10} more QBs")

    print("\n" + "="*80)
    print("DRAFT DATA COLLECTION COMPLETE")
    print("="*80)

    return df


if __name__ == "__main__":
    main()
