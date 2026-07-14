"""
Fix Missing College Names in QB List

The QB list has 62 QBs (2015-2023) with missing college names.
This script fills them in using Pro Football Reference data.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent


# Manually curated college mappings for 2015-2023 QBs
# Source: Pro Football Reference / NFL Draft data
COLLEGE_MAPPINGS = {
    # 2015
    ('Jameis Winston', 2015): 'Florida State',
    ('Marcus Mariota', 2015): 'Oregon',
    ('Sean Mannion', 2015): 'Oregon State',

    # 2016
    ('Jared Goff', 2016): 'California',
    ('Carson Wentz', 2016): 'North Dakota State',
    ('Paxton Lynch', 2016): 'Memphis',
    ('Jacoby Brissett', 2016): 'NC State',
    ('Cody Kessler', 2016): 'USC',
    ('Connor Cook', 2016): 'Michigan State',
    ('Dak Prescott', 2016): 'Mississippi State',
    ('Cardale Jones', 2016): 'Ohio State',
    ('Kevin Hogan', 2016): 'Stanford',

    # 2017
    ('Mitchell Trubisky', 2017): 'North Carolina',
    ('Patrick Mahomes', 2017): 'Texas Tech',
    ('Deshaun Watson', 2017): 'Clemson',
    ('DeShone Kizer', 2017): 'Notre Dame',
    ('C.J. Beathard', 2017): 'Iowa',
    ('Nathan Peterman', 2017): 'Pittsburgh',

    # 2018
    ('Baker Mayfield', 2018): 'Oklahoma',
    ('Sam Darnold', 2018): 'USC',
    ('Josh Allen', 2018): 'Wyoming',
    ('Josh Rosen', 2018): 'UCLA',
    ('Lamar Jackson', 2018): 'Louisville',
    ('Kyle Lauletta', 2018): 'Richmond',

    # 2019
    ('Kyler Murray', 2019): 'Oklahoma',
    ('Daniel Jones', 2019): 'Duke',
    ('Dwayne Haskins', 2019): 'Ohio State',
    ('Drew Lock', 2019): 'Missouri',
    ('Will Grier', 2019): 'West Virginia',
    ('Ryan Finley', 2019): 'NC State',
    ('Jarrett Stidham', 2019): 'Auburn',
    ('Gardner Minshew II', 2019): 'Washington State',

    # 2020
    ('Joe Burrow', 2020): 'LSU',
    ('Tua Tagovailoa', 2020): 'Alabama',
    ('Justin Herbert', 2020): 'Oregon',
    ('Jalen Hurts', 2020): 'Oklahoma',
    ('Jake Luton', 2020): 'Oregon State',
    ('Ben DiNucci', 2020): 'James Madison',

    # 2021
    ('Trevor Lawrence', 2021): 'Clemson',
    ('Zach Wilson', 2021): 'BYU',
    ('Trey Lance', 2021): 'North Dakota State',
    ('Justin Fields', 2021): 'Ohio State',
    ('Mac Jones', 2021): 'Alabama',
    ('Kellen Mond', 2021): 'Texas A&M',
    ('Davis Mills', 2021): 'Stanford',
    ('Ian Book', 2021): 'Notre Dame',

    # 2022
    ('Kenny Pickett', 2022): 'Pittsburgh',
    ('Malik Willis', 2022): 'Liberty',
    ('Desmond Ridder', 2022): 'Cincinnati',
    ('Matt Corral', 2022): 'Ole Miss',
    ('Sam Howell', 2022): 'North Carolina',
    ('Bailey Zappe', 2022): 'Western Kentucky',
    ('Skylar Thompson', 2022): 'Kansas State',
    ('Brock Purdy', 2022): 'Iowa State',

    # 2023
    ('Bryce Young', 2023): 'Alabama',
    ('C.J. Stroud', 2023): 'Ohio State',
    ('Anthony Richardson', 2023): 'Florida',
    ('Will Levis', 2023): 'Kentucky',
    ('Hendon Hooker', 2023): 'Tennessee',
    ('Dorian Thompson-Robinson', 2023): 'UCLA',
    ('Aidan O\'Connell', 2023): 'Purdue',
    ('Clayton Tune', 2023): 'Houston',
    ('Sean Clifford', 2023): 'Penn State',
    ('Jaren Hall', 2023): 'BYU',
}


def main():
    """Main execution"""
    print("="*80)
    print("FIXING MISSING COLLEGE NAMES")
    print("="*80)

    root = get_project_root()
    qb_list_path = root / 'data' / 'processed' / 'qb_list_expanded.csv'

    # Load QB list
    print("\n[1] Loading QB list...")
    df_qbs = pd.read_csv(qb_list_path)
    print(f"    Total QBs: {len(df_qbs)}")

    missing_before = df_qbs['college'].isna().sum()
    print(f"    Missing colleges: {missing_before}")

    # Fill in missing colleges
    print("\n[2] Filling in missing college names...")
    fixed_count = 0

    for idx, row in df_qbs.iterrows():
        if pd.isna(row['college']):
            key = (row['player_name'], row['draft_year'])
            if key in COLLEGE_MAPPINGS:
                df_qbs.at[idx, 'college'] = COLLEGE_MAPPINGS[key]
                print(f"    {row['player_name']} ({row['draft_year']}) -> {COLLEGE_MAPPINGS[key]}")
                fixed_count += 1
            else:
                print(f"    WARNING: No mapping for {row['player_name']} ({row['draft_year']})")

    print(f"\n    Fixed: {fixed_count} QBs")

    # Check if any are still missing
    missing_after = df_qbs['college'].isna().sum()
    if missing_after > 0:
        print(f"\n    WARNING: {missing_after} QBs still have missing colleges:")
        still_missing = df_qbs[df_qbs['college'].isna()][['player_name', 'draft_year']]
        print(still_missing.to_string(index=False))

    # Save updated QB list
    print("\n[3] Saving updated QB list...")
    df_qbs.to_csv(qb_list_path, index=False)
    print(f"    Saved to: {qb_list_path}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n  Total QBs: {len(df_qbs)}")
    print(f"  With colleges: {len(df_qbs) - missing_after}")
    print(f"  Missing colleges: {missing_after}")

    if missing_after == 0:
        print("\n  [SUCCESS] All college names filled in!")
        print("\n  Next: Run sportsdataverse collection for remaining QBs")
    else:
        print("\n  [WARNING] Some colleges still missing - manual intervention needed")


if __name__ == "__main__":
    main()
