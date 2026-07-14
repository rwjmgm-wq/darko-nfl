"""
Add 2025 QB draft class to drafted_qbs.csv
"""
import pandas as pd
from pathlib import Path

# Team abbreviations mapping (NFL standard 3-letter codes)
TEAM_MAP = {
    'Tennessee Titans': 'TEN',
    'New York Giants': 'NYG',
    'New Orleans Saints': 'NOR',
    'Seattle Seahawks': 'SEA',
    'Cleveland Browns': 'CLE',
    'Philadelphia Eagles': 'PHI',
    'Pittsburgh Steelers': 'PIT',
    'Indianapolis Colts': 'IND',
    'Houston Texans': 'HOU',
    'Las Vegas Raiders': 'LVR',
    'San Francisco 49ers': 'SFO',
    'Miami Dolphins': 'MIA'
}

# 2025 QB draft class data
qbs_2025 = [
    {'player': 'Cam Ward', 'college': 'Miami (FL)', 'round': 1, 'pick': 1, 'team': 'Tennessee Titans'},
    {'player': 'Jaxson Dart', 'college': 'Mississippi', 'round': 1, 'pick': 25, 'team': 'New York Giants'},
    {'player': 'Tyler Shough', 'college': 'Louisville', 'round': 2, 'pick': 40, 'team': 'New Orleans Saints'},
    {'player': 'Jalen Milroe', 'college': 'Alabama', 'round': 3, 'pick': 92, 'team': 'Seattle Seahawks'},
    {'player': 'Dillon Gabriel', 'college': 'Oregon', 'round': 3, 'pick': 94, 'team': 'Cleveland Browns'},
    {'player': 'Shedeur Sanders', 'college': 'Colorado', 'round': 5, 'pick': 144, 'team': 'Cleveland Browns'},
    {'player': 'Kyle McCord', 'college': 'Syracuse', 'round': 6, 'pick': 181, 'team': 'Philadelphia Eagles'},
    {'player': 'Will Howard', 'college': 'Ohio St.', 'round': 6, 'pick': 185, 'team': 'Pittsburgh Steelers'},
    {'player': 'Riley Leonard', 'college': 'Notre Dame', 'round': 6, 'pick': 189, 'team': 'Indianapolis Colts'},
    {'player': 'Graham Mertz', 'college': 'Florida', 'round': 6, 'pick': 197, 'team': 'Houston Texans'},
    {'player': 'Tommy Mellott', 'college': 'Montana St.', 'round': 6, 'pick': 213, 'team': 'Las Vegas Raiders'},
    {'player': 'Cam Miller', 'college': 'North Dakota St.', 'round': 6, 'pick': 215, 'team': 'Las Vegas Raiders'},
    {'player': 'Kurtis Rourke', 'college': 'Indiana', 'round': 7, 'pick': 227, 'team': 'San Francisco 49ers'},
    {'player': 'Quinn Ewers', 'college': 'Texas', 'round': 7, 'pick': 231, 'team': 'Miami Dolphins'},
]

def main():
    # Load existing data
    root = Path(__file__).parent
    csv_path = root / 'data' / 'raw' / 'drafted_qbs.csv'

    df = pd.read_csv(csv_path)
    print(f"Current QB count: {len(df)}")
    print(f"Draft years: {df['draft_year'].min():.0f}-{df['draft_year'].max():.0f}")

    # Prepare new rows
    new_rows = []
    for qb in qbs_2025:
        new_row = {
            'draft_year': 2025,
            'round': qb['round'],
            'pick': qb['pick'],
            'nfl_team': TEAM_MAP[qb['team']],
            'player_name': qb['player'],
            'cfb_id': '',  # Will be filled during data collection
            'pfr_id': '',  # Will be filled during data collection
            'position': 'QB',
            'age': '',  # Will be filled if needed
            'college': qb['college']
        }
        new_rows.append(new_row)

    # Append new rows
    df_new = pd.DataFrame(new_rows)
    df_updated = pd.concat([df, df_new], ignore_index=True)

    # Save updated file
    df_updated.to_csv(csv_path, index=False)

    print(f"\n[OK] Added {len(new_rows)} QBs from 2025 draft")
    print(f"New total: {len(df_updated)} QBs")
    print(f"Draft years: {df_updated['draft_year'].min():.0f}-{df_updated['draft_year'].max():.0f}")

    print("\n2025 QBs added:")
    for qb in qbs_2025:
        print(f"  R{qb['round']}-{qb['pick']:3d}: {qb['player']:20s} ({qb['college']})")

if __name__ == '__main__':
    main()
