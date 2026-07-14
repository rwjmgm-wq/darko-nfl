"""
Update qb_list_expanded.csv with 2024 and 2025 draft classes
"""
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    root = Path(__file__).parent

    # Load both files
    drafted_path = root / 'data' / 'raw' / 'drafted_qbs.csv'
    expanded_path = root / 'data' / 'processed' / 'qb_list_expanded.csv'

    df_drafted = pd.read_csv(drafted_path)
    df_expanded = pd.read_csv(expanded_path)

    print(f"Current qb_list_expanded.csv: {len(df_expanded)} QBs")
    print(f"Draft years: {df_expanded['draft_year'].min():.0f}-{df_expanded['draft_year'].max():.0f}")

    # Find QBs in drafted_qbs but not in qb_list_expanded
    expanded_keys = set(zip(df_expanded['player_name'], df_expanded['draft_year']))

    new_qbs = []
    for _, row in df_drafted.iterrows():
        key = (row['player_name'], row['draft_year'])
        if key not in expanded_keys:
            new_qbs.append(row)

    print(f"\nFound {len(new_qbs)} new QBs to add")

    if len(new_qbs) == 0:
        print("  No new QBs to add!")
        return

    # Create new rows for qb_list_expanded
    new_rows = []
    for qb in new_qbs:
        new_row = {
            'player_name': qb['player_name'],
            'draft_year': qb['draft_year'],
            'draft_pick': qb['pick'],
            'college': qb['college'],
            'college_college': qb['college'],  # Duplicate
            'final_season': qb['draft_year'] - 1,
            'draft_round': qb['round'],
            'nfl_team': qb['nfl_team'],
            'round': qb['round'],
            'pick': qb['pick'],
        }

        # Add NaN for columns we don't have yet
        for col in df_expanded.columns:
            if col not in new_row:
                new_row[col] = np.nan

        new_rows.append(new_row)
        print(f"  Adding: {qb['player_name']} ({qb['draft_year']}, {qb['college']})")

    # Append new rows
    df_new = pd.DataFrame(new_rows)
    df_updated = pd.concat([df_expanded, df_new], ignore_index=True)

    # Save updated file
    df_updated.to_csv(expanded_path, index=False)

    print(f"\n[OK] Updated qb_list_expanded.csv")
    print(f"New total: {len(df_updated)} QBs")
    print(f"Draft years: {df_updated['draft_year'].min():.0f}-{df_updated['draft_year'].max():.0f}")

if __name__ == '__main__':
    main()
