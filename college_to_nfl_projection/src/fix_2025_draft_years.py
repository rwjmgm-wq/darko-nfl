"""
Fix Draft Years for 2025 QBs

Many QBs who were draft eligible in 2025 are incorrectly marked as 2026 prospects.
This script corrects their draft years.
"""

import pandas as pd
from pathlib import Path


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent


# QBs who were actually drafted in 2025 NFL Draft
QBS_2025 = [
    'Cameron Ward',     # Miami - #1 overall, Titans
    'Jaxson Dart',      # Ole Miss - #25, Giants
    'Tyler Shough',     # Louisville - Saints
    'Jalen Milroe',     # Alabama
    'Shedeur Sanders',  # Colorado - Browns (Day 3)
    'Dillon Gabriel',   # Oregon - Browns
    'Kyle McCord',      # Syracuse - Eagles
    'Quinn Ewers'       # Texas - Dolphins
]


def main():
    print("="*80)
    print("FIXING 2025 DRAFT YEARS")
    print("="*80)

    root = get_project_root()

    # Fix the qbs_2026_prospects.csv file
    print("\n[1] Updating qbs_2026_prospects.csv...")
    prospects_path = root / 'data' / 'raw' / 'qbs_2026_prospects.csv'

    if prospects_path.exists():
        df = pd.read_csv(prospects_path)

        # Update draft years
        updates = 0
        for qb in QBS_2025:
            mask = df['player_name'] == qb
            if mask.any():
                df.loc[mask, 'draft_year'] = 2025
                updates += 1
                print(f"    {qb}: 2026 -> 2025")

        df.to_csv(prospects_path, index=False)
        print(f"\n    [OK] Updated {updates} QBs to 2025 draft class")

    # Fix aggregated stats files
    print("\n[2] Updating aggregated stats files...")
    stats_dir = root / 'data' / 'processed' / 'aggregated_stats_sp_adjusted'

    if stats_dir.exists():
        for stats_file in stats_dir.glob('stats_*.csv'):
            df = pd.read_csv(stats_file)

            updates = 0
            for qb in QBS_2025:
                mask = df['player_name'] == qb
                if mask.any():
                    df.loc[mask, 'draft_year'] = 2025
                    updates += 1

            if updates > 0:
                df.to_csv(stats_file, index=False)
                print(f"    {stats_file.name}: Updated {updates} QBs")

    print("\n[3] Rebuilding projections database...")
    print("    Run: python src/build_projection_database.py")
    print("\n" + "="*80)
    print("DONE")
    print("="*80)


if __name__ == "__main__":
    main()
