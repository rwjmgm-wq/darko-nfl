"""
Fetch Historical SP+ Ratings from CFBD API

Collects SP+ ratings from 2005 (or earlier if available) through 2023
to enable expanded historical QB analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import cfbd
from cfbd.rest import ApiException
import time


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent


def load_api_key():
    """Load CFBD API key from .env file"""
    root = get_project_root()
    api_key_file = root / '.env'

    if not api_key_file.exists():
        raise FileNotFoundError("No .env file found. Create one with CFBD_API_KEY=your_key")

    with open(api_key_file) as f:
        for line in f:
            if line.startswith('CFBD_API_KEY='):
                return line.strip().split('=')[1]

    raise ValueError("CFBD_API_KEY not found in .env file")


def fetch_sp_plus_for_season(api_instance, year):
    """
    Fetch SP+ ratings for all teams in a given season

    Returns DataFrame with columns: team, overall_rating, offense_rating, defense_rating
    """
    try:
        print(f"  Fetching {year}...", end=' ', flush=True)

        # Get SP+ ratings for the season
        ratings = api_instance.get_sp(year=year)

        if not ratings or len(ratings) == 0:
            print("[--] No data")
            return pd.DataFrame()

        # Extract ratings
        data = []
        for rating in ratings:
            data.append({
                'season': year,
                'team': rating.team,
                'conference': rating.conference,
                'overall_rating': rating.rating,
                'offense_rating': rating.offense.rating if rating.offense else np.nan,
                'defense_rating': rating.defense.rating if rating.defense else np.nan,
                'special_teams_rating': rating.special_teams.rating if rating.special_teams else np.nan
            })

        df = pd.DataFrame(data)
        print(f"[OK] {len(df)} teams")
        return df

    except ApiException as e:
        if e.status == 401:
            print("[ERROR] Unauthorized - check API key")
        elif e.status == 429:
            print("[ERROR] Rate limit exceeded")
        else:
            print(f"[ERROR] API error: {e.status}")
        return pd.DataFrame()

    except Exception as e:
        print(f"[ERROR] {str(e)[:50]}")
        return pd.DataFrame()


def main():
    """Main execution"""
    print("="*80)
    print("FETCHING HISTORICAL SP+ RATINGS")
    print("="*80)
    print("\nCollecting SP+ data from CFBD API to expand dataset backward")

    # Load API key and configure
    print("\n[1] Configuring API...")
    try:
        api_key = load_api_key()
        # Use access_token parameter (recommended by CFBD support)
        configuration = cfbd.Configuration(
            access_token=api_key
        )
        print("    [OK] API configured")
    except Exception as e:
        print(f"    [ERROR] {str(e)}")
        return

    # Create API instance
    api_instance = cfbd.RatingsApi(cfbd.ApiClient(configuration))

    # Determine year range
    # SP+ officially started in 2005, but let's test back to 2003
    start_year = 2003
    end_year = 2024

    print(f"\n[2] Fetching SP+ ratings {start_year}-{end_year}...")
    print("-"*80)

    all_ratings = []

    for year in range(start_year, end_year + 1):
        df_year = fetch_sp_plus_for_season(api_instance, year)

        if len(df_year) > 0:
            all_ratings.append(df_year)

        # Respectful rate limiting
        time.sleep(0.5)

    if not all_ratings:
        print("\n[ERROR] No SP+ data collected")
        return

    # Combine all years
    print(f"\n[3] Combining data...")
    df_all = pd.concat(all_ratings, ignore_index=True)
    print(f"    [OK] {len(df_all)} team-seasons collected")

    # Summary by year
    print(f"\n    Seasons covered:")
    years_covered = sorted(df_all['season'].unique())
    for year in years_covered:
        count = len(df_all[df_all['season'] == year])
        print(f"      {year}: {count} teams")

    print(f"\n    Date range: {min(years_covered)} - {max(years_covered)}")
    print(f"    Total years: {len(years_covered)}")

    # Save to CSV
    print(f"\n[4] Saving historical SP+ data...")
    root = get_project_root()
    output_dir = root / 'data' / 'processed' / 'sp_plus_historical'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / 'sp_plus_all_years.csv'
    df_all.to_csv(output_path, index=False)
    print(f"    [OK] Saved to: {output_path}")

    # Also save by year for easy lookup
    print(f"\n[5] Saving individual year files...")
    for year in years_covered:
        year_data = df_all[df_all['season'] == year]
        year_path = output_dir / f'sp_plus_{year}.csv'
        year_data.to_csv(year_path, index=False)
    print(f"    [OK] Saved {len(years_covered)} year files")

    # Merge with existing cached data
    print(f"\n[6] Checking for cached betting model data...")
    betting_model_path = Path("C:/Users/rwjmg/OneDrive/Pictures/Writing/new_cfb_betting_model")

    if betting_model_path.exists():
        print(f"    [OK] Found betting model cache")

        # Load cached data
        import glob
        pattern = f"{betting_model_path}/data/raw/sp_plus_*_week*.csv"
        files = glob.glob(pattern)

        cached_years = set()
        for file in files:
            import re
            match = re.search(r'sp_plus_(\d{4})_week', file)
            if match:
                cached_years.add(int(match.group(1)))

        cached_years = sorted(cached_years)
        print(f"    Cached years: {cached_years}")
        print(f"    Historical years: {years_covered}")

        # Calculate coverage
        all_years = sorted(set(years_covered) | set(cached_years))
        print(f"\n    Combined coverage: {min(all_years)}-{max(all_years)}")
        print(f"    Total coverage: {len(all_years)} years")
    else:
        print(f"    [--] Betting model cache not found")
        all_years = years_covered

    # Summary
    print("\n" + "="*80)
    print("HISTORICAL SP+ FETCH COMPLETE")
    print("="*80)

    print(f"\nData collected:")
    print(f"  Years: {min(years_covered)}-{max(years_covered)}")
    print(f"  Team-seasons: {len(df_all)}")
    print(f"  Average teams per year: {len(df_all) / len(years_covered):.1f}")

    print(f"\nNext steps:")
    print(f"  1. Expand QB list to {min(all_years)+4}-2023 draft classes")
    print(f"  2. Collect full college careers (not just final season)")
    print(f"  3. Apply SP+ adjustments throughout")

    # Check data quality
    print(f"\n" + "="*80)
    print("DATA QUALITY CHECK")
    print("="*80)

    # Count complete records
    complete = df_all[
        df_all['overall_rating'].notna() &
        df_all['offense_rating'].notna() &
        df_all['defense_rating'].notna()
    ]

    print(f"\nComplete records (all ratings present):")
    print(f"  {len(complete)}/{len(df_all)} ({len(complete)/len(df_all)*100:.1f}%)")

    # Missing data by component
    print(f"\nMissing data by component:")
    print(f"  Overall rating:  {df_all['overall_rating'].isna().sum()} missing")
    print(f"  Offense rating:  {df_all['offense_rating'].isna().sum()} missing")
    print(f"  Defense rating:  {df_all['defense_rating'].isna().sum()} missing")

    if len(complete) < len(df_all):
        print(f"\n[WARNING] Some incomplete records - may need cleanup")


if __name__ == "__main__":
    main()
