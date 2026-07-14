"""
Fetch Historical SP+ Ratings with Automatic Retry

Keeps trying every 10 minutes until all data is collected.
Resumes from where it left off if interrupted.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import cfbd
from cfbd.rest import ApiException
import time
from datetime import datetime


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


def get_already_fetched_years(output_dir):
    """Check which years we've already successfully fetched"""
    fetched_years = set()

    if output_dir.exists():
        for file in output_dir.glob('sp_plus_*.csv'):
            # Extract year from filename
            import re
            match = re.search(r'sp_plus_(\d{4})\.csv', file.name)
            if match:
                year = int(match.group(1))
                # Verify file has data
                try:
                    df = pd.read_csv(file)
                    if len(df) > 0:
                        fetched_years.add(year)
                except:
                    pass

    return fetched_years


def fetch_sp_plus_for_season(api_instance, year):
    """
    Fetch SP+ ratings for all teams in a given season

    Returns DataFrame with columns: team, overall_rating, offense_rating, defense_rating
    """
    try:
        # Get SP+ ratings for the season
        ratings = api_instance.get_sp(year=year)

        if not ratings or len(ratings) == 0:
            return None, "No data"

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
        return df, None

    except ApiException as e:
        if e.status == 401:
            return None, "Unauthorized"
        elif e.status == 429:
            return None, "Rate limit"
        else:
            return None, f"API error {e.status}"

    except Exception as e:
        return None, f"Error: {str(e)[:50]}"


def main():
    """Main execution with retry logic"""
    print("="*80)
    print("FETCHING HISTORICAL SP+ RATINGS (WITH RETRY)")
    print("="*80)
    print("\nWill retry every 10 minutes until all data is collected")
    print("Press Ctrl+C to stop")

    # Setup
    root = get_project_root()
    output_dir = root / 'data' / 'processed' / 'sp_plus_historical'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load API key and configure
    print("\n[1] Configuring API...")
    try:
        api_key = load_api_key()
        configuration = cfbd.Configuration()
        configuration.api_key['Authorization'] = api_key
        configuration.api_key_prefix['Authorization'] = 'Bearer'
        print("    [OK] API configured")
    except Exception as e:
        print(f"    [ERROR] {str(e)}")
        return

    api_instance = cfbd.RatingsApi(cfbd.ApiClient(configuration))

    # Target years
    start_year = 2003
    end_year = 2023
    target_years = list(range(start_year, end_year + 1))

    retry_count = 0
    max_retries = 100  # Maximum number of 10-minute cycles

    while retry_count < max_retries:
        print(f"\n" + "="*80)
        print(f"ATTEMPT #{retry_count + 1} - {datetime.now().strftime('%H:%M:%S')}")
        print("="*80)

        # Check what we already have
        fetched_years = get_already_fetched_years(output_dir)

        if fetched_years:
            print(f"\n[2] Already fetched: {sorted(fetched_years)}")

        # Determine what we still need
        remaining_years = [y for y in target_years if y not in fetched_years]

        if not remaining_years:
            print("\n[SUCCESS] All years fetched!")
            break

        print(f"\n[3] Still need: {remaining_years}")
        print(f"    ({len(remaining_years)} years remaining)")

        # Try to fetch remaining years
        print(f"\n[4] Fetching...")
        print("-"*80)

        newly_fetched = []
        hit_rate_limit = False

        for year in remaining_years:
            print(f"  {year}...", end=' ', flush=True)

            df, error = fetch_sp_plus_for_season(api_instance, year)

            if df is not None:
                # Success! Save immediately
                year_path = output_dir / f'sp_plus_{year}.csv'
                df.to_csv(year_path, index=False)
                newly_fetched.append(year)
                print(f"[OK] {len(df)} teams saved")

            elif error == "Rate limit":
                print(f"[RATE LIMIT]")
                hit_rate_limit = True
                break  # Stop trying, wait 10 minutes

            elif error == "Unauthorized":
                print(f"[AUTH ERROR]")
                hit_rate_limit = True
                break

            else:
                print(f"[ERROR] {error}")

            # Polite delay between requests
            time.sleep(0.5)

        # Status update
        if newly_fetched:
            print(f"\n    New this round: {newly_fetched}")

        fetched_years = get_already_fetched_years(output_dir)
        remaining_years = [y for y in target_years if y not in fetched_years]

        print(f"\n    Progress: {len(fetched_years)}/{len(target_years)} years")
        print(f"    Complete: {len(fetched_years)/len(target_years)*100:.1f}%")

        if not remaining_years:
            print("\n[SUCCESS] All data collected!")
            break

        if hit_rate_limit or not newly_fetched:
            # Wait 10 minutes before retry
            wait_seconds = 600  # 10 minutes
            print(f"\n    Waiting 10 minutes before retry...")
            print(f"    Will resume at {datetime.now().replace(second=0, microsecond=0) + pd.Timedelta(seconds=wait_seconds)}")

            try:
                for i in range(wait_seconds // 60):
                    time.sleep(60)
                    print(f"    {wait_seconds//60 - i - 1} minutes remaining...")
            except KeyboardInterrupt:
                print("\n\n[INTERRUPTED] Stopping retry loop")
                print(f"Progress saved: {len(fetched_years)}/{len(target_years)} years")
                return

        retry_count += 1

    # Final summary
    print("\n" + "="*80)
    print("FETCH COMPLETE")
    print("="*80)

    fetched_years = get_already_fetched_years(output_dir)

    if len(fetched_years) == len(target_years):
        print("\n[SUCCESS] All years collected!")
    else:
        print(f"\n[PARTIAL] Collected {len(fetched_years)}/{len(target_years)} years")
        remaining = [y for y in target_years if y not in fetched_years]
        print(f"Still missing: {remaining}")

    # Combine all years into single file
    if fetched_years:
        print(f"\n[5] Combining all years into single file...")
        all_data = []

        for year in sorted(fetched_years):
            year_path = output_dir / f'sp_plus_{year}.csv'
            df_year = pd.read_csv(year_path)
            all_data.append(df_year)

        df_all = pd.concat(all_data, ignore_index=True)

        combined_path = output_dir / 'sp_plus_all_years.csv'
        df_all.to_csv(combined_path, index=False)

        print(f"    [OK] Combined file: {combined_path}")
        print(f"    Total team-seasons: {len(df_all)}")
        print(f"    Years: {sorted(fetched_years)}")


if __name__ == "__main__":
    main()
