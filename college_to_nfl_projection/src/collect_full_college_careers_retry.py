"""
Collect Full College Career Data with Automatic Retry

Keeps trying every 30 minutes until all QB career data is collected.
Saves progress incrementally and resumes from where it left off.
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


def get_already_collected_qbs(output_dir):
    """Check which QBs already have career data collected"""
    collected = set()

    if output_dir.exists():
        for file in output_dir.glob('*_*.csv'):
            # Skip summary file
            if file.name == 'career_summary.csv':
                continue

            # Extract player name and year from filename
            # Format: "Patrick_Mahomes_2017.csv"
            try:
                parts = file.stem.rsplit('_', 1)
                if len(parts) == 2:
                    year = int(parts[1])
                    player_name = parts[0].replace('_', ' ')

                    # Verify file has data
                    df = pd.read_csv(file)
                    if len(df) > 0:
                        collected.add((player_name, year))
            except:
                pass

    return collected


def determine_college_seasons(draft_year, college):
    """Estimate all college seasons for a QB based on draft year"""
    final_season = draft_year - 1
    potential_seasons = list(range(final_season - 4, final_season + 1))
    return potential_seasons


def fetch_player_season_stats(api_instance, player_name, team, season):
    """Fetch a player's stats for a specific season"""
    try:
        stats = api_instance.get_player_season_stats(
            year=season,
            team=team,
            category='passing'
        )

        if not stats:
            return None

        for player_stats in stats:
            if player_stats.player and player_name.lower() in player_stats.player.lower():
                if player_stats.stat and float(player_stats.stat) > 20:
                    return {
                        'season': season,
                        'team': team,
                        'attempts': float(player_stats.stat) if player_stats.stat else 0
                    }

        return None

    except ApiException:
        return None
    except Exception:
        return None


def fetch_player_pbp_for_season(api_instance, player_name, team, season):
    """Fetch all play-by-play data for a player in a specific season"""
    all_plays = []

    try:
        for week in range(1, 16):
            try:
                plays = api_instance.get_plays(
                    year=season,
                    week=week,
                    team=team,
                    play_type='pass'
                )

                if plays:
                    for play in plays:
                        if (play.offense == team and
                            play.passer_player_name and
                            player_name.lower() in play.passer_player_name.lower()):
                            all_plays.append(play)

                time.sleep(0.3)

            except ApiException as e:
                if e.status == 401:
                    return None, "AUTH"
                elif e.status == 429:
                    return None, "RATE_LIMIT"
                continue
            except Exception:
                continue

        return all_plays, None

    except Exception as e:
        return None, str(e)[:50]


def extract_play_data(play):
    """Extract relevant data from a play object"""
    return {
        'down': play.down if hasattr(play, 'down') else None,
        'distance': play.distance if hasattr(play, 'distance') else None,
        'yards_gained': play.yards_gained if hasattr(play, 'yards_gained') else None,
        'play_type': play.play_type if hasattr(play, 'play_type') else None,
        'epa': play.epa if hasattr(play, 'epa') else None,
        'scoring': play.scoring if hasattr(play, 'scoring') else False,
        'yards_to_goal': play.yards_to_goal if hasattr(play, 'yards_to_goal') else None,
    }


def collect_qb_career(row, stats_api, plays_api, output_dir):
    """
    Collect career data for a single QB

    Returns: (success, error_type)
    """
    player_name = row['player_name']
    draft_year = row['draft_year']
    college = row['college']

    # Determine potential college seasons
    potential_seasons = determine_college_seasons(draft_year, college)

    # Find which seasons they actually played
    active_seasons = []

    for season in potential_seasons:
        season_check = fetch_player_season_stats(stats_api, player_name, college, season)

        if season_check:
            active_seasons.append(season)

        time.sleep(0.3)

    if not active_seasons:
        active_seasons = [draft_year - 1]

    # Collect play-by-play for all active seasons
    all_career_plays = []
    hit_error = None

    for season in active_seasons:
        plays, error = fetch_player_pbp_for_season(plays_api, player_name, college, season)

        if error:
            hit_error = error
            break

        if plays:
            season_plays = []
            for play in plays:
                play_data = extract_play_data(play)
                play_data['season'] = season
                play_data['player_name'] = player_name
                play_data['college'] = college
                play_data['draft_year'] = draft_year
                season_plays.append(play_data)

            all_career_plays.extend(season_plays)

    if hit_error:
        return False, hit_error

    if all_career_plays:
        # Save this QB's career data
        df_career = pd.DataFrame(all_career_plays)
        career_file = output_dir / f"{player_name.replace(' ', '_')}_{draft_year}.csv"
        df_career.to_csv(career_file, index=False)
        return True, None
    else:
        # No plays found, but not an error - save empty marker
        df_empty = pd.DataFrame([{
            'player_name': player_name,
            'draft_year': draft_year,
            'college': college,
            'note': 'No plays found'
        }])
        career_file = output_dir / f"{player_name.replace(' ', '_')}_{draft_year}.csv"
        df_empty.to_csv(career_file, index=False)
        return True, None


def main():
    """Main execution with retry logic"""
    print("="*80)
    print("COLLECTING FULL COLLEGE CAREERS (WITH RETRY)")
    print("="*80)
    print("\nWill retry every 30 minutes until all data is collected")
    print("Press Ctrl+C to stop")

    root = get_project_root()

    # Load expanded QB list
    print("\n[1] Loading QB list...")
    qb_list_path = root / 'data' / 'processed' / 'qb_list_expanded.csv'

    if not qb_list_path.exists():
        print("    [ERROR] QB list not found")
        return

    df_qbs = pd.read_csv(qb_list_path)
    print(f"    [OK] {len(df_qbs)} QBs to collect")

    # Setup output directory
    output_dir = root / 'data' / 'processed' / 'full_college_careers'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load API key
    print("\n[2] Configuring CFBD API...")
    try:
        api_key = load_api_key()
        configuration = cfbd.Configuration()
        configuration.api_key['Authorization'] = api_key
        configuration.api_key_prefix['Authorization'] = 'Bearer'
        print("    [OK] API configured")
    except Exception as e:
        print(f"    [ERROR] {str(e)}")
        return

    stats_api = cfbd.StatsApi(cfbd.ApiClient(configuration))
    plays_api = cfbd.PlaysApi(cfbd.ApiClient(configuration))

    retry_count = 0
    max_retries = 100

    while retry_count < max_retries:
        print(f"\n" + "="*80)
        print(f"ATTEMPT #{retry_count + 1} - {datetime.now().strftime('%H:%M:%S')}")
        print("="*80)

        # Check what we already have
        collected_qbs = get_already_collected_qbs(output_dir)

        if collected_qbs:
            print(f"\n[3] Already collected: {len(collected_qbs)} QBs")

        # Determine what we still need
        remaining_qbs = []
        for _, row in df_qbs.iterrows():
            if (row['player_name'], row['draft_year']) not in collected_qbs:
                remaining_qbs.append(row)

        if not remaining_qbs:
            print("\n[SUCCESS] All QBs collected!")
            break

        print(f"\n[4] Still need: {len(remaining_qbs)} QBs")

        # Try to collect remaining QBs
        print(f"\n[5] Collecting...")
        print("-"*80)

        newly_collected = []
        hit_error = False
        error_type = None

        for idx, row in enumerate(remaining_qbs):
            player_name = row['player_name']
            draft_year = row['draft_year']

            print(f"  [{idx+1}/{len(remaining_qbs)}] {player_name} ({draft_year})...", end=' ', flush=True)

            success, error = collect_qb_career(row, stats_api, plays_api, output_dir)

            if success:
                newly_collected.append(player_name)
                print("[OK]")
            elif error == "AUTH":
                print("[AUTH ERROR]")
                hit_error = True
                error_type = "AUTH"
                break
            elif error == "RATE_LIMIT":
                print("[RATE LIMIT]")
                hit_error = True
                error_type = "RATE_LIMIT"
                break
            else:
                print(f"[ERROR] {error}")

            time.sleep(0.5)

        # Status update
        if newly_collected:
            print(f"\n    New this round: {len(newly_collected)} QBs")

        collected_qbs = get_already_collected_qbs(output_dir)
        remaining_count = len(df_qbs) - len(collected_qbs)

        print(f"\n    Progress: {len(collected_qbs)}/{len(df_qbs)} QBs")
        print(f"    Complete: {len(collected_qbs)/len(df_qbs)*100:.1f}%")

        if not remaining_count:
            print("\n[SUCCESS] All data collected!")
            break

        if hit_error or not newly_collected:
            wait_seconds = 1800  # 30 minutes
            print(f"\n    Waiting 30 minutes before retry...")
            print(f"    Will resume at {(datetime.now().replace(second=0, microsecond=0) + pd.Timedelta(seconds=wait_seconds)).strftime('%H:%M')}")

            try:
                for i in range(wait_seconds // 60):
                    time.sleep(60)
                    remaining_mins = (wait_seconds // 60) - i - 1
                    if remaining_mins > 0:
                        print(f"    {remaining_mins} minutes remaining...")
            except KeyboardInterrupt:
                print("\n\n[INTERRUPTED] Stopping retry loop")
                print(f"Progress saved: {len(collected_qbs)}/{len(df_qbs)} QBs")
                return

        retry_count += 1

    # Final summary and create summary file
    print("\n" + "="*80)
    print("COLLECTION COMPLETE")
    print("="*80)

    collected_qbs = get_already_collected_qbs(output_dir)

    if len(collected_qbs) == len(df_qbs):
        print("\n[SUCCESS] All QBs collected!")
    else:
        print(f"\n[PARTIAL] Collected {len(collected_qbs)}/{len(df_qbs)} QBs")

    # Build career summary
    print("\n[6] Building career summary...")
    career_summary = []

    for file in output_dir.glob('*_*.csv'):
        if file.name == 'career_summary.csv':
            continue

        try:
            df = pd.read_csv(file)

            if len(df) > 0 and 'season' in df.columns:
                player_name = df['player_name'].iloc[0]
                draft_year = df['draft_year'].iloc[0]
                college = df['college'].iloc[0]
                seasons = sorted(df['season'].unique())

                career_summary.append({
                    'player_name': player_name,
                    'draft_year': draft_year,
                    'college': college,
                    'seasons_played': len(seasons),
                    'first_season': min(seasons),
                    'final_season': max(seasons),
                    'total_plays': len(df),
                    'file': file.name
                })
        except:
            pass

    if career_summary:
        df_summary = pd.DataFrame(career_summary)
        summary_path = output_dir / 'career_summary.csv'
        df_summary.to_csv(summary_path, index=False)

        print(f"    [OK] Summary saved: {len(df_summary)} QBs")
        print(f"    Average seasons: {df_summary['seasons_played'].mean():.1f}")
        print(f"    Total plays: {df_summary['total_plays'].sum():,}")

    print("\n    Next: Run create_multi_year_aggregations.py")


if __name__ == "__main__":
    main()
