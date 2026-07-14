"""
Collect Full College Careers for 2026 QB Prospects using CFBD API

Reads the list of 2026 prospects and collects their play-by-play data.
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
        print("[ERROR] No .env file found")
        print("Create a .env file with: CFBD_API_KEY=your_key")
        print("Get a free API key at https://collegefootballdata.com/")
        return None

    with open(api_key_file) as f:
        for line in f:
            if line.startswith('CFBD_API_KEY='):
                return line.strip().split('=')[1].strip()

    print("[ERROR] CFBD_API_KEY not found in .env file")
    return None


def fetch_qb_plays_for_season(api_instance, player_name, team, season):
    """Fetch all passing plays for a QB in a season (week by week)"""
    all_plays = []

    print(f"      {season}...", end=' ', flush=True)

    play_count = 0

    # Fetch week by week (weeks 1-15 for regular season)
    for week in range(1, 16):
        try:
            plays = api_instance.get_plays(
                year=season,
                week=week,
                team=team
            )

            if plays:
                for play in plays:
                    # Check if this is a passing play by this QB
                    # Player name is in play_text field, e.g. "Caleb Williams pass complete..."
                    if (play.offense == team and
                        hasattr(play, 'play_text') and play.play_text and
                        hasattr(play, 'play_type') and play.play_type):

                        # Check if this is a passing play
                        passing_play_types = ['Pass Reception', 'Passing Touchdown', 'Pass Incompletion',
                                             'Pass Interception Return', 'Sack']
                        if play.play_type not in passing_play_types:
                            continue

                        # Extract passer name from play_text (e.g. "Caleb Williams pass complete...")
                        play_text = play.play_text.lower()
                        player_name_lower = player_name.lower()

                        # Check if player name appears at start of play text followed by " pass"
                        if not play_text.startswith(player_name_lower + ' pass'):
                            # Try last name only
                            last_name = player_name.split()[-1].lower()
                            if not (play_text.startswith(last_name + ' pass') and len(last_name) > 3):
                                continue

                        # Determine completion status
                        is_completion = 1 if play.play_type in ['Pass Reception', 'Passing Touchdown'] else 0
                        is_interception = 1 if 'Interception' in str(play.play_type) else 0
                        is_touchdown = 1 if 'Touchdown' in str(play.play_type) else 0

                        # Get EPA (called 'ppa' in CFBD)
                        epa = play.ppa if hasattr(play, 'ppa') and play.ppa else None

                        play_data = {
                            'season': season,
                            'player_name': player_name,
                            'college': team,
                            'down': play.down if hasattr(play, 'down') and play.down else None,
                            'distance': play.distance if hasattr(play, 'distance') and play.distance else None,
                            'yards_gained': play.yards_gained if hasattr(play, 'yards_gained') and play.yards_gained else 0,
                            'epa': epa,
                            'success': 1 if (epa and epa > 0) else 0,
                            'pass': 1,
                            'completion': is_completion,
                            'interception': is_interception,
                            'touchdown': is_touchdown
                        }
                        all_plays.append(play_data)
                        play_count += 1

            time.sleep(0.2)  # Rate limiting

        except ApiException:
            continue
        except Exception:
            continue

    if play_count > 0:
        print(f"[{play_count} plays]", end='')
    else:
        print("[No plays]", end='')

    return all_plays


def collect_qb_career_cfbd(api_instance, player_name, team, draft_year):
    """Collect full career for a QB using CFBD"""
    print(f"\n  {player_name} ({team}, {draft_year})")

    # Determine seasons to collect (2020-2024 for 2026 prospects)
    final_season = 2024
    seasons = list(range(final_season - 4, final_season + 1))  # 5 years back

    all_plays = []
    for season in seasons:
        plays = fetch_qb_plays_for_season(api_instance, player_name, team, season)
        all_plays.extend(plays)

    print()  # New line

    if len(all_plays) == 0:
        print(f"    [FAILED] No plays found")
        return None

    df = pd.DataFrame(all_plays)
    print(f"    [SUCCESS] {len(df)} career plays across {df['season'].nunique()} seasons")

    return df


def main():
    """Collect 2026 QB data using CFBD"""
    print("=" * 80)
    print("COLLECTING 2026 QB PROSPECT DATA USING CFBD API")
    print("=" * 80)

    root = get_project_root()

    # Load API key
    print("\n[1] Loading CFBD API key...")
    api_key = load_api_key()
    if not api_key:
        return

    # Configure API (updated method for cfbd-python)
    configuration = cfbd.Configuration(
        access_token=api_key
    )

    api_client = cfbd.ApiClient(configuration)
    api_instance = cfbd.PlaysApi(api_client)

    print("    [OK] API configured")

    # Load 2026 prospects list
    print("\n[2] Loading 2026 QB prospects...")
    prospects_file = root / 'data' / 'raw' / 'qbs_2026_prospects.csv'

    if not prospects_file.exists():
        print("    [ERROR] qbs_2026_prospects.csv not found")
        print("    Run identify_2026_qbs.py first")
        return

    prospects = pd.read_csv(prospects_file)
    print(f"    [OK] Found {len(prospects)} QB prospects")

    # Check which already have data
    careers_dir = root / 'data' / 'processed' / 'full_college_careers'
    careers_dir.mkdir(parents=True, exist_ok=True)

    existing = set(f.stem.replace('_', ' ') for f in careers_dir.glob('*.csv'))

    qbs_to_collect = []
    for _, qb in prospects.iterrows():
        if qb['player_name'] not in existing:
            qbs_to_collect.append(qb)

    print(f"\n[3] Already collected: {len(prospects) - len(qbs_to_collect)} QBs")
    print(f"    Need to collect: {len(qbs_to_collect)} QBs")

    if len(qbs_to_collect) == 0:
        print("\n[OK] All QBs already collected!")
        return

    # Collect data
    print(f"\n[4] Collecting QB careers...")
    print("-" * 80)

    collected = 0
    failed = 0

    for idx, qb in enumerate(qbs_to_collect, 1):
        print(f"[{idx}/{len(qbs_to_collect)}]")

        df_career = collect_qb_career_cfbd(
            api_instance,
            qb['player_name'],
            qb['team'],
            int(qb['draft_year'])
        )

        if df_career is not None and len(df_career) > 0:
            # Save career data
            filename = qb['player_name'].replace(' ', '_') + '.csv'
            career_path = careers_dir / filename
            df_career.to_csv(career_path, index=False)
            collected += 1
        else:
            failed += 1

        # Progress update every 10 QBs
        if idx % 10 == 0:
            print(f"\n    Progress: {collected} collected, {failed} failed\n")

    # Summary
    print("\n" + "=" * 80)
    print("COLLECTION SUMMARY")
    print("=" * 80)
    print(f"\nSuccessfully collected: {collected} QBs")
    print(f"Failed: {failed} QBs")
    print(f"\nCareer data saved to: {careers_dir}")


if __name__ == "__main__":
    main()
