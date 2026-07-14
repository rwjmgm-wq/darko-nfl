"""
Collect Missing QB Data

Collects career data for QBs that are missing from the database:
- 2025 draft QBs without career files (Cameron Ward, etc.)
- High-profile 2026 prospects (Ty Simpson, Dante Moore)
"""

import pandas as pd
import cfbd
from pathlib import Path
import time


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent


def collect_qb_career(player_name, team, seasons, api_key):
    """
    Collect career data for a QB using CFBD API
    """
    configuration = cfbd.Configuration(access_token=api_key)
    api_client = cfbd.ApiClient(configuration)
    api_instance = cfbd.PlaysApi(api_client)

    all_plays = []

    for season in seasons:
        print(f"    Season {season}...", end=' ')
        season_plays = 0

        for week in range(1, 16):
            try:
                plays = api_instance.get_plays(
                    year=season,
                    week=week,
                    season_type='regular',
                    team=team
                )

                if not plays:
                    continue

                # Filter for QB passes
                for play in plays:
                    if not hasattr(play, 'play_text') or not play.play_text:
                        continue

                    play_text = play.play_text.lower()
                    player_name_lower = player_name.lower()

                    # Check for player name in play text
                    if play_text.startswith(player_name_lower + ' pass'):
                        play_found = True
                    else:
                        # Try last name only (must be >3 chars)
                        last_name = player_name.split()[-1].lower()
                        play_found = (
                            play_text.startswith(last_name + ' pass') and
                            len(last_name) > 3
                        )

                    if not play_found:
                        continue

                    # Add play data
                    play_data = {
                        'season': season,
                        'week': week,
                        'offense': play.offense,
                        'defense': play.defense,
                        'down': play.down,
                        'distance': play.distance,
                        'yards_to_goal': play.yards_to_goal if hasattr(play, 'yards_to_goal') else None,
                        'play_type': 'pass',
                        'epa': play.ppa if hasattr(play, 'ppa') else None,
                        'yards_gained': play.yards_gained if hasattr(play, 'yards_gained') else None,
                        'play_text': play.play_text,
                        'success': play.ppa > 0 if hasattr(play, 'ppa') and play.ppa is not None else None,
                        'touchdown': 'touchdown' in play.play_text.lower() if play.play_text else False
                    }
                    all_plays.append(play_data)
                    season_plays += 1

                time.sleep(0.1)  # Rate limiting

            except Exception as e:
                if '429' in str(e):  # Rate limit
                    print(f"[Rate limit, waiting 60s]", end=' ')
                    time.sleep(60)
                    continue
                # Ignore other errors (week doesn't exist, etc.)
                pass

        print(f"{season_plays} plays")

    return all_plays


def main():
    print("="*80)
    print("COLLECTING MISSING QB DATA")
    print("="*80)

    root = get_project_root()

    # Load API key from .env file
    api_key_file = root / '.env'
    if not api_key_file.exists():
        print("\n[ERROR] No .env file found")
        print("        Create a .env file with: CFBD_API_KEY=your_key")
        return

    api_key = None
    with open(api_key_file) as f:
        for line in f:
            if line.startswith('CFBD_API_KEY='):
                api_key = line.strip().split('=')[1].strip()
                break

    if not api_key:
        print("\n[ERROR] CFBD_API_KEY not found in .env file")
        return

    # QBs to collect
    missing_qbs = [
        # 2025 draft QBs without career files
        {'player_name': 'Cameron Ward', 'team': 'Miami', 'seasons': [2020, 2021, 2022, 2023, 2024], 'draft_year': 2025},

        # High-profile 2026 prospects
        {'player_name': 'Ty Simpson', 'team': 'Alabama', 'seasons': [2023, 2024], 'draft_year': 2026},
        {'player_name': 'Dante Moore', 'team': 'Oregon', 'seasons': [2023, 2024], 'draft_year': 2026},
    ]

    print(f"\n[1] Collecting {len(missing_qbs)} QBs...")
    print("-"*80)

    careers_dir = root / 'data' / 'processed' / 'full_college_careers'
    careers_dir.mkdir(parents=True, exist_ok=True)

    summary_data = []

    for idx, qb in enumerate(missing_qbs):
        player_name = qb['player_name']
        team = qb['team']
        seasons = qb['seasons']
        draft_year = qb['draft_year']

        print(f"\n  [{idx+1}/{len(missing_qbs)}] {player_name} ({team})")

        # Collect plays
        plays = collect_qb_career(player_name, team, seasons, api_key)

        if len(plays) == 0:
            print(f"       [FAILED] No plays found")
            continue

        # Save career file
        df_career = pd.DataFrame(plays)
        filename = f"{player_name.lower().replace(' ', '_')}_{draft_year}.csv"
        career_path = careers_dir / filename
        df_career.to_csv(career_path, index=False)

        # Add to summary
        summary_data.append({
            'player_name': player_name,
            'draft_year': draft_year,
            'college': team,
            'seasons_played': len(seasons),
            'total_attempts': len(plays),
            'file': filename
        })

        print(f"       [OK] {len(plays)} plays -> {filename}")

    # Update career summary
    if summary_data:
        summary_path = careers_dir / 'career_summary.csv'

        if summary_path.exists():
            df_existing = pd.read_csv(summary_path)
            df_new = pd.DataFrame(summary_data)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            # Remove duplicates
            df_combined = df_combined.drop_duplicates(['player_name', 'draft_year'], keep='last')
            df_combined.to_csv(summary_path, index=False)
        else:
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_csv(summary_path, index=False)

        print(f"\n[2] Updated career summary")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n  1. Add missing QBs to qbs_2026_prospects.csv (Ty Simpson, Dante Moore)")
    print("  2. Re-run: python src/apply_sp_plus_to_careers.py")
    print("  3. Re-run: python src/build_projection_database.py")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
