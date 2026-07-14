"""
Collect Full College Career Data for All QBs

Expands beyond final college season to capture complete college careers.
This addresses the critical limitation that we've been discarding years of data
for multi-year starters like Mahomes, Mayfield, etc.
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


def determine_college_seasons(draft_year, college):
    """
    Estimate all college seasons for a QB based on draft year

    Most QBs play 3-4 years. Final college season is typically draft_year - 1.
    We'll search backward to find all seasons.
    """
    final_season = draft_year - 1

    # Most QBs: 3-5 year careers (including redshirt)
    # Start searching from 5 years before draft
    potential_seasons = list(range(final_season - 4, final_season + 1))

    return potential_seasons


def fetch_player_season_stats(api_instance, player_name, team, season):
    """
    Fetch a player's stats for a specific season

    Returns None if player didn't play that season
    """
    try:
        stats = api_instance.get_player_season_stats(
            year=season,
            team=team,
            category='passing'
        )

        if not stats:
            return None

        # Find this player in the results
        for player_stats in stats:
            if player_stats.player and player_name.lower() in player_stats.player.lower():
                # Check if they had meaningful playing time
                if player_stats.stat and float(player_stats.stat) > 20:  # At least 20 attempts
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
    """
    Fetch all play-by-play data for a player in a specific season
    """
    all_plays = []

    try:
        # Fetch week by week (weeks 1-15 for regular season)
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
                        # Check if this player was the passer
                        if (play.offense == team and
                            play.passer_player_name and
                            player_name.lower() in play.passer_player_name.lower()):
                            all_plays.append(play)

                time.sleep(0.3)  # Rate limiting

            except ApiException:
                continue
            except Exception:
                continue

        return all_plays

    except Exception as e:
        print(f"      [ERROR] {str(e)[:50]}")
        return []


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


def main():
    """Main execution"""
    print("="*80)
    print("COLLECTING FULL COLLEGE CAREER DATA")
    print("="*80)
    print("\nGathering complete college careers (all seasons, not just final year)")

    root = get_project_root()

    # Load expanded QB list
    print("\n[1] Loading expanded QB list...")
    qb_list_path = root / 'data' / 'processed' / 'qb_list_expanded.csv'

    if not qb_list_path.exists():
        print("    [ERROR] Expanded QB list not found. Run create_expanded_qb_list.py first")
        return

    df_qbs = pd.read_csv(qb_list_path)
    print(f"    [OK] Loaded {len(df_qbs)} QBs")
    print(f"    Draft years: {df_qbs['draft_year'].min()}-{df_qbs['draft_year'].max()}")

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

    # Create output directory
    output_dir = root / 'data' / 'processed' / 'full_college_careers'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track progress
    print(f"\n[3] Collecting career data for {len(df_qbs)} QBs...")
    print("-"*80)

    career_summary = []
    failed_qbs = []

    for idx, row in df_qbs.iterrows():
        player_name = row['player_name']
        draft_year = row['draft_year']
        college = row['college']

        print(f"\n  [{idx+1}/{len(df_qbs)}] {player_name} ({college}, {draft_year} draft)")

        # Determine potential college seasons
        potential_seasons = determine_college_seasons(draft_year, college)

        # Find which seasons they actually played
        print(f"    Checking seasons: {potential_seasons}")
        active_seasons = []

        for season in potential_seasons:
            # Quick check: did they have stats this season?
            season_check = fetch_player_season_stats(stats_api, player_name, college, season)

            if season_check:
                active_seasons.append(season)
                print(f"      {season}: Found ({season_check['attempts']:.0f} attempts)")

            time.sleep(0.3)  # Rate limiting

        if not active_seasons:
            print(f"    [WARN] No active seasons found - using final season only")
            active_seasons = [draft_year - 1]
            failed_qbs.append(player_name)

        # Collect play-by-play for all active seasons
        all_career_plays = []

        for season in active_seasons:
            print(f"    Fetching PBP for {season}...", end=' ', flush=True)

            plays = fetch_player_pbp_for_season(plays_api, player_name, college, season)

            if plays:
                # Convert to dictionaries
                season_plays = []
                for play in plays:
                    play_data = extract_play_data(play)
                    play_data['season'] = season
                    play_data['player_name'] = player_name
                    play_data['college'] = college
                    play_data['draft_year'] = draft_year
                    season_plays.append(play_data)

                all_career_plays.extend(season_plays)
                print(f"[OK] {len(plays)} plays")
            else:
                print(f"[WARN] No plays found")

        if all_career_plays:
            # Save this QB's career data
            df_career = pd.DataFrame(all_career_plays)
            career_file = output_dir / f"{player_name.replace(' ', '_')}_{draft_year}.csv"
            df_career.to_csv(career_file, index=False)

            career_summary.append({
                'player_name': player_name,
                'draft_year': draft_year,
                'college': college,
                'seasons_played': len(active_seasons),
                'first_season': min(active_seasons),
                'final_season': max(active_seasons),
                'total_plays': len(all_career_plays),
                'file': career_file.name
            })

            print(f"    [COMPLETE] {len(active_seasons)} seasons, {len(all_career_plays)} total plays")
        else:
            print(f"    [FAILED] No play-by-play data collected")
            failed_qbs.append(player_name)

    # Save summary
    print("\n" + "="*80)
    print("COLLECTION SUMMARY")
    print("="*80)

    if career_summary:
        df_summary = pd.DataFrame(career_summary)
        summary_path = output_dir / 'career_summary.csv'
        df_summary.to_csv(summary_path, index=False)

        print(f"\n[OK] Collected career data for {len(df_summary)} QBs")
        print(f"    Average seasons per QB: {df_summary['seasons_played'].mean():.1f}")
        print(f"    Total plays collected: {df_summary['total_plays'].sum():,}")
        print(f"    Saved to: {output_dir}")

        # Distribution by number of seasons
        print(f"\n    QBs by number of college seasons:")
        season_counts = df_summary['seasons_played'].value_counts().sort_index()
        for seasons, count in season_counts.items():
            print(f"      {seasons} seasons: {count} QBs")

        # Multi-year starters we're now capturing
        multi_year = df_summary[df_summary['seasons_played'] >= 3]
        print(f"\n    Multi-year starters (3+ seasons): {len(multi_year)} QBs")
        if len(multi_year) > 0:
            print(f"    Example multi-year careers:")
            for _, qb in multi_year.head(5).iterrows():
                print(f"      {qb['player_name']}: {qb['seasons_played']} seasons, {qb['total_plays']} plays")

    if failed_qbs:
        print(f"\n[WARN] Failed to collect data for {len(failed_qbs)} QBs:")
        for name in failed_qbs[:10]:
            print(f"      {name}")
        if len(failed_qbs) > 10:
            print(f"      ... and {len(failed_qbs) - 10} more")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n  1. Load SP+ ratings (when available)")
    print("  2. Calculate stats for each aggregation method:")
    print("     - Final season only (baseline)")
    print("     - Career average")
    print("     - Recency-weighted")
    print("     - Cumulative totals")
    print("     - Best single season")
    print("     - Volume-weighted")
    print("  3. Test which aggregation predicts NFL outcomes best")


if __name__ == "__main__":
    main()
