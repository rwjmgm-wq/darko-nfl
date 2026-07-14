"""
Extract SP+ Ratings from Cached CFB Betting Model Data

Uses cached SP+ CSV files from the betting model to supplement
opponent defensive ratings for play-by-play data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import ast
import glob


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent


def parse_dict_field(field_str):
    """Parse string representation of dict to extract rating"""
    if pd.isna(field_str) or not field_str:
        return np.nan

    try:
        # Parse string representation of dict
        field_dict = ast.literal_eval(field_str)
        return field_dict.get('rating', np.nan)
    except:
        return np.nan


def load_cached_sp_plus(betting_model_path, seasons):
    """
    Load SP+ ratings from cached CSV files in betting model

    Returns dict: {season: {team: {'overall': X, 'offense': Y, 'defense': Z}}}
    """
    sp_plus_data = {}

    for season in seasons:
        season_int = int(season)
        sp_plus_data[season_int] = {}

        # Find all SP+ files for this season
        pattern = f"{betting_model_path}/data/raw/sp_plus_{season_int}_week*.csv"
        files = glob.glob(pattern)

        if not files:
            print(f"    [--] No cached SP+ data for {season_int}")
            continue

        # Use the last week file (most complete data)
        files.sort()
        last_file = files[-1]

        try:
            df = pd.read_csv(last_file)

            # Parse each team's ratings
            for _, row in df.iterrows():
                team = row['team']
                overall = row.get('rating', np.nan)

                # Parse offense/defense dicts
                offense_rating = parse_dict_field(row.get('offense'))
                defense_rating = parse_dict_field(row.get('defense'))

                sp_plus_data[season_int][team] = {
                    'overall': overall,
                    'offense': offense_rating,
                    'defense': defense_rating
                }

            print(f"    [OK] Loaded {len(sp_plus_data[season_int])} teams for {season_int} from cache")

        except Exception as e:
            print(f"    [ERROR] Failed to load {last_file}: {str(e)[:100]}")

    return sp_plus_data


def normalize_team_name(team_name):
    """Normalize team names for SP+ matching"""
    if pd.isna(team_name):
        return None

    # Common abbreviations and variations
    replacements = {
        'Florida St.': 'Florida State',
        'North Carolina St.': 'NC State',
        'Mississippi St.': 'Mississippi State',
        'Colorado St.': 'Colorado State',
        'Oregon St.': 'Oregon State',
        'Iowa St.': 'Iowa State',
        'Ohio St.': 'Ohio State',
        'Penn St.': 'Penn State',
        'Michigan St.': 'Michigan State',
        'Boise St.': 'Boise State',
        'San Diego St.': 'San Diego State',
        'Fresno St.': 'Fresno State',
        'San Jose St.': 'San Jose State',
        'Washington St.': 'Washington State',
        'Kansas St.': 'Kansas State',
        'Miami (FL)': 'Miami',
        'Louisiana State': 'LSU',
        'USC': 'Southern California',
        'UCF': 'Central Florida',
        'Southern Miss': 'Southern Mississippi',
        'UConn': 'Connecticut',
        'UMass': 'Massachusetts',
    }

    return replacements.get(team_name, team_name)


def main():
    """Main execution"""
    print("="*80)
    print("EXTRACTING SP+ FROM CACHED BETTING MODEL DATA")
    print("="*80)

    root = get_project_root()
    betting_model_path = Path("C:/Users/rwjmg/OneDrive/Pictures/Writing/new_cfb_betting_model")

    # Load play-by-play data
    print("\n[1] Loading play-by-play data...")
    plays_path = root / 'data' / 'processed' / 'play_by_play' / 'all_qb_plays_full.csv'
    df_plays = pd.read_csv(plays_path)
    print(f"    [OK] Loaded {len(df_plays):,} plays")

    # Identify seasons we need
    seasons = sorted(df_plays['season'].unique())
    print(f"\n[2] Need SP+ data for seasons: {list(seasons)}")

    # Load cached SP+ data
    print(f"\n[3] Loading cached SP+ data from betting model...")
    sp_plus_cache = load_cached_sp_plus(betting_model_path, seasons)

    available_seasons = list(sp_plus_cache.keys())
    print(f"\n    Available seasons: {available_seasons}")
    print(f"    Missing seasons: {sorted(set(seasons) - set(available_seasons))}")

    # Normalize team names
    print(f"\n[4] Normalizing team names...")
    df_plays['opponent_normalized'] = df_plays['opponent'].apply(normalize_team_name)
    df_plays['team_normalized'] = df_plays['team'].apply(normalize_team_name)

    # Merge opponent defensive SP+
    print(f"\n[5] Merging opponent SP+ into plays...")

    def get_opponent_defense_sp(row):
        """Get opponent defensive SP+ for a play"""
        season = row['season']
        opponent = row['opponent_normalized']

        if season not in sp_plus_cache or opponent not in sp_plus_cache[season]:
            return np.nan

        return sp_plus_cache[season][opponent].get('defense', np.nan)

    def get_team_offense_sp(row):
        """Get team offensive SP+ for a play"""
        season = row['season']
        team = row['team_normalized']

        if season not in sp_plus_cache or team not in sp_plus_cache[season]:
            return np.nan

        return sp_plus_cache[season][team].get('offense', np.nan)

    # Add SP+ columns (create if they don't exist)
    print("    Adding opponent defensive SP+...")
    new_opponent_sp = df_plays.apply(get_opponent_defense_sp, axis=1)
    if 'opponent_defense_sp' in df_plays.columns:
        df_plays['opponent_defense_sp'] = df_plays['opponent_defense_sp'].fillna(new_opponent_sp)
    else:
        df_plays['opponent_defense_sp'] = new_opponent_sp

    print("    Adding team offensive SP+...")
    new_team_sp = df_plays.apply(get_team_offense_sp, axis=1)
    if 'team_offense_sp' in df_plays.columns:
        df_plays['team_offense_sp'] = df_plays['team_offense_sp'].fillna(new_team_sp)
    else:
        df_plays['team_offense_sp'] = new_team_sp

    # Calculate coverage
    total_plays = len(df_plays)
    plays_with_opp_sp = df_plays['opponent_defense_sp'].notna().sum()
    plays_with_team_sp = df_plays['team_offense_sp'].notna().sum()

    print(f"\n    Coverage AFTER adding cached data:")
    print(f"        Opponent defense SP+: {plays_with_opp_sp:,}/{total_plays:,} ({plays_with_opp_sp/total_plays*100:.1f}%)")
    print(f"        Team offense SP+:     {plays_with_team_sp:,}/{total_plays:,} ({plays_with_team_sp/total_plays*100:.1f}%)")

    # Coverage by season
    print(f"\n    Coverage by season:")
    for season in sorted(seasons):
        season_plays = df_plays[df_plays['season'] == season]
        season_coverage = season_plays['opponent_defense_sp'].notna().sum() / len(season_plays) * 100
        print(f"        {season}: {season_coverage:.1f}%")

    # Save enhanced dataset
    print(f"\n[6] Saving enhanced play-by-play data...")
    output_path = root / 'data' / 'processed' / 'play_by_play' / 'all_qb_plays_with_sp_plus.csv'
    df_plays.to_csv(output_path, index=False)
    print(f"    [OK] Saved to: {output_path}")

    # Summary
    print("\n" + "="*80)
    print("SP+ EXTRACTION FROM CACHE COMPLETE")
    print("="*80)
    print(f"\nDataset Summary:")
    print(f"    Total plays: {len(df_plays):,}")
    print(f"    Plays with opponent SP+: {plays_with_opp_sp:,} ({plays_with_opp_sp/len(df_plays)*100:.1f}%)")
    print(f"    Cached seasons used: {available_seasons}")
    print(f"    Missing seasons: {sorted(set(seasons) - set(available_seasons))}")

    print(f"\nNext Step: Re-run College LEAF calculation with improved SP+ coverage")


if __name__ == "__main__":
    main()
