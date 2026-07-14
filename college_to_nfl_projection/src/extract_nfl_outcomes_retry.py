"""
Extract NFL Outcomes with Automatic Retry

Keeps trying every 30 minutes if NFL data download fails.
Useful for handling network issues or temporary API problems.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import nfl_data_py as nfl
import time
from datetime import datetime


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent


def convert_to_nfl_name(full_name):
    """Convert full name to nfl_data_py abbreviated format"""
    parts = full_name.split()
    if len(parts) < 2:
        return full_name

    first_initial = parts[0][0]
    last_name = parts[-1]

    return f"{first_initial}.{last_name}"


def calculate_leaf_rating(df_games):
    """Calculate LEAF rating"""
    if len(df_games) == 0 or df_games['attempts'].sum() == 0:
        return np.nan

    total_attempts = df_games['attempts'].sum()
    weights = df_games['attempts'] / total_attempts
    leaf = (df_games['epa'] * weights).sum()

    return leaf


def extract_rookie_leaf(df_pbp, player_name, draft_year):
    """Outcome 1: Rookie season LEAF performance"""
    rookie_season = draft_year

    df_rookie = df_pbp[
        (df_pbp['season'] == rookie_season) &
        (df_pbp['passer_player_name'] == player_name)
    ]

    if len(df_rookie) == 0:
        return np.nan

    game_stats = df_rookie.groupby('game_id').agg({
        'epa': 'mean',
        'passer_player_name': 'count'
    }).rename(columns={'passer_player_name': 'attempts'})

    leaf = calculate_leaf_rating(game_stats)
    return leaf


def extract_career_starts(df_pbp, player_name):
    """Count total career starts (10+ attempts)"""
    df_player = df_pbp[df_pbp['passer_player_name'] == player_name]

    if len(df_player) == 0:
        return 0

    game_attempts = df_player.groupby(['season', 'week']).size()
    starts = (game_attempts >= 10).sum()

    return starts


def extract_85_starts_outcome(df_pbp, player_name, draft_year):
    """Outcome 2: Reached 85+ career starts"""
    if draft_year >= 2020:
        return np.nan

    starts = extract_career_starts(df_pbp, player_name)
    return 1 if starts >= 85 else 0


def calculate_rolling_epa(df_pbp, player_name, window=10):
    """Calculate rolling window EPA per play"""
    df_player = df_pbp[df_pbp['passer_player_name'] == player_name].copy()

    if len(df_player) == 0:
        return np.nan, np.nan

    df_player = df_player.sort_values(['season', 'week'])

    game_stats = df_player.groupby(['season', 'week']).agg({
        'epa': 'mean'
    }).reset_index()

    if len(game_stats) < window:
        return np.nan, np.nan

    game_stats['rolling_epa'] = game_stats['epa'].rolling(window=window, min_periods=window).mean()

    max_rolling = game_stats['rolling_epa'].max()
    min_rolling = game_stats['rolling_epa'].min()

    return max_rolling, min_rolling


def extract_bust_outcome(df_pbp, player_name):
    """Outcome 3: Bust classification"""
    starts = extract_career_starts(df_pbp, player_name)

    if starts <= 35:
        return 1

    max_rolling, _ = calculate_rolling_epa(df_pbp, player_name, window=10)

    if pd.isna(max_rolling):
        return 1

    if max_rolling <= -0.035:
        return 1

    return 0


def extract_elite_peak_outcome(df_pbp, player_name):
    """Outcome 4: Elite peak"""
    max_rolling, _ = calculate_rolling_epa(df_pbp, player_name, window=10)

    if pd.isna(max_rolling):
        return 0

    return 1 if max_rolling > 0.15 else 0


def extract_sustained_starter_outcome(df_pbp, player_name):
    """Outcome 5: Sustained starter (30+ consecutive starts)"""
    df_player = df_pbp[df_pbp['passer_player_name'] == player_name]

    if len(df_player) == 0:
        return 0

    game_attempts = df_player.groupby(['season', 'week']).size().reset_index(name='attempts')
    game_attempts['is_start'] = (game_attempts['attempts'] >= 10).astype(int)

    max_streak = 0
    current_streak = 0

    for is_start in game_attempts['is_start']:
        if is_start == 1:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    return 1 if max_streak >= 30 else 0


def load_nfl_data(min_year, max_year):
    """
    Load NFL play-by-play data

    Returns: (df_pbp, error)
    """
    try:
        years = list(range(min_year, max_year + 1))
        print(f"    Loading NFL PBP data ({min_year}-{max_year})...")
        print(f"    This may take several minutes...")

        df_pbp = nfl.import_pbp_data(years, downcast=False)

        if df_pbp is None or len(df_pbp) == 0:
            return None, "No data returned"

        # Filter to passing plays with EPA
        df_pbp = df_pbp[
            (df_pbp['play_type'] == 'pass') &
            (df_pbp['epa'].notna()) &
            (df_pbp['passer_player_name'].notna())
        ]

        print(f"    [OK] {len(df_pbp):,} passing plays loaded")
        return df_pbp, None

    except Exception as e:
        return None, str(e)


def main():
    """Main execution with retry logic"""
    print("="*80)
    print("EXTRACTING NFL OUTCOMES (WITH RETRY)")
    print("="*80)
    print("\nWill retry every 30 minutes if data loading fails")
    print("Press Ctrl+C to stop")

    root = get_project_root()

    # Load QB list
    print("\n[1] Loading QB list...")
    qb_list_path = root / 'data' / 'processed' / 'qb_list_expanded.csv'

    if not qb_list_path.exists():
        print("    [ERROR] QB list not found")
        return

    df_qbs = pd.read_csv(qb_list_path)
    print(f"    [OK] {len(df_qbs)} QBs")

    min_year = df_qbs['draft_year'].min()
    max_year = 2024

    retry_count = 0
    max_retries = 20

    while retry_count < max_retries:
        print(f"\n" + "="*80)
        print(f"ATTEMPT #{retry_count + 1} - {datetime.now().strftime('%H:%M:%S')}")
        print("="*80)

        # Try to load NFL data
        print("\n[2] Loading NFL play-by-play data...")
        df_pbp, error = load_nfl_data(min_year, max_year)

        if error:
            print(f"    [ERROR] {error}")
            print(f"\n    Waiting 30 minutes before retry...")

            wait_seconds = 1800
            try:
                for i in range(wait_seconds // 60):
                    time.sleep(60)
                    remaining_mins = (wait_seconds // 60) - i - 1
                    if remaining_mins > 0:
                        print(f"    {remaining_mins} minutes remaining...")
            except KeyboardInterrupt:
                print("\n\n[INTERRUPTED] Stopping retry loop")
                return

            retry_count += 1
            continue

        # NFL data loaded successfully - proceed with extraction
        print("\n[3] Extracting outcomes for all QBs...")
        print("-"*80)

        results = []

        for idx, row in df_qbs.iterrows():
            player_name = row['player_name']
            draft_year = row['draft_year']

            nfl_name = convert_to_nfl_name(player_name)

            print(f"  [{idx+1}/{len(df_qbs)}] {player_name} ({draft_year})", end=' ')

            # Check which name format exists
            if player_name in df_pbp['passer_player_name'].values:
                search_name = player_name
            elif nfl_name in df_pbp['passer_player_name'].values:
                search_name = nfl_name
            else:
                print("[NOT FOUND]")
                results.append({
                    'player_name': player_name,
                    'draft_year': draft_year,
                    'nfl_name_used': None,
                    'rookie_leaf': np.nan,
                    'reached_85_starts': np.nan,
                    'is_bust': np.nan,
                    'is_elite': np.nan,
                    'is_sustained_starter': np.nan,
                    'total_starts': 0,
                    'max_rolling_epa': np.nan,
                    'min_rolling_epa': np.nan
                })
                continue

            # Extract all 5 outcomes
            try:
                rookie_leaf = extract_rookie_leaf(df_pbp, search_name, draft_year)
                reached_85 = extract_85_starts_outcome(df_pbp, search_name, draft_year)
                is_bust = extract_bust_outcome(df_pbp, search_name)
                is_elite = extract_elite_peak_outcome(df_pbp, search_name)
                is_sustained = extract_sustained_starter_outcome(df_pbp, search_name)

                total_starts = extract_career_starts(df_pbp, search_name)
                max_rolling, min_rolling = calculate_rolling_epa(df_pbp, search_name, window=10)

                results.append({
                    'player_name': player_name,
                    'draft_year': draft_year,
                    'nfl_name_used': search_name,
                    'rookie_leaf': rookie_leaf,
                    'reached_85_starts': reached_85,
                    'is_bust': is_bust,
                    'is_elite': is_elite,
                    'is_sustained_starter': is_sustained,
                    'total_starts': total_starts,
                    'max_rolling_epa': max_rolling,
                    'min_rolling_epa': min_rolling
                })

                # Status
                status = []
                if not pd.isna(rookie_leaf):
                    status.append(f"LEAF={rookie_leaf:+.3f}")
                if reached_85 == 1:
                    status.append("85+")
                if is_elite == 1:
                    status.append("ELITE")
                if is_bust == 1:
                    status.append("BUST")

                print(f"[OK] {', '.join(status) if status else 'Done'}")

            except Exception as e:
                print(f"[ERROR] {str(e)[:30]}")
                results.append({
                    'player_name': player_name,
                    'draft_year': draft_year,
                    'nfl_name_used': search_name,
                    'rookie_leaf': np.nan,
                    'reached_85_starts': np.nan,
                    'is_bust': np.nan,
                    'is_elite': np.nan,
                    'is_sustained_starter': np.nan,
                    'total_starts': 0,
                    'max_rolling_epa': np.nan,
                    'min_rolling_epa': np.nan
                })

        # Save results
        print("\n[4] Saving NFL outcomes...")
        df_outcomes = pd.DataFrame(results)
        output_path = root / 'data' / 'processed' / 'nfl_outcomes_comprehensive.csv'
        df_outcomes.to_csv(output_path, index=False)
        print(f"    [OK] Saved to: {output_path}")

        # Summary
        print("\n" + "="*80)
        print("EXTRACTION COMPLETE")
        print("="*80)

        qbs_with_data = df_outcomes[df_outcomes['nfl_name_used'].notna()]
        print(f"\n  QBs found: {len(qbs_with_data)}/{len(df_outcomes)}")

        for outcome_col, outcome_name in [
            ('rookie_leaf', 'Rookie LEAF'),
            ('reached_85_starts', '85+ Starts'),
            ('is_bust', 'Bust'),
            ('is_elite', 'Elite'),
            ('is_sustained_starter', 'Sustained Starter')
        ]:
            valid = df_outcomes[outcome_col].dropna()
            if len(valid) > 0:
                if outcome_col == 'rookie_leaf':
                    print(f"  {outcome_name}: {len(valid)} QBs (mean={valid.mean():+.3f})")
                else:
                    pos = (valid == 1).sum()
                    print(f"  {outcome_name}: {pos}/{len(valid)} ({pos/len(valid)*100:.1f}%)")

        print("\n  Next: Run create_multi_year_aggregations.py")
        return  # Success - exit

    print("\n[ERROR] Maximum retries reached")


if __name__ == "__main__":
    main()
