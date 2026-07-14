"""
Identify 2026 QB Prospects with 6+ Games in 2024 Season

Uses CFBD API to find current college QBs who are likely 2026 draft prospects.
"""

import pandas as pd
import cfbd
from pathlib import Path
from collections import defaultdict


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent


def load_api_key():
    """Load CFBD API key from .env file"""
    root = get_project_root()
    api_key_file = root / '.env'

    if not api_key_file.exists():
        print("[ERROR] No .env file found")
        return None

    with open(api_key_file) as f:
        for line in f:
            if line.startswith('CFBD_API_KEY='):
                return line.strip().split('=')[1].strip()

    print("[ERROR] CFBD_API_KEY not found in .env file")
    return None


def main():
    """Identify 2026 QB prospects from 2024 season"""
    print("=" * 80)
    print("IDENTIFYING 2026 QB PROSPECTS (6+ GAMES IN 2024)")
    print("=" * 80)

    root = get_project_root()

    # Load API key
    print("\n[1] Loading CFBD API key...")
    api_key = load_api_key()
    if not api_key:
        return

    configuration = cfbd.Configuration(access_token=api_key)
    api_client = cfbd.ApiClient(configuration)
    stats_api = cfbd.StatsApi(api_client)

    print("    [OK] API configured")

    # Get 2024 passing stats for all QBs
    print("\n[2] Fetching 2024 QB stats from CFBD...")

    try:
        stats = stats_api.get_player_season_stats(
            year=2024,
            category='passing'
        )

        print(f"    [OK] Found {len(stats)} QB stat records")

    except Exception as e:
        print(f"    [ERROR] {type(e).__name__}: {str(e)}")
        return

    # Aggregate by player (may have multiple stat types)
    print("\n[3] Aggregating QB statistics...")

    qb_stats = defaultdict(lambda: {
        'team': None,
        'conference': None,
        'attempts': 0,
        'completions': 0,
        'yards': 0,
        'tds': 0,
        'ints': 0
    })

    for stat in stats:
        player = stat.player
        team = stat.team
        conf = stat.conference if hasattr(stat, 'conference') else None
        stat_type = stat.stat_type
        value = stat.stat

        if player and team:
            qb_stats[player]['team'] = team
            qb_stats[player]['conference'] = conf

            # Map stat types
            if stat_type == 'ATT':
                qb_stats[player]['attempts'] = int(value)
            elif stat_type == 'COMPLETIONS':
                qb_stats[player]['completions'] = int(value)
            elif stat_type == 'YDS':
                qb_stats[player]['yards'] = int(value)
            elif stat_type == 'TD':
                qb_stats[player]['tds'] = int(value)
            elif stat_type == 'INT':
                qb_stats[player]['ints'] = int(value)

    print(f"    [OK] Aggregated {len(qb_stats)} unique QBs")

    # Filter to QBs with significant playing time
    # Use 150+ attempts as proxy for 6+ games (roughly 25 attempts per game)
    print("\n[4] Filtering to QBs with 150+ attempts (6+ games)...")

    qualified_qbs = []

    for player, stats in qb_stats.items():
        if stats['attempts'] >= 150:
            comp_pct = (stats['completions'] / stats['attempts'] * 100) if stats['attempts'] > 0 else 0

            qualified_qbs.append({
                'player_name': player,
                'team': stats['team'],
                'conference': stats['conference'],
                'attempts': stats['attempts'],
                'completions': stats['completions'],
                'comp_pct': round(comp_pct, 1),
                'yards': stats['yards'],
                'tds': stats['tds'],
                'ints': stats['ints'],
                'draft_year': 2026
            })

    # Sort by attempts (most active QBs first)
    qualified_qbs.sort(key=lambda x: x['attempts'], reverse=True)

    print(f"    [OK] Found {len(qualified_qbs)} QBs with 150+ attempts")

    # Save to CSV
    print("\n[5] Saving 2026 QB prospects list...")

    output_dir = root / 'data' / 'raw'
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(qualified_qbs)
    output_path = output_dir / 'qbs_2026_prospects.csv'
    df.to_csv(output_path, index=False)

    print(f"    [OK] Saved {len(qualified_qbs)} prospects to: {output_path}")

    # Display top 25 QBs
    print("\n" + "=" * 80)
    print("TOP 25 QB PROSPECTS FOR 2026 NFL DRAFT")
    print("=" * 80)

    for idx, qb in enumerate(qualified_qbs[:25], 1):
        print(f"\n{idx:2d}. {qb['player_name']:25s} ({qb['team']})")
        print(f"    {qb['attempts']} ATT, {qb['completions']} CMP ({qb['comp_pct']}%), " +
              f"{qb['yards']} YDS, {qb['tds']} TD, {qb['ints']} INT")

    print(f"\n\n[OK] Total 2026 prospects identified: {len(qualified_qbs)}")
    print(f"[OK] List saved to: {output_path}")


if __name__ == "__main__":
    main()
