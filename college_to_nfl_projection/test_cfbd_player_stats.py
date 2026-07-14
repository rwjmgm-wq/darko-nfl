"""
Test CFBD Player Stats API to get QB statistics
"""

import cfbd
from pathlib import Path


def load_api_key():
    """Load CFBD API key from .env file"""
    api_key_file = Path('.env')

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
    """Test CFBD Player Stats API"""
    print("Testing CFBD Player Stats API...")

    # Load API key
    api_key = load_api_key()
    if not api_key:
        return

    # Configure API
    configuration = cfbd.Configuration(
        access_token=api_key
    )

    api_client = cfbd.ApiClient(configuration)
    stats_api = cfbd.StatsApi(api_client)

    # Test with Caleb Williams at USC in 2023
    print("\nTesting: Caleb Williams, USC, 2023")
    print("-" * 60)

    try:
        # Get player season stats
        stats = stats_api.get_player_season_stats(
            year=2023,
            team='USC',
            category='passing'
        )

        print(f"API returned {len(stats) if stats else 0} player stats")

        if stats and len(stats) > 0:
            print("\nQBs found:")
            for player_stat in stats:
                player = player_stat.player
                stat_value = player_stat.stat
                print(f"  {player}: {stat_value}")

                # Show first player's full structure
                if player == stats[0].player:
                    print(f"\n  Sample player_stat structure:")
                    print(f"    Attributes: {[a for a in dir(player_stat) if not a.startswith('_')]}")

        else:
            print("\n[WARNING] No player stats returned")

    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {str(e)}")


if __name__ == "__main__":
    main()
