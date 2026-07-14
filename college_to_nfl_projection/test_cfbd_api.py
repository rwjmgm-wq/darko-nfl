"""
Test CFBD API to understand data structure
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
    """Test CFBD API with a known QB"""
    print("Testing CFBD API...")

    # Load API key
    api_key = load_api_key()
    if not api_key:
        return

    # Configure API
    configuration = cfbd.Configuration(
        access_token=api_key
    )

    api_client = cfbd.ApiClient(configuration)
    api_instance = cfbd.PlaysApi(api_client)
    teams_api = cfbd.TeamsApi(api_client)

    # First, check what team names CFBD uses
    print("\nChecking CFBD team names for USC...")
    print("-" * 60)

    try:
        teams = teams_api.get_teams()
        usc_teams = [t for t in teams if 'USC' in t.school or 'Southern' in t.school or 'California' in t.school]
        print(f"Found {len(usc_teams)} teams matching USC/Southern/California:")
        for team in usc_teams:
            print(f"  - {team.school}")
    except Exception as e:
        print(f"[ERROR] Could not fetch teams: {e}")

    # Test without play_type filter first
    print(f"\nTesting: USC, 2023, Week 1 (ALL plays)")
    print("-" * 60)

    try:
        plays = api_instance.get_plays(
            year=2023,
            week=1,
            team='USC'
        )

        print(f"API returned {len(plays) if plays else 0} plays")

        if plays and len(plays) > 0:
            print("\nFirst play structure:")
            first_play = plays[0]
            print(f"  All attributes: {[a for a in dir(first_play) if not a.startswith('_')]}")
            print(f"\n  Sample fields:")
            for attr in ['offense', 'defense', 'play_type', 'yards_gained', 'ppa', 'play_text']:
                if hasattr(first_play, attr):
                    val = getattr(first_play, attr)
                    if val is not None:
                        print(f"    {attr}: {val}")

            # Check play_text for passing plays
            print("\n  Checking play_text field for passing plays:")
            passing_play_types = ['Pass Reception', 'Passing Touchdown', 'Pass Incompletion', 'Pass Interception Return', 'Sack']
            pass_plays = [p for p in plays if hasattr(p, 'play_type') and p.play_type in passing_play_types]
            print(f"  Found {len(pass_plays)} passing plays")

            if len(pass_plays) > 0:
                print(f"\n  Sample play_text from passing plays:")
                for i, play in enumerate(pass_plays[:5]):
                    if hasattr(play, 'play_text') and play.play_text:
                        print(f"    {i+1}. [{play.play_type}] {play.play_text}")

                # Try to extract player names from play_text
                print(f"\n  Looking for 'Caleb Williams' or 'Williams' in play_text:")
                williams_plays = [p for p in pass_plays if hasattr(p, 'play_text') and p.play_text and
                                 ('Caleb Williams' in p.play_text or 'C.Williams' in p.play_text or
                                  (p.offense == 'USC' and 'Williams' in p.play_text))]
                print(f"    Found {len(williams_plays)} plays with Williams in play_text")

                if len(williams_plays) > 0:
                    for i, play in enumerate(williams_plays[:3]):
                        print(f"    Example {i+1}: {play.play_text}")

    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {str(e)}")


if __name__ == "__main__":
    main()
