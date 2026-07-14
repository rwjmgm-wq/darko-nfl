"""Test CFBD API connection"""
import cfbd
from dotenv import load_dotenv
import os
from pathlib import Path

# Load API key
load_dotenv(Path(__file__).parent / '.env')
api_key = os.getenv('CFBD_API_KEY')

print(f"API Key loaded: {api_key[:20]}... (length: {len(api_key)})")

# Configure API client
configuration = cfbd.Configuration()
configuration.api_key['Authorization'] = api_key.strip()
configuration.api_key_prefix['Authorization'] = 'Bearer'

api_client = cfbd.ApiClient(configuration)

# Test connection
games_api = cfbd.GamesApi(api_client)

try:
    print("\nTesting API connection...")
    games = games_api.get_games(year=2023, week=1)
    print(f"SUCCESS! Retrieved {len(games)} games from Week 1, 2023")
    for game in games[:3]:
        print(f"  - {game.away_team} @ {game.home_team}")
except Exception as e:
    print(f"FAILED: {e}")
