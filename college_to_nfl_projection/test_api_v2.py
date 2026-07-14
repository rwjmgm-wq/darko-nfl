"""Test CFBD API connection - Alternative methods"""
import requests

api_key = "7n0Ht9AGwlvlE29sc6LWnKJRp9b5N3YqleJzFdodK+jkrci73O/Riv9vw+e/V3Kg"

print(f"API Key: {api_key[:20]}... (length: {len(api_key)})")

# Test 1: Direct HTTP request with Bearer token
print("\n=== Test 1: Direct HTTP request ===")
headers = {
    'Authorization': f'Bearer {api_key}',
    'Accept': 'application/json'
}

try:
    response = requests.get(
        'https://api.collegefootballdata.com/games?year=2023&week=1',
        headers=headers
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        games = response.json()
        print(f"SUCCESS! Retrieved {len(games)} games")
        if games:
            print(f"  Sample: {games[0].get('away_team')} @ {games[0].get('home_team')}")
    else:
        print(f"FAILED: {response.text}")
except Exception as e:
    print(f"ERROR: {e}")

# Test 2: Try with cfbd library but different config
print("\n=== Test 2: CFBD library (alternative config) ===")
import cfbd

try:
    configuration = cfbd.Configuration()
    configuration.host = "https://api.collegefootballdata.com"

    # Try setting the full "Bearer <key>" in api_key
    configuration.api_key['Authorization'] = f'Bearer {api_key}'
    configuration.api_key_prefix['Authorization'] = ''  # Empty since we included Bearer above

    api_client = cfbd.ApiClient(configuration)
    games_api = cfbd.GamesApi(api_client)

    games = games_api.get_games(year=2023, week=1)
    print(f"SUCCESS! Retrieved {len(games)} games")
    for game in games[:2]:
        print(f"  - {game.away_team} @ {game.home_team}")
except Exception as e:
    print(f"FAILED: {e}")

# Test 3: Original method but verify
print("\n=== Test 3: Original method (verify headers) ===")
try:
    configuration = cfbd.Configuration()
    configuration.api_key['Authorization'] = api_key
    configuration.api_key_prefix['Authorization'] = 'Bearer'

    # Print what the library will send
    print(f"Library will send: 'Bearer {api_key[:20]}...'")

    api_client = cfbd.ApiClient(configuration)
    games_api = cfbd.GamesApi(api_client)
    games = games_api.get_games(year=2023, week=1)
    print(f"SUCCESS! Retrieved {len(games)} games")
except Exception as e:
    print(f"FAILED: {e}")
