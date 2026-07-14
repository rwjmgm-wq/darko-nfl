"""Test CFBD play-by-play API"""
import requests
import os
from dotenv import load_dotenv
from pathlib import Path
import json

# Load API key
root = Path(__file__).parent
env_path = root / '.env'
load_dotenv(env_path)
api_key = os.getenv('CFBD_API_KEY')

headers = {
    'Authorization': f'Bearer {api_key}',
    'Accept': 'application/json'
}

# Test 1: Get games for Florida State 2014
print("Test 1: Get Florida State 2014 games")
print("="*80)

response = requests.get(
    'https://api.collegefootballdata.com/games',
    headers=headers,
    params={'year': 2014, 'team': 'Florida State', 'seasonType': 'regular'},
    timeout=10
)

print(f"Status: {response.status_code}")

if response.status_code == 200:
    games = response.json()
    print(f"Found {len(games)} games")

    if games:
        print("\nFirst game structure:")
        print(json.dumps(games[0], indent=2))

        print("\nGame weeks:")
        for game in games:
            week = game.get('week')
            home = game.get('home_team')
            away = game.get('away_team')
            print(f"  Week: {week}, Home: {home}, Away: {away}")
else:
    print(f"Error: {response.text}")

# Test 2: Get plays for a specific game
print("\n\nTest 2: Get plays for Florida State Week 1, 2014")
print("="*80)

response = requests.get(
    'https://api.collegefootballdata.com/plays',
    headers=headers,
    params={
        'seasonType': 'regular',
        'year': 2014,
        'week': 1,
        'team': 'Florida State'
    },
    timeout=15
)

print(f"Status: {response.status_code}")

if response.status_code == 200:
    plays = response.json()
    print(f"Found {len(plays)} total plays")

    if plays:
        print("\nFirst play structure (checking field names):")
        print(json.dumps(plays[0], indent=2))

    # Filter to passing plays - try different field name variations
    print("\nTrying to find pass plays...")
    pass_plays_v1 = [p for p in plays if 'pass' in str(p.get('play_type', '')).lower()]
    print(f"  Plays with 'pass' in play_type: {len(pass_plays_v1)}")

    pass_plays_v2 = [p for p in plays if 'pass' in str(p.get('playType', '')).lower()]
    print(f"  Plays with 'pass' in playType (camelCase): {len(pass_plays_v2)}")

    # Check offense field
    fsu_plays_v1 = [p for p in plays if p.get('offense') == 'Florida State']
    print(f"  Plays where offense='Florida State': {len(fsu_plays_v1)}")

    fsu_plays_v2 = [p for p in plays if p.get('offenseTeam') == 'Florida State']
    print(f"  Plays where offenseTeam='Florida State': {len(fsu_plays_v2)}")

    # Combine
    pass_plays = [p for p in plays if 'pass' in str(p.get('playType', '')).lower()
                  and p.get('offense') == 'Florida State']
    print(f"\nFlorida State pass plays (final): {len(pass_plays)}")

    if pass_plays:
        print("\nFirst pass play structure:")
        print(json.dumps(pass_plays[0], indent=2))
else:
    print(f"Error: {response.text}")
