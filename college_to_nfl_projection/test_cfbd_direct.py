"""
Test CFBD API with direct HTTP request to see exact header format needed
"""

import requests
from pathlib import Path


def load_api_key():
    """Load API key from .env"""
    root = Path(__file__).parent
    with open(root / '.env') as f:
        for line in f:
            if line.startswith('CFBD_API_KEY='):
                return line.strip().split('=', 1)[1].strip('"').strip("'")
    return None


def test_direct_api_call():
    """Test with direct HTTP request"""
    print("="*60)
    print("DIRECT CFBD API TEST")
    print("="*60)

    api_key = load_api_key()
    print(f"\n[1] API Key: {api_key[:10]}...{api_key[-4:]}")

    # Try different header formats
    tests = [
        ("Bearer with space", f"Bearer {api_key}"),
        ("Just Bearer (no space)", f"Bearer{api_key}"),
        ("Just the key", api_key),
    ]

    for test_name, auth_value in tests:
        print(f"\n[2] Testing: {test_name}")
        print(f"    Authorization header: {auth_value[:20]}...")

        headers = {
            "Authorization": auth_value,
            "accept": "application/json"
        }

        try:
            response = requests.get(
                "https://api.collegefootballdata.com/ratings/sp",
                headers=headers,
                params={"year": 2023}
            )

            print(f"    Status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                print(f"    [SUCCESS!] Got {len(data)} results")
                print(f"    Sample: {data[0]['team']} - {data[0]['rating']}")
                print(f"\n    CORRECT FORMAT: {test_name}")
                print(f"    Use: Authorization: {auth_value[:30]}...")
                return True

            elif response.status_code == 401:
                print(f"    [FAILED] 401 Unauthorized")

            elif response.status_code == 429:
                print(f"    [FAILED] 429 Rate Limited")

        except Exception as e:
            print(f"    [ERROR] {e}")

    return False


if __name__ == "__main__":
    success = test_direct_api_call()

    print("\n" + "="*60)
    if success:
        print("[SUCCESS] Found working auth format!")
        print("\nNow we know how to configure the cfbd library")
    else:
        print("[FAILED] All auth formats failed")
        print("\nPossible issues:")
        print("  - API key might be expired or invalid")
        print("  - API key might need to be refreshed/regenerated")
        print("  - Contact CFBD support for help")
    print("="*60)
