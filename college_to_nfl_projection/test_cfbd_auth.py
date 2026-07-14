"""
Quick test to verify CFBD API authentication is working
"""

import cfbd
from cfbd.rest import ApiException
from pathlib import Path


def load_api_key():
    """Load CFBD API key from .env file"""
    root = Path(__file__).parent
    api_key_file = root / '.env'

    if not api_key_file.exists():
        print("ERROR: No .env file found")
        print(f"Create one at: {api_key_file}")
        print("With contents: CFBD_API_KEY=your_api_key_here")
        return None

    with open(api_key_file) as f:
        for line in f:
            if line.startswith('CFBD_API_KEY='):
                key = line.strip().split('=', 1)[1]
                # Remove quotes if present
                key = key.strip('"').strip("'")
                return key

    print("ERROR: CFBD_API_KEY not found in .env file")
    return None


def test_api():
    """Test CFBD API with proper authentication"""
    print("="*60)
    print("CFBD API AUTHENTICATION TEST")
    print("="*60)

    # Load API key
    print("\n[1] Loading API key from .env...")
    api_key = load_api_key()

    if not api_key:
        return False

    print(f"    API key loaded: {api_key[:10]}...{api_key[-4:]}")

    # Configure API client using CFBD support's recommended method
    print("\n[2] Configuring API client...")

    # Use access_token parameter (as recommended by CFBD support)
    configuration = cfbd.Configuration(
        access_token=api_key
    )

    print("    Configuration set:")
    print(f"      Host: {configuration.host}")
    print(f"      Access token configured: Yes")

    # Test with a simple API call using context manager
    print("\n[3] Testing API call (SP+ ratings for 2023)...")
    try:
        with cfbd.ApiClient(configuration) as api_client:
            api_instance = cfbd.RatingsApi(api_client)
            ratings = api_instance.get_sp(year=2023)

        if ratings and len(ratings) > 0:
            print(f"    SUCCESS! Retrieved {len(ratings)} team ratings")
            print(f"\n    Sample: {ratings[0].team} - Rating: {ratings[0].rating}")
            return True
        else:
            print("    WARNING: API call succeeded but returned no data")
            return False

    except ApiException as e:
        print(f"    FAILED: API Exception")
        print(f"      Status: {e.status}")
        print(f"      Reason: {e.reason}")
        if e.status == 401:
            print("\n    This is an AUTHENTICATION error (401)")
            print("    Possible causes:")
            print("      - API key is invalid or expired")
            print("      - API key not properly formatted in .env")
            print("      - Authorization header not being sent correctly")
        elif e.status == 429:
            print("\n    This is a RATE LIMIT error (429)")
            print("    You've exceeded your API quota")
        return False

    except Exception as e:
        print(f"    FAILED: {type(e).__name__}: {str(e)}")
        return False


if __name__ == "__main__":
    success = test_api()

    print("\n" + "="*60)
    if success:
        print("[SUCCESS] API AUTHENTICATION WORKING!")
        print("\nYou can now use the CFBD API in your scripts")
    else:
        print("[FAILED] API AUTHENTICATION FAILED")
        print("\nPlease check your API key and try again")
    print("="*60)
