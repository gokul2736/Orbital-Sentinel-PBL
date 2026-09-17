import os
import requests
from dotenv import load_dotenv

load_dotenv()

username = os.getenv("SPACE_TRACK_USERNAME")
password = os.getenv("SPACE_TRACK_PASSWORD")
base_url = os.getenv("SPACE_TRACK_BASE_URL")

if not username or not password:
    raise RuntimeError("Space-Track credentials were not loaded.")

login_url = f"{base_url}/ajaxauth/login"

session = requests.Session()

response = session.post(
    login_url,
    data={
        "identity": username,
        "password": password,
    },
    timeout=30,
)

print("HTTP status:", response.status_code)
print("Authentication response received.")

if response.ok:
    print("✅ Space-Track authentication request succeeded.")
else:
    print("❌ Authentication failed.")
    print(response.text[:500])