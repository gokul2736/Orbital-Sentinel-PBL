import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

username = os.getenv("SPACE_TRACK_USERNAME")
password = os.getenv("SPACE_TRACK_PASSWORD")
base_url = os.getenv("SPACE_TRACK_BASE_URL")

if not username or not password or not base_url:
    raise RuntimeError("Space-Track configuration is missing.")

session = requests.Session()

# Authenticate
login_response = session.post(
    f"{base_url}/ajaxauth/login",
    data={
        "identity": username,
        "password": password,
    },
    timeout=30,
)

login_response.raise_for_status()

print("✅ Authentication successful.")

# ONE small CDM request
url = (
    f"{base_url}/expandedspacedata/query/"
    "class/cdm/"
    "limit/1/"
    "format/json"
)

response = session.get(url, timeout=30)

print("CDM HTTP status:", response.status_code)

print("Response content type:", response.headers.get("content-type"))
print("Response body:")
print(response.text[:2000])

response.raise_for_status()

data = response.json()

print("✅ Live CDM response received.")
print("Number of records:", len(data))

if data:
    print("\nAvailable CDM fields:")
    print(list(data[0].keys()))

    output_dir = Path("data/raw/cdm")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "live_cdm_test.json"

    import json

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"\n💾 Saved to: {output_file}")
else:
    print("⚠️ Authentication worked, but no CDM record was returned.")
