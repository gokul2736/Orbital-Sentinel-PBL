import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv


class SpaceTrackClient:
    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        load_dotenv()

        self._username = username or os.getenv("SPACE_TRACK_USERNAME")
        self._password = password or os.getenv("SPACE_TRACK_PASSWORD")
        self._base_url = (
            base_url
            or os.getenv("SPACE_TRACK_BASE_URL")
            or "https://www.space-track.org"
        )
        self._session: Optional[requests.Session] = None
        self._authenticated = False

    def login(self, timeout: int = 30) -> bool:
        if not self._username or not self._password:
            raise RuntimeError(
                "Space-Track credentials not configured. "
                "Set SPACE_TRACK_USERNAME and SPACE_TRACK_PASSWORD in .env"
            )

        self._session = requests.Session()

        try:
            response = self._session.post(
                f"{self._base_url}/ajaxauth/login",
                data={
                    "identity": self._username,
                    "password": self._password,
                },
                timeout=timeout,
            )
            response.raise_for_status()
            self._authenticated = True
            return True
        except requests.exceptions.Timeout:
            raise RuntimeError("Space-Track login timed out")
        except requests.exceptions.ConnectionError:
            raise RuntimeError("Could not connect to Space-Track")
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"Space-Track authentication failed: {e}")

    def fetch_cdm(
        self,
        limit: int = 1,
        format: str = "json",
        timeout: int = 30,
    ) -> List[Dict[str, Any]]:
        if not self._authenticated or self._session is None:
            self.login()

        url = (
            f"{self._base_url}/expandedspacedata/query/"
            f"class/cdm/"
            f"limit/{limit}/"
            f"format/{format}"
        )

        try:
            response = self._session.get(url, timeout=timeout)
            response.raise_for_status()
        except requests.exceptions.Timeout:
            raise RuntimeError("CDM fetch timed out")
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"CDM fetch failed: {e}")

        if format == "json":
            return response.json()
        return [{"raw": response.text}]

    def save_cdm(
        self,
        data: List[Dict[str, Any]],
        output_path: Optional[str] = None,
    ) -> Path:
        if output_path is None:
            output_dir = Path("data/raw/cdm")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / "live_cdm.json")

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return path

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None
            self._authenticated = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
