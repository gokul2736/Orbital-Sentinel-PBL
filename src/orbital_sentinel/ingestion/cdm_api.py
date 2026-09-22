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

            body = response.text.strip().strip('"')
            if body.lower() == "failed":
                raise RuntimeError(
                    "Space-Track authentication failed: invalid credentials"
                )

            self._authenticated = True
            return True
        except requests.exceptions.Timeout:
            raise RuntimeError("Space-Track login timed out")
        except requests.exceptions.ConnectionError:
            raise RuntimeError("Could not connect to Space-Track")
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"Space-Track authentication failed: {e}")

    def _ensure_auth(self) -> None:
        if not self._authenticated or self._session is None:
            self.login()

    def fetch_cdm(
        self,
        limit: int = 1,
        format: str = "json",
        timeout: int = 30,
        enrich_gp: bool = True,
    ) -> List[Dict[str, Any]]:
        """Fetch latest CDMs from Space-Track cdm_public endpoint.

        Uses basicspacedata/cdm_public (publicly accessible) with fallback
        to expandedspacedata/cdm for accounts with expanded access.

        If enrich_gp=True, fetches GP orbital element data for all satellites
        in the CDMs and merges it in (orbital elements, TLE, etc).
        """
        self._ensure_auth()

        cdms = self._fetch_cdm_public(limit, format, timeout)

        if cdms is None:
            cdms = self._fetch_cdm_expanded(limit, format, timeout)

        if cdms and enrich_gp:
            cdms = self._enrich_with_gp(cdms, timeout)

        return cdms

    def _fetch_cdm_public(
        self, limit: int, format: str, timeout: int
    ) -> Optional[List[Dict[str, Any]]]:
        url = (
            f"{self._base_url}/basicspacedata/query/"
            f"class/cdm_public/"
            f"limit/{limit}/"
            f"orderby/CDM_ID%20desc/"
            f"format/{format}"
        )
        try:
            response = self._session.get(url, timeout=timeout)
            response.raise_for_status()
            if format == "json":
                data = response.json()
                if isinstance(data, list) and data and "error" not in data[0]:
                    return data
            return None
        except (requests.exceptions.RequestException, ValueError):
            return None

    def _fetch_cdm_expanded(
        self, limit: int, format: str, timeout: int
    ) -> List[Dict[str, Any]]:
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

    def fetch_gp(
        self,
        norad_ids: List[str],
        timeout: int = 30,
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch latest GP (General Perturbation) data for given NORAD IDs.

        Returns a dict keyed by NORAD_CAT_ID with the latest GP record.
        """
        self._ensure_auth()

        if not norad_ids:
            return {}

        ids_str = ",".join(str(nid) for nid in norad_ids)
        url = (
            f"{self._base_url}/basicspacedata/query/"
            f"class/gp/"
            f"NORAD_CAT_ID/{ids_str}/"
            f"orderby/EPOCH%20desc/"
            f"format/json"
        )

        try:
            response = self._session.get(url, timeout=timeout)
            response.raise_for_status()
            records = response.json()
        except (requests.exceptions.RequestException, ValueError):
            return {}

        gp_map = {}
        for rec in records:
            nid = str(rec.get("NORAD_CAT_ID", ""))
            if nid and nid not in gp_map:
                gp_map[nid] = rec

        return gp_map

    def _enrich_with_gp(
        self, cdms: List[Dict[str, Any]], timeout: int
    ) -> List[Dict[str, Any]]:
        norad_ids = set()
        for cdm in cdms:
            for key in ("SAT_1_ID", "SAT_2_ID"):
                val = cdm.get(key)
                if val:
                    norad_ids.add(str(val))

        if not norad_ids:
            return cdms

        gp_map = self.fetch_gp(list(norad_ids), timeout)

        gp_to_cdm_obj1 = {
            "SEMIMAJOR_AXIS": "OBJECT1_SEMI_MAJOR_AXIS",
            "ECCENTRICITY": "OBJECT1_ECCENTRICITY",
            "INCLINATION": "OBJECT1_INCLINATION",
            "RA_OF_ASC_NODE": "OBJECT1_RA_OF_ASC_NODE",
            "ARG_OF_PERICENTER": "OBJECT1_ARG_OF_PERICENTER",
            "MEAN_ANOMALY": "OBJECT1_MEAN_ANOMALY",
            "MEAN_MOTION": "OBJECT1_MEAN_MOTION",
            "APOAPSIS": "OBJECT1_APOAPSIS",
            "PERIAPSIS": "OBJECT1_PERIAPSIS",
            "PERIOD": "OBJECT1_PERIOD",
            "BSTAR": "OBJECT1_BSTAR",
            "OBJECT_TYPE": "SAT1_OBJECT_TYPE_GP",
            "EPOCH": "OBJECT1_EPOCH",
            "TLE_LINE1": "OBJECT1_TLE_LINE1",
            "TLE_LINE2": "OBJECT1_TLE_LINE2",
        }
        gp_to_cdm_obj2 = {
            k: v.replace("OBJECT1_", "OBJECT2_").replace("SAT1_", "SAT2_")
            for k, v in gp_to_cdm_obj1.items()
        }

        enriched = []
        for cdm in cdms:
            cdm = dict(cdm)
            sat1_id = str(cdm.get("SAT_1_ID", ""))
            sat2_id = str(cdm.get("SAT_2_ID", ""))

            gp1 = gp_map.get(sat1_id, {})
            for gp_key, cdm_key in gp_to_cdm_obj1.items():
                if gp_key in gp1 and cdm_key not in cdm:
                    cdm[cdm_key] = gp1[gp_key]

            gp2 = gp_map.get(sat2_id, {})
            for gp_key, cdm_key in gp_to_cdm_obj2.items():
                if gp_key in gp2 and cdm_key not in cdm:
                    cdm[cdm_key] = gp2[gp_key]

            cdm["_enriched_gp"] = bool(gp1 or gp2)
            enriched.append(cdm)

        return enriched

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
