import sys
import os
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orbital_sentinel.ingestion.cdm_api import SpaceTrackClient


def test_spacetrack_client_instantiation():
    client = SpaceTrackClient(username="test_user", password="test_pass")
    assert client._username == "test_user"
    assert client._password == "test_pass"
    assert client._authenticated is False
    assert client._session is None


def test_spacetrack_client_default_base_url():
    client = SpaceTrackClient(username="u", password="p")
    assert "space-track.org" in client._base_url


@patch.dict(os.environ, {}, clear=True)
def test_login_without_credentials_raises_runtime_error():
    client = SpaceTrackClient(username=None, password=None)
    client._username = None
    client._password = None
    with pytest.raises(RuntimeError, match="credentials not configured"):
        client.login()


def test_context_manager():
    client = SpaceTrackClient(username="u", password="p")
    with client as c:
        assert c is client
    assert client._session is None


def test_close_resets_state():
    client = SpaceTrackClient(username="u", password="p")
    import requests
    client._session = requests.Session()
    client._authenticated = True
    client.close()
    assert client._authenticated is False
    assert client._session is None
