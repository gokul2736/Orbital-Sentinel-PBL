# Space-Track API Integration

## Overview

Orbital Sentinel integrates with the [Space-Track.org](https://www.space-track.org) API to fetch live Conjunction Data Messages (CDMs). This enables real-time collision risk assessment on incoming conjunction events, beyond the offline ESA Kelvins dataset.

## Authentication

Space-Track requires a registered account. Set credentials in your `.env` file:

```
SPACE_TRACK_USERNAME=your_username
SPACE_TRACK_PASSWORD=your_password
```

Optionally override the base URL (defaults to `https://www.space-track.org`):

```
SPACE_TRACK_BASE_URL=https://www.space-track.org
```

## Client Usage

```python
from orbital_sentinel.ingestion import SpaceTrackClient

# Context manager handles login and cleanup
with SpaceTrackClient() as client:
    cdms = client.fetch_cdm(limit=5)
    path = client.save_cdm(cdms, "data/raw/cdm/latest.json")
    print(f"Saved {len(cdms)} CDMs to {path}")
```

### Methods

| Method | Description |
|---|---|
| `login(timeout=30)` | Authenticate with Space-Track. Called automatically by `fetch_cdm` if needed. |
| `fetch_cdm(limit=1, format="json", timeout=30)` | Fetch CDM records. Returns a list of dicts. |
| `save_cdm(data, output_path=None)` | Save fetched CDMs to JSON. Defaults to `data/raw/cdm/live_cdm.json`. |
| `close()` | Close the session. Called automatically when using `with`. |

## CDM Mapping

Raw Space-Track CDM fields use different names than the ESA Kelvins dataset. The `cdm_mapper` module handles this translation:

```python
from orbital_sentinel.ingestion.cdm_mapper import map_cdm_to_features

# Convert Space-Track CDM dict to model-ready feature vector
features_df = map_cdm_to_features(raw_cdm_list)
```

Key field mappings:

| Space-Track Field | ESA Kelvins Field |
|---|---|
| `CDM_ID` | `event_id` |
| `TCA` | `tca` |
| `MISS_DISTANCE` | `miss_distance` |
| `RELATIVE_SPEED` | `relative_speed` |
| `RELATIVE_POSITION_R` | `relative_position_r` |
| `RELATIVE_POSITION_T` | `relative_position_t` |
| `RELATIVE_POSITION_N` | `relative_position_n` |
| `RELATIVE_VELOCITY_R` | `relative_velocity_r` |

## End-to-End Pipeline

See `scripts/test_live_cdm.py` for a complete example that fetches live CDMs, maps them to model features, runs inference through the XGBoost pipeline, and outputs risk assessments.

## Rate Limits

Space-Track enforces rate limits on API requests. The client uses a single persistent session to minimize authentication calls. For batch operations, keep requests under 30 per minute and 300 per hour.
