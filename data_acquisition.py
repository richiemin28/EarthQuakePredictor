# =============================================================================
# data_acquisition.py
# Handles all data retrieval from the USGS Earthquake Catalog API and the
# USGS real-time ATOM syndication feed.
#
# Two functions are exposed:
#   fetch_historical_data() - Pulls the full training/test catalog from USGS
#   fetch_live_events()     - Polls the ATOM feed for the most recent events
# =============================================================================

import os
import time
import requests
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime, timedelta, timezone
from config import (
    GEO_BOUNDS, MIN_MAGNITUDE,
    HISTORICAL_START, HISTORICAL_END,
    USGS_API_URL, USGS_ATOM_FEED,
    RAW_DATA_PATH, HISTORICAL_CHUNK_DAYS
)


# ---------------------------------------------------------------------------
# USGS API returns events in batches of max 20,000. For a 30-year catalog
# this function splits the request into yearly chunks to stay within limits.
# ---------------------------------------------------------------------------
def fetch_historical_data(start: str = HISTORICAL_START,
                          end:   str = HISTORICAL_END,
                          force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch the full historical earthquake catalog for the Myanmar region
    from the USGS Earthquake Catalog API.

    Parameters
    ----------
    start         : ISO date string, e.g. "1990-01-01"
    end           : ISO date string, e.g. "2019-12-31"
    force_refresh : If False and cached file exists, load from disk instead

    Returns
    -------
    pd.DataFrame with columns: time, latitude, longitude, depth, magnitude
    """

    # Return cached data if it already exists and refresh is not forced
    if not force_refresh and os.path.exists(RAW_DATA_PATH):
        print(f"[DATA] Loading cached catalog from {RAW_DATA_PATH}")
        return pd.read_csv(RAW_DATA_PATH, parse_dates=["time"])

    os.makedirs("data", exist_ok=True)
    all_events = []

    # Split into yearly chunks to avoid hitting the 20,000-event API limit
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d")
    current  = start_dt

    while current < end_dt:
        chunk_end = min(current + timedelta(days=HISTORICAL_CHUNK_DAYS), end_dt)

        params = {
            "format":        "geojson",
            "starttime":     current.strftime("%Y-%m-%d"),
            "endtime":       chunk_end.strftime("%Y-%m-%d"),
            "minmagnitude":  MIN_MAGNITUDE,
            "minlatitude":   GEO_BOUNDS["min_latitude"],
            "maxlatitude":   GEO_BOUNDS["max_latitude"],
            "minlongitude":  GEO_BOUNDS["min_longitude"],
            "maxlongitude":  GEO_BOUNDS["max_longitude"],
            "orderby":       "time-asc",
        }

        print(f"[DATA] Fetching {current.strftime('%Y-%m-%d')} to "
              f"{chunk_end.strftime('%Y-%m-%d')} ... ", end="", flush=True)

        try:
            response = requests.get(USGS_API_URL, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

            features = data.get("features", [])
            print(f"{len(features)} events")
            if len(features) >= 20000:
                print(f"[DATA] WARNING: chunk hit the 20,000-event API cap - "
                      f"some events in this window were likely dropped. "
                      f"Lower HISTORICAL_CHUNK_DAYS for this region.")

            for feat in features:
                props = feat["properties"]
                coords = feat["geometry"]["coordinates"]
                all_events.append({
                    "time":      datetime.fromtimestamp(props["time"] / 1000, timezone.utc).replace(tzinfo=None),
                    "latitude":  coords[1],
                    "longitude": coords[0],
                    "depth":     coords[2],
                    "magnitude": props["mag"],
                    "place":     props.get("place", ""),
                    "id":        feat["id"],
                })

        except requests.RequestException as e:
            print(f"ERROR: {e}. Retrying in 10 seconds...")
            time.sleep(10)
            continue

        current = chunk_end
        time.sleep(1)   # Be polite to the USGS API

    df = pd.DataFrame(all_events)
    df = df.sort_values("time").reset_index(drop=True)

    # Remove any duplicate event IDs
    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)

    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"\n[DATA] Saved {len(df)} events to {RAW_DATA_PATH}")
    return df


# ---------------------------------------------------------------------------
# Fetch the most recent events from the USGS ATOM feed.
# Called repeatedly by the live updater on a scheduled cycle.
# ---------------------------------------------------------------------------
def fetch_live_events(last_seen_time: datetime = None) -> pd.DataFrame:
    """
    Poll the USGS ATOM feed for new earthquake events.
    Filters to the Myanmar bounding box and returns only events
    newer than last_seen_time (if provided).

    Parameters
    ----------
    last_seen_time : datetime, the timestamp of the last event already processed

    Returns
    -------
    pd.DataFrame with same columns as fetch_historical_data(), may be empty
    """

    print("[LIVE] Polling USGS ATOM feed...", flush=True)
    new_events = []

    try:
        response = requests.get(USGS_ATOM_FEED, timeout=30)
        response.raise_for_status()

        # ATOM feed is XML - parse it
        root = ET.fromstring(response.content)
        ns   = {"atom": "http://www.w3.org/2005/Atom",
                "georss": "http://www.georss.org/georss"}

        for entry in root.findall("atom:entry", ns):

            # Extract magnitude from title, e.g. "M 5.1 - 10 km NE of Mandalay"
            title = entry.find("atom:title", ns)
            if title is None:
                continue

            title_text = title.text or ""
            try:
                mag = float(title_text.split("M")[1].strip().split()[0])
            except (IndexError, ValueError):
                continue

            # Extract coordinates from georss:point  "lat lon"
            point = entry.find("georss:point", ns)
            if point is None:
                continue
            try:
                lat, lon = map(float, point.text.strip().split())
            except ValueError:
                continue

            # Filter to bounding box
            if not (GEO_BOUNDS["min_latitude"]  <= lat <= GEO_BOUNDS["max_latitude"]  and
                    GEO_BOUNDS["min_longitude"] <= lon <= GEO_BOUNDS["max_longitude"]):
                continue

            # Filter to minimum magnitude
            if mag < MIN_MAGNITUDE:
                continue

            # Parse event time
            updated = entry.find("atom:updated", ns)
            if updated is None:
                continue
            try:
                event_time = datetime.strptime(
                    updated.text.strip(), "%Y-%m-%dT%H:%M:%S.%fZ"
                )
            except ValueError:
                try:
                    event_time = datetime.strptime(
                        updated.text.strip(), "%Y-%m-%dT%H:%M:%SZ"
                    )
                except ValueError:
                    continue

            # Skip events already processed
            if last_seen_time and event_time <= last_seen_time:
                continue

            link = entry.find("atom:id", ns)
            event_id = link.text.strip() if link is not None else ""

            new_events.append({
                "time":      event_time,
                "latitude":  lat,
                "longitude": lon,
                "depth":     0.0,       # ATOM feed does not include depth
                "magnitude": mag,
                "place":     title_text,
                "id":        event_id,
            })

    except Exception as e:
        print(f"[LIVE] Feed error: {e}")
        return pd.DataFrame()

    if new_events:
        print(f"[LIVE] {len(new_events)} new events in bounding box")
    else:
        print("[LIVE] No new events since last check")

    return pd.DataFrame(new_events)
