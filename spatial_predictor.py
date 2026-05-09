# =============================================================================
# spatial_predictor.py
# Estimates the most probable epicenter location, coordinates, and
# uncertainty radius for upcoming significant seismic events.
#
# Approach:
#   Recent seismic activity clusters spatially along active fault segments.
#   By computing the centroid and spread of recent M>=3.0 events in each
#   geographic zone, weighted by recency and magnitude, the system can
#   estimate both a most-likely epicenter coordinate and an uncertainty
#   radius that reflects how dispersed the recent seismicity is.
#
#   Smaller radius = tightly clustered activity = more location certainty
#   Larger radius  = diffuse activity = broader search area
#
# This is analogous to the spatial hazard mapping approach used in
# operational seismological practice, adapted here for ML-based output.
# =============================================================================

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from config import LOCATION_ZONES


# ---------------------------------------------------------------------------
# Haversine distance between two lat/lon points (returns km)
# ---------------------------------------------------------------------------
def haversine_km(lat1: float, lon1: float,
                 lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = (np.sin(dphi / 2) ** 2 +
         np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Identify zone for a coordinate pair
# ---------------------------------------------------------------------------
def identify_zone(lat: float, lon: float) -> str:
    for zone_name, bounds in LOCATION_ZONES.items():
        if (bounds["lat"][0] <= lat <= bounds["lat"][1] and
                bounds["lon"][0] <= lon <= bounds["lon"][1]):
            return zone_name
    return "Myanmar Region (General)"


# ---------------------------------------------------------------------------
# Compute zone-level spatial statistics from recent catalog
# ---------------------------------------------------------------------------
def compute_zone_spatial_stats(recent_events: pd.DataFrame,
                                lookback_days: int = 90,
                                min_magnitude:  float = 3.0) -> dict:
    """
    For each geographic zone, compute:
      - event_count    : number of recent events
      - centroid_lat   : magnitude-weighted mean latitude
      - centroid_lon   : magnitude-weighted mean longitude
      - radius_km      : 1-sigma spread of events around centroid (km)
      - max_magnitude  : largest recent event in zone
      - last_event_time: most recent event time

    Parameters
    ----------
    recent_events : Raw catalog DataFrame
    lookback_days : How many days back to consider (default 90)
    min_magnitude : Minimum magnitude to include (default 3.0)

    Returns
    -------
    dict: {zone_name: {stat_name: value}}
    """

    if recent_events is None or len(recent_events) == 0:
        return {}

    df = recent_events.copy()
    df["time"] = pd.to_datetime(df["time"])
    cutoff = datetime.utcnow() - timedelta(days=lookback_days)

    # Filter by time and magnitude
    df = df[
        (df["time"] >= cutoff) &
        (df["magnitude"] >= min_magnitude)
    ].copy()

    if len(df) == 0:
        return {}

    # Assign recency weight: events in the last 7 days get 3x weight,
    # last 30 days get 2x, older get 1x
    now = datetime.utcnow()
    def recency_weight(t):
        days_ago = (now - t.to_pydatetime()).total_seconds() / 86400
        if days_ago <= 7:
            return 3.0
        elif days_ago <= 30:
            return 2.0
        return 1.0

    df["recency_w"] = df["time"].apply(recency_weight)
    df["mag_w"]     = df["magnitude"] ** 2   # magnitude-squared weighting
    df["weight"]    = df["recency_w"] * df["mag_w"]

    # Assign zones
    df["zone"] = df.apply(
        lambda r: identify_zone(r["latitude"], r["longitude"]), axis=1
    )

    zone_stats = {}

    for zone_name, group in df.groupby("zone"):
        if len(group) == 0:
            continue

        weights = group["weight"].values
        total_w = weights.sum()

        # Weighted centroid
        centroid_lat = float(np.average(group["latitude"],  weights=weights))
        centroid_lon = float(np.average(group["longitude"], weights=weights))

        # Compute radius as weighted RMS distance from centroid
        distances = group.apply(
            lambda r: haversine_km(
                centroid_lat, centroid_lon,
                r["latitude"], r["longitude"]
            ),
            axis=1
        ).values

        if len(distances) > 1:
            radius_km = float(
                np.sqrt(np.average(distances ** 2, weights=weights))
            )
        else:
            # Single event - use zone half-diagonal as radius
            bounds = LOCATION_ZONES.get(zone_name, {})
            if bounds:
                lat_span = bounds["lat"][1] - bounds["lat"][0]
                lon_span = bounds["lon"][1] - bounds["lon"][0]
                radius_km = haversine_km(
                    bounds["lat"][0], bounds["lon"][0],
                    bounds["lat"][1], bounds["lon"][1]
                ) / 2.0
            else:
                radius_km = 150.0

        # Apply minimum and maximum radius bounds
        # Min 50 km (instrumental location uncertainty)
        # Max 400 km (zone is too diffuse to narrow further)
        radius_km = max(50.0, min(radius_km, 400.0))

        zone_stats[zone_name] = {
            "event_count":     int(len(group)),
            "centroid_lat":    round(centroid_lat, 3),
            "centroid_lon":    round(centroid_lon, 3),
            "radius_km":       round(radius_km, 1),
            "max_magnitude":   round(float(group["magnitude"].max()), 1),
            "mean_magnitude":  round(float(group["magnitude"].mean()), 2),
            "last_event_time": str(group["time"].max()),
            "total_weight":    round(float(total_w), 2),
        }

    return zone_stats


# ---------------------------------------------------------------------------
# Rank zones by seismic hazard potential
# Uses a composite score: event count + recency + magnitude weighting
# ---------------------------------------------------------------------------
def rank_zones(zone_stats: dict,
               top_n: int = 3) -> list:
    """
    Rank zones by their composite hazard score and return top N.
    Returns list of (zone_name, stats_dict) tuples.
    """
    if not zone_stats:
        return []

    scored = []
    for zone_name, stats in zone_stats.items():
        # Composite score:
        # - Count contributes linearly
        # - Total weight (recency + magnitude) is the main driver
        score = stats["total_weight"] * np.log1p(stats["event_count"])
        scored.append((score, zone_name, stats))

    scored.sort(reverse=True)
    return [(name, stats) for _, name, stats in scored[:top_n]]


# ---------------------------------------------------------------------------
# Generate location prediction for a single zone
# ---------------------------------------------------------------------------
def build_location_prediction(zone_name: str,
                               zone_stats: dict,
                               confidence_note: str = "") -> dict:
    """
    Build a structured location prediction dict for one zone.
    """
    stats = zone_stats.get(zone_name, {})

    if not stats:
        # Fallback: use zone centroid from config
        bounds = LOCATION_ZONES.get(zone_name, {})
        if bounds:
            centroid_lat = (bounds["lat"][0] + bounds["lat"][1]) / 2
            centroid_lon = (bounds["lon"][0] + bounds["lon"][1]) / 2
        else:
            centroid_lat, centroid_lon = 20.0, 96.0
        return {
            "zone":           zone_name,
            "centroid_lat":   round(centroid_lat, 3),
            "centroid_lon":   round(centroid_lon, 3),
            "radius_km":      300.0,
            "event_count":    0,
            "max_magnitude":  None,
            "confidence_note": "No recent events in zone - using zone centroid",
        }

    return {
        "zone":            zone_name,
        "centroid_lat":    stats["centroid_lat"],
        "centroid_lon":    stats["centroid_lon"],
        "radius_km":       stats["radius_km"],
        "event_count":     stats["event_count"],
        "max_magnitude":   stats["max_magnitude"],
        "mean_magnitude":  stats["mean_magnitude"],
        "last_event_time": stats["last_event_time"],
        "confidence_note": confidence_note,
    }


# ---------------------------------------------------------------------------
# Format coordinates for display
# ---------------------------------------------------------------------------
def format_coords(lat: float, lon: float) -> str:
    lat_dir = "N" if lat >= 0 else "S"
    lon_dir = "E" if lon >= 0 else "W"
    return f"{abs(lat):.3f}{lat_dir}, {abs(lon):.3f}{lon_dir}"


def format_radius(radius_km: float) -> str:
    if radius_km < 100:
        return f"~{radius_km:.0f} km radius  [HIGH location precision]"
    elif radius_km < 200:
        return f"~{radius_km:.0f} km radius  [MODERATE location precision]"
    elif radius_km < 300:
        return f"~{radius_km:.0f} km radius  [LOW-MODERATE location precision]"
    else:
        return f"~{radius_km:.0f} km radius  [LOW location precision - broad area]"
