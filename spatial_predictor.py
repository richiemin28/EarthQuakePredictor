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
from config import LOCATION_ZONES, COUNTRY_NAME, ZONE_PRIORITY


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
    return f"{COUNTRY_NAME} Region (General)"


# ---------------------------------------------------------------------------
# Confidence-tiered precision floors.
#
# A tight radius (or depth spread) computed from only 2-3 events could just
# be coincidence, not genuine confidence - it isn't allowed to claim the
# same precision as the same tight number backed by dozens of clustered
# events. Floors loosen as event_count drops, so "how precise" is honestly
# tied to "how much real data actually supports it", not one fixed number
# applied regardless of sample size. 5km / 3km (the tightest tiers) are
# reachable only when a zone has genuinely dense recent clustering; most
# zones today land in the middle tiers - see the README's location
# precision section for where the wider floor actually comes from.
# ---------------------------------------------------------------------------
def _radius_floor_km(n_events: int) -> float:
    if n_events >= 40:
        return 5.0
    if n_events >= 25:
        return 10.0
    if n_events >= 15:
        return 20.0
    return 30.0


def _depth_floor_km(n_events: int) -> float:
    if n_events >= 40:
        return 3.0
    if n_events >= 25:
        return 5.0
    if n_events >= 15:
        return 8.0
    return 10.0


# ---------------------------------------------------------------------------
# Compute zone-level spatial statistics from recent catalog
# ---------------------------------------------------------------------------
def compute_zone_spatial_stats(recent_events: pd.DataFrame,
                                lookback_days: int = 90,
                                min_magnitude:  float = 3.0) -> dict:
    """
    For each geographic zone, compute:
      - event_count      : number of recent events
      - centroid_lat      : magnitude-weighted mean latitude
      - centroid_lon      : magnitude-weighted mean longitude
      - radius_km         : 1-sigma spread of events around centroid (km)
      - centroid_depth_km : magnitude-weighted mean depth
      - depth_range_km    : 1-sigma spread of event depths (km)
      - max_magnitude     : largest recent event in zone
      - last_event_time   : most recent event time

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

    # Use the most recent event date in the supplied data as the reference
    # point rather than datetime.utcnow(). This is critical when the catalog
    # ends before the current date (e.g. historical data ending Dec 2025
    # being queried in May 2026) - using utcnow() would return zero events.
    reference_date = df["time"].max()

    # If the requested lookback is too quiet for a stable estimate,
    # progressively widen the *time* window rather than falling back to an
    # arbitrary fixed event count. The old fallback (last 200 events by
    # count, no matter how far back that reached) meant a short lookback
    # -  e.g. the ~14-day window behind a 7-day forecast - could silently
    # balloon into over a year of old activity whenever the region had a
    # quiet stretch, which defeats tying "how recent" to "how near-term the
    # forecast is" at all. Capped at 180 days, double the old fixed default,
    # so even a very quiet stretch never reaches back further than that.
    MIN_EVENTS = 30
    mag_df = df[df["magnitude"] >= min_magnitude]
    df = mag_df[mag_df["time"] >= reference_date - timedelta(days=lookback_days)]
    for expanded_days in (lookback_days * 2, lookback_days * 4, 90, 180):
        if len(df) >= MIN_EVENTS:
            break
        df = mag_df[mag_df["time"] >= reference_date - timedelta(days=expanded_days)]
    df = df.copy()

    if len(df) == 0:
        return {}

    # Assign recency weight relative to the most recent event in data
    # Events in last 7 days (relative) get 3x, last 30 days get 2x, else 1x
    now = reference_date
    def recency_weight(t):
        days_ago = (now - t).total_seconds() / 86400
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

        n_events = len(group)

        # Weighted centroid
        centroid_lat = float(np.average(group["latitude"],  weights=weights))
        centroid_lon = float(np.average(group["longitude"], weights=weights))
        centroid_depth = float(np.average(group["depth"], weights=weights))

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
            # Depth spread: same weighted-RMS-from-mean logic as the
            # horizontal radius, just on the depth axis instead of the
            # haversine distance.
            depth_deviations = (group["depth"].values - centroid_depth) ** 2
            depth_range_km = float(np.sqrt(np.average(depth_deviations, weights=weights)))
        else:
            # Single event - use zone half-diagonal as radius, and the
            # loosest depth-confidence tier (one point says nothing about
            # spread).
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
            depth_range_km = _depth_floor_km(1)

        # Apply minimum and maximum bounds, confidence-tiered by how many
        # events actually back the estimate (see _radius_floor_km /
        # _depth_floor_km above) rather than one fixed floor regardless of
        # sample size. Max 400km / 60km: too diffuse to narrow further.
        radius_km      = max(_radius_floor_km(n_events), min(radius_km, 400.0))
        depth_range_km = max(_depth_floor_km(n_events), min(depth_range_km, 60.0))

        zone_stats[zone_name] = {
            "event_count":       n_events,
            "centroid_lat":      round(centroid_lat, 3),
            "centroid_lon":      round(centroid_lon, 3),
            "radius_km":         round(radius_km, 1),
            "centroid_depth_km": round(centroid_depth, 1),
            "depth_range_km":    round(depth_range_km, 1),
            "max_magnitude":     round(float(group["magnitude"].max()), 1),
            "mean_magnitude":    round(float(group["magnitude"].mean()), 2),
            "last_event_time":   str(group["time"].max()),
            "total_weight":      round(float(total_w), 2),
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

    # The generic "Region (General)" catch-all (anything that didn't fall
    # inside a named fault zone's box) is excluded from ranking entirely -
    # it's never returned as a location prediction, primary or secondary.
    # It used to be included with a heavy penalty instead of excluded, which
    # meant it could still occasionally surface as a vague "near <Country>
    # Region (General), radius 400km" answer; that's not a useful location
    # prediction for anyone, precise named zones only. compute_zone_spatial_
    # stats still computes it internally (identify_zone still needs
    # somewhere to put events outside every named box), it just never
    # reaches rank_zones' output.
    named_zone_stats = {
        name: stats for name, stats in zone_stats.items() if "General" not in name
    }
    if not named_zone_stats:
        return []

    # Known high-priority seismic zones get a bonus multiplier (per-country,
    # from config's ZONE_PRIORITY).
    PRIORITY_ZONES = ZONE_PRIORITY

    scored = []
    for zone_name, stats in named_zone_stats.items():

        # Base score: total seismic weight times log of event count
        base = stats["total_weight"] * np.log1p(stats["event_count"])

        # Precision bonus: tighter clusters get higher scores.
        # Radius of 30km = bonus 1.0, 400km = bonus 0.0
        max_radius = 400.0
        precision_bonus = max(
            0.0, 1.0 - (stats["radius_km"] / max_radius)
        )
        base *= (1.0 + precision_bonus)
        base *= PRIORITY_ZONES.get(zone_name, 1.0)

        scored.append((base, zone_name, stats))

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
        # Fallback: use the pre-defined centre coordinates from config
        # which are set to known fault-zone centroids, not just bounding
        # box midpoints.
        bounds = LOCATION_ZONES.get(zone_name, {})
        if bounds:
            centroid_lat = bounds.get(
                "centre_lat",
                (bounds["lat"][0] + bounds["lat"][1]) / 2
            )
            centroid_lon = bounds.get(
                "centre_lon",
                (bounds["lon"][0] + bounds["lon"][1]) / 2
            )
            radius_km = bounds.get("radius_km", 300.0)
        else:
            centroid_lat, centroid_lon = 20.0, 96.0
            radius_km = 300.0
        return {
            "zone":              zone_name,
            "centroid_lat":      round(centroid_lat, 3),
            "centroid_lon":      round(centroid_lon, 3),
            "radius_km":         radius_km,
            # No recent events to estimate depth from - reported as
            # unknown rather than guessed, unlike the horizontal position
            # which at least has a real fault-zone centroid to fall back on.
            "centroid_depth_km": None,
            "depth_range_km":    None,
            "event_count":       0,
            "max_magnitude":     None,
            "confidence_note":   "Using zone centroid - no events in computed window",
        }

    return {
        "zone":              zone_name,
        "centroid_lat":      stats["centroid_lat"],
        "centroid_lon":      stats["centroid_lon"],
        "radius_km":         stats["radius_km"],
        "centroid_depth_km": stats["centroid_depth_km"],
        "depth_range_km":    stats["depth_range_km"],
        "event_count":       stats["event_count"],
        "max_magnitude":     stats["max_magnitude"],
        "mean_magnitude":    stats["mean_magnitude"],
        "last_event_time":   stats["last_event_time"],
        "confidence_note":   confidence_note,
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


def format_depth(depth_km, depth_range_km) -> str:
    if depth_km is None:
        return "unknown - no recent events to estimate from"
    return f"~{depth_km:.0f} km (+/- {depth_range_km:.0f} km)"
