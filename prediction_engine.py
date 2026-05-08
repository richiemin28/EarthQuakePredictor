# =============================================================================
# prediction_engine.py
# Generates structured earthquake predictions with:
#   - Approximate zone name and description
#   - Estimated centre coordinates (lat, lon)
#   - Estimated search radius in km
#   - Primary fault structure
#   - Probability and confidence level
#   - Date range window
#   - Recent seismic activity context
# =============================================================================

import json
import math
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from config import (
    LOCATION_ZONES,
    MAGNITUDE_THRESHOLDS,
    PREDICTION_WINDOWS,
    PREDICTIONS_PATH,
)
from feature_engineering import FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Confidence labels
# ---------------------------------------------------------------------------
def _confidence_label(prob: float) -> str:
    if prob >= 0.85:
        return "HIGH"
    elif prob >= 0.65:
        return "MODERATE"
    elif prob >= 0.45:
        return "LOW-MODERATE"
    else:
        return "LOW"


def _confidence_bar(prob: float, width: int = 20) -> str:
    filled = int(prob * width)
    filled = min(filled, width)
    bar = "[" + "=" * filled + ">" + " " * max(0, width - filled) + "]"
    return f"{bar} {prob:.0%}"


# ---------------------------------------------------------------------------
# Magnitude descriptions
# ---------------------------------------------------------------------------
MAGNITUDE_INFO = {
    4.5: {
        "label":  "MODERATE (M4.5+)",
        "effect": "Felt strongly, minor damage to weak structures possible",
        "action": "Awareness level",
    },
    5.0: {
        "label":  "STRONG (M5.0+)",
        "effect": "Significant shaking, damage likely to weak structures",
        "action": "Preparedness level - check emergency readiness",
    },
    5.5: {
        "label":  "STRONG+ (M5.5+)",
        "effect": "Destructive shaking, serious infrastructure risk",
        "action": "HIGH ALERT - significant damage potential",
    },
}


# ---------------------------------------------------------------------------
# Haversine distance between two lat/lon points in km
# ---------------------------------------------------------------------------
def _haversine_km(lat1: float, lon1: float,
                  lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Zone identification
# ---------------------------------------------------------------------------
def identify_zone(latitude: float, longitude: float) -> str:
    """Return the zone name for a given coordinate."""
    for zone_name, z in LOCATION_ZONES.items():
        if (z["lat"][0] <= latitude <= z["lat"][1] and
                z["lon"][0] <= longitude <= z["lon"][1]):
            return zone_name
    return "Myanmar Region (General)"


def get_zone_info(zone_name: str) -> dict:
    """Return the full zone metadata dict."""
    return LOCATION_ZONES.get(zone_name, {
        "centre_lat": 21.0,
        "centre_lon": 96.0,
        "radius_km":  500,
        "fault":      "Unknown",
        "description": "Myanmar and surrounding tectonic region",
    })


# ---------------------------------------------------------------------------
# Zone activity from recent catalog
# ---------------------------------------------------------------------------
def get_zone_activity(recent_events: pd.DataFrame) -> dict:
    """Count recent events per zone, sorted by activity descending."""
    zone_counts = {z: 0 for z in LOCATION_ZONES}
    if recent_events is None or len(recent_events) == 0:
        return zone_counts
    for _, row in recent_events.iterrows():
        zone = identify_zone(row["latitude"], row["longitude"])
        if zone in zone_counts:
            zone_counts[zone] += 1
    return dict(sorted(zone_counts.items(), key=lambda x: x[1], reverse=True))


def get_active_zones(recent_events: pd.DataFrame, top_n: int = 3) -> list:
    activity = get_zone_activity(recent_events)
    active = [z for z, c in activity.items() if c > 0]
    if not active:
        active = list(LOCATION_ZONES.keys())
    return active[:top_n]


# ---------------------------------------------------------------------------
# Compute weighted centre coordinates from recent events in a zone
# More precise than just using the zone centre when events cluster tightly
# ---------------------------------------------------------------------------
def _weighted_centre(recent_events: pd.DataFrame,
                     zone_name: str) -> tuple:
    """
    Compute a weighted average lat/lon for recent events in a zone,
    weighted by magnitude. Falls back to zone centre if no events.
    Returns (lat, lon, radius_km, n_events_used).
    """
    zone_info = get_zone_info(zone_name)
    default = (
        zone_info["centre_lat"],
        zone_info["centre_lon"],
        zone_info["radius_km"],
        0,
    )

    if recent_events is None or len(recent_events) == 0:
        return default

    # Filter to events in this zone
    zone_events = recent_events[
        recent_events.apply(
            lambda r: identify_zone(r["latitude"], r["longitude"]) == zone_name,
            axis=1,
        )
    ]

    if len(zone_events) == 0:
        return default

    # Weight by magnitude (larger events anchor the centre more)
    weights = zone_events["magnitude"].values
    weights = weights / weights.sum()

    w_lat = float(np.average(zone_events["latitude"].values,  weights=weights))
    w_lon = float(np.average(zone_events["longitude"].values, weights=weights))

    # Radius = max distance from weighted centre to any event in zone
    # plus a minimum of 100 km to reflect prediction uncertainty
    distances = [
        _haversine_km(w_lat, w_lon, r["latitude"], r["longitude"])
        for _, r in zone_events.iterrows()
    ]
    radius = max(max(distances) * 1.5, 150.0)
    # Cap at the zone's default radius
    radius = min(radius, zone_info["radius_km"])

    return (round(w_lat, 3), round(w_lon, 3), round(radius, 1), len(zone_events))


# ---------------------------------------------------------------------------
# Core prediction generation
# ---------------------------------------------------------------------------
def generate_predictions(model,
                         current_features: pd.DataFrame,
                         recent_events: pd.DataFrame = None,
                         reference_date: datetime = None,
                         min_threshold: float = 4.5,
                         verbose: bool = True) -> list:
    """
    Generate structured earthquake predictions from the adaptive model.

    Each prediction includes:
      - Zone name and description
      - Estimated centre coordinates (weighted from recent activity)
      - Estimated search radius in km
      - Primary fault structure
      - Probability score and confidence label
      - Date range window
    """

    if reference_date is None:
        reference_date = datetime.utcnow()

    zone_activity = get_zone_activity(recent_events)
    active_zones  = get_active_zones(recent_events, top_n=4)

    predictions = []
    report_thresholds = [t for t in MAGNITUDE_THRESHOLDS if t >= min_threshold]

    for threshold in report_thresholds:
        for window in PREDICTION_WINDOWS:

            try:
                probs = model.predict_proba(current_features, threshold, window)
                prob  = float(np.mean(probs))
                prob  = max(0.0, min(1.0, prob))
            except Exception:
                continue

            date_end = reference_date + timedelta(days=window)
            mag_info = MAGNITUDE_INFO.get(threshold, {})

            # Build per-zone location details for this prediction
            zone_details = []
            report_zones = active_zones if active_zones else list(LOCATION_ZONES.keys())[:3]

            for zone_name in report_zones:
                w_lat, w_lon, radius, n_ev = _weighted_centre(
                    recent_events, zone_name
                )
                zone_info = get_zone_info(zone_name)

                zone_details.append({
                    "zone":          zone_name,
                    "description":   zone_info.get("description", ""),
                    "fault":         zone_info.get("fault", "Unknown"),
                    "centre_lat":    w_lat,
                    "centre_lon":    w_lon,
                    "radius_km":     radius,
                    "recent_events": n_ev,
                    "coords_str":    f"{abs(w_lat):.3f}{'N' if w_lat>=0 else 'S'}, "
                                     f"{abs(w_lon):.3f}{'E' if w_lon>=0 else 'W'}",
                })

            primary_zone = zone_details[0] if zone_details else {}

            predictions.append({
                "generated_at":           reference_date.strftime(
                    "%Y-%m-%d %H:%M UTC"),
                "magnitude_threshold":    threshold,
                "magnitude_label":        mag_info.get("label", f"M{threshold}+"),
                "magnitude_effect":       mag_info.get("effect", ""),
                "magnitude_action":       mag_info.get("action", ""),
                "prediction_window_days": window,
                "date_range_start":       reference_date.strftime("%Y-%m-%d"),
                "date_range_end":         date_end.strftime("%Y-%m-%d"),
                "probability":            round(prob, 4),
                "confidence":             _confidence_label(prob),
                "primary_zone":           primary_zone,
                "all_zones":              zone_details,
                "disclaimer": (
                    "Probabilistic approximation only. Coordinates indicate "
                    "the weighted centre of recent seismic activity in the "
                    "most active zone. Exact time, location and magnitude "
                    "cannot be predicted deterministically. Not for "
                    "operational early-warning use without independent "
                    "seismological validation."
                ),
            })

    predictions.sort(key=lambda x: x["probability"], reverse=True)

    if verbose:
        print_predictions(predictions, recent_events)

    return predictions


# ---------------------------------------------------------------------------
# Console display
# ---------------------------------------------------------------------------
def print_predictions(predictions: list,
                      recent_events: pd.DataFrame = None):
    """Full dashboard-style prediction output with coordinates."""

    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    print("\n" + "=" * 72)
    print("  MYANMAR EARTHQUAKE PREDICTION SYSTEM")
    print(f"  Generated : {now_str}")
    print("  Coverage  : Myanmar + Surrounding Tectonic Region")
    print("=" * 72)

    # -----------------------------------------------------------------------
    # Section 1: Recent seismic activity
    # -----------------------------------------------------------------------
    if recent_events is not None and len(recent_events) > 0:
        recent_sig = recent_events[
            recent_events["magnitude"] >= 4.0
        ].sort_values("time", ascending=False).head(8)

        print("\n  RECENT SEISMIC ACTIVITY (M4.0+, most recent first)")
        print("  " + "-" * 68)

        if len(recent_sig) == 0:
            print("  No M4.0+ events in recent catalog window.")
        else:
            for _, row in recent_sig.iterrows():
                t    = pd.to_datetime(row["time"])
                mag  = row["magnitude"]
                lat  = row["latitude"]
                lon  = row["longitude"]
                zone = identify_zone(lat, lon)

                if mag >= 5.5:
                    marker = "  [!!!]"
                elif mag >= 5.0:
                    marker = "  [>> ]"
                elif mag >= 4.5:
                    marker = "  [>  ]"
                else:
                    marker = "  [   ]"

                lat_str = f"{abs(lat):.3f}{'N' if lat >= 0 else 'S'}"
                lon_str = f"{abs(lon):.3f}{'E' if lon >= 0 else 'W'}"

                print(f"{marker} M{mag:.1f}  "
                      f"{t.strftime('%Y-%m-%d %H:%M')} UTC  "
                      f"{zone}")
                print(f"         Coords: {lat_str}, {lon_str}")

        print()

    # -----------------------------------------------------------------------
    # Section 2: Zone activity summary
    # -----------------------------------------------------------------------
    if recent_events is not None and len(recent_events) > 0:
        activity = get_zone_activity(recent_events)
        active = [(z, c) for z, c in activity.items() if c > 0]

        if active:
            print("  ZONE ACTIVITY SUMMARY (recent catalog window)")
            print("  " + "-" * 68)
            for zone, count in active[:5]:
                bar = "#" * min(count, 28)
                z_info = get_zone_info(zone)
                clat = z_info.get("centre_lat", 0)
                clon = z_info.get("centre_lon", 0)
                print(f"  {zone:<28}  {bar:<30} ({count} events)")
                print(f"  {'':28}  Centre: "
                      f"{abs(clat):.1f}{'N' if clat>=0 else 'S'}, "
                      f"{abs(clon):.1f}{'E' if clon>=0 else 'W'}")
            print()

    # -----------------------------------------------------------------------
    # Section 3: Forward predictions
    # -----------------------------------------------------------------------
    if not predictions:
        print("  No predictions available.")
        print("=" * 72)
        return

    print("  FORWARD PREDICTIONS")
    print("  " + "-" * 68)

    thresholds_shown = sorted(
        set(p["magnitude_threshold"] for p in predictions), reverse=True
    )

    for threshold in thresholds_shown:
        thresh_preds = sorted(
            [p for p in predictions if p["magnitude_threshold"] == threshold],
            key=lambda x: x["prediction_window_days"],
        )
        if not thresh_preds:
            continue

        mag_info = MAGNITUDE_INFO.get(threshold, {})
        print(f"\n  {mag_info.get('label', f'M{threshold}+')}")
        print(f"  {mag_info.get('effect', '')}")
        print(f"  Action: {mag_info.get('action', '')}")
        print()

        for pred in thresh_preds:
            window   = pred["prediction_window_days"]
            prob     = pred["probability"]
            conf     = pred["confidence"]
            d_end    = pred["date_range_end"]
            primary  = pred.get("primary_zone", {})
            all_zones = pred.get("all_zones", [])

            print(f"    {'='*62}")
            print(f"    Prediction Window : {window} days  "
                  f"(now through {d_end})")
            print(f"    Probability       : {_confidence_bar(prob, 18)}  "
                  f"[{conf}]")
            print()

            # Primary zone with full location detail
            if primary:
                print(f"    PRIMARY ZONE : {primary.get('zone','Unknown')}")
                print(f"    Description  : {primary.get('description','')}")
                print(f"    Fault        : {primary.get('fault','Unknown')}")
                print(f"    Est. Centre  : {primary.get('coords_str','N/A')}")
                lat = primary.get('centre_lat', 0)
                lon = primary.get('centre_lon', 0)
                print(f"    Coordinates  : {lat:.3f}N, {lon:.3f}E")
                print(f"    Search Radius: ~{primary.get('radius_km', 0):.0f} km "
                      f"from centre")
                if primary.get("recent_events", 0) > 0:
                    print(f"    Based on     : {primary['recent_events']} "
                          f"recent events in this zone")

            # Additional zones being watched
            others = [z for z in all_zones if z != primary]
            if others:
                print()
                print(f"    ALSO MONITORING:")
                for z in others[:2]:
                    zlat = z.get('centre_lat', 0)
                    zlon = z.get('centre_lon', 0)
                    print(f"      {z.get('zone',''):<28} "
                          f"{abs(zlat):.2f}{'N' if zlat>=0 else 'S'}, "
                          f"{abs(zlon):.2f}{'E' if zlon>=0 else 'W'}  "
                          f"(~{z.get('radius_km',0):.0f} km radius)  "
                          f"Fault: {z.get('fault','')}")

            print()

    # -----------------------------------------------------------------------
    # Disclaimer
    # -----------------------------------------------------------------------
    print("  " + "-" * 68)
    print("  IMPORTANT: Coordinates show the weighted centre of recent")
    print("  seismic activity in each zone. The search radius reflects")
    print("  the spatial spread of that activity plus prediction uncertainty.")
    print("  These are probabilistic estimates, not deterministic predictions.")
    print("  Not for operational use without seismological validation.")
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Save / load / summarise
# ---------------------------------------------------------------------------
def save_predictions(predictions: list, path: str = PREDICTIONS_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(predictions, f, indent=2, default=str)
    print(f"[PREDICT] {len(predictions)} predictions saved to {path}")


def load_predictions(path: str = PREDICTIONS_PATH) -> list:
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


def summarise_results(static_results: dict, adaptive_results: dict):
    print("\n" + "=" * 92)
    print(f"  {'Label':<22} {'Model':<12} {'Acc':<8} {'Prec':<8} "
          f"{'Recall':<8} {'F1':<8} {'AUC':<8}")
    print("  " + "-" * 88)

    all_base_cols = sorted(set(
        list(static_results.keys()) +
        [k.split(":")[0] for k in adaptive_results.keys()]
    ))

    for base_col in all_base_cols:
        s = static_results.get(base_col, {})
        a = adaptive_results.get(base_col, {})
        if not a:
            a = adaptive_results.get(f"{base_col}:final", {})

        def fmt(d):
            return (f"{d.get('accuracy','N/A'):<8} "
                    f"{d.get('precision','N/A'):<8} "
                    f"{d.get('recall','N/A'):<8} "
                    f"{d.get('f1','N/A'):<8} "
                    f"{d.get('auc','N/A'):<8}")

        if s:
            print(f"  {base_col:<22} {'Static':<12} {fmt(s)}")
        if a:
            print(f"  {base_col:<22} {'Adaptive':<12} {fmt(a)}")

    print("=" * 92)
