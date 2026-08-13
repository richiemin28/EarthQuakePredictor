# =============================================================================
# prediction_engine.py
# Generates structured earthquake predictions with:
#   - Probability score per magnitude threshold and time window
#   - Estimated epicenter coordinates (latitude, longitude)
#   - Estimated uncertainty radius in kilometres
#   - Geographic zone name
#   - Recent seismic activity context
#
# Location estimation uses magnitude-weighted spatial clustering of
# recent seismicity to identify the most active fault-zone areas.
# Smaller radius = tighter clustering = higher location confidence.
# =============================================================================

import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

from config import (
    LOCATION_ZONES,
    MAGNITUDE_THRESHOLDS,
    PREDICTION_WINDOWS,
    PREDICTIONS_PATH,
    ZONE_PRIORITY,
)
from feature_engineering import FEATURE_COLUMNS
from spatial_predictor import (
    compute_zone_spatial_stats,
    rank_zones,
    build_location_prediction,
    identify_zone,
    format_coords,
    format_radius,
    format_depth,
)


# ---------------------------------------------------------------------------
# Confidence labels
# ---------------------------------------------------------------------------
def _confidence_label(prob: float) -> str:
    if prob >= 0.80:
        return "SEVERE"
    elif prob >= 0.60:
        return "HIGH"
    elif prob >= 0.40:
        return "ELEVATED"
    elif prob >= 0.20:
        return "GUARDED"
    else:
        return "LOW"


def _prob_bar(prob: float, width: int = 20) -> str:
    filled = int(prob * width)
    bar = "=" * filled + ">" + " " * (width - filled)
    return f"[{bar}] {prob:.0%}"


# ---------------------------------------------------------------------------
# Magnitude descriptions
# ---------------------------------------------------------------------------
MAGNITUDE_INFO = {
    4.5: {
        "label":  "MODERATE (M >= 4.5)",
        "effect": "Felt strongly nearby, possible minor structural damage",
        "action": "AWARENESS - monitor updates",
    },
    5.0: {
        "label":  "STRONG (M >= 5.0)",
        "effect": "Significant shaking, damage likely to weak structures",
        "action": "PREPAREDNESS - check emergency readiness",
    },
    5.5: {
        "label":  "STRONG+ (M >= 5.5)",
        "effect": "Destructive shaking, serious infrastructure risk",
        "action": "HIGH ALERT - significant damage potential",
    },
    # 6.0-7.0 are only ever reported for regions whose config lists these
    # thresholds (currently Japan - see config_japan.py's own comment for
    # why: real signal at this range, unlike Myanmar's much sparser
    # high-magnitude catalog).
    6.0: {
        "label":  "MAJOR (M >= 6.0)",
        "effect": "Serious damage to vulnerable buildings, injuries likely",
        "action": "SEVERE ALERT - activate emergency preparedness",
    },
    6.5: {
        "label":  "MAJOR+ (M >= 6.5)",
        "effect": "Widespread structural damage, significant casualty risk",
        "action": "SEVERE ALERT - review evacuation and shelter plans",
    },
    7.0: {
        "label":  "GREAT (M >= 7.0)",
        "effect": "Catastrophic damage over a wide area, major casualty and infrastructure risk",
        "action": "CRITICAL ALERT - full emergency response readiness",
    },
}


# ---------------------------------------------------------------------------
# Core prediction generation
# ---------------------------------------------------------------------------
def generate_predictions(model,
                         current_features: pd.DataFrame,
                         recent_events:    pd.DataFrame = None,
                         reference_date:   datetime = None,
                         min_threshold:    float = 4.0,
                         verbose:          bool = True) -> list:
    """
    Generate full structured predictions including location estimates.

    For each (magnitude_threshold, prediction_window) combination:
      1. Get probability from the adaptive model
      2. Compute spatial stats from recent catalog
      3. Identify top 3 most active zones as likely locations
      4. Attach coordinates + radius to each zone

    Parameters
    ----------
    model            : Fitted AdaptiveModel
    current_features : Recent feature rows (last N rows of features_df)
    recent_events    : Raw catalog events for spatial context
    reference_date   : Prediction start date (default: now UTC)
    min_threshold    : Minimum magnitude threshold to report
    verbose          : Print to console

    Returns
    -------
    List of prediction dicts sorted by probability descending
    """

    if reference_date is None:
        reference_date = datetime.now(timezone.utc).replace(tzinfo=None)

    # Spatial stats are computed per prediction window, not once for the
    # whole run. A 7-day forecast should point at a "specific spot" tied to
    # roughly the last one to two weeks of actual clustering; reusing the
    # same fixed 90-day snapshot for every window (the old behaviour) meant
    # the 7-day and 30-day forecasts always showed the identical circle,
    # which isn't what "upcoming, close" should mean. Longer windows get a
    # proportionally longer lookback (still well under the old fixed 90
    # days) so the near-term answer is the tightest one, not a coincidence.
    def _lookback_for_window(window_days: int) -> int:
        return max(14, min(45, round(window_days * 1.5)))

    zone_stats_by_window = {
        w: compute_zone_spatial_stats(
            recent_events, lookback_days=_lookback_for_window(w), min_magnitude=3.0
        )
        for w in PREDICTION_WINDOWS
    }
    ranked_zones_by_window = {
        w: rank_zones(zone_stats_by_window[w], top_n=3)
        for w in PREDICTION_WINDOWS
    }

    predictions = []
    report_thresholds = [t for t in MAGNITUDE_THRESHOLDS if t >= min_threshold]

    for threshold in report_thresholds:
        for window in PREDICTION_WINDOWS:

            # Get probability from model
            try:
                probs = model.predict_proba(
                    current_features, threshold, window
                )
                prob = float(np.mean(probs))
                prob = max(0.0, min(1.0, prob))
            except Exception:
                continue

            date_end = reference_date + timedelta(days=window)
            mag_info = MAGNITUDE_INFO.get(threshold, {})

            zone_stats   = zone_stats_by_window[window]
            ranked_zones = ranked_zones_by_window[window]

            # Build location predictions for top 3 zones.
            # If ranked_zones has results (computed from real events),
            # use those. Otherwise fall back to the top 3 zones from this
            # country's own ZONE_PRIORITY table (highest-priority first) -
            # e.g. Sagaing Fault Zone leads for Myanmar, Nankai Trough for
            # Japan. This path used to hardcode Myanmar's zone names
            # regardless of which country's pipeline was actually running,
            # which broke silently for Japan since none of those names
            # exist in config_japan.py's LOCATION_ZONES.
            location_predictions = []
            if ranked_zones:
                for zone_name, stats in ranked_zones:
                    loc = build_location_prediction(zone_name, zone_stats)
                    location_predictions.append(loc)
            else:
                priority_zones = sorted(
                    ZONE_PRIORITY, key=ZONE_PRIORITY.get, reverse=True
                )[:3]
                for zone_name in priority_zones:
                    loc = build_location_prediction(zone_name, zone_stats)
                    location_predictions.append(loc)

            primary_loc = location_predictions[0]

            predictions.append({
                "generated_at":           reference_date.strftime(
                                              "%Y-%m-%d %H:%M UTC"
                                          ),
                # How many times the adaptive model has retrained itself on
                # confirmed new earthquake data since it was first built.
                # 0 is a legitimate value (freshly trained, no updates yet).
                "model_update_count":     getattr(model, "update_count", 0),
                "magnitude_threshold":    threshold,
                "magnitude_label":        mag_info.get(
                                              "label", f"M{threshold}+"
                                          ),
                "magnitude_effect":       mag_info.get("effect", ""),
                "magnitude_action":       mag_info.get("action", ""),
                "prediction_window_days": window,
                "date_range_start":       reference_date.strftime("%Y-%m-%d"),
                "date_range_end":         date_end.strftime("%Y-%m-%d"),
                "probability":            round(prob, 4),
                "confidence":             _confidence_label(prob),
                # Primary (most likely) location
                "primary_zone":           primary_loc["zone"],
                "primary_lat":            primary_loc["centroid_lat"],
                "primary_lon":            primary_loc["centroid_lon"],
                "primary_radius_km":      primary_loc["radius_km"],
                "primary_depth_km":       primary_loc.get("centroid_depth_km"),
                "primary_depth_range_km": primary_loc.get("depth_range_km"),
                # True when primary_lat/lon/radius describe a real DBSCAN-
                # found cluster of nearby recent events, not an average
                # over every event anywhere in the zone (see
                # spatial_predictor._find_hotspot_mask).
                "primary_is_hotspot":     primary_loc.get("is_hotspot", False),
                # All top locations
                "location_predictions":   location_predictions,
                "disclaimer": (
                    "Probabilistic estimate only. Coordinates indicate the "
                    "centroid of recent seismic clustering, not a precise "
                    "epicenter. Radius and depth reflect spatial "
                    "uncertainty, both tighter where recent activity is "
                    "well-clustered and wider where it is not. "
                    "Not for operational early-warning use."
                ),
            })

    predictions.sort(key=lambda x: x["probability"], reverse=True)

    if verbose:
        # Console "recent activity" overview uses its own fixed-lookback
        # snapshot (see print_predictions' default) - it's a human-readable
        # summary, independent of the per-window stats used for the JSON
        # output above.
        print_predictions(predictions, recent_events)

    return predictions


# ---------------------------------------------------------------------------
# Console display
# ---------------------------------------------------------------------------
def print_predictions(predictions: list,
                      recent_events: pd.DataFrame = None,
                      zone_stats: dict = None):
    """
    Print the full prediction dashboard to the terminal.
    Shows recent activity, zone stats, and forward predictions
    with coordinates and radius for each location.
    """

    now_str = datetime.now(timezone.utc).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S UTC")
    W = 72

    print("\n" + "=" * W)
    print("  MYANMAR EARTHQUAKE PREDICTION SYSTEM")
    print(f"  Generated  : {now_str}")
    print(f"  Region     : Myanmar + Surrounding Tectonic Region")
    print(f"  Bounds     : 10-30N, 90-105E")
    print("=" * W)

    # -----------------------------------------------------------------------
    # Recent seismic activity
    # -----------------------------------------------------------------------
    if recent_events is not None and len(recent_events) > 0:

        sig = recent_events[
            recent_events["magnitude"] >= 4.0
        ].sort_values("time", ascending=False).head(10)

        print(f"\n  RECENT SEISMIC ACTIVITY  (M4.0+ events, last 90 days)")
        print("  " + "-" * (W - 2))

        if len(sig) == 0:
            print("  No M4.0+ events in recent window.")
        else:
            for _, row in sig.iterrows():
                t    = pd.to_datetime(row["time"])
                mag  = row["magnitude"]
                lat  = row["latitude"]
                lon  = row["longitude"]
                zone = identify_zone(lat, lon)
                coords = format_coords(lat, lon)

                if mag >= 5.5:
                    tag = "[!!!]"
                elif mag >= 5.0:
                    tag = "[>> ]"
                elif mag >= 4.5:
                    tag = "[>  ]"
                else:
                    tag = "[   ]"

                print(f"  {tag} M{mag:.1f}  "
                      f"{t.strftime('%Y-%m-%d %H:%M')}  "
                      f"{coords}")
                print(f"          Zone: {zone}")

        print()

        # Zone activity summary
        if zone_stats is None:
            zone_stats = compute_zone_spatial_stats(
                recent_events, lookback_days=90, min_magnitude=3.0
            )
        active_zones = rank_zones(zone_stats, top_n=5)

        if active_zones:
            print(f"  ZONE ACTIVITY SUMMARY  (weighted by recency + magnitude)")
            print("  " + "-" * (W - 2))
            for zone_name, stats in active_zones:
                count   = stats["event_count"]
                max_mag = stats["max_magnitude"]
                lat     = stats["centroid_lat"]
                lon     = stats["centroid_lon"]
                radius  = stats["radius_km"]
                coords  = format_coords(lat, lon)
                bar     = "#" * min(count, 25)
                print(f"  {zone_name:<36} {bar} ({count} events)")
                print(f"    Centroid: {coords}  |  "
                      f"Radius: {radius:.0f} km  |  "
                      f"Max: M{max_mag}")
            print()

    # -----------------------------------------------------------------------
    # Forward predictions grouped by magnitude threshold
    # -----------------------------------------------------------------------
    if not predictions:
        print("  No predictions generated.")
        print("=" * W + "\n")
        return

    print(f"  FORWARD EARTHQUAKE PREDICTIONS")
    print("=" * W)

    thresholds = sorted(
        set(p["magnitude_threshold"] for p in predictions), reverse=True
    )

    for threshold in thresholds:
        t_preds = sorted(
            [p for p in predictions if p["magnitude_threshold"] == threshold],
            key=lambda x: x["prediction_window_days"]
        )
        if not t_preds:
            continue

        mag_info = MAGNITUDE_INFO.get(threshold, {})
        print(f"\n  {'='*66}")
        print(f"  {mag_info.get('label', f'M{threshold}+')}")
        print(f"  {mag_info.get('effect', '')}")
        print(f"  Action: {mag_info.get('action', '')}")
        print(f"  {'='*66}")

        for pred in t_preds:
            window  = pred["prediction_window_days"]
            prob    = pred["probability"]
            conf    = pred["confidence"]
            d_start = pred["date_range_start"]
            d_end   = pred["date_range_end"]
            p_zone  = pred["primary_zone"]
            p_lat   = pred["primary_lat"]
            p_lon   = pred["primary_lon"]
            p_rad   = pred["primary_radius_km"]
            p_depth = pred.get("primary_depth_km")
            p_drange = pred.get("primary_depth_range_km")

            print(f"\n  {window}-DAY WINDOW  ({d_start} to {d_end})")
            print(f"  Probability : {_prob_bar(prob)}  [{conf}]")
            print()

            # Primary location
            print(f"  PRIMARY LOCATION")
            print(f"    Zone        : {p_zone}")
            print(f"    Coordinates : {format_coords(p_lat, p_lon)}")
            print(f"    Est. Lat    : {p_lat:.3f} N")
            print(f"    Est. Lon    : {p_lon:.3f} E")
            print(f"    Radius      : {format_radius(p_rad)}")
            print(f"    Depth       : {format_depth(p_depth, p_drange)}")
            print(f"    Basis       : {'real event cluster (DBSCAN)' if pred.get('primary_is_hotspot') else 'whole-zone average (no distinct cluster found)'}")

            # Secondary locations
            other_locs = pred.get("location_predictions", [])[1:]
            if other_locs:
                print(f"\n  ALSO WATCH")
                for i, loc in enumerate(other_locs[:2], 2):
                    print(f"    #{i} {loc['zone']}")
                    print(f"       {format_coords(loc['centroid_lat'], loc['centroid_lon'])}  "
                          f"Radius: {loc['radius_km']:.0f} km")

            print("  " + "-" * 66)

    # -----------------------------------------------------------------------
    # Disclaimer
    # -----------------------------------------------------------------------
    print()
    print("  IMPORTANT NOTICE")
    print("  " + "-" * (W - 2))
    print("  Coordinates show the centroid of recent seismic clustering,")
    print("  not a precise predicted epicenter. Radius reflects spatial")
    print("  uncertainty derived from event dispersion. These are")
    print("  probabilistic research outputs, not official warnings.")
    print("=" * W + "\n")


# ---------------------------------------------------------------------------
# Save / load
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


# ---------------------------------------------------------------------------
# Results comparison table
# ---------------------------------------------------------------------------
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
