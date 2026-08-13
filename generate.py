"""
generate.py
Lightweight prediction refresh script — runs in GitHub Actions every 6 hours.
Does NOT retrain the model. Loads the saved adaptive model, fetches the last
90 days of USGS data and merges it with the historical catalog's most recent
500 events (spanning several years, not just 90 days) so the spatial stats
in prediction_engine.py have enough runway to expand their lookback window
on a quiet region without running out of data, then saves new predictions.
Runs in under 60 seconds.
"""

import os
import sys
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

from config import (
    GEO_BOUNDS, MIN_MAGNITUDE, USGS_API_URL,
    FULL_CATALOG_PATH, PROCESSED_DATA_PATH,
    MODEL_ADAPTIVE_PATH, PREDICTIONS_PATH,
)
from models import AdaptiveModel
from prediction_engine import generate_predictions, save_predictions


def fetch_recent_events(days: int = 90) -> pd.DataFrame:
    """Fetch the last N days from USGS API for up-to-date spatial stats."""
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    params = {
        "format":       "geojson",
        "starttime":    start.strftime("%Y-%m-%d"),
        "endtime":      end.strftime("%Y-%m-%d"),
        "minmagnitude": MIN_MAGNITUDE,
        "minlatitude":  GEO_BOUNDS["min_latitude"],
        "maxlatitude":  GEO_BOUNDS["max_latitude"],
        "minlongitude": GEO_BOUNDS["min_longitude"],
        "maxlongitude": GEO_BOUNDS["max_longitude"],
    }
    print(f"[GENERATE] Fetching last {days} days from USGS...", flush=True)
    try:
        r = requests.get(USGS_API_URL, params=params, timeout=30)
        r.raise_for_status()
        feats = r.json().get("features", [])
        rows = []
        for f in feats:
            p = f["properties"]
            c = f["geometry"]["coordinates"]
            rows.append({
                "time":      datetime.fromtimestamp(p["time"] / 1000, timezone.utc).replace(tzinfo=None),
                "latitude":  c[1],
                "longitude": c[0],
                "depth":     c[2],
                "magnitude": p["mag"],
                "id":        f["id"],
            })
        df = pd.DataFrame(rows)
        print(f"[GENERATE] Got {len(df)} recent events from USGS.")
        return df
    except Exception as e:
        print(f"[GENERATE] USGS fetch failed: {e}. Falling back to catalog.")
        return pd.DataFrame()


def main():
    # ── Load pre-trained model ────────────────────────────────
    if not os.path.exists(MODEL_ADAPTIVE_PATH):
        print(f"[GENERATE] ERROR: {MODEL_ADAPTIVE_PATH} not found.")
        print("[GENERATE] Run `python main.py --mode train` locally first.")
        sys.exit(1)

    print("[GENERATE] Loading adaptive model...", flush=True)
    model = AdaptiveModel.load()

    # ── Load processed features (last 50 rows used for prediction) ──
    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"[GENERATE] ERROR: {PROCESSED_DATA_PATH} not found.")
        sys.exit(1)

    features_df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["time"])
    features_df["time"] = pd.to_datetime(features_df["time"])
    recent_features = features_df.tail(50)
    print(f"[GENERATE] Features loaded. Using last {len(recent_features)} rows.")

    # ── Build spatial context: fresh USGS data + historical catalog ──
    recent_events = fetch_recent_events(days=90)

    if recent_events.empty and os.path.exists(FULL_CATALOG_PATH):
        print("[GENERATE] Using historical catalog for spatial stats.")
        cat = pd.read_csv(FULL_CATALOG_PATH, parse_dates=["time"])
        recent_events = cat.sort_values("time").tail(500)
    elif not recent_events.empty and os.path.exists(FULL_CATALOG_PATH):
        # Merge fresh events with the tail of the historical catalog
        cat = pd.read_csv(FULL_CATALOG_PATH, parse_dates=["time"])
        tail = cat.sort_values("time").tail(500)
        recent_events = pd.concat([tail, recent_events], ignore_index=True)
        if "id" in recent_events.columns:
            recent_events = recent_events.drop_duplicates(subset=["id"])
        recent_events = recent_events.sort_values("time")

    # ── Generate predictions with today's date as reference ──────
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    print(f"[GENERATE] Generating predictions for {now.strftime('%Y-%m-%d %H:%M UTC')}...")

    predictions = generate_predictions(
        model=model,
        current_features=recent_features,
        recent_events=recent_events if len(recent_events) > 0 else None,
        reference_date=now,
        # 4.0 to include Myanmar's lowest trained threshold (already in
        # MAGNITUDE_THRESHOLDS, previously filtered out here) - harmless
        # for Japan, whose own MAGNITUDE_THRESHOLDS starts at 4.5 anyway.
        min_threshold=4.0,
        verbose=True,
    )

    save_predictions(predictions)
    print(f"[GENERATE] Done. {len(predictions)} predictions saved.")


if __name__ == "__main__":
    main()
