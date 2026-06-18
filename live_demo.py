# =============================================================================
# live_demo.py
# Real-time live prediction dashboard with continuous adaptive learning.
#
# Each cycle:
#   1. Polls USGS ATOM feed for new Myanmar-region events
#   2. Displays any new events with magnitude, coordinates, zone
#   3. Recomputes seismic features for the updated catalog
#   4. Triggers adaptive model update (replay-based continual learning)
#   5. Recomputes spatial clustering stats for location estimation
#   6. Displays refreshed predictions with:
#        - Probability score per threshold and window
#        - Estimated epicenter coordinates
#        - Uncertainty radius in km
#        - Zone name
#        - Ranked secondary locations
#   7. Saves predictions to JSON and logs all events to JSONL
#
# Usage:
#   python live_demo.py                  Default: poll every 5 minutes
#   python live_demo.py --interval 60   Fast mode: poll every 60 seconds
#   python live_demo.py --interval 30   Demo mode: poll every 30 seconds
#
# Press Ctrl+C to stop. Model is saved automatically on exit.
# =============================================================================

import argparse
import os
import sys
import time
import json
import traceback
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

from config import (
    PROCESSED_DATA_PATH,
    MODEL_ADAPTIVE_PATH,
    FULL_CATALOG_PATH,
    PREDICTIONS_PATH,
    LIVE_LOG_PATH,
    GEO_BOUNDS,
)
from data_acquisition    import fetch_live_events
from feature_engineering import (
    compute_features,
    build_labeled_dataset,
    FEATURE_COLUMNS,
)
from models              import AdaptiveModel
from spatial_predictor   import (
    compute_zone_spatial_stats,
    rank_zones,
    identify_zone,
    format_coords,
    format_radius,
)
from prediction_engine   import (
    generate_predictions,
    save_predictions,
    print_predictions,
)

# Number of recent feature rows used for prediction context
CONTEXT_ROWS = 50

# Number of days of recent events used for spatial clustering
SPATIAL_LOOKBACK_DAYS = 90


# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------
def clear():
    os.system("cls" if os.name == "nt" else "clear")


def hr(char="=", w=72):
    print(char * w)


def section(title: str, w: int = 72):
    print(f"\n  {title}")
    print("  " + "-" * (w - 2))


# ---------------------------------------------------------------------------
# Display new events as they are detected
# ---------------------------------------------------------------------------
def display_new_events(new_events: pd.DataFrame):
    if new_events is None or len(new_events) == 0:
        return

    print("\n")
    hr("*")
    print(f"  *** {len(new_events)} NEW EVENT(S) DETECTED IN MYANMAR REGION ***")
    hr("*")

    for _, row in new_events.sort_values(
        "magnitude", ascending=False
    ).iterrows():
        t      = pd.to_datetime(row["time"])
        mag    = row["magnitude"]
        lat    = row["latitude"]
        lon    = row["longitude"]
        zone   = identify_zone(lat, lon)
        coords = format_coords(lat, lon)
        place  = row.get("place", "Unknown location")

        if mag >= 5.5:
            flag = "  [!!!] MAJOR"
        elif mag >= 5.0:
            flag = "  [>> ] STRONG"
        elif mag >= 4.5:
            flag = "  [>  ] MODERATE"
        else:
            flag = "  [   ] MINOR"

        print()
        print(f"{flag}  M{mag:.1f}")
        print(f"         Time        : {t.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"         Coordinates : {coords}")
        print(f"         Latitude    : {lat:.4f} N")
        print(f"         Longitude   : {lon:.4f} E")
        print(f"         Zone        : {zone}")
        print(f"         Location    : {place}")

    hr("*")
    print()


# ---------------------------------------------------------------------------
# Display model update status
# ---------------------------------------------------------------------------
def display_update_status(update_num: int, n_events: int, n_features: int):
    print(f"\n  [ADAPTIVE MODEL UPDATE #{update_num}]")
    print(f"  New events ingested  : {n_events}")
    print(f"  New feature rows     : {n_features}")
    print(f"  Strategy             : Replay-based continual learning")
    print(f"  Catastrophic forgetting mitigated via replay buffer")
    print(f"  Retraining classifiers... ", end="", flush=True)


# ---------------------------------------------------------------------------
# Append a record to the JSONL log
# ---------------------------------------------------------------------------
def log(entry: dict):
    os.makedirs(os.path.dirname(LIVE_LOG_PATH), exist_ok=True)
    with open(LIVE_LOG_PATH, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


# ---------------------------------------------------------------------------
# Feature update - recompute only new rows
# ---------------------------------------------------------------------------
def update_feature_df(catalog_df:          pd.DataFrame,
                      existing_features_df: pd.DataFrame) -> tuple:
    """
    Recompute features on the full updated catalog.
    Returns (full_updated_features_df, new_rows_only_df)
    """
    updated = compute_features(catalog_df)
    updated = build_labeled_dataset(updated)

    if existing_features_df is not None and len(existing_features_df) > 0:
        existing_times = set(
            pd.to_datetime(
                existing_features_df["time"]
            ).dt.strftime("%Y-%m-%dT%H:%M:%S")
        )
        new_mask = ~pd.to_datetime(updated["time"]).dt.strftime(
            "%Y-%m-%dT%H:%M:%S"
        ).isin(existing_times)
        new_rows = updated[new_mask].copy()
    else:
        new_rows = updated.copy()

    return updated, new_rows


# ---------------------------------------------------------------------------
# Main dashboard class
# ---------------------------------------------------------------------------
class LiveDashboard:

    def __init__(self, poll_interval: int = 300):
        self.interval      = poll_interval
        self.cycle         = 0
        self.update_count  = 0
        self.last_update   = "Not yet"
        self.total_new     = 0

        print("\n" + "=" * 72)
        print("  MYANMAR EARTHQUAKE PREDICTION SYSTEM")
        print("  Starting live mode...")
        print("=" * 72)

        # Load features
        if not os.path.exists(PROCESSED_DATA_PATH):
            print("\n[ERROR] No processed features found.")
            print("Run: python main.py --mode train --refresh")
            sys.exit(1)
        self.features_df = pd.read_csv(
            PROCESSED_DATA_PATH, parse_dates=["time"]
        )
        self.features_df["time"] = pd.to_datetime(self.features_df["time"])
        print(f"  Features  : {len(self.features_df)} rows loaded")

        # Load catalog
        cat_path = FULL_CATALOG_PATH if os.path.exists(
            FULL_CATALOG_PATH
        ) else "data/raw_catalog.csv"
        if not os.path.exists(cat_path):
            print("\n[ERROR] No catalog file found.")
            print("Run: python main.py --mode train --refresh")
            sys.exit(1)
        self.catalog_df = pd.read_csv(cat_path, parse_dates=["time"])
        self.catalog_df["time"] = pd.to_datetime(self.catalog_df["time"])
        print(f"  Catalog   : {len(self.catalog_df)} events loaded")

        # Set last known event time
        self.last_event_time = (
            self.catalog_df["time"].max().to_pydatetime()
        )
        print(f"  Latest    : {self.last_event_time.strftime('%Y-%m-%d %H:%M UTC')}")

        # Load adaptive model
        if not os.path.exists(MODEL_ADAPTIVE_PATH):
            print("\n[ERROR] No adaptive model found.")
            print("Run: python main.py --mode train")
            sys.exit(1)
        self.model = AdaptiveModel.load()
        print(f"  Model     : Adaptive model loaded")
        print(f"  Interval  : Every {poll_interval}s")
        print("\n  Starting in 3 seconds... (Ctrl+C to stop)")
        time.sleep(3)

    def run(self):
        try:
            # Initial display on startup
            self._refresh_display(new_events=None)

            while True:
                self._countdown()
                self.cycle += 1
                self._poll_and_process()

        except KeyboardInterrupt:
            print("\n\n  Saving model and shutting down...")
            self.model.save()
            print(f"  Model saved to {MODEL_ADAPTIVE_PATH}")
            print(f"  Total updates : {self.update_count}")
            print(f"  Total new events detected: {self.total_new}")
            print(f"  Log saved to  : {LIVE_LOG_PATH}")
            print("  Goodbye.\n")

    def _countdown(self):
        for s in range(self.interval, 0, -1):
            print(
                f"\r  [LIVE] Next USGS poll in {s:>4}s  "
                f"| Cycle #{self.cycle}  "
                f"| Updates: {self.update_count}  "
                f"| Ctrl+C to stop",
                end="",
                flush=True,
            )
            time.sleep(1)
        print()

    def _poll_and_process(self):
        print(f"\n[LIVE] Cycle #{self.cycle}: Polling USGS ATOM feed...",
              flush=True)

        try:
            new_events = fetch_live_events(
                last_seen_time=self.last_event_time
            )
        except Exception as e:
            print(f"[LIVE] Feed error: {e}. Will retry next cycle.")
            self._refresh_display(new_events=None)
            return

        if new_events.empty:
            print("[LIVE] No new events in Myanmar region.")
            self._refresh_display(new_events=None)
            return

        # New events found
        self.total_new += len(new_events)
        self.last_event_time = pd.to_datetime(
            new_events["time"]
        ).max().to_pydatetime()
        self.last_update = datetime.now(timezone.utc).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M UTC")

        # Append to catalog
        self.catalog_df = pd.concat(
            [self.catalog_df, new_events], ignore_index=True
        ).drop_duplicates(subset=["id"]).sort_values("time").reset_index(drop=True)
        self.catalog_df.to_csv("data/live_catalog.csv", index=False)

        # Recompute features
        print("[LIVE] Recomputing features...", flush=True)
        try:
            updated_features, new_rows = update_feature_df(
                self.catalog_df, self.features_df
            )
            self.features_df = updated_features
        except Exception as e:
            print(f"[LIVE] Feature error: {e}")
            self._refresh_display(new_events=new_events)
            return

        # Update adaptive model if new feature rows exist
        if len(new_rows) > 0:
            self.update_count += 1
            display_update_status(
                self.update_count, len(new_events), len(new_rows)
            )
            try:
                self.model.update(new_rows, verbose=False)
                print("Done.")
                log({
                    "type":          "model_update",
                    "timestamp":     datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
                    "update_number": self.update_count,
                    "new_events":    len(new_events),
                    "new_features":  len(new_rows),
                })
            except Exception as e:
                print(f"Error: {e}")

            # Save checkpoint every 5 updates
            if self.update_count % 5 == 0:
                self.model.save()
                print(f"[LIVE] Checkpoint saved (update #{self.update_count})")
        else:
            print("[LIVE] No new feature rows generated from new events.")

        # Log each new event
        for _, row in new_events.iterrows():
            log({
                "type":       "new_event",
                "timestamp":  datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
                "event_time": str(row["time"]),
                "magnitude":  row["magnitude"],
                "latitude":   row["latitude"],
                "longitude":  row["longitude"],
                "zone":       identify_zone(row["latitude"], row["longitude"]),
                "place":      row.get("place", ""),
            })

        self._refresh_display(new_events=new_events)

    def _refresh_display(self, new_events: pd.DataFrame = None):
        """Clear and redraw the full dashboard."""
        clear()

        # Header
        now = datetime.now(timezone.utc).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S UTC")
        print("=" * 72)
        print("  MYANMAR EARTHQUAKE PREDICTION SYSTEM  |  LIVE MODE")
        print(f"  Current Time   : {now}")
        print(f"  Last Update    : {self.last_update}")
        print(f"  Poll Interval  : Every {self.interval}s")
        print(f"  Catalog Size   : {len(self.catalog_df)} events")
        print(f"  Model Updates  : {self.update_count}")
        print(f"  New Events     : {self.total_new} total detected this session")
        print("=" * 72)

        # Show new events if any
        if new_events is not None and len(new_events) > 0:
            display_new_events(new_events)

        # Spatial context: use the most recent 500 events from the
        # full historical + live catalog to ensure enough events for
        # meaningful spatial clustering regardless of ATOM feed recency.
        # This means the location estimates always draw on a rich
        # earthquake history rather than only today's events.
        recent = self.catalog_df.sort_values("time").tail(500)

        # Also try to get events from the last 90 days if there are enough
        cutoff = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(
            days=SPATIAL_LOOKBACK_DAYS
        ))
        recent_90d = self.catalog_df[
            pd.to_datetime(self.catalog_df["time"]) >= cutoff
        ]
        # Use whichever gives more events (at least 50 for good clustering)
        if len(recent_90d) >= 50:
            recent = recent_90d
        elif len(recent_90d) > 0:
            # Blend: last 90 days + fill up to 500 from historical
            historical_fill = self.catalog_df.sort_values(
                "time"
            ).tail(500 - len(recent_90d))
            recent = pd.concat(
                [historical_fill, recent_90d]
            ).drop_duplicates().sort_values("time")

        # Generate predictions
        context = self.features_df.tail(CONTEXT_ROWS)

        try:
            preds = generate_predictions(
                model=self.model,
                current_features=context,
                recent_events=recent if len(recent) > 0 else None,
                reference_date=datetime.now(timezone.utc).replace(tzinfo=None),
                min_threshold=4.5,
                verbose=True,
            )
            save_predictions(preds)

        except Exception as e:
            print(f"\n[LIVE] Prediction error: {e}")
            traceback.print_exc()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Myanmar Earthquake Prediction - Live Dashboard"
    )
    p.add_argument(
        "--interval", type=int, default=300,
        help="Poll interval in seconds (default 300 = 5 min)"
    )
    return p.parse_args()


def main():
    args = parse_args()
    dash = LiveDashboard(poll_interval=args.interval)
    dash.run()


if __name__ == "__main__":
    main()
