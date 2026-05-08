# =============================================================================
# live_demo.py
# Real-time live updating earthquake prediction dashboard.
#
# This script runs independently of main.py. It:
#   1. Loads the trained adaptive model from disk
#   2. Displays the current predictions based on the latest catalog data
#   3. Polls the USGS ATOM feed every POLL_INTERVAL seconds
#   4. When new events are detected in the Myanmar region:
#      a. Displays the new events clearly
#      b. Computes new features for the updated catalog
#      c. Triggers an adaptive model update (continual learning step)
#      d. Regenerates and displays updated predictions
#   5. Logs all events and updates to a timestamped log file
#
# Usage:
#   python live_demo.py                    # Poll every 5 minutes (default)
#   python live_demo.py --interval 60      # Poll every 60 seconds (fast demo)
#   python live_demo.py --interval 300     # Poll every 5 minutes
#
# Press Ctrl+C to stop cleanly. The updated model is saved on exit.
# =============================================================================

import argparse
import os
import sys
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from config import (
    PROCESSED_DATA_PATH,
    MODEL_ADAPTIVE_PATH,
    FULL_CATALOG_PATH,
    PREDICTIONS_PATH,
    LIVE_LOG_PATH,
    GEO_BOUNDS,
    MIN_MAGNITUDE,
)
from data_acquisition    import fetch_live_events
from feature_engineering import compute_features, build_labeled_dataset, FEATURE_COLUMNS
from models              import AdaptiveModel
from prediction_engine   import (
    generate_predictions,
    save_predictions,
    print_predictions,
    identify_zone,
    get_zone_activity,
)



# How many recent feature rows to use for prediction generation
PREDICTION_CONTEXT_ROWS = 50


# ---------------------------------------------------------------------------
# Helper: clear the terminal screen
# ---------------------------------------------------------------------------
def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


# ---------------------------------------------------------------------------
# Helper: print a separator line
# ---------------------------------------------------------------------------
def sep(char="-", width=70):
    print("  " + char * width)


# ---------------------------------------------------------------------------
# Helper: print the live dashboard header
# ---------------------------------------------------------------------------
def print_header(cycle: int, last_update: str, next_check_in: int,
                 total_events: int, new_events_this_cycle: int):
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    print("\n" + "=" * 72)
    print("  MYANMAR EARTHQUAKE PREDICTION SYSTEM  |  LIVE MODE")
    print(f"  Current Time : {now}")
    print(f"  Last Update  : {last_update}")
    print(f"  Cycle        : #{cycle}")
    print(f"  Catalog Size : {total_events} events")
    if new_events_this_cycle > 0:
        print(f"  NEW EVENTS   : {new_events_this_cycle} detected this cycle")
    print(f"  Next Check   : in {next_check_in}s")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Helper: display newly detected events
# ---------------------------------------------------------------------------
def print_new_events(new_events_df: pd.DataFrame):
    if new_events_df is None or len(new_events_df) == 0:
        return

    print("\n  *** NEW EARTHQUAKE EVENTS DETECTED ***")
    sep("*")

    for _, row in new_events_df.sort_values(
        "magnitude", ascending=False
    ).iterrows():
        t    = pd.to_datetime(row["time"])
        mag  = row["magnitude"]
        lat  = row["latitude"]
        lon  = row["longitude"]
        zone = identify_zone(lat, lon)
        loc  = row.get("place", f"Lat {lat:.2f}, Lon {lon:.2f}")

        if mag >= 5.5:
            prefix = "  [!!!] "
        elif mag >= 5.0:
            prefix = "  [>> ] "
        elif mag >= 4.5:
            prefix = "  [>  ] "
        else:
            prefix = "  [   ] "

        print(f"{prefix}M{mag:.1f}  |  {t.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"         Zone: {zone}")
        print(f"         Location: {loc}")
        print(f"         Coordinates: {lat:.3f}N, {lon:.3f}E")
        print()

    sep("*")


# ---------------------------------------------------------------------------
# Helper: display model update status
# ---------------------------------------------------------------------------
def print_update_status(n_new: int, update_number: int):
    print(f"\n  [MODEL UPDATE #{update_number}]")
    print(f"  Incorporating {n_new} new event(s) into adaptive model...")
    print(f"  Replay buffer: retaining historical patterns (Van de Ven et al., 2024)")
    print(f"  Retraining classifiers on combined new + replay data...")


# ---------------------------------------------------------------------------
# Helper: append to log file
# ---------------------------------------------------------------------------
def log_event(entry: dict):
    os.makedirs(os.path.dirname(LIVE_LOG_PATH), exist_ok=True)
    with open(LIVE_LOG_PATH, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


# ---------------------------------------------------------------------------
# Helper: compute features for a catalog and return labeled df
# Does not recompute from scratch - only adds rows for new events
# ---------------------------------------------------------------------------
def update_features(catalog_df: pd.DataFrame,
                    existing_features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute features on the full updated catalog and return
    only the rows that are new (not already in existing_features_df).
    """

    # Recompute on full catalog to get correct rolling window for new rows
    updated = compute_features(catalog_df)
    updated = build_labeled_dataset(updated)

    # Find genuinely new rows by time
    if existing_features_df is not None and len(existing_features_df) > 0:
        existing_times = set(
            pd.to_datetime(existing_features_df["time"]).dt.strftime(
                "%Y-%m-%dT%H:%M:%S"
            )
        )
        new_mask = ~pd.to_datetime(updated["time"]).dt.strftime(
            "%Y-%m-%dT%H:%M:%S"
        ).isin(existing_times)
        new_rows = updated[new_mask].copy()
    else:
        new_rows = updated.copy()

    return updated, new_rows


# ---------------------------------------------------------------------------
# Main live demo loop
# ---------------------------------------------------------------------------
class LiveDashboard:

    def __init__(self, poll_interval: int = 300):
        self.poll_interval    = poll_interval
        self.cycle_count      = 0
        self.update_count     = 0
        self.last_event_time  = None
        self.last_update_str  = "Never"
        self.new_events_count = 0

        print("\n[LIVE] Loading components...")

        # Load processed features
        if not os.path.exists(PROCESSED_DATA_PATH):
            print("[LIVE] ERROR: No processed features found.")
            print("[LIVE] Run: python main.py --mode train --refresh")
            sys.exit(1)

        self.features_df = pd.read_csv(
            PROCESSED_DATA_PATH, parse_dates=["time"]
        )
        self.features_df["time"] = pd.to_datetime(self.features_df["time"])
        print(f"[LIVE] Features loaded: {len(self.features_df)} rows")

        # Load raw catalog
        catalog_path = FULL_CATALOG_PATH if os.path.exists(
            FULL_CATALOG_PATH
        ) else "data/raw_catalog.csv"

        if not os.path.exists(catalog_path):
            print("[LIVE] ERROR: No catalog file found.")
            print("[LIVE] Run: python main.py --mode train --refresh")
            sys.exit(1)

        self.catalog_df = pd.read_csv(catalog_path, parse_dates=["time"])
        self.catalog_df["time"] = pd.to_datetime(self.catalog_df["time"])
        print(f"[LIVE] Catalog loaded: {len(self.catalog_df)} events")

        # Set last known event time
        self.last_event_time = self.catalog_df["time"].max().to_pydatetime()
        print(f"[LIVE] Catalog runs to: "
              f"{self.last_event_time.strftime('%Y-%m-%d %H:%M UTC')}")

        # Load adaptive model
        if not os.path.exists(MODEL_ADAPTIVE_PATH):
            print("[LIVE] ERROR: No adaptive model found.")
            print("[LIVE] Run: python main.py --mode train")
            sys.exit(1)

        self.model = AdaptiveModel.load()
        print("[LIVE] Adaptive model loaded.")
        print(f"[LIVE] Poll interval: every {poll_interval}s")
        print("[LIVE] Starting in 3 seconds... (Press Ctrl+C to stop)\n")
        time.sleep(3)

    def run(self):
        """Main loop - runs until Ctrl+C."""
        try:
            # Generate and show initial predictions on startup
            self._show_current_state(new_events=None)

            while True:
                # Wait for next poll
                self._countdown(self.poll_interval)

                # Poll for new events
                self.cycle_count += 1
                print(f"\n[LIVE] Cycle #{self.cycle_count}: "
                      f"Polling USGS feed...", flush=True)

                new_events = fetch_live_events(
                    last_seen_time=self.last_event_time
                )

                if new_events.empty:
                    # No new events - just refresh the display
                    self._show_current_state(new_events=None)
                else:
                    # New events detected
                    self.new_events_count = len(new_events)
                    self._process_new_events(new_events)

        except KeyboardInterrupt:
            print("\n\n[LIVE] Stopping... saving updated model.")
            self.model.save()
            print(f"[LIVE] Model saved. Total updates: {self.update_count}")
            print(f"[LIVE] Log saved to: {LIVE_LOG_PATH}")
            print("[LIVE] Goodbye.\n")

    def _countdown(self, seconds: int):
        """Show a countdown timer while waiting for next poll."""
        for remaining in range(seconds, 0, -1):
            print(
                f"\r  [LIVE] Next USGS poll in {remaining:>4}s  "
                f"(Ctrl+C to stop)",
                end="",
                flush=True,
            )
            time.sleep(1)
        print()

    def _process_new_events(self, new_events: pd.DataFrame):
        """
        Handle new events: append to catalog, recompute features,
        update model, regenerate predictions.
        """

        # Update last event time
        self.last_event_time = pd.to_datetime(
            new_events["time"]
        ).max().to_pydatetime()
        self.last_update_str = datetime.utcnow().strftime(
            "%Y-%m-%d %H:%M UTC"
        )

        # Append to catalog
        self.catalog_df = pd.concat(
            [self.catalog_df, new_events], ignore_index=True
        ).drop_duplicates(subset=["id"]).sort_values("time").reset_index(drop=True)

        # Save updated catalog
        self.catalog_df.to_csv("data/live_catalog.csv", index=False)

        # Recompute features
        print("[LIVE] Computing features for updated catalog...", flush=True)
        try:
            updated_features, new_feature_rows = update_features(
                self.catalog_df, self.features_df
            )
            self.features_df = updated_features
        except Exception as e:
            print(f"[LIVE] Feature computation error: {e}")
            self._show_current_state(new_events=new_events)
            return

        # Update model if new feature rows exist
        if len(new_feature_rows) > 0:
            self.update_count += 1
            print_update_status(len(new_feature_rows), self.update_count)
            try:
                self.model.update(new_feature_rows, verbose=False)
                print(f"  Model update #{self.update_count} complete.")

                # Log the update
                log_event({
                    "type":         "model_update",
                    "timestamp":    datetime.utcnow().isoformat(),
                    "update_number": self.update_count,
                    "new_events":   len(new_events),
                    "new_features": len(new_feature_rows),
                })
            except Exception as e:
                print(f"[LIVE] Model update error: {e}")

        # Log new events
        for _, row in new_events.iterrows():
            log_event({
                "type":      "new_event",
                "timestamp": datetime.utcnow().isoformat(),
                "event_time": str(row["time"]),
                "magnitude":  row["magnitude"],
                "latitude":   row["latitude"],
                "longitude":  row["longitude"],
                "zone":       identify_zone(row["latitude"], row["longitude"]),
                "place":      row.get("place", ""),
            })

        # Show full updated dashboard
        self._show_current_state(new_events=new_events)

        # Save updated model periodically (every 5 updates)
        if self.update_count > 0 and self.update_count % 5 == 0:
            self.model.save()
            print("[LIVE] Model checkpoint saved.")

    def _show_current_state(self, new_events: pd.DataFrame = None):
        """Clear screen and redraw the full dashboard."""
        clear_screen()

        # Header
        print_header(
            cycle=self.cycle_count,
            last_update=self.last_update_str,
            next_check_in=self.poll_interval,
            total_events=len(self.catalog_df),
            new_events_this_cycle=len(new_events) if new_events is not None
            and len(new_events) > 0 else 0,
        )

        # Show new events if any
        if new_events is not None and len(new_events) > 0:
            print_new_events(new_events)

        # Get recent catalog events for context
        cutoff = datetime.utcnow() - timedelta(days=90)
        recent_catalog = self.catalog_df[
            pd.to_datetime(self.catalog_df["time"]) >= cutoff
        ]

        # Generate and display predictions
        recent_features = self.features_df.tail(PREDICTION_CONTEXT_ROWS)

        try:
            predictions = generate_predictions(
                model=self.model,
                current_features=recent_features,
                recent_events=recent_catalog if len(recent_catalog) > 0
                else None,
                reference_date=datetime.utcnow(),
                min_threshold=4.5,
                verbose=True,
            )
            save_predictions(predictions)
        except Exception as e:
            print(f"\n[LIVE] Prediction error: {e}")
            import traceback
            traceback.print_exc()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Myanmar Earthquake Prediction System - Live Dashboard"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Poll interval in seconds (default: 300 = 5 minutes)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 72)
    print("  MYANMAR EARTHQUAKE PREDICTION SYSTEM")
    print("  LIVE CONTINUOUS LEARNING MODE")
    print(f"  Poll Interval: {args.interval}s "
          f"({'%.1f' % (args.interval/60)} minutes)")
    print("=" * 72)

    dashboard = LiveDashboard(poll_interval=args.interval)
    dashboard.run()


if __name__ == "__main__":
    main()
