# =============================================================================
# live_updater.py
# Manages the real-time continuous learning loop.
#
# Once initial training is complete, this module:
#   1. Polls the USGS ATOM feed on a scheduled cycle
#   2. Appends new events to the running catalog
#   3. Recomputes features for the updated catalog
#   4. Triggers an adaptive model update using the new data + replay buffer
#   5. Generates updated predictions and saves them to disk
#
# The update cycle runs indefinitely until manually stopped (Ctrl+C).
# For dissertation evaluation purposes, a simulation mode is also provided
# that replays the 2020-2024 test period year by year, producing the
# longitudinal performance curve needed for Chapter 5 results.
# =============================================================================

import time
import os
import pandas as pd
from datetime import datetime, timezone

from config import RETRAIN_INTERVAL, PROCESSED_DATA_PATH
from feature_engineering import FEATURE_COLUMNS
from data_acquisition import fetch_live_events
from feature_engineering import compute_features, build_labeled_dataset
from prediction_engine import generate_predictions, save_predictions


# ---------------------------------------------------------------------------
# Real-time live updating loop (production mode)
# ---------------------------------------------------------------------------
class LiveUpdater:
    """
    Manages the continuous learning and prediction cycle.

    Parameters
    ----------
    adaptive_model  : Fitted AdaptiveModel instance
    catalog_df      : Current full catalog DataFrame (grows over time)
    features_df     : Current full feature DataFrame (grows over time)
    update_interval : Seconds between update cycles (default 24 hours)
    """

    def __init__(self, adaptive_model,
                 catalog_df: pd.DataFrame,
                 features_df: pd.DataFrame,
                 update_interval: int = RETRAIN_INTERVAL):

        self.model           = adaptive_model
        self.catalog_df      = catalog_df.copy()
        self.features_df     = features_df.copy()
        self.update_interval = update_interval
        self.last_event_time = None
        self.cycle_count     = 0

        # Set last_event_time from most recent event in catalog
        if len(catalog_df) > 0:
            self.last_event_time = pd.to_datetime(
                catalog_df["time"]
            ).max().to_pydatetime()

    def run(self):
        """
        Start the continuous update loop.
        Runs until KeyboardInterrupt (Ctrl+C).
        """
        print("\n[LIVE] Starting real-time update loop.")
        print(f"[LIVE] Update interval: every "
              f"{self.update_interval // 3600} hours")
        print("[LIVE] Press Ctrl+C to stop.\n")

        try:
            while True:
                self._run_cycle()
                print(f"\n[LIVE] Next update in "
                      f"{self.update_interval // 3600} hours. "
                      f"Sleeping...\n")
                time.sleep(self.update_interval)

        except KeyboardInterrupt:
            print("\n[LIVE] Update loop stopped by user.")
            self.model.save()
            print("[LIVE] Model saved.")

    def _run_cycle(self):
        """Execute one update cycle."""
        self.cycle_count += 1
        now = datetime.now(timezone.utc).replace(tzinfo=None)

        print(f"\n[LIVE] --- Cycle #{self.cycle_count} "
              f"({now.strftime('%Y-%m-%d %H:%M UTC')}) ---")

        # Step 1: Fetch new events from USGS ATOM feed
        new_events = fetch_live_events(
            last_seen_time=self.last_event_time
        )

        if new_events.empty:
            print("[LIVE] No new events. Generating predictions from "
                  "current model...")
            self._generate_and_save_predictions()
            return

        # Step 2: Append new events to catalog
        self.catalog_df = pd.concat(
            [self.catalog_df, new_events], ignore_index=True
        ).drop_duplicates(subset=["id"]).sort_values("time").reset_index(drop=True)

        # Update last seen time
        self.last_event_time = pd.to_datetime(
            self.catalog_df["time"]
        ).max().to_pydatetime()

        # Save updated catalog
        os.makedirs("data", exist_ok=True)
        self.catalog_df.to_csv("data/live_catalog.csv", index=False)

        # Step 3: Recompute features for the updated catalog
        print("[LIVE] Recomputing features...")
        updated_features = compute_features(self.catalog_df)
        updated_features = build_labeled_dataset(updated_features)

        # Extract only the genuinely new feature rows
        existing_times = set(
            pd.to_datetime(self.features_df["time"]).astype(str)
        )
        new_mask = ~pd.to_datetime(
            updated_features["time"]
        ).astype(str).isin(existing_times)
        new_feature_rows = updated_features[new_mask]

        if len(new_feature_rows) == 0:
            print("[LIVE] No new feature rows. Skipping model update.")
            self._generate_and_save_predictions()
            return

        # Append to full feature DataFrame
        self.features_df = pd.concat(
            [self.features_df, new_feature_rows], ignore_index=True
        ).reset_index(drop=True)

        # Step 4: Trigger adaptive model update
        print(f"[LIVE] Triggering model update with "
              f"{len(new_feature_rows)} new feature rows...")
        self.model.update(new_feature_rows, verbose=True)

        # Save updated model
        self.model.save()

        # Step 5: Generate and save predictions
        self._generate_and_save_predictions()

    def _generate_and_save_predictions(self):
        """Generate predictions from the latest feature window."""
        # Use the most recent 50 feature rows as the current window
        recent_features = self.features_df.tail(50)
        recent_events   = self.catalog_df.tail(200)

        try:
            predictions = generate_predictions(
                model=self.model,
                current_features=recent_features,
                recent_events=recent_events,
                reference_date=datetime.now(timezone.utc).replace(tzinfo=None),
                min_threshold=4.0,
                verbose=True,
            )
            save_predictions(predictions)
        except Exception as e:
            print(f"[LIVE] Prediction error: {e}")


# ---------------------------------------------------------------------------
# Pseudo-prospective simulation mode (for dissertation evaluation)
# Replays the 2020-2024 test period year by year, recording performance
# at each stage to produce the longitudinal performance curve for Chapter 5.
# ---------------------------------------------------------------------------
def run_pseudo_prospective_evaluation(adaptive_model,
                                      static_model,
                                      full_features_df: pd.DataFrame,
                                      test_years: list = None) -> dict:
    """
    Simulate the adaptive learning process over the 2020-2024 test period.
    For each year:
      1. Evaluate both models on that year's data
      2. Update the adaptive model with that year's data
      3. Record performance metrics for both models

    This produces the longitudinal performance comparison needed to
    demonstrate whether the adaptive advantage grows over time
    (Dascher-Cousineau et al., 2023).

    Parameters
    ----------
    adaptive_model  : Fitted AdaptiveModel (trained on 1990-2019)
    static_model    : Fitted StaticModel (trained on 1990-2019)
    full_features_df: Full feature DataFrame including 2020-2024
    test_years      : List of years to evaluate (default [2020..2024])

    Returns
    -------
    dict: {year: {"static": metrics, "adaptive": metrics}}
    """

    if test_years is None:
        test_years = [2020, 2021, 2022, 2023, 2024, 2025]

    full_features_df = full_features_df.copy()
    full_features_df["time"] = pd.to_datetime(full_features_df["time"])
    full_features_df["year"] = full_features_df["time"].dt.year

    longitudinal_results = {}

    print("\n[SIM] Starting pseudo-prospective evaluation...")
    print(f"[SIM] Years: {test_years}\n")

    for year in test_years:
        print(f"\n[SIM] ===== Year {year} =====")

        year_mask = full_features_df["year"] == year
        year_data = full_features_df[year_mask].copy()

        if len(year_data) == 0:
            print(f"[SIM] No data for {year}, skipping.")
            continue

        print(f"[SIM] {len(year_data)} events in {year}")

        # Evaluate static model on this year (it never changes)
        static_year_results = static_model.evaluate(
            full_features_df,
            year_mask,
            verbose=True
        )

        # Evaluate adaptive model BEFORE updating with this year's data
        # This is the "fair" evaluation point
        adaptive_year_results = adaptive_model.evaluate(
            full_features_df,
            year_mask,
            tag=str(year),
            verbose=True,
        )

        longitudinal_results[year] = {
            "static":   static_year_results,
            "adaptive": adaptive_year_results,
        }

        # Update adaptive model with this year's data
        year_data_features = year_data.drop(columns=["year"], errors="ignore")
        adaptive_model.update(year_data_features, verbose=True)

        print(f"[SIM] Year {year} complete.")

    print("\n[SIM] Pseudo-prospective evaluation complete.")
    return longitudinal_results


def print_longitudinal_summary(results: dict):
    """Print a clean year-by-year comparison for the thesis results table."""

    print("\n" + "=" * 75)
    print("  LONGITUDINAL PERFORMANCE COMPARISON (Pseudo-Prospective)")
    print("  Static vs Adaptive Model - F1 Score by Year")
    print("=" * 75)

    # Show one representative label as a summary metric
    # (M45_30d is typically the most informative: M>=4.5, 30-day window)
    sample_col = "label_M45_30d"

    print(f"\n  Label: {sample_col}")
    print(f"  {'Year':<8} {'Static F1':<14} {'Adaptive F1':<14} {'Difference'}")
    print("  " + "-" * 50)

    for year in sorted(results.keys()):
        s = results[year]["static"].get(sample_col, {})
        a = results[year]["adaptive"].get(sample_col, {})

        s_f1 = s.get("f1", float("nan"))
        a_f1 = a.get("f1", float("nan"))

        if isinstance(s_f1, float) and isinstance(a_f1, float):
            diff = round(a_f1 - s_f1, 4)
            diff_str = f"+{diff}" if diff >= 0 else str(diff)
        else:
            diff_str = "N/A"

        print(f"  {year:<8} {str(s_f1):<14} {str(a_f1):<14} {diff_str}")

    print("=" * 75 + "\n")
