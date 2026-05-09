# =============================================================================
# main.py
# Master entry point for the Myanmar Earthquake Prediction System.
#
# Usage modes:
#   python main.py --mode train       Fetch data and train both models
#   python main.py --mode evaluate    Run pseudo-prospective evaluation
#   python main.py --mode live        Start real-time live updating loop
#   python main.py --mode predict     Generate predictions from saved model
#   python main.py --mode full        Run everything: train -> evaluate -> live
#
# KEY FIX: The full 1990-2025 catalog is fetched in ONE SINGLE PASS and
# stored in data/full_catalog_1990_2025.csv. The train/test split is
# applied by year mask AFTER features are computed. This ensures
# processed_features.csv always contains all years 1990-2025.
# =============================================================================

import argparse
import os
import pandas as pd

from config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    MODEL_STATIC_PATH,
    MODEL_ADAPTIVE_PATH,
)

from data_acquisition    import fetch_historical_data
from feature_engineering import compute_features, build_labeled_dataset
from data_augmentation   import augment_all_labels
from models              import StaticModel, AdaptiveModel
from prediction_engine   import (
    generate_predictions,
    save_predictions,
    summarise_results,
)
from live_updater import (
    LiveUpdater,
    run_pseudo_prospective_evaluation,
    print_longitudinal_summary,
)

FULL_CATALOG_PATH = "data/full_catalog_1990_2025.csv"
TEST_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]


# ---------------------------------------------------------------------------
# Step 1-3: Data acquisition and feature computation
# ---------------------------------------------------------------------------
def prepare_data(force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch the complete 1990-2025 catalog in a single pass and compute
    all seismic indicator features. Caches results to disk.
    """

    # Check if valid cache exists (must have data through at least 2022)
    if not force_refresh and os.path.exists(PROCESSED_DATA_PATH):
        print(f"[MAIN] Checking cached features at {PROCESSED_DATA_PATH}...")
        df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["time"])
        df["time"] = pd.to_datetime(df["time"])
        years_present = sorted(df["time"].dt.year.unique().tolist())
        max_year = max(years_present)

        if max_year >= 2022:
            print(f"[MAIN] Cache valid: {years_present[0]}-{max_year}, "
                  f"{len(df)} rows. Loading.")
            return df
        else:
            print(f"[MAIN] Cache only covers to {max_year}. Rebuilding...")

    os.makedirs("data", exist_ok=True)

    # Fetch full raw catalog 1990-2025 in one pass
    if not force_refresh and os.path.exists(FULL_CATALOG_PATH):
        print(f"[MAIN] Loading full catalog from {FULL_CATALOG_PATH}...")
        full_catalog = pd.read_csv(FULL_CATALOG_PATH, parse_dates=["time"])
        full_catalog["time"] = pd.to_datetime(full_catalog["time"])
        print(f"[MAIN] Loaded {len(full_catalog)} events.")
    else:
        print("[MAIN] Step 1: Fetching full catalog 1990-2025 from USGS...")
        print("[MAIN] Downloads ~35 years in yearly chunks. Please wait...\n")
        full_catalog = fetch_historical_data(
            start="1990-01-01",
            end="2025-12-31",
            force_refresh=True,
        )
        full_catalog.to_csv(FULL_CATALOG_PATH, index=False)
        print(f"\n[MAIN] Saved to {FULL_CATALOG_PATH}")

    full_catalog["time"] = pd.to_datetime(full_catalog["time"])

    # Print year-by-year coverage so you can verify completeness
    year_counts = full_catalog.groupby(
        full_catalog["time"].dt.year
    ).size().sort_index()
    print(f"\n[MAIN] Catalog coverage ({len(full_catalog)} total events):")
    for yr, cnt in year_counts.items():
        tag = " <- TRAIN" if yr < 2020 else " <- TEST"
        print(f"  {yr}: {cnt:>5} events{tag}")

    # Compute features on the full catalog
    print("\n[MAIN] Step 2: Computing seismic indicator features...")
    print("[MAIN] Iterating through every event - takes 10-20 minutes...")
    features_df = compute_features(full_catalog)

    print("\n[MAIN] Step 3: Building classification labels...")
    features_df = build_labeled_dataset(features_df)

    # Save and summarise
    features_df.to_csv(PROCESSED_DATA_PATH, index=False)
    features_df["time"] = pd.to_datetime(features_df["time"])
    y_min   = features_df["time"].dt.year.min()
    y_max   = features_df["time"].dt.year.max()
    n_train = (features_df["time"].dt.year < 2020).sum()
    n_test  = (features_df["time"].dt.year >= 2020).sum()

    print(f"\n[MAIN] Saved {len(features_df)} feature rows ({y_min}-{y_max})")
    print(f"[MAIN] Training rows (1990-2019): {n_train}")
    print(f"[MAIN] Test rows     (2020-2025): {n_test}")

    return features_df


# ---------------------------------------------------------------------------
# Step 4: Train static and adaptive models on 1990-2019 data
# ---------------------------------------------------------------------------
def train_models(features_df: pd.DataFrame,
                 use_augmentation: bool = True) -> tuple:

    features_df = features_df.copy()
    features_df["time"] = pd.to_datetime(features_df["time"])

    train_mask = features_df["time"].dt.year < 2020
    test_mask  = ~train_mask

    print(f"\n[MAIN] Training set: {train_mask.sum()} rows (1990-2019)")
    print(f"[MAIN] Test set:     {test_mask.sum()} rows  (2020-2025)")

    if test_mask.sum() == 0:
        print("\n[MAIN] ERROR: No test period data in features file.")
        print("[MAIN] Delete cached files and run:")
        print("[MAIN]   del data\\processed_features.csv")
        print("[MAIN]   del data\\raw_catalog.csv")
        print("[MAIN]   python main.py --mode train --refresh")
        return None, None

    train_df = features_df[train_mask].copy().reset_index(drop=True)

    augmented_sets = None
    if use_augmentation:
        print("\n[MAIN] Step 3: Running CTGAN data augmentation...")
        augmented_sets = augment_all_labels(
            train_df, pd.Series([True] * len(train_df))
        )

    print("\n[MAIN] Step 4a: Training Static Baseline Model (XGBoost)...")
    static_model = StaticModel(classifier="xgboost")
    static_model.train(train_df, augmented_sets=augmented_sets)
    static_model.save()

    print("\n[MAIN] Step 4b: Training Adaptive Model (XGBoost)...")
    adaptive_model = AdaptiveModel(classifier="xgboost")
    adaptive_model.initial_train(train_df, augmented_sets=augmented_sets)
    adaptive_model.save()

    return static_model, adaptive_model


# ---------------------------------------------------------------------------
# Step 5: Pseudo-prospective evaluation 2020-2025
# ---------------------------------------------------------------------------
def run_evaluation(static_model, adaptive_model,
                   features_df: pd.DataFrame):

    features_df = features_df.copy()
    features_df["time"] = pd.to_datetime(features_df["time"])
    years_in_data = features_df["time"].dt.year.unique().tolist()
    test_years_available = [y for y in TEST_YEARS if y in years_in_data]

    if not test_years_available:
        print("\n[MAIN] ERROR: No test years found in features file.")
        print("[MAIN] Delete data/processed_features.csv and rerun with --refresh")
        return {}

    print(f"\n[MAIN] Test years available: {test_years_available}")
    print("[MAIN] Step 5: Running pseudo-prospective evaluation (2020-2025)...")

    longitudinal_results = run_pseudo_prospective_evaluation(
        adaptive_model=adaptive_model,
        static_model=static_model,
        full_features_df=features_df,
        test_years=test_years_available,
    )

    print_longitudinal_summary(longitudinal_results)

    # Overall combined evaluation
    test_mask = features_df["time"].dt.year >= 2020
    print(f"\n[MAIN] Combined test evaluation: {test_mask.sum()} events (2020-2025)")
    static_results   = static_model.evaluate(features_df, test_mask)
    adaptive_results = adaptive_model.evaluate(
        features_df, test_mask, tag="final"
    )
    summarise_results(static_results, adaptive_results)

    return longitudinal_results


# ---------------------------------------------------------------------------
# Step 6: Generate structured predictions
# ---------------------------------------------------------------------------
def generate_current_predictions(adaptive_model, features_df: pd.DataFrame):
    features_df = features_df.copy()
    features_df["time"] = pd.to_datetime(features_df["time"])
    recent_features = features_df.tail(50)

    # Use last 500 catalog events for spatial clustering context.
    # This ensures location estimates are based on rich seismic history
    # rather than only the most recent ATOM feed events.
    catalog_path = FULL_CATALOG_PATH if os.path.exists(
        FULL_CATALOG_PATH
    ) else RAW_DATA_PATH
    if os.path.exists(catalog_path):
        catalog_df = pd.read_csv(catalog_path, parse_dates=["time"])
        catalog_df["time"] = pd.to_datetime(catalog_df["time"])
        recent_events = catalog_df.sort_values("time").tail(500)
    else:
        recent_events = None

    print("\n[MAIN] Generating current earthquake predictions...")
    predictions = generate_predictions(
        model=adaptive_model,
        current_features=recent_features,
        recent_events=recent_events,
        min_threshold=4.5,
        verbose=True,
    )
    save_predictions(predictions)


# ---------------------------------------------------------------------------
# Step 7: Live USGS feed loop
# ---------------------------------------------------------------------------
def start_live_loop(adaptive_model, features_df, catalog_df):
    print("\n[MAIN] Starting live continuous update loop...")
    updater = LiveUpdater(
        adaptive_model=adaptive_model,
        catalog_df=catalog_df,
        features_df=features_df,
    )
    updater.run()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Myanmar Earthquake Prediction System"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "live", "predict", "full"],
        default="full",
    )
    parser.add_argument("--no-augment", action="store_true",
                        help="Skip CTGAN augmentation")
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-download from USGS")
    return parser.parse_args()


def main():
    args = parse_args()
    use_augmentation = not args.no_augment

    print("\n" + "=" * 65)
    print("  MYANMAR EARTHQUAKE PREDICTION SYSTEM")
    print("  MSc Data Science Dissertation - University of Chester")
    print(f"  Mode: {args.mode.upper()}")
    print("=" * 65 + "\n")

    if args.mode in ("train", "full"):
        features_df = prepare_data(force_refresh=args.refresh)
        result = train_models(features_df, use_augmentation=use_augmentation)
        if result[0] is None:
            return
        static_model, adaptive_model = result

    elif args.mode in ("evaluate", "live", "predict"):
        if not os.path.exists(PROCESSED_DATA_PATH):
            print("[MAIN] No processed features. Run:")
            print("[MAIN]   python main.py --mode train --refresh")
            return

        features_df = pd.read_csv(
            PROCESSED_DATA_PATH, parse_dates=["time"]
        )
        features_df["time"] = pd.to_datetime(features_df["time"])
        years = sorted(features_df["time"].dt.year.unique().tolist())
        print(f"[MAIN] Loaded: {len(features_df)} rows, {years[0]}-{years[-1]}")

        if not any(y >= 2020 for y in years):
            print(f"\n[MAIN] ERROR: Cache only covers up to {max(years)}.")
            print("[MAIN] You have the old incomplete cache file.")
            print("[MAIN] In your terminal run these commands then retry:")
            print("[MAIN]")
            print("[MAIN]   del data\\processed_features.csv")
            print("[MAIN]   del data\\raw_catalog.csv")
            print("[MAIN]   python main.py --mode train --refresh")
            return

        if not os.path.exists(MODEL_STATIC_PATH) or \
           not os.path.exists(MODEL_ADAPTIVE_PATH):
            print("[MAIN] No saved models. Run:")
            print("[MAIN]   python main.py --mode train --refresh")
            return

        static_model   = StaticModel.load()
        adaptive_model = AdaptiveModel.load()
        print("[MAIN] Models loaded.")

    if args.mode in ("evaluate", "full"):
        run_evaluation(static_model, adaptive_model, features_df)

    if args.mode in ("predict", "full"):
        generate_current_predictions(adaptive_model, features_df)

    if args.mode in ("live", "full"):
        catalog_path = FULL_CATALOG_PATH \
            if os.path.exists(FULL_CATALOG_PATH) else RAW_DATA_PATH
        catalog_df = pd.read_csv(catalog_path, parse_dates=["time"]) \
            if os.path.exists(catalog_path) else features_df
        start_live_loop(adaptive_model, features_df, catalog_df)


if __name__ == "__main__":
    main()
