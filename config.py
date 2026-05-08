# =============================================================================
# config.py
# Central configuration file for the Myanmar Earthquake Prediction System.
# All constants, API endpoints, geographic bounds, and model parameters
# are defined here so they can be changed in one place.
# =============================================================================

# ---------------------------------------------------------------------------
# Geographic bounding box for Myanmar and surrounding tectonic region.
# Covers: Myanmar, parts of Yunnan (China), northern Thailand,
#         Andaman Sea subduction zone, and Indo-Burman fold-thrust belt.
# ---------------------------------------------------------------------------
GEO_BOUNDS = {
    "min_latitude":  10.0,
    "max_latitude":  30.0,
    "min_longitude": 90.0,
    "max_longitude": 105.0,
}

# ---------------------------------------------------------------------------
# Data collection settings
# ---------------------------------------------------------------------------
MIN_MAGNITUDE       = 2.0       # Minimum magnitude to fetch (background seismicity)
HISTORICAL_START    = "1990-01-01"
HISTORICAL_END      = "2019-12-31"
TEST_START          = "2020-01-01"
TEST_END            = "2025-12-31"

# USGS API endpoint (FDSN-compliant)
USGS_API_URL        = "https://earthquake.usgs.gov/fdsnws/event/1/query"

# USGS real-time ATOM feed (updated every minute, M2.5+)
USGS_ATOM_FEED      = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_day.atom"

# Local file paths
RAW_DATA_PATH       = "data/raw_catalog.csv"
FULL_CATALOG_PATH   = "data/full_catalog_1990_2025.csv"
PROCESSED_DATA_PATH = "data/processed_features.csv"
MODEL_STATIC_PATH   = "models/static_model.pkl"
MODEL_ADAPTIVE_PATH = "models/adaptive_model.pkl"
PREDICTIONS_PATH    = "predictions/latest_predictions.json"
LIVE_LOG_PATH       = "predictions/live_log.jsonl"

# ---------------------------------------------------------------------------
# Feature engineering settings
# ---------------------------------------------------------------------------
ROLLING_WINDOW_N    = 50        # Number of past events for rolling features

# ---------------------------------------------------------------------------
# Prediction experiment settings
# Mirrors Mukherjee et al. (2025) for direct performance comparison
# ---------------------------------------------------------------------------
MAGNITUDE_THRESHOLDS = [4.0, 4.5, 5.0, 5.5]
PREDICTION_WINDOWS   = [7, 10, 15, 30]     # Days ahead to predict

# ---------------------------------------------------------------------------
# Adaptive learning settings
# ---------------------------------------------------------------------------
REPLAY_BUFFER_RATIO  = 0.20     # 20% of historical data kept in replay buffer
RETRAIN_INTERVAL     = 86400    # Seconds between retraining cycles (24 hours)

# ---------------------------------------------------------------------------
# CTGAN augmentation settings
# ---------------------------------------------------------------------------
CTGAN_EPOCHS         = 50
CTGAN_TARGET_SAMPLES = 500      # Synthetic samples per minority magnitude class
MIN_SAMPLES_FOR_AUG  = 100      # Augment only if class has fewer than this

# ---------------------------------------------------------------------------
# Spatial zones for location predictions.
# Each zone includes bounding box, centre coordinates, radius, fault and
# plain-text description for human-readable prediction output.
# Radii are approximate (half diagonal of bounding box, 1deg lat ~ 111km).
# ---------------------------------------------------------------------------
LOCATION_ZONES = {
    "Sagaing Fault Zone": {
        "lat": (18.0, 25.0), "lon": (94.5, 96.5),
        "centre_lat": 21.5,  "centre_lon": 95.5,
        "radius_km": 430,
        "fault": "Sagaing Fault",
        "description": "Central Myanmar along the Sagaing Fault corridor (Mandalay to Sagaing region)",
    },
    "Central Myanmar": {
        "lat": (18.5, 23.0), "lon": (94.0, 100.0),
        "centre_lat": 20.75, "centre_lon": 97.0,
        "radius_km": 420,
        "fault": "Sagaing Fault / Central Myanmar Basin",
        "description": "Central Myanmar basin including Mandalay, Naypyidaw and surrounding regions",
    },
    "Northern Myanmar": {
        "lat": (23.0, 28.5), "lon": (95.0, 101.0),
        "centre_lat": 25.75, "centre_lon": 98.0,
        "radius_km": 500,
        "fault": "Northern Sagaing / Naga Thrust",
        "description": "Northern Myanmar including Kachin State and northern Shan State",
    },
    "Indo-Burman Range": {
        "lat": (20.0, 27.0), "lon": (92.0, 95.0),
        "centre_lat": 23.5,  "centre_lon": 93.5,
        "radius_km": 450,
        "fault": "Indo-Burman Fold-Thrust Belt",
        "description": "Western Myanmar including Chin State, Rakhine State and fold-thrust belt",
    },
    "Southern Myanmar": {
        "lat": (14.0, 18.5), "lon": (97.0, 102.0),
        "centre_lat": 16.25, "centre_lon": 99.5,
        "radius_km": 390,
        "fault": "Southern Sagaing / Andaman Spreading",
        "description": "Southern Myanmar including Tanintharyi Region and northern Andaman coast",
    },
    "Andaman Sea Region": {
        "lat": (10.0, 16.0), "lon": (92.0, 99.0),
        "centre_lat": 13.0,  "centre_lon": 95.5,
        "radius_km": 530,
        "fault": "Andaman Spreading Centre / Sunda Subduction",
        "description": "Andaman Sea and offshore regions including the Andaman spreading centre",
    },
    "Yunnan Border Region": {
        "lat": (23.0, 27.0), "lon": (99.0, 105.0),
        "centre_lat": 25.0,  "centre_lon": 102.0,
        "radius_km": 410,
        "fault": "Xianshuihe / Red River Fault System",
        "description": "Eastern Myanmar and Yunnan Province border including Red River Fault",
    },
    "Northern Thailand": {
        "lat": (15.0, 20.0), "lon": (98.0, 105.0),
        "centre_lat": 17.5,  "centre_lon": 101.5,
        "radius_km": 430,
        "fault": "Mae Chan / Phrae Fault Zone",
        "description": "Northern Thailand including Chiang Rai, Chiang Mai and Mae Chan fault",
    },
}
