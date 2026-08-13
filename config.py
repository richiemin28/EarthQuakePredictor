# =============================================================================
# config.py
# Central configuration file for the Myanmar Earthquake Prediction System.
# All constants, API endpoints, geographic bounds, and model parameters
# are defined here so they can be changed in one place.
# =============================================================================

COUNTRY_NAME = "Myanmar"

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
HISTORICAL_CHUNK_DAYS = 365     # yearly chunks; denser catalogs (e.g. Japan) use smaller chunks

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
#
# Boundaries are data-driven: for each zone, computed the magnitude^2-weighted
# 15th-85th percentile box of real M>=3.0 events (1990-2025) that fell inside
# the zone's original (much larger, administrative-region-sized) box - i.e.
# the box tightened down to the middle 70% of that zone's actual seismicity in
# each of the lat/lon dimensions. This roughly halves the displayed
# uncertainty radius on average versus the original boxes, at the cost of
# more events now falling outside any named zone into the generic
# "Myanmar Region (General)" catch-all (46% overall coverage vs. 79% before -
# see scratch analysis in the EarthQuakePredictor session history if this
# needs re-deriving). That tradeoff was a deliberate choice: a zone that's
# "precise" only because it's drawn as big as the whole country isn't
# actually precise, and General was already the deprioritised fallback zone.
# ---------------------------------------------------------------------------
LOCATION_ZONES = {
    "Sagaing Fault Zone": {
        "lat": (19.72, 24.63), "lon": (94.62, 96.0),
        "centre_lat": 22.724, "centre_lon": 95.207,
        "radius_km": 282,
        "fault": "Sagaing Fault",
        "description": "Central Myanmar along the Sagaing Fault corridor (Mandalay to Sagaing region)",
    },
    "Central Myanmar": {
        "lat": (19.64, 22.75), "lon": (94.42, 98.91),
        "centre_lat": 21.379, "centre_lon": 95.707,
        "radius_km": 290,
        "fault": "Sagaing Fault / Central Myanmar Basin",
        "description": "Central Myanmar basin including Mandalay, Naypyidaw and surrounding regions",
    },
    "Northern Myanmar": {
        "lat": (24.56, 27.08), "lon": (95.33, 100.19),
        "centre_lat": 25.688, "centre_lon": 97.482,
        "radius_km": 280,
        "fault": "Northern Sagaing / Naga Thrust",
        "description": "Northern Myanmar including Kachin State and northern Shan State",
    },
    "Indo-Burman Range": {
        "lat": (22.04, 24.94), "lon": (92.96, 94.72),
        "centre_lat": 23.478, "centre_lon": 94.169,
        "radius_km": 184,
        "fault": "Indo-Burman Fold-Thrust Belt",
        "description": "Western Myanmar including Chin State, Rakhine State and fold-thrust belt",
    },
    "Southern Myanmar": {
        "lat": (14.84, 18.36), "lon": (97.8, 100.7),
        "centre_lat": 15.607, "centre_lon": 98.955,
        "radius_km": 250,
        "fault": "Southern Sagaing / Andaman Spreading",
        "description": "Southern Myanmar including Tanintharyi Region and northern Andaman coast",
    },
    "Andaman Sea Region": {
        "lat": (10.52, 13.94), "lon": (92.54, 94.82),
        "centre_lat": 12.21, "centre_lon": 93.455,
        "radius_km": 227,
        "fault": "Andaman Spreading Centre / Sunda Subduction",
        "description": "Andaman Sea and offshore regions including the Andaman spreading centre",
    },
    "Yunnan Border Region": {
        "lat": (23.62, 26.32), "lon": (99.82, 102.77),
        "centre_lat": 25.585, "centre_lon": 101.202,
        "radius_km": 211,
        "fault": "Xianshuihe / Red River Fault System",
        "description": "Eastern Myanmar and Yunnan Province border including Red River Fault",
    },
    "Northern Thailand": {
        "lat": (18.91, 19.68), "lon": (99.17, 101.37),
        "centre_lat": 19.554, "centre_lon": 100.126,
        "radius_km": 123,
        "fault": "Mae Chan / Phrae Fault Zone",
        "description": "Northern Thailand including Chiang Rai, Chiang Mai and Mae Chan fault",
    },
    "Chin Hills / Assam Region": {
        "lat": (23.18, 27.31), "lon": (91.11, 93.12),
        "centre_lat": 25.842, "centre_lon": 92.23,
        "radius_km": 251,
        "fault": "Dauki Fault / Shillong Plateau / Chin Hills Thrust",
        "description": "Western edge including Chin Hills, Assam, Meghalaya and Bangladesh border",
    },
    "Bay of Bengal North": {
        "lat": (19.86, 22.83), "lon": (91.75, 92.39),
        "centre_lat": 21.989, "centre_lon": 92.14,
        "radius_km": 169,
        "fault": "Indo-Burman Subduction / Bengal Basin",
        "description": "Northern Bay of Bengal and Bangladesh coastal region",
    },
}

# ---------------------------------------------------------------------------
# Per-zone priority multiplier used when ranking zones for the headline
# "most likely location" prediction (spatial_predictor.rank_zones). Ensures
# well-established, well-studied fault zones outrank the generic
# "Region (General)" catch-all even when General has more raw recent events -
# reflects known seismic hazard significance, not just recent event count.
# ---------------------------------------------------------------------------
ZONE_PRIORITY = {
    "Sagaing Fault Zone":   2.5,
    "Central Myanmar":      2.0,
    "Northern Myanmar":     1.8,
    "Indo-Burman Range":    1.8,
    "Southern Myanmar":     1.6,
    "Andaman Sea Region":   1.5,
    "Yunnan Border Region": 1.4,
    "Northern Thailand":    1.3,
    "Chin Hills / Assam Region": 1.2,
    "Bay of Bengal North":  1.1,
}
