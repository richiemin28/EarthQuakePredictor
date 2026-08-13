# =============================================================================
# config_japan.py
# Country-specific configuration for Japan, mirroring config.py's structure
# and variable names so it can be swapped in via sys.modules (see
# run_pipeline.py) without touching any of the pipeline modules.
#
# Japan's USGS catalog is far denser than Myanmar's (confirmed via the USGS
# count API: ~31,900 events at M3.5+ for 1990-2019 vs Myanmar's ~5,500 at
# M2.0+), so MIN_MAGNITUDE is raised to 4.5 to keep feature computation and
# training tractable while still preserving a real background-seismicity
# signal (~15,500 events, roughly 2.3x Myanmar's dataset). Historical fetch
# chunking is quarterly rather than yearly since a single quarter (e.g. the
# 2011 Tohoku aftershock sequence) can otherwise approach the USGS API's
# 20,000-events-per-request limit.
# =============================================================================

COUNTRY_NAME = "Japan"

# ---------------------------------------------------------------------------
# Geographic bounding box for the Japanese archipelago and its subduction
# zones: Ryukyu Trench (SW) through Hokkaido/Kuril Trench (NE), including
# the Nankai Trough, Japan Trench, and Sagami Trough.
# ---------------------------------------------------------------------------
GEO_BOUNDS = {
    "min_latitude":  24.0,
    "max_latitude":  46.0,
    "min_longitude": 122.0,
    "max_longitude": 146.0,
}

# ---------------------------------------------------------------------------
# Data collection settings
# ---------------------------------------------------------------------------
MIN_MAGNITUDE       = 4.5       # raised from Myanmar's 2.0 - see module docstring
HISTORICAL_START    = "1990-01-01"
HISTORICAL_END      = "2019-12-31"
TEST_START           = "2020-01-01"
TEST_END             = "2025-12-31"
HISTORICAL_CHUNK_DAYS = 90       # quarterly, not yearly - see module docstring

# USGS API endpoint (FDSN-compliant) - same service, different query params
USGS_API_URL        = "https://earthquake.usgs.gov/fdsnws/event/1/query"

# USGS real-time ATOM feed (updated every minute, M2.5+)
USGS_ATOM_FEED      = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_day.atom"

# Local file paths - all japan_-prefixed so they never collide with or
# overwrite the existing Myanmar data/models/predictions.
RAW_DATA_PATH       = "data/japan_raw_catalog.csv"
FULL_CATALOG_PATH   = "data/japan_full_catalog_1990_2025.csv"
PROCESSED_DATA_PATH = "data/japan_processed_features.csv"
MODEL_STATIC_PATH   = "models/japan_static_model.pkl"
MODEL_ADAPTIVE_PATH = "models/japan_adaptive_model.pkl"
PREDICTIONS_PATH    = "predictions/japan_latest_predictions.json"
LIVE_LOG_PATH        = "predictions/japan_live_log.jsonl"

# ---------------------------------------------------------------------------
# Feature engineering settings
# ---------------------------------------------------------------------------
ROLLING_WINDOW_N    = 50        # same window as Myanmar for a fair comparison

# ---------------------------------------------------------------------------
# Prediction experiment settings
#
# Extended beyond Myanmar's [4.0, 4.5, 5.0, 5.5]: Japan's catalog carries
# real signal much higher up the scale (confirmed via the USGS count API,
# 1990-2019: 378 events M6.0+, 106 events M6.5+, 34 events M7.0+ - vs.
# Myanmar's entire M5.5+ count of 182 over the same period). Below M5.5
# most Japan predictions were already saturating near 100%, which is a
# real result (Japan's background seismicity really is that dense at
# moderate magnitudes) but it left the higher-consequence, genuinely
# differentiating range - where a "major" or "great" earthquake is more
# or less likely from one window to the next - completely untracked.
# ---------------------------------------------------------------------------
MAGNITUDE_THRESHOLDS = [4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
PREDICTION_WINDOWS   = [7, 10, 15, 30]

# ---------------------------------------------------------------------------
# Adaptive learning settings
# ---------------------------------------------------------------------------
REPLAY_BUFFER_RATIO  = 0.20
RETRAIN_INTERVAL     = 86400

# ---------------------------------------------------------------------------
# CTGAN augmentation settings
# ---------------------------------------------------------------------------
CTGAN_EPOCHS         = 50
CTGAN_TARGET_SAMPLES = 500
MIN_SAMPLES_FOR_AUG  = 100

# ---------------------------------------------------------------------------
# Spatial zones for location predictions - Japan's principal seismic source
# zones, matched in granularity (10 zones) to the Myanmar configuration.
# Boundaries and fault names reflect established Japanese seismotectonics
# (Nankai Trough, Japan Trench/Tohoku, Sagami Trough, etc.), not invented.
#
# Boundaries are data-driven: for each zone, computed the magnitude^2-weighted
# 15th-85th percentile box of real M>=3.0 events (1990-2025) that fell inside
# the zone's original (much larger) box - i.e. tightened down to the middle
# 70% of that zone's actual seismicity in each of the lat/lon dimensions.
# Halves the displayed uncertainty radius on average (down to under 100km for
# tightly-clustered zones like Kanto Triple Junction and Sagami Trough),
# at the cost of more events falling outside any named zone into the
# generic "Japan Region (General)" catch-all (39% overall coverage vs. 77%
# before). Deliberate tradeoff, same reasoning as config.py's Myanmar zones.
# ---------------------------------------------------------------------------
LOCATION_ZONES = {
    "Nankai Trough": {
        "lat": (31.6, 33.93), "lon": (131.54, 137.98),
        "centre_lat": 32.878, "centre_lon": 134.668,
        "radius_km": 327,
        "fault": "Nankai Trough Megathrust",
        "description": "Subduction zone south of Shikoku and the Kii Peninsula; source of historical M8+ megathrust earthquakes (1944 Tonankai, 1946 Nankai) and Japan's highest-priority near-term hazard",
    },
    "Japan Trench (Tohoku)": {
        "lat": (36.41, 39.72), "lon": (141.53, 143.7),
        "centre_lat": 38.128, "centre_lon": 142.535,
        "radius_km": 207,
        "fault": "Japan Trench",
        "description": "Subduction zone off the Pacific coast of Tohoku; source of the 2011 Mw 9.0 Tohoku earthquake and tsunami",
    },
    "Sagami Trough (Kanto)": {
        "lat": (34.93, 35.84), "lon": (139.36, 140.35),
        "centre_lat": 35.511, "centre_lon": 140.056,
        "radius_km": 67,
        "fault": "Sagami Trough",
        "description": "South of Tokyo and the Kanto Plain; source of the 1923 Great Kanto earthquake",
    },
    "Izu-Bonin Arc": {
        "lat": (27.34, 33.11), "lon": (139.81, 141.77),
        "centre_lat": 30.617, "centre_lon": 140.869,
        "radius_km": 334,
        "fault": "Izu-Bonin Trench",
        "description": "Volcanic island arc and subduction zone south of Tokyo, extending to the Ogasawara Islands",
    },
    "Median Tectonic Line (SW Japan)": {
        "lat": (32.4, 33.72), "lon": (130.61, 132.53),
        "centre_lat": 32.96, "centre_lon": 131.385,
        "radius_km": 116,
        "fault": "Median Tectonic Line",
        "description": "Major inland strike-slip fault system running through Shikoku and Kyushu",
    },
    "Hokkaido / Kuril Trench": {
        "lat": (41.49, 43.13), "lon": (141.83, 144.87),
        "centre_lat": 42.165, "centre_lon": 143.25,
        "radius_km": 155,
        "fault": "Kuril Trench",
        "description": "Northern Japan subduction zone off Hokkaido, extending toward the Kuril Islands",
    },
    "Ryukyu Trench (Okinawa)": {
        "lat": (24.45, 28.82), "lon": (122.65, 129.93),
        "centre_lat": 26.222, "centre_lon": 126.765,
        "radius_km": 436,
        "fault": "Ryukyu Trench",
        "description": "Southwestern subduction zone along the Ryukyu (Okinawa) island chain",
    },
    "Niigata-Kobe Tectonic Zone": {
        "lat": (35.75, 37.52), "lon": (136.88, 139.04),
        "centre_lat": 36.978, "centre_lon": 137.993,
        "radius_km": 138,
        "fault": "Niigata-Kobe Tectonic Zone",
        "description": "Central Honshu seismic belt of concentrated crustal strain, from the Sea of Japan coast to the Kobe area",
    },
    "Itoigawa-Shizuoka Line (Fossa Magna)": {
        "lat": (35.5, 36.81), "lon": (137.67, 138.87),
        "centre_lat": 36.382, "centre_lon": 137.992,
        "radius_km": 90,
        "fault": "Itoigawa-Shizuoka Tectonic Line",
        "description": "Major fault boundary marking the western edge of the Fossa Magna rift in central Honshu",
    },
    "Kanto Triple Junction": {
        "lat": (35.41, 36.13), "lon": (139.68, 140.34),
        "centre_lat": 35.782, "centre_lon": 140.032,
        "radius_km": 50,
        "fault": "Kanto Triple Junction",
        "description": "Complex zone beneath the Tokyo metropolitan area where the Pacific, Philippine Sea, and Okhotsk plates meet",
    },
}

# ---------------------------------------------------------------------------
# Per-zone priority multiplier used when ranking zones for the headline
# "most likely location" prediction (spatial_predictor.rank_zones). Ensures
# well-established, well-studied fault zones outrank the generic
# "Region (General)" catch-all even when General has more raw recent events.
# Ordered by documented seismic hazard significance: Nankai Trough and the
# Japan Trench (Tohoku) carry Japan's highest-consequence megathrust hazard
# (1944/1946 Nankai-Tonankai, 2011 Tohoku); Kanto-area zones get a bonus for
# population exposure beneath Tokyo; the rest follow established hazard
# literature. Previously this table only had Myanmar's zone names (a bug -
# it silently fell back to a 1.0x multiplier for every Japan zone, meaning
# a well-known fault zone had no ranking advantage over the catch-all).
# ---------------------------------------------------------------------------
ZONE_PRIORITY = {
    "Nankai Trough":                        2.5,
    "Japan Trench (Tohoku)":                2.3,
    "Kanto Triple Junction":                1.9,
    "Sagami Trough (Kanto)":                1.8,
    "Hokkaido / Kuril Trench":              1.7,
    "Median Tectonic Line (SW Japan)":      1.6,
    "Niigata-Kobe Tectonic Zone":           1.5,
    "Ryukyu Trench (Okinawa)":              1.4,
    "Izu-Bonin Arc":                        1.3,
    "Itoigawa-Shizuoka Line (Fossa Magna)": 1.2,
}
