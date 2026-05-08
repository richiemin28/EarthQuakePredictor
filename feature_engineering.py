# =============================================================================
# feature_engineering.py
# Computes 15 seismic indicator features from the raw earthquake catalog
# using a rolling window of the last N events, following the methodology
# of Mukherjee et al. (2025).
#
# Features are divided into:
#   Non-parametric: time interval, mean magnitude, b-value, a-value,
#                   energy release, seismic rate change (z-value),
#                   max magnitude in last 7 days
#   Parametric:     probability of M>=6.0, deviation from G-R law,
#                   std of b-value, magnitude deficit, recurrence time
# =============================================================================

import numpy as np
import pandas as pd
from config import ROLLING_WINDOW_N, MAGNITUDE_THRESHOLDS, PREDICTION_WINDOWS


# ---------------------------------------------------------------------------
# Helper: Gutenberg-Richter b-value and a-value via maximum likelihood
# ---------------------------------------------------------------------------
def _compute_b_value_ml(magnitudes: np.ndarray, m_min: float = 2.0):
    """
    Maximum likelihood estimate of b-value.
    Bender (1983) formula: b = log10(e) / (mean(M) - M_min)
    """
    if len(magnitudes) < 2:
        return np.nan, np.nan
    mean_m = np.mean(magnitudes)
    if mean_m <= m_min:
        return np.nan, np.nan
    b = np.log10(np.e) / (mean_m - m_min)
    a = np.log10(len(magnitudes)) + b * m_min
    return round(b, 4), round(a, 4)


def _compute_b_value_ls(magnitudes: np.ndarray):
    """
    Least-squares estimate of b-value from Gutenberg-Richter relationship.
    Fits log10(N) = a - b*M by linear regression.
    """
    if len(magnitudes) < 5:
        return np.nan, np.nan
    mag_bins = np.arange(np.floor(magnitudes.min()), np.ceil(magnitudes.max()) + 0.5, 0.5)
    counts = np.array([np.sum(magnitudes >= m) for m in mag_bins])
    valid = counts > 0
    if valid.sum() < 2:
        return np.nan, np.nan
    x = mag_bins[valid]
    y = np.log10(counts[valid])
    coeffs = np.polyfit(x, y, 1)
    b = -coeffs[0]
    a =  coeffs[1]
    return round(b, 4), round(a, 4)


# ---------------------------------------------------------------------------
# Helper: Seismic energy release (Richter 1958 formula)
# ---------------------------------------------------------------------------
def _energy_release_rate(magnitudes: np.ndarray, time_days: float) -> float:
    """
    Rate of square root of seismic energy release.
    log10(E) = 11.8 + 1.5 * M  (energy in ergs)
    """
    if time_days <= 0:
        return 0.0
    total = np.sum(np.sqrt(10 ** (11.8 + 1.5 * magnitudes)))
    return round(total / time_days, 4)


# ---------------------------------------------------------------------------
# Helper: Seismic rate change z-value (Habermann 1988)
# ---------------------------------------------------------------------------
def _z_value(mags_15d: np.ndarray, mags_30d: np.ndarray) -> float:
    """
    Z-value measures seismic rate change between two time intervals.
    z = (R1 - R2) / sqrt(S1/n1 + S2/n2)
    """
    n1, n2 = len(mags_15d), len(mags_30d)
    if n1 < 2 or n2 < 2:
        return 0.0
    r1 = n1 / 15.0
    r2 = n2 / 30.0
    s1 = np.std(mags_15d, ddof=1)
    s2 = np.std(mags_30d, ddof=1)
    denom = np.sqrt((s1 / n1) + (s2 / n2))
    if denom == 0:
        return 0.0
    return round((r1 - r2) / denom, 4)


# ---------------------------------------------------------------------------
# Main feature computation function
# ---------------------------------------------------------------------------
def compute_features(df: pd.DataFrame,
                     n: int = ROLLING_WINDOW_N) -> pd.DataFrame:
    """
    Compute 15 seismic indicator features for each event in the catalog
    using a rolling window of the previous n events.

    Parameters
    ----------
    df : Raw catalog DataFrame with columns: time, latitude, longitude,
         depth, magnitude
    n  : Rolling window size (default 50, following Mukherjee et al. 2025)

    Returns
    -------
    DataFrame with feature columns added, first n rows dropped (insufficient
    history to compute features)
    """

    df = df.sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"])

    records = []
    total   = len(df)

    print(f"[FEATURES] Computing features for {total} events (window={n})...")

    for i in range(n, total):
        window = df.iloc[i - n: i]
        current = df.iloc[i]

        mags = window["magnitude"].values
        times = window["time"].values.astype("datetime64[s]").astype(float)

        # --- Time interval (days between first and last event in window) ---
        t_first = window["time"].iloc[0]
        t_last  = window["time"].iloc[-1]
        t_days  = max((t_last - t_first).total_seconds() / 86400.0, 1e-6)

        # --- Mean magnitude ---
        mean_mag = round(np.mean(mags), 4)

        # --- b-value and a-value (maximum likelihood and least squares) ---
        b_ml, a_ml = _compute_b_value_ml(mags)
        b_ls, a_ls = _compute_b_value_ls(mags)

        # --- Energy release rate ---
        energy_rate = _energy_release_rate(mags, t_days)

        # --- Seismic rate change (z-value) ---
        now = current["time"]
        mags_15d = df[
            (df["time"] >= now - pd.Timedelta(days=15)) &
            (df["time"] < now)
        ]["magnitude"].values
        mags_30d = df[
            (df["time"] >= now - pd.Timedelta(days=30)) &
            (df["time"] < now)
        ]["magnitude"].values
        z_val = _z_value(mags_15d, mags_30d)

        # --- Maximum magnitude in last 7 days ---
        mags_7d = df[
            (df["time"] >= now - pd.Timedelta(days=7)) &
            (df["time"] < now)
        ]["magnitude"].values
        max_mag_7d = round(float(np.max(mags_7d)), 4) if len(mags_7d) > 0 else 0.0

        # --- Probability of M>=6.0 (using b-value ML) ---
        prob_m6 = round(np.exp(-3 * b_ml * np.log(10)), 6) if not np.isnan(b_ml) else 0.0

        # --- Deviation from G-R law (eta value) ---
        if not np.isnan(a_ml) and not np.isnan(b_ml):
            predicted = a_ml - b_ml * mags
            actual    = np.log10(np.arange(len(mags), 0, -1))
            eta = round(float(np.sum((actual - predicted) ** 2) / max(len(mags) - 1, 1)), 6)
        else:
            eta = 0.0

        # --- Standard deviation of b-value ---
        std_b = round(float(np.std(mags, ddof=1) * 2.3 * (b_ml ** 2)
                      if not np.isnan(b_ml) else 0.0), 6)

        # --- Magnitude deficit ---
        if not np.isnan(a_ml) and not np.isnan(b_ml) and b_ml != 0:
            m_max_expected = a_ml / b_ml
            m_max_observed = float(np.max(mags))
            mag_deficit = round(m_max_expected - m_max_observed, 4)
        else:
            mag_deficit = 0.0

        # --- Recurrence time for M>=4.5 and M>=5.0 ---
        def recurrence_time(a, b, m0):
            if np.isnan(a) or np.isnan(b) or b == 0:
                return 0.0
            return round(t_days / (10 ** (a - b * m0)), 4)

        t_rec_45 = recurrence_time(a_ml, b_ml, 4.5)
        t_rec_50 = recurrence_time(a_ml, b_ml, 5.0)

        records.append({
            # Raw event info (kept for label construction and tracking)
            "time":         current["time"],
            "latitude":     current["latitude"],
            "longitude":    current["longitude"],
            "depth":        current["depth"],
            "magnitude":    current["magnitude"],
            # Non-parametric features
            "t_days":       round(t_days, 4),
            "mean_mag":     mean_mag,
            "b_ml":         b_ml,
            "a_ml":         a_ml,
            "b_ls":         b_ls,
            "a_ls":         a_ls,
            "energy_rate":  energy_rate,
            "z_value":      z_val,
            "max_mag_7d":   max_mag_7d,
            # Parametric features
            "prob_m6":      prob_m6,
            "eta":          eta,
            "std_b":        std_b,
            "mag_deficit":  mag_deficit,
            "t_rec_45":     t_rec_45,
            "t_rec_50":     t_rec_50,
        })

    features_df = pd.DataFrame(records).reset_index(drop=True)
    features_df = features_df.fillna(0.0)

    print(f"[FEATURES] Computed {len(features_df)} feature rows")
    return features_df


# ---------------------------------------------------------------------------
# Build labeled dataset: add binary classification labels for each
# magnitude threshold and prediction window combination
# ---------------------------------------------------------------------------
def build_labeled_dataset(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds binary label columns to the feature DataFrame.
    Label = 1 if an earthquake >= threshold occurs within the next
    window_days days, 0 otherwise.

    Column naming: label_M{threshold}_{window}d
    e.g. label_M45_7d = 1 if M>=4.5 within next 7 days
    """

    df = features_df.copy()
    df["time"] = pd.to_datetime(df["time"])

    print("[LABELS] Building classification labels...")

    for threshold in MAGNITUDE_THRESHOLDS:
        for window in PREDICTION_WINDOWS:
            col_name = f"label_M{int(threshold*10)}_{window}d"
            labels = []

            for idx, row in df.iterrows():
                t_now   = row["time"]
                t_end   = t_now + pd.Timedelta(days=window)
                future  = df[
                    (df["time"] > t_now) &
                    (df["time"] <= t_end) &
                    (df["magnitude"] >= threshold)
                ]
                labels.append(1 if len(future) > 0 else 0)

            df[col_name] = labels
            pos = sum(labels)
            print(f"  {col_name}: {pos} positive / {len(labels) - pos} negative")

    return df


# ---------------------------------------------------------------------------
# Return feature column names only (excludes metadata and label columns)
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "t_days", "mean_mag", "b_ml", "a_ml", "b_ls", "a_ls",
    "energy_rate", "z_value", "max_mag_7d",
    "prob_m6", "eta", "std_b", "mag_deficit",
    "t_rec_45", "t_rec_50",
]
