# =============================================================================
# data_augmentation.py
# CTGAN-based synthetic data augmentation for rare high-magnitude seismic
# events, following the methodology of Joshi et al. (2025).
#
# Class imbalance is a major challenge in earthquake prediction: large
# magnitude events are intrinsically rare, meaning classifiers trained on
# raw catalog data see very few positive class examples for thresholds
# such as M>=5.0 or M>=5.5.
#
# CTGAN (Conditional Tabular GAN) learns the distributional properties of
# the minority class and generates synthetic samples that preserve feature
# correlations, helping the classifier learn to recognise rare events.
# =============================================================================

import os
import numpy as np
import pandas as pd

from config import (
    CTGAN_EPOCHS,
    CTGAN_TARGET_SAMPLES,
    MIN_SAMPLES_FOR_AUG,
    MAGNITUDE_THRESHOLDS,
    PREDICTION_WINDOWS,
)
from feature_engineering import FEATURE_COLUMNS

# CTGAN is provided by the 'sdv' (Synthetic Data Vault) library.
# Install with: pip install sdv
try:
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata
    CTGAN_AVAILABLE = True
except ImportError:
    CTGAN_AVAILABLE = False
    print("[AUGMENT] Warning: sdv library not installed. "
          "Run: pip install sdv")
    print("[AUGMENT] Augmentation will be skipped.")


def augment_minority_class(X_train: pd.DataFrame,
                            y_train: pd.Series,
                            label_col: str,
                            verbose: bool = True) -> tuple:
    """
    Apply CTGAN augmentation to the minority class (label=1) in a
    training dataset for a specific label column.

    Parameters
    ----------
    X_train   : Feature DataFrame
    y_train   : Binary label Series
    label_col : Name of the label column (for logging)
    verbose   : Print progress information

    Returns
    -------
    (X_augmented, y_augmented) tuple with synthetic samples added
    """

    n_positive = y_train.sum()
    n_negative = len(y_train) - n_positive

    if verbose:
        print(f"[AUGMENT] {label_col}: {n_positive} positive / "
              f"{n_negative} negative before augmentation")

    # Do not augment if positive class already has enough samples
    if n_positive >= MIN_SAMPLES_FOR_AUG:
        if verbose:
            print(f"[AUGMENT] Sufficient samples, skipping augmentation.")
        return X_train, y_train

    # Do not augment if CTGAN is not available
    if not CTGAN_AVAILABLE:
        if verbose:
            print("[AUGMENT] CTGAN unavailable, returning original data.")
        return X_train, y_train

    # Do not augment if there are fewer than 5 positive samples
    # (CTGAN cannot learn from tiny datasets)
    if n_positive < 5:
        if verbose:
            print(f"[AUGMENT] Too few positive samples ({n_positive}) "
                  f"for CTGAN training. Skipping.")
        return X_train, y_train

    # Extract positive class samples
    positive_mask = y_train == 1
    X_positive = X_train[positive_mask].copy()
    X_positive["__label__"] = 1

    # Set up CTGAN metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(X_positive)

    synthesizer = CTGANSynthesizer(
        metadata,
        epochs=CTGAN_EPOCHS,
        verbose=verbose,
    )

    if verbose:
        print(f"[AUGMENT] Training CTGAN on {len(X_positive)} positive samples "
              f"for {CTGAN_EPOCHS} epochs...")

    synthesizer.fit(X_positive)

    # Generate synthetic samples
    n_to_generate = min(
        CTGAN_TARGET_SAMPLES - int(n_positive),
        CTGAN_TARGET_SAMPLES,
    )
    n_to_generate = max(n_to_generate, 0)

    if n_to_generate == 0:
        return X_train, y_train

    synthetic = synthesizer.sample(num_rows=n_to_generate)
    synthetic = synthetic.drop(columns=["__label__"], errors="ignore")

    # Align columns to feature set
    for col in FEATURE_COLUMNS:
        if col not in synthetic.columns:
            synthetic[col] = 0.0
    synthetic = synthetic[FEATURE_COLUMNS]

    # Clip synthetic values to realistic ranges to prevent out-of-distribution
    for col in FEATURE_COLUMNS:
        if col in X_train.columns:
            col_min = X_train[col].min()
            col_max = X_train[col].max()
            synthetic[col] = synthetic[col].clip(col_min, col_max)

    # Create synthetic labels
    y_synthetic = pd.Series(
        [1] * len(synthetic),
        name=y_train.name
    )

    # Concatenate original and synthetic data
    X_combined = pd.concat([X_train, synthetic], ignore_index=True)
    y_combined = pd.concat([y_train, y_synthetic], ignore_index=True)

    if verbose:
        new_pos = y_combined.sum()
        new_neg = len(y_combined) - new_pos
        print(f"[AUGMENT] After augmentation: {new_pos} positive / "
              f"{new_neg} negative")

    return X_combined, y_combined


def augment_all_labels(features_df: pd.DataFrame,
                        train_mask: pd.Series) -> dict:
    """
    Run CTGAN augmentation for every magnitude threshold and prediction
    window combination and return a dictionary of augmented training sets.

    Parameters
    ----------
    features_df : Full feature DataFrame with label columns
    train_mask  : Boolean Series indicating which rows are training data

    Returns
    -------
    dict mapping label_col -> (X_augmented, y_augmented)
    """

    X_base = features_df.loc[train_mask, FEATURE_COLUMNS]
    augmented_sets = {}

    for threshold in MAGNITUDE_THRESHOLDS:
        for window in PREDICTION_WINDOWS:
            col = f"label_M{int(threshold * 10)}_{window}d"
            if col not in features_df.columns:
                continue

            y_base = features_df.loc[train_mask, col]
            X_aug, y_aug = augment_minority_class(X_base, y_base, col)
            augmented_sets[col] = (X_aug, y_aug)

    return augmented_sets
