# =============================================================================
# models.py
# Implements two model classes:
#
#   StaticModel   - Trained once on historical data, never updated.
#                   Represents the conventional ML earthquake prediction
#                   approach used in virtually all prior literature.
#
#   AdaptiveModel - Begins with identical historical training, then
#                   incrementally retrains using a replay buffer as new
#                   seismic data arrives from the USGS feed.
#                   Addresses catastrophic forgetting via replay-based
#                   continual learning (Van de Ven et al., 2024).
#
# Both classes use Random Forest and XGBoost as base classifiers,
# matching the top-performing architectures identified by
# Mukherjee et al. (2025)
# =============================================================================

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score,
    classification_report,
)
from sklearn.utils import resample
from xgboost import XGBClassifier

from config import (
    MAGNITUDE_THRESHOLDS,
    PREDICTION_WINDOWS,
    REPLAY_BUFFER_RATIO,
    MODEL_STATIC_PATH,
    MODEL_ADAPTIVE_PATH,
)
from feature_engineering import FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# StratifiedKFold needs at least this many examples of *each* class - used
# both by the actual CV split below and by the pre-flight guard that skips
# a label rather than letting GridSearchCV crash on it.
# ---------------------------------------------------------------------------
CV_FOLDS = 5

# ---------------------------------------------------------------------------
# Hyperparameter grids for grid search cross-validation
# (smaller than full grid for MSc-level runtime feasibility)
# ---------------------------------------------------------------------------
RF_PARAM_GRID = {
    "n_estimators":      [100, 200],
    "max_depth":         [6, 10, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf":  [1, 2],
}

XGB_PARAM_GRID = {
    "n_estimators":  [100, 200],
    "max_depth":     [4, 6, 8],
    "learning_rate": [0.05, 0.1],
    "subsample":     [0.7, 1.0],
}


def _evaluate(y_true, y_pred, y_prob=None, label="") -> dict:
    """Compute and return standard classification metrics."""
    metrics = {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["auc"] = round(roc_auc_score(y_true, y_prob), 4)
    else:
        metrics["auc"] = np.nan

    if label:
        print(f"  [{label}] Acc={metrics['accuracy']} | "
              f"P={metrics['precision']} | "
              f"R={metrics['recall']} | "
              f"F1={metrics['f1']} | "
              f"AUC={metrics['auc']}")
    return metrics


def _undersample(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Undersample majority class to balance the training set.
    Mirrors the approach of Mukherjee et al. (2025).
    Only applied when minority class has fewer than half the majority class.
    """
    n_pos = y.sum()
    n_neg = len(y) - n_pos

    # No undersampling needed if already roughly balanced
    if n_pos == 0 or n_neg / max(n_pos, 1) < 2:
        return X, y

    X_pos = X[y == 1]
    y_pos = y[y == 1]
    X_neg = X[y == 0]
    y_neg = y[y == 0]

    # Undersample majority class to 2x the minority class size
    target_neg = min(n_neg, 2 * n_pos)
    X_neg_us, y_neg_us = resample(
        X_neg, y_neg,
        replace=False,
        n_samples=target_neg,
        random_state=42,
    )

    X_balanced = pd.concat([X_pos, X_neg_us])
    y_balanced = pd.concat([y_pos, y_neg_us])
    return X_balanced.reset_index(drop=True), y_balanced.reset_index(drop=True)


# =============================================================================
# Static Model
# =============================================================================
class StaticModel:
    """
    Trained once on historical data (1990-2019) and never updated.
    Serves as the baseline for comparison against the adaptive model.
    One classifier is trained per (magnitude_threshold, prediction_window)
    combination, giving 4 x 4 = 16 classifiers in total.
    """

    def __init__(self, classifier: str = "xgboost"):
        """
        Parameters
        ----------
        classifier : "xgboost" or "random_forest"
        """
        assert classifier in ("xgboost", "random_forest"), \
            "classifier must be 'xgboost' or 'random_forest'"
        self.classifier = classifier
        self.models = {}       # key: label_col -> fitted classifier
        self.results = {}      # key: label_col -> evaluation metrics dict

    def _build_estimator(self):
        if self.classifier == "xgboost":
            return XGBClassifier(
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )
        else:
            return RandomForestClassifier(random_state=42, n_jobs=-1)

    def _param_grid(self):
        return XGB_PARAM_GRID if self.classifier == "xgboost" else RF_PARAM_GRID

    def train(self, features_df: pd.DataFrame,
              augmented_sets: dict = None,
              verbose: bool = True):
        """
        Train classifiers on all 16 (threshold, window) label combinations.

        Parameters
        ----------
        features_df    : Full labeled feature DataFrame
        augmented_sets : Optional dict from data_augmentation.augment_all_labels()
        verbose        : Print training progress
        """

        if verbose:
            print(f"\n[STATIC] Training {self.classifier.upper()} "
                  f"on historical data...")

        for threshold in MAGNITUDE_THRESHOLDS:
            for window in PREDICTION_WINDOWS:
                col = f"label_M{int(threshold * 10)}_{window}d"
                if col not in features_df.columns:
                    continue

                X = features_df[FEATURE_COLUMNS]
                y = features_df[col]

                # Use augmented data if available, else raw data
                if augmented_sets and col in augmented_sets:
                    X_train, y_train = augmented_sets[col]
                else:
                    X_train, y_train = _undersample(X, y)

                if verbose:
                    print(f"\n  Training {col} "
                          f"({y_train.sum()} pos / "
                          f"{len(y_train) - y_train.sum()} neg)")

                # StratifiedKFold needs at least CV_FOLDS examples of *each*
                # class, not just both classes present. Myanmar's M4.0/30d
                # (2 negative) and Japan's M4.0/15d (0 negative) are both
                # instances of the same underlying issue: some regions have
                # background seismicity so continuous that a window is
                # almost always positive, leaving too few (or zero)
                # negative examples. Undersampling and CTGAN augmentation
                # both only ever act on the positive class, so neither
                # helps when the *negative* class is the tiny one - this
                # guard is the actual fix, for both directions of imbalance.
                minority_n = min(int(y_train.sum()), len(y_train) - int(y_train.sum()))
                if minority_n < CV_FOLDS:
                    if verbose:
                        print(f"  {col}: minority class has only {minority_n} "
                              f"example(s) ({int(y_train.sum())} pos / "
                              f"{len(y_train) - int(y_train.sum())} neg), "
                              f"fewer than the {CV_FOLDS} needed for "
                              f"cross-validation - skipping, this "
                              f"threshold/window has no reliable "
                              f"discriminative signal for this region")
                    continue

                cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
                estimator = self._build_estimator()

                grid = GridSearchCV(
                    estimator,
                    self._param_grid(),
                    cv=cv,
                    scoring="f1",
                    n_jobs=-1,
                    verbose=0,
                )
                grid.fit(X_train, y_train)
                self.models[col] = grid.best_estimator_

                if verbose:
                    print(f"  Best params: {grid.best_params_}")

        print(f"\n[STATIC] Training complete. "
              f"{len(self.models)} classifiers trained.")

    def evaluate(self, features_df: pd.DataFrame,
                 test_mask: pd.Series,
                 verbose: bool = True) -> dict:
        """
        Evaluate all trained classifiers on held-out test data.

        Parameters
        ----------
        features_df : Full feature DataFrame
        test_mask   : Boolean Series selecting test rows

        Returns
        -------
        dict of dicts: {label_col: {metric: value}}
        """

        X_test = features_df.loc[test_mask, FEATURE_COLUMNS]

        if verbose:
            print(f"\n[STATIC] Evaluating on {test_mask.sum()} test events...")

        for col, model in self.models.items():
            if col not in features_df.columns:
                continue

            y_test = features_df.loc[test_mask, col]

            if len(y_test) == 0 or y_test.sum() == 0:
                continue

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            self.results[col] = _evaluate(
                y_test, y_pred, y_prob,
                label=f"STATIC {col}" if verbose else ""
            )

        return self.results

    def predict(self, X: pd.DataFrame, threshold: float,
                window: int) -> np.ndarray:
        """Return predictions for a specific threshold and window."""
        col = f"label_M{int(threshold * 10)}_{window}d"
        if col not in self.models:
            raise ValueError(f"No model trained for {col}")
        return self.models[col].predict(X[FEATURE_COLUMNS])

    def save(self, path: str = MODEL_STATIC_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[STATIC] Saved to {path}")

    @classmethod
    def load(cls, path: str = MODEL_STATIC_PATH):
        with open(path, "rb") as f:
            return pickle.load(f)


# =============================================================================
# Adaptive Model
# =============================================================================
class AdaptiveModel:
    """
    Begins with the same historical training as the StaticModel but is
    subsequently updated incrementally as new seismic data arrives.

    Catastrophic forgetting is mitigated through replay-based continual
    learning: each retraining cycle combines the new data batch with a
    randomly sampled subset of the historical training data (Van de Ven
    et al., 2024). The proportion of historical data retained in the
    replay buffer is controlled by REPLAY_BUFFER_RATIO (default 0.20).
    """

    def __init__(self, classifier: str = "xgboost",
                 replay_ratio: float = REPLAY_BUFFER_RATIO):
        assert classifier in ("xgboost", "random_forest")
        self.classifier   = classifier
        self.replay_ratio = replay_ratio
        self.models       = {}
        self.results      = {}
        self.replay_buffer = {}    # key: label_col -> (X_replay, y_replay)
        self.update_count  = 0

    def _build_estimator(self):
        if self.classifier == "xgboost":
            return XGBClassifier(
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )
        return RandomForestClassifier(random_state=42, n_jobs=-1)

    def _param_grid(self):
        return XGB_PARAM_GRID if self.classifier == "xgboost" else RF_PARAM_GRID

    def initial_train(self, features_df: pd.DataFrame,
                      augmented_sets: dict = None,
                      verbose: bool = True):
        """
        Initial training on the historical dataset (identical to StaticModel).
        Also initialises the replay buffer from the training data.
        """

        if verbose:
            print(f"\n[ADAPTIVE] Initial training on historical data...")

        for threshold in MAGNITUDE_THRESHOLDS:
            for window in PREDICTION_WINDOWS:
                col = f"label_M{int(threshold * 10)}_{window}d"
                if col not in features_df.columns:
                    continue

                X = features_df[FEATURE_COLUMNS]
                y = features_df[col]

                if augmented_sets and col in augmented_sets:
                    X_train, y_train = augmented_sets[col]
                else:
                    X_train, y_train = _undersample(X, y)

                if verbose:
                    print(f"  Training {col} "
                          f"({y_train.sum()} pos / "
                          f"{len(y_train) - y_train.sum()} neg)")

                # See the matching guard in StaticModel.train(): StratifiedKFold
                # needs CV_FOLDS examples of *each* class, not just both
                # classes present - undersampling/CTGAN only ever act on the
                # positive class, so neither helps when negative is the tiny
                # one. Skip it - update() already skips any column missing
                # from self.models, so no replay buffer is needed for it either.
                minority_n = min(int(y_train.sum()), len(y_train) - int(y_train.sum()))
                if minority_n < CV_FOLDS:
                    if verbose:
                        print(f"  {col}: minority class has only {minority_n} "
                              f"example(s) ({int(y_train.sum())} pos / "
                              f"{len(y_train) - int(y_train.sum())} neg), "
                              f"fewer than the {CV_FOLDS} needed for "
                              f"cross-validation - skipping, this "
                              f"threshold/window has no reliable "
                              f"discriminative signal for this region")
                    continue

                cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
                estimator = self._build_estimator()
                grid = GridSearchCV(
                    estimator, self._param_grid(),
                    cv=cv, scoring="f1",
                    n_jobs=-1, verbose=0,
                )
                grid.fit(X_train, y_train)
                self.models[col] = grid.best_estimator_

                # Initialise replay buffer: sample REPLAY_BUFFER_RATIO
                # of training data to retain for future updates
                n_replay = max(int(len(X_train) * self.replay_ratio), 10)
                idx = np.random.choice(len(X_train), size=n_replay, replace=False)
                self.replay_buffer[col] = (
                    X_train.iloc[idx].reset_index(drop=True),
                    y_train.iloc[idx].reset_index(drop=True),
                )

        print(f"\n[ADAPTIVE] Initial training complete. "
              f"Replay buffer initialised.")

    def update(self, new_features_df: pd.DataFrame,
               verbose: bool = True):
        """
        Incremental update: retrain all classifiers on new data combined
        with the replay buffer. This is the core continual learning step.

        Parameters
        ----------
        new_features_df : Feature DataFrame containing only the new batch
                          of events (e.g., one year of data)
        """

        self.update_count += 1

        if verbose:
            print(f"\n[ADAPTIVE] Update #{self.update_count} "
                  f"with {len(new_features_df)} new events...")

        for threshold in MAGNITUDE_THRESHOLDS:
            for window in PREDICTION_WINDOWS:
                col = f"label_M{int(threshold * 10)}_{window}d"
                if col not in new_features_df.columns:
                    continue
                if col not in self.models:
                    continue

                X_new = new_features_df[FEATURE_COLUMNS]
                y_new = new_features_df[col]

                # Combine new data with replay buffer
                if col in self.replay_buffer:
                    X_replay, y_replay = self.replay_buffer[col]
                    X_combined = pd.concat(
                        [X_replay, X_new], ignore_index=True
                    )
                    y_combined = pd.concat(
                        [y_replay, y_new], ignore_index=True
                    )
                else:
                    X_combined, y_combined = X_new, y_new

                X_combined, y_combined = _undersample(X_combined, y_combined)

                if len(y_combined) < 10 or y_combined.sum() == 0:
                    if verbose:
                        print(f"  {col}: insufficient data, skipping update")
                    continue

                # XGBoost requires both classes present in training data.
                # Skip update if only one class exists in the combined batch.
                # This is expected for M4.0 labels in Myanmar where the
                # negative class is extremely rare (Mukherjee et al., 2025).
                if y_combined.nunique() < 2:
                    if verbose:
                        print(f"  {col}: only one class in batch "
                              f"(all positive), skipping update")
                    continue

                if verbose:
                    print(f"  Updating {col}: "
                          f"{len(X_combined)} samples "
                          f"({int(y_combined.sum())} pos / "
                          f"{int((y_combined==0).sum())} neg)")

                # Retrain with same architecture, no hyperparameter re-search
                # (this keeps update cycles fast for real-time operation)
                self.models[col].fit(X_combined, y_combined)

                # Update replay buffer: merge old buffer with new data
                # and resample to maintain fixed buffer size
                X_new_replay = pd.concat([X_replay, X_new], ignore_index=True) \
                    if col in self.replay_buffer else X_new
                y_new_replay = pd.concat([y_replay, y_new], ignore_index=True) \
                    if col in self.replay_buffer else y_new

                n_keep = max(int(len(X_new_replay) * self.replay_ratio), 10)
                idx = np.random.choice(
                    len(X_new_replay), size=n_keep, replace=False
                )
                self.replay_buffer[col] = (
                    X_new_replay.iloc[idx].reset_index(drop=True),
                    y_new_replay.iloc[idx].reset_index(drop=True),
                )

        if verbose:
            print(f"[ADAPTIVE] Update #{self.update_count} complete.")

    def evaluate(self, features_df: pd.DataFrame,
                 test_mask: pd.Series,
                 tag: str = "",
                 verbose: bool = True) -> dict:
        """Evaluate on held-out test data."""

        X_test = features_df.loc[test_mask, FEATURE_COLUMNS]

        if verbose:
            print(f"\n[ADAPTIVE{' ' + tag if tag else ''}] "
                  f"Evaluating on {test_mask.sum()} test events...")

        round_results = {}
        for col, model in self.models.items():
            if col not in features_df.columns:
                continue

            y_test = features_df.loc[test_mask, col]
            if len(y_test) == 0 or y_test.sum() == 0:
                continue

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            m = _evaluate(
                y_test, y_pred, y_prob,
                label=f"ADAPTIVE{' ' + tag} {col}" if verbose else ""
            )

            # Store with year tag for longitudinal tracking
            key = f"{col}{':' + tag if tag else ''}"
            self.results[key] = m
            round_results[col] = m

        return round_results

    def predict(self, X: pd.DataFrame, threshold: float,
                window: int) -> np.ndarray:
        col = f"label_M{int(threshold * 10)}_{window}d"
        if col not in self.models:
            raise ValueError(f"No model trained for {col}")
        return self.models[col].predict(X[FEATURE_COLUMNS])

    def predict_proba(self, X: pd.DataFrame, threshold: float,
                      window: int) -> np.ndarray:
        col = f"label_M{int(threshold * 10)}_{window}d"
        if col not in self.models:
            raise ValueError(f"No model trained for {col}")
        return self.models[col].predict_proba(X[FEATURE_COLUMNS])[:, 1]

    def save(self, path: str = MODEL_ADAPTIVE_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[ADAPTIVE] Saved to {path}")

    @classmethod
    def load(cls, path: str = MODEL_ADAPTIVE_PATH):
        with open(path, "rb") as f:
            return pickle.load(f)
