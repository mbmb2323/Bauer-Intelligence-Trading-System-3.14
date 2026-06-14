"""
Data preprocessor.

Normalises OHLCV data, computes forward-return labels for supervised
training, and builds the sliding-window sequence tensors that the LSTM
expects.  All heavy NumPy work is vectorised so it can be dispatched
to the Jetson's CUDA cores when called from the feature pipeline.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from ml_stock_screener.config import CFG

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------

def compute_labels(
    close: pd.Series,
    horizon: Optional[int] = None,
    threshold: Optional[float] = None,
) -> pd.Series:
    """
    Compute discrete directional labels for supervised training.

    Labels:
        2 = forward return > +threshold  (UP)
        1 = 0 < forward return <= threshold  (MILD UP)
        0 = |forward return| <= threshold  (NEUTRAL)
       -1 = -threshold <= forward return < 0  (MILD DOWN)
       -2 = forward return < -threshold  (DOWN)

    For the 3-class LSTM model we collapse to {0, 1, 2} = {DOWN, NEUTRAL, UP}
    using ``to_three_class``.
    """
    h = horizon or CFG["model"]["training"]["label_horizon"]
    t = threshold or CFG["model"]["training"]["label_threshold"]

    fwd_return = close.shift(-h) / close - 1.0
    labels = pd.cut(
        fwd_return,
        bins=[-np.inf, -t, -t / 2, t / 2, t, np.inf],
        labels=[-2, -1, 0, 1, 2],
    ).astype(float)
    return labels.rename("label")


def to_three_class(labels: pd.Series) -> pd.Series:
    """Collapse 5-class labels to 3: {0=DOWN, 1=NEUTRAL, 2=UP}."""
    mapping = {-2: 0, -1: 0, 0: 1, 1: 2, 2: 2}
    return labels.map(mapping).astype("Int64")


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def fit_scaler(features: np.ndarray) -> RobustScaler:
    """Fit a RobustScaler on the feature matrix (2-D: samples × features)."""
    scaler = RobustScaler()
    scaler.fit(features)
    return scaler


def scale_features(features: np.ndarray, scaler: RobustScaler) -> np.ndarray:
    return scaler.transform(features).astype(np.float32)


# ---------------------------------------------------------------------------
# Sequence builder
# ---------------------------------------------------------------------------

def build_sequences(
    feature_matrix: np.ndarray,
    labels: Optional[np.ndarray] = None,
    seq_len: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Convert a 2-D feature matrix (T × F) into 3-D sequences (N × seq_len × F).

    Parameters
    ----------
    feature_matrix : shape (T, F)
    labels : shape (T,) or None
    seq_len : sliding window length; defaults to config value

    Returns
    -------
    X : np.ndarray  shape (N, seq_len, F)  dtype float32
    y : np.ndarray  shape (N,)  dtype int64  — or None if labels not provided
    """
    seq_len = seq_len if seq_len is not None else CFG["features"]["sequence_length"]
    T, F = feature_matrix.shape

    if T < seq_len:
        raise ValueError(
            f"Feature matrix has only {T} rows; not enough for seq_len={seq_len}."
        )

    n_seqs = T - seq_len + 1

    X = np.lib.stride_tricks.sliding_window_view(
        feature_matrix, window_shape=(seq_len, F)
    ).reshape(n_seqs, seq_len, F).astype(np.float32)

    if labels is not None:
        y = labels[seq_len - 1 :].astype(np.int64)
        return X, y

    return X, None


# ---------------------------------------------------------------------------
# Train/val/test split
# ---------------------------------------------------------------------------

def chronological_split(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: Optional[float] = None,
    val_frac: Optional[float] = None,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """
    Split sequences chronologically into train / val / test sets.

    Returns
    -------
    (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    train_cfg = CFG["model"]["training"]
    t_frac = train_frac or train_cfg["train_split"]
    v_frac = val_frac or train_cfg["val_split"]

    n = len(X)
    i_val = int(n * t_frac)
    i_test = int(n * (t_frac + v_frac))

    return (
        (X[:i_val], y[:i_val]),
        (X[i_val:i_test], y[i_val:i_test]),
        (X[i_test:], y[i_test:]),
    )


# ---------------------------------------------------------------------------
# Convenience: process a ticker's raw OHLCV + feature DataFrame end-to-end
# ---------------------------------------------------------------------------

def prepare_for_training(
    feature_df: pd.DataFrame,
    close: pd.Series,
) -> Dict:
    """
    Full preprocessing pipeline for a single ticker.

    Parameters
    ----------
    feature_df : DataFrame of technical features (no NaN rows)
    close : Close price series aligned with feature_df

    Returns
    -------
    dict with keys: X_train, y_train, X_val, y_val, X_test, y_test, scaler
    """
    labels_raw = compute_labels(close)
    labels_3c = to_three_class(labels_raw)

    # Align
    valid_idx = feature_df.index.intersection(labels_3c.dropna().index)
    feat = feature_df.loc[valid_idx]
    lbl = labels_3c.loc[valid_idx]

    scaler = fit_scaler(feat.values)
    scaled = scale_features(feat.values, scaler)
    X, y = build_sequences(scaled, lbl.values.astype(np.int64))

    # Drop any samples where label is NaN (end-of-series)
    valid_mask = ~np.isnan(y.astype(float))
    X, y = X[valid_mask], y[valid_mask]

    train, val, test = chronological_split(X, y)

    return dict(
        X_train=train[0], y_train=train[1],
        X_val=val[0],   y_val=val[1],
        X_test=test[0],  y_test=test[1],
        scaler=scaler,
        feature_names=list(feat.columns),
    )
