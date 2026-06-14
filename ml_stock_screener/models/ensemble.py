"""
LightGBM + LSTM ensemble model.

The ensemble combines:
  1. A LightGBM classifier trained on the flat feature vector (last row
     of the sequence window) — fast, interpretable technical-signal scoring.
  2. An LSTM (or its TensorRT-accelerated equivalent) trained on the full
     sequence window — captures temporal dependencies.

At inference time both models score each ticker and their probabilities
are blended using the weights defined in config.yaml.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ml_stock_screener.config import CFG, MODELS_DIR

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb  # type: ignore

    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False
    logger.warning("LightGBM not installed; ensemble will use LSTM only.")

try:
    import joblib  # type: ignore

    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False


# ---------------------------------------------------------------------------
# LightGBM wrapper
# ---------------------------------------------------------------------------

class LGBMSignalModel:
    """
    Gradient-boosting classifier for technical-indicator signals.

    Trained on the **flat** feature vector of the most-recent timestep
    in each window (no sequence dimension needed).
    """

    def __init__(self) -> None:
        lgbm_cfg = CFG["model"]["lgbm"]
        self._model: Optional["lgb.LGBMClassifier"] = None
        self._cfg = lgbm_cfg

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """Train the LightGBM classifier."""
        if not _HAS_LGB:
            raise RuntimeError("LightGBM is required.")

        # Use last timestep of each sequence as flat feature vector
        X_tr_flat = _last_step(X_train)
        X_va_flat = _last_step(X_val)

        cfg = self._cfg
        self._model = lgb.LGBMClassifier(
            n_estimators=cfg["n_estimators"],
            learning_rate=cfg["learning_rate"],
            num_leaves=cfg["num_leaves"],
            max_depth=cfg["max_depth"],
            min_child_samples=cfg["min_child_samples"],
            subsample=cfg["subsample"],
            colsample_bytree=cfg["colsample_bytree"],
            reg_alpha=cfg["reg_alpha"],
            reg_lambda=cfg["reg_lambda"],
            n_jobs=-1,
            verbose=-1,
        )
        self._model.fit(
            X_tr_flat,
            y_train,
            eval_set=[(X_va_flat, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        logger.info(
            "LightGBM trained: best iteration %d", self._model.best_iteration_
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities, shape (N, n_classes)."""
        if self._model is None:
            raise RuntimeError("Model not trained or loaded.")
        X_flat = _last_step(X)
        return self._model.predict_proba(X_flat).astype(np.float32)

    def save(self, path: Optional[Path] = None) -> Path:
        if not _HAS_JOBLIB:
            raise RuntimeError(
                "joblib is required to save the LightGBM model. "
                "Install it with: pip install joblib"
            )
        if self._model is None:
            raise RuntimeError("Model has not been trained yet; call fit() first.")
        path = path or MODELS_DIR / "lgbm_weights.pkl"
        joblib.dump(self._model, str(path))
        logger.info("LightGBM model saved to %s", path)
        return path

    def load(self, path: Optional[Path] = None) -> "LGBMSignalModel":
        if not _HAS_JOBLIB:
            raise RuntimeError(
                "joblib is required to load the LightGBM model. "
                "Install it with: pip install joblib"
            )
        path = path or MODELS_DIR / "lgbm_weights.pkl"
        self._model = joblib.load(str(path))
        logger.info("LightGBM model loaded from %s", path)
        return self


# ---------------------------------------------------------------------------
# Ensemble scorer
# ---------------------------------------------------------------------------

class EnsembleScorer:
    """
    Blend LSTM and LightGBM probability outputs into a composite score.

    The composite score is the weighted average of the probability assigned
    to class 2 (UP), normalised so that a score of 0.5 is neutral.

    A finalised EnsembleScorer can rank an entire universe in <1 second
    on the Jetson Orin Nano using GPU inference (LSTM via TRT) + CPU
    LightGBM.
    """

    def __init__(
        self,
        lstm_inference_fn,
        lgbm_model: Optional[LGBMSignalModel] = None,
    ) -> None:
        """
        Parameters
        ----------
        lstm_inference_fn:
            Callable that accepts a numpy array of shape
            (N, seq_len, n_features) and returns probabilities
            (N, n_classes) as a numpy float32 array.
            Can be a plain PyTorch forward pass or the TensorRT engine.
        lgbm_model:
            Optional LGBMSignalModel.  If None, only LSTM is used.
        """
        self._lstm_fn = lstm_inference_fn
        self._lgbm = lgbm_model
        self._lstm_w = CFG["model"]["ensemble"]["lstm_weight"]
        self._lgbm_w = CFG["model"]["ensemble"]["lgbm_weight"]

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute composite bull scores for a batch of windows.

        Parameters
        ----------
        X : shape (N, seq_len, n_features)

        Returns
        -------
        scores : shape (N,)  values in [0, 1]  — higher = more bullish
        """
        lstm_proba = self._lstm_fn(X)   # (N, 3)

        if self._lgbm is not None and _HAS_LGB:
            lgbm_proba = self._lgbm.predict_proba(X)  # (N, 3)
            # Align class count (LGBM may have fewer classes if some absent in training)
            lgbm_proba = _pad_proba(lgbm_proba, n_classes=3)
            combined = self._lstm_w * lstm_proba + self._lgbm_w * lgbm_proba
        else:
            combined = lstm_proba

        # Bull score = P(UP) + 0.5 * P(NEUTRAL)
        up_prob = combined[:, 2]
        neutral_prob = combined[:, 1]
        return (up_prob + 0.5 * neutral_prob).astype(np.float32)

    def classify(self, scores: np.ndarray) -> np.ndarray:
        """
        Convert continuous scores to discrete signal integers.

        Returns
        -------
        signals : np.ndarray of int  in {-2, -1, 0, 1, 2}
        """
        signals = np.zeros(len(scores), dtype=np.int8)
        signals[scores >= 0.75] = 2    # STRONG BUY
        signals[(scores >= 0.60) & (scores < 0.75)] = 1   # BUY
        signals[(scores <= 0.40) & (scores > 0.25)] = -1  # SELL
        signals[scores <= 0.25] = -2   # STRONG SELL
        return signals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _last_step(X: np.ndarray) -> np.ndarray:
    """Extract the last timestep from a sequence array (N, T, F) -> (N, F)."""
    if X.ndim == 3:
        return X[:, -1, :]
    return X


def _pad_proba(proba: np.ndarray, n_classes: int) -> np.ndarray:
    """Pad probability array to n_classes columns if needed."""
    if proba.shape[1] >= n_classes:
        return proba[:, :n_classes]
    pad = np.zeros((len(proba), n_classes - proba.shape[1]), dtype=np.float32)
    return np.concatenate([proba, pad], axis=1)
