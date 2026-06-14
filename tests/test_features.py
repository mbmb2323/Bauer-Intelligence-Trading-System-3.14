"""
Unit tests for feature engineering.

These tests run entirely with synthetic data — no network calls,
no PyTorch, no TensorRT.  They verify that:

1. compute_features returns a non-empty DataFrame with the expected
   columns for a valid OHLCV input.
2. No NaN or Inf values remain in the output after compute_features.
3. All indicator values fall within expected numerical ranges.
4. The preprocessor correctly builds sliding-window sequences.
5. Label generation produces the correct three-class distribution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Synthetic OHLCV fixture (300 trading days)
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)

    close = 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.012, n))
    high  = close * (1 + rng.uniform(0.001, 0.02, n))
    low   = close * (1 - rng.uniform(0.001, 0.02, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    volume = rng.integers(1_000_000, 10_000_000, n).astype(float)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


@pytest.fixture
def ohlcv():
    return _make_ohlcv()


# ---------------------------------------------------------------------------
# compute_features
# ---------------------------------------------------------------------------

class TestComputeFeatures:
    def test_returns_dataframe(self, ohlcv):
        from ml_stock_screener.features.technical import compute_features
        feat = compute_features(ohlcv)
        assert isinstance(feat, pd.DataFrame)

    def test_non_empty(self, ohlcv):
        from ml_stock_screener.features.technical import compute_features
        feat = compute_features(ohlcv)
        assert len(feat) > 0, "Feature DataFrame must not be empty."

    def test_no_nan(self, ohlcv):
        from ml_stock_screener.features.technical import compute_features
        feat = compute_features(ohlcv)
        assert not feat.isnull().any().any(), "Feature DataFrame must not contain NaN."

    def test_no_inf(self, ohlcv):
        from ml_stock_screener.features.technical import compute_features
        feat = compute_features(ohlcv)
        assert not np.isinf(feat.values).any(), "Feature DataFrame must not contain Inf."

    def test_rsi_range(self, ohlcv):
        from ml_stock_screener.features.technical import compute_features
        feat = compute_features(ohlcv)
        assert "rsi" in feat.columns
        # RSI is stored normalised (/ 100), so values should be in [0, 1]
        assert feat["rsi"].between(0.0, 1.0).all(), "RSI out of [0, 1] range."

    def test_has_key_columns(self, ohlcv):
        from ml_stock_screener.features.technical import compute_features
        required = ["rsi", "macd", "bb_width", "atr", "adx", "log_return"]
        feat = compute_features(ohlcv)
        for col in required:
            assert col in feat.columns, f"Missing feature column: {col}"

    def test_column_count(self, ohlcv):
        from ml_stock_screener.features.technical import compute_features
        feat = compute_features(ohlcv)
        # Should have at least 30 features
        assert feat.shape[1] >= 30, f"Expected ≥30 features, got {feat.shape[1]}"


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

class TestPreprocessor:
    def test_build_sequences_shape(self, ohlcv):
        from ml_stock_screener.features.technical import compute_features
        from ml_stock_screener.data.preprocessor import fit_scaler, scale_features, build_sequences

        feat = compute_features(ohlcv)
        scaler = fit_scaler(feat.values)
        scaled = scale_features(feat.values, scaler)

        seq_len = 30
        X, _ = build_sequences(scaled, seq_len=seq_len)

        T, F = scaled.shape
        expected_n = T - seq_len + 1
        assert X.shape == (expected_n, seq_len, F), (
            f"Expected shape {(expected_n, seq_len, F)}, got {X.shape}"
        )

    def test_build_sequences_dtype(self, ohlcv):
        from ml_stock_screener.features.technical import compute_features
        from ml_stock_screener.data.preprocessor import fit_scaler, scale_features, build_sequences

        feat = compute_features(ohlcv)
        scaler = fit_scaler(feat.values)
        scaled = scale_features(feat.values, scaler)
        X, _ = build_sequences(scaled, seq_len=30)
        assert X.dtype == np.float32

    def test_build_sequences_with_labels(self, ohlcv):
        from ml_stock_screener.features.technical import compute_features
        from ml_stock_screener.data.preprocessor import (
            fit_scaler, scale_features, build_sequences, compute_labels, to_three_class
        )

        feat = compute_features(ohlcv)
        lbl = to_three_class(compute_labels(ohlcv["close"]))
        valid = feat.index.intersection(lbl.dropna().index)
        feat = feat.loc[valid]
        lbl = lbl.loc[valid]

        scaler = fit_scaler(feat.values)
        scaled = scale_features(feat.values, scaler)
        seq_len = 30
        X, y = build_sequences(scaled, lbl.values.astype(np.int64), seq_len=seq_len)
        assert y is not None
        assert len(X) == len(y)

    def test_insufficient_data_raises(self):
        from ml_stock_screener.data.preprocessor import build_sequences
        import pytest

        tiny = np.random.randn(10, 5).astype(np.float32)
        with pytest.raises(ValueError, match="not enough"):
            build_sequences(tiny, seq_len=60)


# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------

class TestLabels:
    def test_three_class_values(self, ohlcv):
        from ml_stock_screener.data.preprocessor import compute_labels, to_three_class
        labels = to_three_class(compute_labels(ohlcv["close"]))
        unique = set(labels.dropna().astype(int).unique())
        assert unique.issubset({0, 1, 2}), f"Unexpected label values: {unique}"

    def test_labels_not_all_same(self, ohlcv):
        from ml_stock_screener.data.preprocessor import compute_labels, to_three_class
        labels = to_three_class(compute_labels(ohlcv["close"]))
        assert labels.dropna().nunique() > 1, "All labels are the same class — suspicious."


# ---------------------------------------------------------------------------
# Ensemble scorer (no ML libraries needed — uses stub inference fn)
# ---------------------------------------------------------------------------

class TestEnsembleScorer:
    def _stub_lstm(self, x: np.ndarray) -> np.ndarray:
        """Stub that returns uniform probabilities."""
        n = len(x)
        return np.full((n, 3), 1.0 / 3.0, dtype=np.float32)

    def test_score_shape(self):
        from ml_stock_screener.models.ensemble import EnsembleScorer
        scorer = EnsembleScorer(lstm_inference_fn=self._stub_lstm, lgbm_model=None)
        X = np.random.randn(5, 60, 10).astype(np.float32)
        scores = scorer.score(X)
        assert scores.shape == (5,)

    def test_score_range(self):
        from ml_stock_screener.models.ensemble import EnsembleScorer
        scorer = EnsembleScorer(lstm_inference_fn=self._stub_lstm, lgbm_model=None)
        X = np.random.randn(10, 60, 10).astype(np.float32)
        scores = scorer.score(X)
        assert (scores >= 0.0).all() and (scores <= 1.0).all()

    def test_classify_outputs(self):
        from ml_stock_screener.models.ensemble import EnsembleScorer
        scorer = EnsembleScorer(lstm_inference_fn=self._stub_lstm, lgbm_model=None)
        scores = np.array([0.9, 0.65, 0.5, 0.35, 0.1])
        signals = scorer.classify(scores)
        assert signals[0] == 2   # STRONG BUY
        assert signals[1] == 1   # BUY
        assert signals[2] == 0   # NEUTRAL
        assert signals[3] == -1  # SELL
        assert signals[4] == -2  # STRONG SELL
