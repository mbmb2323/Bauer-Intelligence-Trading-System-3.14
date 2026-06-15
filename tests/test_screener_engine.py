from __future__ import annotations

import sys
import types

import numpy as np
import pytest

FEATURE_COUNT = 8


def test_engine_rejects_nonpositive_batch_size(monkeypatch):
    from ml_stock_screener.config import CFG

    monkeypatch.setitem(
        sys.modules,
        "yfinance",
        types.SimpleNamespace(download=lambda *args, **kwargs: None),
    )
    from ml_stock_screener.screener.engine import ScreenerEngine

    for bad_size in (0, -1, -100):
        monkeypatch.setitem(CFG["tensorrt"], "max_batch_size", bad_size)
        with pytest.raises(ValueError, match="max_batch_size"):
            ScreenerEngine(lstm_inference_fn=lambda x: x, lgbm_model=None)


def test_run_inference_chunks_large_universe(monkeypatch):
    from ml_stock_screener.config import CFG

    monkeypatch.setitem(
        sys.modules,
        "yfinance",
        types.SimpleNamespace(download=lambda *args, **kwargs: None),
    )

    from ml_stock_screener.screener.engine import ScreenerEngine

    monkeypatch.setitem(CFG["tensorrt"], "max_batch_size", 32)

    calls: list[int] = []

    def stub_lstm(x: np.ndarray) -> np.ndarray:
        calls.append(len(x))
        out = np.zeros((len(x), 3), dtype=np.float32)
        out[:, 2] = 0.8
        return out

    engine = ScreenerEngine(lstm_inference_fn=stub_lstm, lgbm_model=None)

    n_tickers = 10_000
    seq_len = CFG["features"]["sequence_length"]
    window = np.ones((1, seq_len, FEATURE_COUNT), dtype=np.float32)

    windows = {f"T{i:05d}": window.copy() for i in range(n_tickers)}
    meta = {
        ticker: {"close": 100.0, "rsi": 50.0, "adx": 25.0, "vol_ratio": 1.0}
        for ticker in windows
    }

    results = engine._run_inference(windows, meta)

    assert len(results) == n_tickers
    assert max(calls) <= 32
    assert sum(calls) == n_tickers


def test_run_inference_falls_back_when_lgbm_fails(monkeypatch):
    from ml_stock_screener.config import CFG
    from ml_stock_screener.models import ensemble as ensemble_module

    monkeypatch.setitem(
        sys.modules,
        "yfinance",
        types.SimpleNamespace(download=lambda *args, **kwargs: None),
    )
    monkeypatch.setitem(CFG["tensorrt"], "max_batch_size", 16)
    monkeypatch.setattr(ensemble_module, "_HAS_LGB", True)

    from ml_stock_screener.screener.engine import ScreenerEngine

    class FlakyLGBM:
        def __init__(self):
            self.calls = 0

        def predict_proba(self, x: np.ndarray) -> np.ndarray:
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("simulated lgbm failure")
            out = np.zeros((len(x), 3), dtype=np.float32)
            out[:, 2] = 0.7
            out[:, 1] = 0.2
            out[:, 0] = 0.1
            return out

    def stub_lstm(x: np.ndarray) -> np.ndarray:
        out = np.zeros((len(x), 3), dtype=np.float32)
        out[:, 2] = 0.8
        out[:, 1] = 0.15
        out[:, 0] = 0.05
        return out

    lgbm = FlakyLGBM()
    engine = ScreenerEngine(lstm_inference_fn=stub_lstm, lgbm_model=lgbm)

    n_tickers = 100
    seq_len = CFG["features"]["sequence_length"]
    window = np.ones((1, seq_len, FEATURE_COUNT), dtype=np.float32)
    windows = {f"T{i:03d}": window.copy() for i in range(n_tickers)}
    meta = {
        ticker: {"close": 100.0, "rsi": 50.0, "adx": 25.0, "vol_ratio": 1.0}
        for ticker in windows
    }

    results = engine._run_inference(windows, meta)
    assert len(results) == n_tickers
    assert lgbm.calls >= 2

    # The first batch (16 tickers) had a successful LGBM call → ensemble scoring.
    # score = lstm_w * P_lstm(up) + lgbm_w * P_lgbm(up)
    #       + 0.5 * (lstm_w * P_lstm(neutral) + lgbm_w * P_lgbm(neutral))
    #       = 0.6 * 0.8 + 0.4 * 0.7 + 0.5 * (0.6 * 0.15 + 0.4 * 0.2)
    #       = 0.76 + 0.5 * 0.17 = 0.845
    expected_combined = 0.6 * 0.8 + 0.4 * 0.7 + 0.5 * (0.6 * 0.15 + 0.4 * 0.2)
    # The second batch onward had no LGBM (failure) → LSTM-only scoring.
    # score = P(up) + 0.5 * P(neutral) = 0.8 + 0.5 * 0.15 = 0.875
    expected_lstm_only = 0.8 + 0.5 * 0.15
    for i, r in enumerate(results):
        if i < 16:
            assert abs(r.score - expected_combined) < 1e-4, (
                f"Ticker {i}: expected combined score {expected_combined:.4f}, got {r.score}"
            )
        else:
            assert abs(r.score - expected_lstm_only) < 1e-4, (
                f"Ticker {i}: expected LSTM-only score {expected_lstm_only:.4f}, got {r.score}"
            )
