from __future__ import annotations

import numpy as np


def test_run_inference_chunks_large_universe():
    from ml_stock_screener.config import CFG
    from ml_stock_screener.screener.engine import ScreenerEngine

    original_batch_size = CFG["tensorrt"]["max_batch_size"]
    CFG["tensorrt"]["max_batch_size"] = 32

    calls: list[int] = []

    def stub_lstm(x: np.ndarray) -> np.ndarray:
        calls.append(len(x))
        out = np.zeros((len(x), 3), dtype=np.float32)
        out[:, 2] = 0.8
        return out

    try:
        engine = ScreenerEngine(lstm_inference_fn=stub_lstm, lgbm_model=None)

        n_tickers = 10_000
        seq_len = CFG["features"]["sequence_length"]
        n_features = 8
        window = np.ones((1, seq_len, n_features), dtype=np.float32)

        windows = {f"T{i:05d}": window.copy() for i in range(n_tickers)}
        meta = {
            ticker: {"close": 100.0, "rsi": 50.0, "adx": 25.0, "vol_ratio": 1.0}
            for ticker in windows
        }

        results = engine._run_inference(windows, meta)

        assert len(results) == n_tickers
        assert max(calls) <= 32
        assert sum(calls) == n_tickers
    finally:
        CFG["tensorrt"]["max_batch_size"] = original_batch_size
