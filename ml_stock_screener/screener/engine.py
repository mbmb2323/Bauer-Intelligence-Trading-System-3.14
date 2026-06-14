"""
Main screening engine.

Orchestrates the full pipeline:
  1. Fetch OHLCV data for the configured universe.
  2. Compute technical features per ticker.
  3. Build the most-recent sequence window for each ticker.
  4. Run ensemble inference (LSTM/TRT + LightGBM) to score each ticker.
  5. Rank, filter, and return the top-N candidates.

The engine is designed to run on the Jetson Orin Nano 8GB, where it
leverages the GPU for LSTM/TRT inference and the CPU for LightGBM.
Memory is managed carefully: feature computation is sequential per ticker
to avoid holding the full universe in memory simultaneously.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml_stock_screener.config import CFG, MODELS_DIR
from ml_stock_screener.data.fetcher import apply_volume_filter, fetch_universe
from ml_stock_screener.data.preprocessor import build_sequences, fit_scaler, scale_features
from ml_stock_screener.features.technical import compute_features
from ml_stock_screener.models.ensemble import EnsembleScorer, LGBMSignalModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result data class
# ---------------------------------------------------------------------------

@dataclass(order=True)
class ScreenerResult:
    """Holds the screening result for a single ticker."""

    score: float          # composite bull score [0, 1]
    ticker: str = field(compare=False)
    signal: int = field(compare=False)       # -2 … +2
    signal_label: str = field(compare=False)
    lstm_up_prob: float = field(compare=False)
    lgbm_up_prob: float = field(compare=False)
    close: float = field(compare=False)
    rsi: float = field(compare=False)
    adx: float = field(compare=False)
    volume_ratio: float = field(compare=False)


# ---------------------------------------------------------------------------
# Screener engine
# ---------------------------------------------------------------------------

class ScreenerEngine:
    """
    End-to-end ML stock screener for Jetson Orin Nano 8GB.

    Parameters
    ----------
    lstm_inference_fn : callable  (N, seq_len, F) -> (N, 3)
        Inference function — either ``pytorch_inference_fn(model)``
        or ``TRTEngine(path).infer`` for maximum Jetson performance.
    lgbm_model : optional pre-loaded LGBMSignalModel
    """

    def __init__(
        self,
        lstm_inference_fn: Callable,
        lgbm_model: Optional[LGBMSignalModel] = None,
    ) -> None:
        self._ensemble = EnsembleScorer(lstm_inference_fn, lgbm_model)
        self._seq_len: int = CFG["features"]["sequence_length"]
        batch_size = int(CFG["tensorrt"]["max_batch_size"])
        if batch_size <= 0:
            raise ValueError(
                f"Invalid tensorrt.max_batch_size={batch_size}. "
                "Expected a positive integer."
            )
        self._inference_batch_size = batch_size
        self._signal_labels: Dict[int, str] = {
            int(k): v for k, v in CFG["screening"]["signal_labels"].items()
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        tickers: Optional[List[str]] = None,
        force_refresh: bool = False,
        top_n: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> List[ScreenerResult]:
        """
        Run the full screening pipeline.

        Parameters
        ----------
        tickers      : override the universe from config.yaml
        force_refresh: bypass the local data cache
        top_n        : number of top bullish results to return
        min_score    : minimum composite score threshold

        Returns
        -------
        List of ScreenerResult sorted descending by score.
        """
        top_n = top_n or CFG["screening"]["top_n"]
        min_score = min_score or CFG["screening"]["min_score"]

        t_start = time.time()
        logger.info("=== Bauer Intelligence Stock Screener ===")

        # 1. Data
        logger.info("Step 1/4: Fetching market data …")
        raw_data = fetch_universe(tickers=tickers, force_refresh=force_refresh)
        raw_data = apply_volume_filter(raw_data)

        if not raw_data:
            logger.error("No data available after filters. Aborting.")
            return []

        # 2. Features + windows
        logger.info("Step 2/4: Computing features for %d tickers …", len(raw_data))
        windows, meta = self._build_windows(raw_data)

        if not windows:
            logger.error("No valid windows built. Aborting.")
            return []

        # 3. Inference
        logger.info("Step 3/4: Running ensemble inference …")
        results = self._run_inference(windows, meta)

        # 4. Rank & filter
        logger.info("Step 4/4: Ranking results …")
        results = [r for r in results if r.score >= min_score]
        results.sort(reverse=True)
        results = results[:top_n]

        elapsed = time.time() - t_start
        logger.info(
            "Screening complete: %d candidates found in %.1fs", len(results), elapsed
        )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_windows(
        self, raw_data: Dict[str, pd.DataFrame]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
        """
        Compute features and extract the latest sequence window for each ticker.

        Returns
        -------
        windows : dict ticker -> np.ndarray (1, seq_len, n_features)
        meta    : dict ticker -> {close, rsi, adx, vol_ratio}
        """
        windows: Dict[str, np.ndarray] = {}
        meta: Dict[str, dict] = {}

        for ticker, df in raw_data.items():
            try:
                feat_df = compute_features(df)
                if len(feat_df) < self._seq_len:
                    logger.debug("%s: not enough feature rows (%d)", ticker, len(feat_df))
                    continue

                # Normalise using all available data
                scaler = fit_scaler(feat_df.values)
                scaled = scale_features(feat_df.values, scaler)

                # Take only the most recent window
                window = scaled[-self._seq_len:][np.newaxis, ...]  # (1, T, F)
                windows[ticker] = window

                # Capture latest indicator values for display
                last = feat_df.iloc[-1]
                meta[ticker] = {
                    "close": float(df["close"].iloc[-1]),
                    "rsi": float(last.get("rsi", np.nan)) * 100,
                    "adx": float(last.get("adx", np.nan)) * 100,
                    "vol_ratio": float(last.get("vol_ratio", np.nan)),
                }

            except Exception as exc:
                logger.warning("Skipping %s: %s", ticker, exc)

        logger.info("Built %d valid windows", len(windows))
        return windows, meta

    def _run_inference(
        self,
        windows: Dict[str, np.ndarray],
        meta: Dict[str, dict],
    ) -> List[ScreenerResult]:
        """Run inference in chunks and return ScreenerResult list."""
        tickers = list(windows.keys())
        batch_size = self._inference_batch_size

        lstm_parts: List[np.ndarray] = []
        lgbm_parts: List[np.ndarray] = []
        lgbm_failed = False

        for start in range(0, len(tickers), batch_size):
            chunk_tickers = tickers[start : start + batch_size]
            batch = np.concatenate([windows[t] for t in chunk_tickers], axis=0)

            lstm_parts.append(self._ensemble.run_lstm(batch))

            if self._ensemble.has_lgbm() and not lgbm_failed:
                try:
                    chunk_lgbm = self._ensemble.run_lgbm(batch)
                    if chunk_lgbm is not None:
                        lgbm_parts.append(chunk_lgbm)
                except Exception as exc:
                    logger.warning(
                        "LightGBM inference failed for batch starting at %d (%s); "
                        "falling back to LSTM-only scoring for all tickers.",
                        start,
                        exc,
                    )
                    lgbm_failed = True
                    lgbm_parts = []

        lstm_proba = np.concatenate(lstm_parts, axis=0)
        lgbm_proba: Optional[np.ndarray] = (
            np.concatenate(lgbm_parts, axis=0) if lgbm_parts else None
        )

        combined = self._ensemble.combine_probabilities(lstm_proba, lgbm_proba)
        scores = self._ensemble.bull_scores(combined)
        signals = self._ensemble.classify(scores)

        results: List[ScreenerResult] = []
        for i, ticker in enumerate(tickers):
            signal = int(signals[i])
            m = meta.get(ticker, {})

            lgbm_up = float(lgbm_proba[i, 2]) if lgbm_proba is not None else 0.0

            results.append(
                ScreenerResult(
                    ticker=ticker,
                    score=float(scores[i]),
                    signal=signal,
                    signal_label=self._signal_labels.get(signal, "NEUTRAL"),
                    lstm_up_prob=float(lstm_proba[i, 2]),
                    lgbm_up_prob=lgbm_up,
                    close=m.get("close", 0.0),
                    rsi=m.get("rsi", 0.0),
                    adx=m.get("adx", 0.0),
                    volume_ratio=m.get("vol_ratio", 1.0),
                )
            )

        return results
