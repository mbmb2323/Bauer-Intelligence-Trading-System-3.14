"""
Market data fetcher.

Downloads OHLCV data for the configured stock universe using yfinance,
caches results locally as Parquet files, and refreshes stale data
automatically.  Parallel downloads are dispatched via a thread pool to
saturate the network while respecting the Jetson's limited CPU cores.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

from ml_stock_screener.config import CFG, CACHE_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.upper()}.parquet"


def _is_stale(path: Path, max_age_hours: float) -> bool:
    if not path.exists():
        return True
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    return age_hours > max_age_hours


def _download_single(
    ticker: str,
    lookback_days: int,
    interval: str,
    max_age_hours: float,
    force_refresh: bool = False,
) -> Optional[pd.DataFrame]:
    """Download one ticker, using cache when fresh enough."""
    path = _cache_path(ticker)

    if not force_refresh and not _is_stale(path, max_age_hours):
        logger.debug("Cache hit: %s", ticker)
        return pd.read_parquet(path)

    try:
        period = f"{lookback_days}d"
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if df is None or df.empty:
            logger.warning("No data returned for %s", ticker)
            return None

        # Flatten MultiIndex columns that yfinance sometimes returns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        df.index.name = "Date"
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        df.dropna(inplace=True)

        if len(df) < 60:
            logger.warning(
                "Insufficient history for %s (%d rows); skipping.", ticker, len(df)
            )
            return None

        df.to_parquet(path)
        logger.debug("Downloaded and cached: %s (%d rows)", ticker, len(df))
        return df

    except Exception as exc:
        logger.error("Failed to download %s: %s", ticker, exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_universe(
    tickers: Optional[List[str]] = None,
    force_refresh: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for the stock universe.

    Parameters
    ----------
    tickers:
        Override the universe defined in config.yaml.
    force_refresh:
        If True, bypass the local cache and re-download everything.

    Returns
    -------
    dict mapping ticker -> OHLCV DataFrame (columns: open, high, low, close, volume)
    """
    data_cfg = CFG["data"]
    universe_cfg = CFG["universe"]

    tickers = tickers or universe_cfg["tickers"]
    lookback_days = data_cfg["lookback_days"]
    interval = data_cfg["interval"]
    max_age_hours = data_cfg["cache_max_age_hours"]
    workers = data_cfg["download_workers"]

    logger.info(
        "Fetching %d tickers  lookback=%dd  interval=%s  workers=%d",
        len(tickers),
        lookback_days,
        interval,
        workers,
    )

    results: Dict[str, pd.DataFrame] = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _download_single,
                ticker,
                lookback_days,
                interval,
                max_age_hours,
                force_refresh,
            ): ticker
            for ticker in tickers
        }

        for future in as_completed(futures):
            ticker = futures[future]
            try:
                df = future.result()
                if df is not None:
                    results[ticker] = df
            except Exception as exc:
                logger.error("Unexpected error for %s: %s", ticker, exc)

    logger.info("Successfully fetched data for %d / %d tickers", len(results), len(tickers))
    return results


def apply_volume_filter(
    data: Dict[str, pd.DataFrame],
    min_avg_volume: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """Remove tickers whose average daily volume is below the threshold."""
    threshold = min_avg_volume or CFG["universe"]["min_avg_volume"]
    filtered = {
        ticker: df
        for ticker, df in data.items()
        if df["volume"].mean() >= threshold
    }
    removed = len(data) - len(filtered)
    if removed:
        logger.info("Volume filter removed %d tickers (< %d avg daily vol)", removed, threshold)
    return filtered
