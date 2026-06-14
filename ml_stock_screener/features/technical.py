"""
Technical-indicator feature engineering.

Computes a comprehensive set of momentum, trend, volatility, and volume
indicators using the ``pandas_ta`` library (vectorised, no loops).
All indicators are computed on a standard OHLCV DataFrame and the
resulting feature DataFrame is NaN-stripped and ready for the scaler.

The function is written to be called once per ticker; the Jetson's CPU
cores handle the computation while GPU memory is reserved for inference.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from ml_stock_screener.config import CFG

logger = logging.getLogger(__name__)

try:
    import pandas_ta as ta  # type: ignore

    _HAS_PANDAS_TA = True
except ImportError:  # pragma: no cover
    _HAS_PANDAS_TA = False
    logger.warning("pandas_ta not installed; falling back to manual indicators.")


# ---------------------------------------------------------------------------
# Core feature builder
# ---------------------------------------------------------------------------

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical features for a single ticker's OHLCV DataFrame.

    Parameters
    ----------
    df : DataFrame with columns open, high, low, close, volume (lowercase),
         indexed by Date.

    Returns
    -------
    DataFrame of features with NaN rows dropped.
    """
    fc = CFG["features"]
    feat = pd.DataFrame(index=df.index)

    o = df["open"]
    h = df["high"]
    l = df["low"]
    c = df["close"]
    v = df["volume"]

    # ------------------------------------------------------------------
    # Price-based raw features
    # ------------------------------------------------------------------
    feat["log_return"] = np.log(c / c.shift(1))
    feat["hl_range"] = (h - l) / c
    feat["oc_change"] = (c - o) / o

    # ------------------------------------------------------------------
    # Moving averages & crossovers
    # ------------------------------------------------------------------
    for period in fc["ema_periods"]:
        col = f"ema_{period}"
        feat[col] = c.ewm(span=period, adjust=False).mean() / c - 1.0

    for period in fc["sma_periods"]:
        col = f"sma_{period}"
        feat[col] = c.rolling(period).mean() / c - 1.0

    # ------------------------------------------------------------------
    # RSI
    # ------------------------------------------------------------------
    feat["rsi"] = _rsi(c, fc["rsi_period"]) / 100.0

    # ------------------------------------------------------------------
    # MACD
    # ------------------------------------------------------------------
    macd_line, macd_signal, macd_hist = _macd(
        c, fc["macd_fast"], fc["macd_slow"], fc["macd_signal"]
    )
    feat["macd"] = macd_line / c
    feat["macd_signal"] = macd_signal / c
    feat["macd_hist"] = macd_hist / c

    # ------------------------------------------------------------------
    # Bollinger Bands
    # ------------------------------------------------------------------
    bb_mid = c.rolling(fc["bb_period"]).mean()
    bb_std = c.rolling(fc["bb_period"]).std()
    feat["bb_upper_dist"] = (bb_mid + fc["bb_std"] * bb_std - c) / c
    feat["bb_lower_dist"] = (c - (bb_mid - fc["bb_std"] * bb_std)) / c
    feat["bb_width"] = (2 * fc["bb_std"] * bb_std) / bb_mid

    # ------------------------------------------------------------------
    # ATR (Average True Range) — normalised by close
    # ------------------------------------------------------------------
    feat["atr"] = _atr(h, l, c, fc["atr_period"]) / c

    # ------------------------------------------------------------------
    # Stochastic %K / %D
    # ------------------------------------------------------------------
    stoch_k, stoch_d = _stochastic(h, l, c, fc["stoch_k"], fc["stoch_d"])
    feat["stoch_k"] = stoch_k / 100.0
    feat["stoch_d"] = stoch_d / 100.0

    # ------------------------------------------------------------------
    # Williams %R
    # ------------------------------------------------------------------
    feat["williams_r"] = _williams_r(h, l, c, fc["williams_r_period"]) / -100.0

    # ------------------------------------------------------------------
    # CCI
    # ------------------------------------------------------------------
    feat["cci"] = _cci(h, l, c, fc["cci_period"]) / 200.0  # rough normalisation

    # ------------------------------------------------------------------
    # ADX / DI
    # ------------------------------------------------------------------
    adx, plus_di, minus_di = _adx(h, l, c, fc["adx_period"])
    feat["adx"] = adx / 100.0
    feat["plus_di"] = plus_di / 100.0
    feat["minus_di"] = minus_di / 100.0
    feat["di_diff"] = (plus_di - minus_di) / 100.0

    # ------------------------------------------------------------------
    # Momentum
    # ------------------------------------------------------------------
    feat["momentum"] = c / c.shift(fc["momentum_period"]) - 1.0

    # ------------------------------------------------------------------
    # Volume features
    # ------------------------------------------------------------------
    vol_ma = v.rolling(fc["volume_ma_period"]).mean()
    feat["vol_ratio"] = v / vol_ma
    feat["log_volume"] = np.log1p(v) - np.log1p(vol_ma)

    # On-Balance Volume change rate
    obv = _obv(c, v)
    feat["obv_change"] = obv.pct_change(5)

    # ------------------------------------------------------------------
    # Use pandas_ta extras if available (Ichimoku, VWAP components, etc.)
    # ------------------------------------------------------------------
    if _HAS_PANDAS_TA:
        try:
            _add_pandas_ta_extras(df, feat)
        except Exception as exc:
            logger.debug("pandas_ta extras skipped: %s", exc)

    # ------------------------------------------------------------------
    # Clean up
    # ------------------------------------------------------------------
    feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    feat.dropna(inplace=True)
    return feat


# ---------------------------------------------------------------------------
# Convenience: add pandas_ta extras
# ---------------------------------------------------------------------------

def _add_pandas_ta_extras(df: pd.DataFrame, feat: pd.DataFrame) -> None:
    """Append extra indicators available via pandas_ta."""
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    # VWAP (session-level; approximated on daily data)
    try:
        vwap = ta.vwap(h, l, c, v)
        if vwap is not None and not vwap.empty:
            feat["vwap_dist"] = (c / vwap - 1.0).reindex(feat.index)
    except Exception:
        pass

    # PPO (Percentage Price Oscillator)
    try:
        ppo = ta.ppo(c)
        if ppo is not None and not ppo.empty:
            col = [x for x in ppo.columns if "PPO_" in x]
            if col:
                feat["ppo"] = ppo[col[0]].reindex(feat.index) / 100.0
    except Exception:
        pass

    # MFI (Money Flow Index)
    try:
        mfi = ta.mfi(h, l, c, v)
        if mfi is not None and not mfi.empty:
            feat["mfi"] = (mfi / 100.0).reindex(feat.index)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Pure-NumPy / pandas indicator implementations (no external dependency)
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _macd(
    close: pd.Series, fast: int, slow: int, signal: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int
) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def _stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int,
    d_period: int,
) -> tuple[pd.Series, pd.Series]:
    lowest = low.rolling(k_period).min()
    highest = high.rolling(k_period).max()
    k = 100.0 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k, d


def _williams_r(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int
) -> pd.Series:
    highest = high.rolling(period).max()
    lowest = low.rolling(period).min()
    return -100.0 * (highest - close) / (highest - lowest).replace(0, np.nan)


def _cci(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int
) -> pd.Series:
    tp = (high + low + close) / 3.0
    ma = tp.rolling(period).mean()
    md = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def _adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    plus_dm = (high - prev_high).clip(lower=0).where(
        (high - prev_high) > (prev_low - low), 0
    )
    minus_dm = (prev_low - low).clip(lower=0).where(
        (prev_low - low) > (high - prev_high), 0
    )

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    atr_s = tr.ewm(com=period - 1, min_periods=period).mean()
    plus_di = 100.0 * plus_dm.ewm(com=period - 1, min_periods=period).mean() / atr_s
    minus_di = 100.0 * minus_dm.ewm(com=period - 1, min_periods=period).mean() / atr_s
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(com=period - 1, min_periods=period).mean()
    return adx, plus_di, minus_di


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()
