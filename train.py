"""
Training script.

Trains the LSTM + LightGBM models on historical data from the configured
universe and exports:
  - ``models/lstm_weights.pt``  — PyTorch LSTM checkpoint
  - ``models/lstm.onnx``        — ONNX export for TRT conversion
  - ``models/lgbm_weights.pkl`` — LightGBM model

Usage
-----
    python train.py                     # use config.yaml universe
    python train.py --tickers AAPL MSFT TSLA
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from ml_stock_screener.config import CFG, MODELS_DIR
from ml_stock_screener.data.fetcher import apply_volume_filter, fetch_universe
from ml_stock_screener.data.preprocessor import (
    build_sequences,
    chronological_split,
    compute_labels,
    fit_scaler,
    scale_features,
    to_three_class,
)
from ml_stock_screener.features.technical import compute_features
from ml_stock_screener.models.ensemble import LGBMSignalModel
from ml_stock_screener.models.lstm_model import StockLSTM, export_to_onnx, load_model, train_model
from ml_stock_screener.utils.jetson import jetson_init
from ml_stock_screener.utils.logger import setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def build_combined_dataset(tickers: Optional[List[str]] = None):
    """
    Fetch data, compute features, and concatenate sequences from all tickers.

    Returns a dict with keys:
        X_train, y_train, X_val, y_val, X_test, y_test,
        input_size, scaler (from first ticker — used for reference)
    """
    raw_data = fetch_universe(tickers=tickers, force_refresh=False)
    raw_data = apply_volume_filter(raw_data)

    if not raw_data:
        raise RuntimeError("No usable data after filters.")

    seq_len = CFG["features"]["sequence_length"]
    all_X_train, all_y_train = [], []
    all_X_val,   all_y_val   = [], []
    all_X_test,  all_y_test  = [], []
    ref_scaler = None
    input_size = None

    for ticker, df in raw_data.items():
        try:
            feat_df = compute_features(df)
            if len(feat_df) < seq_len + 30:
                continue

            labels_raw = compute_labels(df["close"])
            labels_3c = to_three_class(labels_raw)
            valid_idx = feat_df.index.intersection(labels_3c.dropna().index)
            feat = feat_df.loc[valid_idx]
            lbl = labels_3c.loc[valid_idx]

            scaler = fit_scaler(feat.values)
            if ref_scaler is None:
                ref_scaler = scaler
                input_size = feat.shape[1]

            scaled = scale_features(feat.values, scaler)
            X, y = build_sequences(scaled, lbl.values.astype(np.int64), seq_len=seq_len)

            valid_mask = ~np.isnan(y.astype(float))
            X, y = X[valid_mask], y[valid_mask]

            if len(X) < 20:
                continue

            train, val, test = chronological_split(X, y)
            all_X_train.append(train[0])
            all_y_train.append(train[1])
            all_X_val.append(val[0])
            all_y_val.append(val[1])
            all_X_test.append(test[0])
            all_y_test.append(test[1])

        except Exception as exc:
            logger.warning("Skipping %s: %s", ticker, exc)

    if not all_X_train:
        raise RuntimeError("No training sequences built — check your data.")

    def _cat(arrays):
        return np.concatenate(arrays, axis=0)

    return dict(
        X_train=_cat(all_X_train), y_train=_cat(all_y_train),
        X_val=_cat(all_X_val),     y_val=_cat(all_y_val),
        X_test=_cat(all_X_test),   y_test=_cat(all_y_test),
        input_size=input_size,
        scaler=ref_scaler,
    )


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def main(args):
    setup_logging()
    jetson_init()

    logger.info("Building dataset …")
    ds = build_combined_dataset(tickers=args.tickers or None)

    X_train = ds["X_train"]
    y_train = ds["y_train"]
    X_val   = ds["X_val"]
    y_val   = ds["y_val"]
    X_test  = ds["X_test"]
    y_test  = ds["y_test"]
    input_size = ds["input_size"]

    logger.info(
        "Dataset sizes — train: %d  val: %d  test: %d  features: %d",
        len(X_train), len(X_val), len(X_test), input_size,
    )

    # ---- LSTM ----
    logger.info("Training LSTM …")
    model = StockLSTM(input_size=input_size)
    history = train_model(
        model, X_train, y_train, X_val, y_val,
        save_path=MODELS_DIR / "lstm_weights.pt",
    )
    logger.info("LSTM best epoch %d", history["best_epoch"])

    # Re-load best weights
    model = load_model(
        input_size=input_size,
        weights_path=MODELS_DIR / "lstm_weights.pt",
    )

    # ONNX export
    seq_len = CFG["features"]["sequence_length"]
    onnx_path = export_to_onnx(
        model, seq_len=seq_len, n_features=input_size,
        onnx_path=MODELS_DIR / "lstm.onnx",
    )

    # ---- LightGBM ----
    logger.info("Training LightGBM …")
    lgbm = LGBMSignalModel()
    try:
        lgbm.fit(X_train, y_train, X_val, y_val)
        lgbm.save(MODELS_DIR / "lgbm_weights.pkl")
    except RuntimeError as exc:
        logger.warning("LightGBM training skipped: %s", exc)

    # ---- TensorRT build (Jetson only) ----
    if CFG["tensorrt"]["enabled"]:
        try:
            from ml_stock_screener.inference.tensorrt_engine import TRTEngine
            TRTEngine.build_from_onnx(
                onnx_path=onnx_path,
                engine_path=MODELS_DIR / "lstm_trt.engine",
            )
            logger.info("TensorRT engine built successfully.")
        except Exception as exc:
            logger.warning("TRT build skipped: %s", exc)

    logger.info("Training complete. Models saved to %s", MODELS_DIR)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTM + LightGBM stock screener models.")
    parser.add_argument(
        "--tickers", nargs="*", default=None,
        help="Override the universe tickers (e.g. --tickers AAPL MSFT TSLA).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
