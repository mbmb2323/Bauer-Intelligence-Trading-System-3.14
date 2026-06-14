"""
Main screener entrypoint.

Usage
-----
    # Run the screener with trained models
    python main.py

    # Force-refresh market data
    python main.py --refresh

    # Screen a custom set of tickers
    python main.py --tickers AAPL MSFT NVDA AMD

    # Only show top 10 results
    python main.py --top 10

    # Use ONNX Runtime instead of TensorRT (useful on dev machines)
    python main.py --no-trt

Workflow
--------
1. Initialise Jetson hardware (power mode, CUDA memory).
2. Load the trained LSTM (TRT engine preferred; ONNX Runtime fallback).
3. Load the trained LightGBM model.
4. Run ScreenerEngine.run() — fetches data, computes features, scores.
5. Display a rich terminal table with the top candidates.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ml_stock_screener.config import CFG, MODELS_DIR
from ml_stock_screener.utils.jetson import jetson_init
from ml_stock_screener.utils.logger import setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inference function builder
# ---------------------------------------------------------------------------

def _build_inference_fn(use_trt: bool = True, input_size: int = None):
    """
    Build the LSTM inference callable.

    Preference order:
      1. TensorRT engine (Jetson MAXN performance)
      2. PyTorch CUDA (GPU, no TRT)
      3. ONNX Runtime CPU (x86 dev machines)
    """
    trt_path = MODELS_DIR / "lstm_trt.engine"
    onnx_path = MODELS_DIR / "lstm.onnx"
    pt_path = MODELS_DIR / "lstm_weights.pt"

    if use_trt and trt_path.exists():
        try:
            from ml_stock_screener.inference.tensorrt_engine import TRTEngine
            engine = TRTEngine(trt_path)
            logger.info("Using TensorRT engine for inference.")
            return engine.infer
        except Exception as exc:
            logger.warning("TRT engine load failed (%s); falling back.", exc)

    if pt_path.exists() and input_size is not None:
        try:
            import torch
            from ml_stock_screener.models.lstm_model import load_model
            from ml_stock_screener.inference.tensorrt_engine import pytorch_inference_fn
            model = load_model(input_size=input_size, weights_path=pt_path)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            logger.info("Using PyTorch (%s) for inference.", device)
            return pytorch_inference_fn(model)
        except Exception as exc:
            logger.warning("PyTorch load failed (%s); falling back to ORT.", exc)

    if onnx_path.exists():
        from ml_stock_screener.inference.tensorrt_engine import TRTEngine
        engine = TRTEngine.__new__(TRTEngine)
        engine._path = onnx_path
        logger.info("Using ONNX Runtime (CPU) for inference.")
        return engine._ort_infer

    raise FileNotFoundError(
        "No model weights found. Train the models first:\n"
        "    python train.py\n"
    )


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _display_results(results) -> None:
    """Print a rich terminal table of screening results."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box

        console = Console()
        table = Table(
            title="[bold cyan]Bauer Intelligence — ML Stock Screener[/bold cyan]",
            box=box.ROUNDED,
            show_lines=True,
        )

        table.add_column("Rank", style="dim", width=5)
        table.add_column("Ticker", style="bold white", width=8)
        table.add_column("Signal", width=14)
        table.add_column("Score", justify="right", width=8)
        table.add_column("LSTM ↑%", justify="right", width=9)
        table.add_column("LGBM ↑%", justify="right", width=9)
        table.add_column("Close $", justify="right", width=9)
        table.add_column("RSI", justify="right", width=7)
        table.add_column("ADX", justify="right", width=7)
        table.add_column("Vol×", justify="right", width=7)

        signal_colours = {
            "STRONG BUY": "bold green",
            "BUY": "green",
            "NEUTRAL": "yellow",
            "SELL": "red",
            "STRONG SELL": "bold red",
        }

        for rank, r in enumerate(results, start=1):
            colour = signal_colours.get(r.signal_label, "white")
            table.add_row(
                str(rank),
                r.ticker,
                f"[{colour}]{r.signal_label}[/{colour}]",
                f"{r.score:.3f}",
                f"{r.lstm_up_prob * 100:.1f}%",
                f"{r.lgbm_up_prob * 100:.1f}%",
                f"{r.close:.2f}",
                f"{r.rsi:.1f}",
                f"{r.adx:.1f}",
                f"{r.volume_ratio:.2f}×",
            )

        console.print(table)
        console.print(
            f"\n[dim]{len(results)} candidate(s) with score ≥ "
            f"{CFG['screening']['min_score']}[/dim]"
        )

    except ImportError:
        # Fallback to plain text
        header = (
            f"{'Rank':>4}  {'Ticker':<8}  {'Signal':<12}  "
            f"{'Score':>6}  {'Close':>8}  {'RSI':>6}  {'ADX':>6}"
        )
        print("\n=== Bauer Intelligence — ML Stock Screener ===")
        print(header)
        print("-" * len(header))
        for rank, r in enumerate(results, start=1):
            print(
                f"{rank:>4}  {r.ticker:<8}  {r.signal_label:<12}  "
                f"{r.score:>6.3f}  {r.close:>8.2f}  {r.rsi:>6.1f}  {r.adx:>6.1f}"
            )
        print(f"\n{len(results)} candidate(s) found.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    setup_logging(level=args.log_level)
    jetson_init()

    # Determine feature count from first available feature run (needed for PT)
    # We probe with a single ticker to avoid downloading everything twice.
    input_size = None
    try:
        from ml_stock_screener.data.fetcher import fetch_universe
        from ml_stock_screener.data.preprocessor import fit_scaler
        from ml_stock_screener.features.technical import compute_features

        probe = fetch_universe(tickers=[CFG["universe"]["tickers"][0]])
        if probe:
            feat_df = compute_features(list(probe.values())[0])
            input_size = feat_df.shape[1]
    except Exception as exc:
        logger.debug("Could not probe input size: %s", exc)

    # Build inference callable
    try:
        lstm_fn = _build_inference_fn(use_trt=not args.no_trt, input_size=input_size)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)

    # Load LightGBM
    lgbm = None
    lgbm_path = MODELS_DIR / "lgbm_weights.pkl"
    if lgbm_path.exists():
        from ml_stock_screener.models.ensemble import LGBMSignalModel
        try:
            lgbm = LGBMSignalModel().load(lgbm_path)
        except Exception as exc:
            logger.warning("LightGBM load failed: %s", exc)
    else:
        logger.info("LightGBM weights not found; screening with LSTM only.")

    # Run screener
    from ml_stock_screener.screener.engine import ScreenerEngine
    engine = ScreenerEngine(lstm_fn, lgbm)
    results = engine.run(
        tickers=args.tickers or None,
        force_refresh=args.refresh,
        top_n=args.top,
    )

    if not results:
        logger.warning("No candidates passed the scoring threshold.")
        sys.exit(0)

    _display_results(results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Bauer Intelligence ML Stock Screener — Jetson Orin Nano 8GB"
    )
    parser.add_argument(
        "--tickers", nargs="*", default=None,
        help="Override universe tickers (e.g. --tickers AAPL NVDA TSLA).",
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Force-refresh market data (bypass local cache).",
    )
    parser.add_argument(
        "--top", type=int, default=None,
        help="Number of top results to display.",
    )
    parser.add_argument(
        "--no-trt", action="store_true",
        help="Disable TensorRT; use PyTorch or ONNX Runtime instead.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
