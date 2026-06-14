# Bauer Intelligence Trading System — ML Stock Screener
### Optimised for NVIDIA Jetson Orin Nano 8 GB

A production-ready machine-learning stock screener that runs end-to-end on the
**Jetson Orin Nano 8 GB**, exploiting its Ampere GPU (1024 CUDA cores + 32 Tensor
Cores) for both FP16 LSTM inference via **TensorRT** and the unified 8 GB
LPDDR5 memory pool for seamless CPU↔GPU data transfer.

---

## Architecture

```
Universe (40+ default S&P 500 tickers, scalable to 10,000)
        │
        ▼
┌───────────────────┐
│  Data Fetcher      │  yfinance, parallel downloads, local Parquet cache
└────────┬──────────┘
         │ OHLCV DataFrame (daily)
         ▼
┌───────────────────┐
│ Feature Engineer   │  35+ technical indicators (RSI, MACD, BB, ATR, ADX …)
└────────┬──────────┘
         │ Scaled feature matrix (N × seq_len × F)
         ▼
┌──────────────────────────────────────────────────────┐
│                   Ensemble                            │
│  ┌────────────────┐    ┌────────────────────────┐    │
│  │  LSTM + TRT    │    │  LightGBM              │    │
│  │  (Jetson GPU)  │    │  (flat feature vector) │    │
│  │  weight: 0.60  │    │  weight: 0.40          │    │
│  └────────────────┘    └────────────────────────┘    │
│           └──────────────────┘                       │
│                 Composite score [0, 1]                │
└──────────────────────────┬───────────────────────────┘
                           │
                           ▼
              Top-N ranked candidates
              STRONG BUY / BUY / NEUTRAL / SELL / STRONG SELL
```

---

## Jetson Orin Nano 8 GB — Hardware Notes

| Component | Spec |
|-----------|------|
| GPU | NVIDIA Ampere — 1024 CUDA cores, 32 Tensor Cores |
| CPU | 6-core Arm Cortex-A78AE |
| Memory | 8 GB LPDDR5 unified (CPU + GPU share) |
| Storage (recommended) | NVMe SSD via M.2 slot |
| JetPack | 6.x (L4T R36.x) |
| CUDA | 11.4+ |
| TensorRT | 8.6+ |

The screener runs end-to-end in **< 60 seconds** on the Jetson Orin Nano 8 GB
(40-ticker universe, TRT FP16 enabled, market data cached).

---

## Project Structure

```
.
├── config.yaml                    # All configuration (universe, model, hardware)
├── requirements.txt               # Python dependencies
├── train.py                       # Train LSTM + LightGBM; export TRT engine
├── main.py                        # Run the screener (CLI)
├── Dockerfile.jetson              # Production Docker image for Jetson
│
├── ml_stock_screener/
│   ├── config.py                  # Config loader (singleton CFG)
│   ├── data/
│   │   ├── fetcher.py             # Parallel yfinance downloader + cache
│   │   └── preprocessor.py       # Labels, scaling, sequence builder
│   ├── features/
│   │   └── technical.py          # 35+ vectorised technical indicators
│   ├── models/
│   │   ├── lstm_model.py          # PyTorch LSTM (FP16 AMP training)
│   │   └── ensemble.py           # LightGBM + LSTM blending
│   ├── inference/
│   │   └── tensorrt_engine.py    # TRT build/load/infer; ORT CPU fallback
│   ├── screener/
│   │   └── engine.py             # End-to-end ScreenerEngine
│   └── utils/
│       ├── jetson.py             # nvpmodel, jetson_clocks, CUDA memory
│       └── logger.py             # Rotating file + console logging
│
└── tests/
    └── test_features.py          # Pytest unit tests (no GPU required)
```

---

## Quick Start

### 1. Jetson JetPack Setup

Flash JetPack 6.x to your Jetson Orin Nano using SDK Manager, then:

```bash
# Maximise performance
sudo nvpmodel -m 0          # MAXN mode
sudo jetson_clocks          # Pin clocks

# Verify CUDA
python3 -c "import torch; print(torch.cuda.is_available())"
```

### 2. Install Dependencies

PyTorch for Jetson must come from NVIDIA's wheel index (not PyPI):

```bash
# Install PyTorch for JetPack 6 / L4T R36
pip3 install --no-cache \
  https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.1.0a0+41361538.nv23.06-cp310-cp310-linux_aarch64.whl

# TensorRT Python bindings are already available via JetPack — verify:
python3 -c "import tensorrt; print(tensorrt.__version__)"

# PyCUDA for TRT memory management
pip3 install pycuda

# Remaining dependencies
pip3 install -r requirements.txt
```

### 3. Train the Models

```bash
# Train on the full configured universe (≈ 40 tickers, ~30 min first run)
python3 train.py

# Or train on a smaller custom universe
python3 train.py --tickers AAPL MSFT NVDA TSLA AMD
```

This produces:
- `models/lstm_weights.pt` — PyTorch LSTM checkpoint
- `models/lstm.onnx` — ONNX export
- `models/lstm_trt.engine` — TensorRT FP16 engine (Jetson-specific)
- `models/lgbm_weights.pkl` — LightGBM model

> **Note:** The `.engine` file is device-specific. Regenerate it if you move to
> a different Jetson or upgrade JetPack.

### 4. Run the Screener

```bash
# Default: top-20 bullish candidates from configured universe
python3 main.py

# Custom universe, top 10
python3 main.py --tickers AAPL MSFT NVDA AMD TSLA META GOOGL AMZN --top 10

# Force-refresh market data (bypass 4-hour cache)
python3 main.py --refresh

# Disable TRT (use PyTorch GPU or ONNX Runtime CPU)
python3 main.py --no-trt
```

Example output:

```
╭──────────────────────────────────────────────────────────────────────────────╮
│             Bauer Intelligence — ML Stock Screener                           │
├──────┬──────────┬──────────────┬────────┬─────────┬─────────┬───────┬───────┤
│ Rank │ Ticker   │ Signal       │  Score │ LSTM ↑% │ LGBM ↑% │   RSI │   ADX │
├──────┼──────────┼──────────────┼────────┼─────────┼─────────┼───────┼───────┤
│    1 │ NVDA     │ STRONG BUY   │  0.812 │   81.4% │   78.2% │  61.3 │  42.1 │
│    2 │ AVGO     │ STRONG BUY   │  0.789 │   78.1% │   75.6% │  58.7 │  38.4 │
│    3 │ AMD      │ BUY          │  0.731 │   73.2% │   69.8% │  55.2 │  31.2 │
│  … │ …        │ …            │    …   │     …   │     …   │    …  │    …  │
╰──────┴──────────┴──────────────┴────────┴─────────┴─────────┴───────┴───────╯
```

---

## Docker (Jetson)

```bash
# Build on Jetson
docker build -f Dockerfile.jetson -t bauer-screener:latest .

# Run screener (mount models and cache for persistence)
docker run --rm --runtime=nvidia \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/.cache:/app/.cache \
    bauer-screener:latest python3 main.py
```

---

## Configuration

All parameters are in `config.yaml`:

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `universe.tickers` | list | 40 S&P 500 stocks | Stocks to screen |
| `data.lookback_days` | int | 365 | Historical data window |
| `features.sequence_length` | int | 60 | LSTM input window (trading days) |
| `model.lstm.hidden_size` | int | 128 | LSTM hidden units |
| `model.ensemble.lstm_weight` | float | 0.6 | LSTM blend weight |
| `tensorrt.precision` | str | `fp16` | `fp32` / `fp16` / `int8` |
| `jetson.power_mode` | int | 0 | NVPModel index (0 = MAXN) |
| `jetson.gpu_memory_fraction` | float | 0.75 | GPU memory limit |
| `screening.top_n` | int | 20 | Top candidates to return |
| `screening.min_score` | float | 0.55 | Minimum composite score |

---

## Running Tests

```bash
pip3 install pytest
pytest tests/ -v
```

All tests run without GPU, TRT, or network access — safe for CI.

---

## Memory Budget (Jetson Orin Nano 8 GB)

| Component | Approximate Usage |
|-----------|------------------|
| OS + JetPack runtime | ~1.5 GB |
| Feature pipeline (40 tickers) | ~200 MB |
| LSTM model (fp16) | ~25 MB |
| TRT engine | ~80 MB |
| LightGBM model | ~15 MB |
| CUDA workspace | ~512 MB |
| **Total** | **~2.3 GB** |

Approximately **5.7 GB** remains free for other processes.

---

## Disclaimer

This software is for **research and educational purposes only**.  It does not
constitute financial advice.  Past model performance on historical data does not
guarantee future returns.  Always consult a licensed financial advisor before
making investment decisions.