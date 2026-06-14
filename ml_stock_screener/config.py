"""
Configuration loader.

Reads config.yaml from the project root and exposes a singleton
``CFG`` dict so every module can do ``from ml_stock_screener.config import CFG``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# Project root is two levels above this file:
#   <root>/ml_stock_screener/config.py  ->  <root>
_ROOT = Path(__file__).resolve().parent.parent


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (non-destructive)."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config, optionally merging a user-supplied override file."""
    default_path = _ROOT / "config.yaml"
    with open(default_path, "r") as fh:
        cfg: dict = yaml.safe_load(fh)

    if path is not None:
        with open(path, "r") as fh:
            override = yaml.safe_load(fh) or {}
        cfg = _deep_merge(cfg, override)

    # Allow environment-variable overrides for a handful of sensitive keys.
    # E.g.  SCREENER_UNIVERSE_TICKERS="AAPL,MSFT,GOOG" overrides cfg.universe.tickers
    env_tickers = os.environ.get("SCREENER_UNIVERSE_TICKERS")
    if env_tickers:
        cfg["universe"]["tickers"] = [t.strip() for t in env_tickers.split(",")]

    return cfg


# Singleton loaded at import time
CFG: dict[str, Any] = load_config()

# Convenience: absolute path helpers
ROOT: Path = _ROOT
CACHE_DIR: Path = _ROOT / CFG["data"]["cache_dir"]
LOG_DIR: Path = _ROOT / Path(CFG["logging"]["log_file"]).parent
MODELS_DIR: Path = _ROOT / "models"

# Ensure directories exist
for _d in (CACHE_DIR, LOG_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
