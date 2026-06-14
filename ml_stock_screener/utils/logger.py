"""
Logging configuration helper.

Sets up a rotating file handler and a coloured console handler using the
settings from config.yaml.  Call ``setup_logging()`` once at startup.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

from ml_stock_screener.config import CFG, LOG_DIR

_logging_configured: bool = False


def setup_logging(level: Optional[str] = None) -> None:
    """Configure root logger with console + rotating file handlers."""
    global _logging_configured

    log_cfg = CFG["logging"]
    level_str = level or log_cfg["level"]
    numeric_level = getattr(logging, level_str.upper(), logging.INFO)

    log_path = LOG_DIR / Path(log_cfg["log_file"]).name

    root = logging.getLogger()
    root.setLevel(numeric_level)

    # If a level override is passed after initial setup, propagate it without
    # re-adding handlers.
    if _logging_configured:
        for handler in root.handlers:
            handler.setLevel(numeric_level)
        return

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(numeric_level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # Rotating file
    fh = logging.handlers.RotatingFileHandler(
        str(log_path),
        maxBytes=log_cfg["max_bytes"],
        backupCount=log_cfg["backup_count"],
    )
    fh.setLevel(numeric_level)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Silence noisy libraries
    for noisy in ("yfinance", "urllib3", "requests", "peewee"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _logging_configured = True
