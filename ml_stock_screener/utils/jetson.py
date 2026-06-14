"""
Jetson Orin Nano 8GB hardware utilities.

Provides helpers for:
- Setting the NVPModel power mode (MAXN for full performance)
- Reporting GPU/CPU/memory stats via tegrastats
- Logging Jetson SoC information at startup
- Configuring CUDA memory growth limits for the unified 8 GB pool

These utilities are designed to be called once at startup before the
screening pipeline begins.  They degrade gracefully on non-Jetson
hardware (x86 development machines) so the rest of the codebase is
unaffected.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

def is_jetson() -> bool:
    """Return True if running on an NVIDIA Jetson device."""
    for path in ("/etc/nv_tegra_release", "/proc/device-tree/compatible"):
        if os.path.exists(path):
            return True
    return False


def get_jetson_info() -> Dict[str, str]:
    """Read basic Jetson platform info."""
    info: Dict[str, str] = {}
    try:
        with open("/etc/nv_tegra_release") as fh:
            info["tegra_release"] = fh.read().strip()
    except FileNotFoundError:
        pass

    try:
        result = subprocess.run(
            ["cat", "/proc/device-tree/model"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            info["model"] = result.stdout.strip().rstrip("\x00")
    except Exception:
        pass

    return info


# ---------------------------------------------------------------------------
# Power mode
# ---------------------------------------------------------------------------

def set_power_mode(mode_id: int = 0) -> bool:
    """
    Set NVPModel power mode.

    Parameters
    ----------
    mode_id : int
        0 = MAXN (full performance, all CPU+GPU cores at max clock)
        1 = 10W
        2 = 15W
        3 = 20W

    Returns True on success, False if nvpmodel is unavailable.
    """
    try:
        result = subprocess.run(
            ["nvpmodel", "-m", str(mode_id)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            logger.info("NVPModel set to mode %d (MAXN=0).", mode_id)
            return True
        logger.warning("nvpmodel returned non-zero: %s", result.stderr.strip())
    except FileNotFoundError:
        logger.debug("nvpmodel not found (not running on Jetson).")
    except Exception as exc:
        logger.warning("Failed to set power mode: %s", exc)
    return False


def set_max_clocks() -> bool:
    """
    Run ``jetson_clocks`` to pin CPU/GPU/EMC clocks at maximum frequency.
    Requires root.  Returns True on success.
    """
    try:
        result = subprocess.run(
            ["jetson_clocks"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            logger.info("jetson_clocks: clocks pinned to maximum.")
            return True
        logger.warning("jetson_clocks failed: %s", result.stderr.strip())
    except FileNotFoundError:
        logger.debug("jetson_clocks not found.")
    except Exception as exc:
        logger.warning("Failed to run jetson_clocks: %s", exc)
    return False


# ---------------------------------------------------------------------------
# Memory & GPU stats
# ---------------------------------------------------------------------------

def get_gpu_memory_mb() -> Optional[float]:
    """Return available GPU memory in MB using nvidia-smi (if available)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return float(result.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return None


def get_system_memory_mb() -> Optional[float]:
    """Return total system memory in MB by parsing /proc/meminfo."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb / 1024.0
    except Exception:
        pass
    return None


def log_hardware_stats() -> None:
    """Log a concise hardware summary at startup."""
    info = get_jetson_info()
    if info:
        logger.info("Jetson platform: %s", info.get("model", "unknown"))
        if "tegra_release" in info:
            logger.info("JetPack: %s", info["tegra_release"])
    else:
        logger.info("Not running on Jetson (development mode).")

    gpu_mb = get_gpu_memory_mb()
    sys_mb = get_system_memory_mb()
    if gpu_mb is not None:
        logger.info("GPU memory free : %.0f MB", gpu_mb)
    if sys_mb is not None:
        logger.info("System memory available: %.0f MB", sys_mb)

    # CUDA info via torch if available
    try:
        import torch

        if torch.cuda.is_available():
            dev = torch.cuda.get_device_properties(0)
            logger.info(
                "CUDA device: %s  SM%d.%d  %d MB total",
                dev.name,
                dev.major,
                dev.minor,
                dev.total_memory // (1024 ** 2),
            )
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# CUDA memory configuration
# ---------------------------------------------------------------------------

def configure_cuda_memory(fraction: Optional[float] = None) -> None:
    """
    Set the fraction of GPU memory PyTorch is allowed to use.

    On the Jetson Orin Nano the CPU and GPU share the same 8 GB LPDDR5
    pool.  Capping PyTorch's allocation leaves room for the OS, LightGBM,
    and the feature pipeline.

    Parameters
    ----------
    fraction : 0.0–1.0  (default from config.yaml: jetson.gpu_memory_fraction)
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return

        from ml_stock_screener.config import CFG

        frac = fraction or CFG["jetson"]["gpu_memory_fraction"]
        torch.cuda.set_per_process_memory_fraction(frac, device=0)
        logger.info("CUDA memory fraction set to %.0f%%", frac * 100)
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("Could not set CUDA memory fraction: %s", exc)


# ---------------------------------------------------------------------------
# Startup initialisation
# ---------------------------------------------------------------------------

def jetson_init(power_mode: Optional[int] = None) -> None:
    """
    One-call Jetson initialisation.

    Call this at the top of ``main.py`` before any ML work begins.
    Safe to call on non-Jetson hardware — all steps degrade gracefully.
    """
    from ml_stock_screener.config import CFG

    log_hardware_stats()

    if is_jetson():
        mode = power_mode if power_mode is not None else CFG["jetson"]["power_mode"]
        set_power_mode(mode)
        set_max_clocks()

    configure_cuda_memory()
