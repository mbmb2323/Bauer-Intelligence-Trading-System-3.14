"""
PyTorch LSTM model for stock-direction prediction.

Architecture
------------
Input  : (batch, seq_len, n_features)  float32
LSTM   : configurable hidden_size, num_layers, optional bidirectional
Head   : Linear → output_size (3 classes: DOWN / NEUTRAL / UP)

The model is intentionally kept lightweight so that:
  - It fits comfortably in the Jetson Orin Nano's 8 GB unified memory
    alongside the feature pipeline and LightGBM model.
  - The ONNX export is simple (no dynamic control flow) enabling
    clean TensorRT conversion.

Usage
-----
    from ml_stock_screener.models.lstm_model import StockLSTM, train_model
    model = StockLSTM(input_size=42)
    train_model(model, X_train, y_train, X_val, y_val)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    logger.warning("PyTorch not available; LSTM model disabled.")

from ml_stock_screener.config import CFG, MODELS_DIR


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class StockLSTM(nn.Module if _HAS_TORCH else object):
    """Bidirectional-capable LSTM classifier for stock direction."""

    def __init__(
        self,
        input_size: int,
        hidden_size: Optional[int] = None,
        num_layers: Optional[int] = None,
        dropout: Optional[float] = None,
        bidirectional: Optional[bool] = None,
        output_size: Optional[int] = None,
    ) -> None:
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch is required to use StockLSTM.")
        super().__init__()

        lstm_cfg = CFG["model"]["lstm"]
        self.hidden_size = hidden_size or lstm_cfg["hidden_size"]
        self.num_layers = num_layers or lstm_cfg["num_layers"]
        self.bidirectional = bidirectional if bidirectional is not None else lstm_cfg["bidirectional"]
        out_size = output_size or lstm_cfg["output_size"]
        drop = dropout if dropout is not None else lstm_cfg["dropout"]

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=drop if self.num_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
        )

        lstm_out_dim = self.hidden_size * (2 if self.bidirectional else 1)

        self.head = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Dropout(drop),
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.GELU(),
            nn.Dropout(drop / 2),
            nn.Linear(lstm_out_dim // 2, out_size),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (B, T, F)
        lstm_out, _ = self.lstm(x)   # (B, T, H)
        last = lstm_out[:, -1, :]    # take final timestep
        return self.head(last)       # (B, out_size)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(
    model: "StockLSTM",
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    save_path: Optional[Path] = None,
) -> dict:
    """
    Train the LSTM model and return training history.

    Uses the Jetson's CUDA GPU when available (torch.cuda.is_available()).
    FP16 mixed-precision training is enabled automatically on Ampere.

    Parameters
    ----------
    model : StockLSTM instance
    X_train, y_train : training sequences and labels
    X_val, y_val     : validation sequences and labels
    save_path        : where to save the best checkpoint (.pt)

    Returns
    -------
    dict with keys: train_loss, val_loss, val_acc, best_epoch
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is required.")

    train_cfg = CFG["model"]["training"]
    epochs = train_cfg["epochs"]
    bs = train_cfg["batch_size"]
    lr = train_cfg["learning_rate"]
    wd = train_cfg["weight_decay"]
    patience = train_cfg["early_stopping_patience"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)
    model = model.to(device)

    # Use FP16 autocast on CUDA (Jetson Ampere supports it natively)
    use_amp = device.type == "cuda"
    scaler_amp = torch.amp.GradScaler("cuda") if use_amp else None

    def _to_tensor(arr, dtype):
        return torch.tensor(arr, dtype=dtype)

    ds_train = TensorDataset(
        _to_tensor(X_train, torch.float32),
        _to_tensor(y_train, torch.long),
    )
    ds_val = TensorDataset(
        _to_tensor(X_val, torch.float32),
        _to_tensor(y_val, torch.long),
    )

    pin = CFG["jetson"]["pin_memory"] and device.type == "cuda"
    loader_train = DataLoader(ds_train, batch_size=bs, shuffle=True, pin_memory=pin, num_workers=2)
    loader_val = DataLoader(ds_val, batch_size=bs * 2, shuffle=False, pin_memory=pin, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_epoch = 0
    patience_ctr = 0
    save_path = save_path or MODELS_DIR / "lstm_weights.pt"

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_losses = []

        for xb, yb in loader_train:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optimiser)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler_amp.step(optimiser)
                scaler_amp.update()
            else:
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()

            train_losses.append(loss.item())

        scheduler.step()

        # Validation
        model.eval()
        val_losses, correct, total = [], 0, 0
        with torch.no_grad():
            for xb, yb in loader_val:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        logits = model(xb)
                        loss = criterion(logits, yb)
                else:
                    logits = model(xb)
                    loss = criterion(logits, yb)
                val_losses.append(loss.item())
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += len(yb)

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        val_acc = correct / total if total > 0 else 0.0
        elapsed = time.time() - t0

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_acc"].append(val_acc)

        logger.info(
            "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.3f  (%.1fs)",
            epoch, epochs, avg_train, avg_val, val_acc, elapsed,
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch
            patience_ctr = 0
            torch.save(model.state_dict(), save_path)
            logger.debug("  ↑ New best model saved to %s", save_path)
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                logger.info("Early stopping at epoch %d (best epoch %d)", epoch, best_epoch)
                break

    history["best_epoch"] = best_epoch
    logger.info("Training complete. Best epoch %d  val_loss=%.4f", best_epoch, best_val_loss)
    return history


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_model(input_size: int, weights_path: Optional[Path] = None) -> "StockLSTM":
    """Load a StockLSTM from a saved checkpoint."""
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is required.")
    model = StockLSTM(input_size=input_size)
    path = weights_path or MODELS_DIR / CFG["model"]["weights_path"].split("/")[-1]
    state = torch.load(str(path), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    logger.info("Loaded LSTM weights from %s", path)
    return model


def export_to_onnx(
    model: "StockLSTM",
    seq_len: int,
    n_features: int,
    onnx_path: Optional[Path] = None,
) -> Path:
    """
    Export the trained LSTM to ONNX for TensorRT conversion.

    The exported model takes a single input tensor of shape
    (batch, seq_len, n_features) and returns logits of shape (batch, 3).
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is required.")

    import torch.onnx

    onnx_path = onnx_path or MODELS_DIR / "lstm.onnx"
    device = next(model.parameters()).device
    dummy = torch.randn(1, seq_len, n_features, device=device)

    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy,),
            str(onnx_path),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
            do_constant_folding=True,
        )

    logger.info("ONNX model exported to %s", onnx_path)
    return onnx_path
