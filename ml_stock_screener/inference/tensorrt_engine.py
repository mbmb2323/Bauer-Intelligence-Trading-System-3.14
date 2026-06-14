"""
TensorRT inference engine for NVIDIA Jetson Orin Nano 8GB.

Workflow
--------
1. The LSTM model is trained in PyTorch and exported to ONNX.
2. ``TRTEngine.build_from_onnx`` converts the ONNX model to a serialised
   TensorRT engine, applying FP16 precision (native on Ampere).
3. At inference time ``TRTEngine.infer`` runs the engine on CUDA and
   returns class probabilities as a NumPy array.

When TensorRT is not available (e.g., on an x86 development machine),
``TRTEngine`` transparently falls back to ONNX Runtime (CPU).

Jetson-specific notes
---------------------
* TensorRT and its Python bindings (``tensorrt``) are shipped with JetPack.
  Do not install them via pip on Jetson — use the system Python packages.
* The engine file is device-specific: regenerate it when moving between
  different Jetson SKUs or after a JetPack upgrade.
* Workspace memory is limited to ``tensorrt.workspace_mb`` from config.yaml
  (default 512 MB) to leave room for the rest of the pipeline in the 8 GB
  unified memory pool.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ml_stock_screener.config import CFG, MODELS_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — TensorRT (Jetson JetPack) and ONNX Runtime (fallback)
# ---------------------------------------------------------------------------

try:
    import tensorrt as trt  # type: ignore
    import pycuda.autoinit  # type: ignore  # noqa: F401
    import pycuda.driver as cuda  # type: ignore

    _HAS_TRT = True
    _TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
except ImportError:
    _HAS_TRT = False
    logger.info("TensorRT not available; will use ONNX Runtime fallback.")

try:
    import onnxruntime as ort  # type: ignore

    _HAS_ORT = True
except ImportError:
    _HAS_ORT = False

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# TRT engine wrapper
# ---------------------------------------------------------------------------

class TRTEngine:
    """
    Wraps a serialised TensorRT engine for efficient LSTM inference on Jetson.

    Parameters
    ----------
    engine_path : path to the ``.engine`` file
    """

    def __init__(self, engine_path: Path) -> None:
        self._path = engine_path
        self._engine = None
        self._context = None
        self._input_binding: Optional[int] = None
        self._output_binding: Optional[int] = None
        self._input_shape: Optional[tuple] = None
        self._output_shape: Optional[tuple] = None
        self._d_input = None
        self._d_output = None
        self._d_input_nbytes: int = 0
        self._d_output_nbytes: int = 0
        self._stream = None
        self._h_output: Optional[np.ndarray] = None

        if _HAS_TRT:
            self._load_engine()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    @classmethod
    def build_from_onnx(
        cls,
        onnx_path: Path,
        engine_path: Optional[Path] = None,
        precision: Optional[str] = None,
        max_batch_size: Optional[int] = None,
        workspace_mb: Optional[int] = None,
        force_rebuild: Optional[bool] = None,
    ) -> "TRTEngine":
        """
        Convert an ONNX model to a TensorRT engine and save it.

        Parameters
        ----------
        onnx_path    : ONNX model file
        engine_path  : where to save the ``.engine`` file
        precision    : "fp32" | "fp16" | "int8"
        max_batch_size, workspace_mb, force_rebuild : override config.yaml
        """
        trt_cfg = CFG["tensorrt"]
        engine_path = engine_path or MODELS_DIR / trt_cfg["engine_path"].split("/")[-1]
        precision = precision or trt_cfg["precision"]
        max_batch_size = max_batch_size or trt_cfg["max_batch_size"]
        workspace_mb = workspace_mb or trt_cfg["workspace_mb"]
        force_rebuild = force_rebuild if force_rebuild is not None else trt_cfg["force_rebuild"]

        if not force_rebuild and engine_path.exists():
            if engine_path.stat().st_mtime >= onnx_path.stat().st_mtime:
                logger.info("TRT engine is up-to-date: %s", engine_path)
                return cls(engine_path)

        if not _HAS_TRT:
            raise RuntimeError(
                "TensorRT is not installed. "
                "On Jetson, TRT is provided by JetPack — do not pip-install it. "
                "On x86, install tensorrt via pip."
            )

        logger.info(
            "Building TRT engine: precision=%s  max_batch=%d  workspace=%dMB",
            precision, max_batch_size, workspace_mb,
        )

        with trt.Builder(_TRT_LOGGER) as builder, \
             builder.create_network(
                 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
             ) as network, \
             trt.OnnxParser(network, _TRT_LOGGER) as parser:

            config = builder.create_builder_config()
            config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE,
                workspace_mb * (1 << 20),
            )

            if precision == "fp16" and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("  FP16 mode enabled (Ampere/Orin native).")
            elif precision == "int8":
                config.set_flag(trt.BuilderFlag.INT8)
                logger.warning("  INT8 requires calibration — set up a calibrator.")

            # Dynamic batch axis
            profile = builder.create_optimization_profile()
            # We need to know input name & shape from the ONNX model
            import onnx  # type: ignore

            onnx_model = onnx.load(str(onnx_path))
            inp = onnx_model.graph.input[0]
            dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            # dims = [0 (dynamic batch), seq_len, n_features]
            seq_len = dims[1]
            n_feat = dims[2]

            profile.set_shape(
                inp.name,
                min=(1, seq_len, n_feat),
                opt=(max_batch_size // 2, seq_len, n_feat),
                max=(max_batch_size, seq_len, n_feat),
            )
            config.add_optimization_profile(profile)

            with open(str(onnx_path), "rb") as f:
                if not parser.parse(f.read()):
                    for i in range(parser.num_errors):
                        logger.error("ONNX parse error: %s", parser.get_error(i))
                    raise RuntimeError("Failed to parse ONNX model.")

            serialised_engine = builder.build_serialized_network(network, config)
            if serialised_engine is None:
                raise RuntimeError("TRT engine build failed.")

            with open(str(engine_path), "wb") as f:
                f.write(serialised_engine)
            logger.info("TRT engine saved to %s", engine_path)

        return cls(engine_path)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _load_engine(self) -> None:
        runtime = trt.Runtime(_TRT_LOGGER)
        with open(str(self._path), "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        self._context = self._engine.create_execution_context()
        logger.info("TRT engine loaded: %s", self._path)

        # Inspect bindings
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            mode = self._engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self._input_binding = i
                self._input_name = name
            else:
                self._output_binding = i
                self._output_name = name

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(self, x: np.ndarray) -> np.ndarray:
        """
        Run inference on a batch of sequences.

        Parameters
        ----------
        x : float32 array  shape (N, seq_len, n_features)

        Returns
        -------
        proba : float32 array  shape (N, n_classes)  — softmax probabilities
        """
        if not _HAS_TRT or self._engine is None:
            return self._ort_infer(x)

        x = x.astype(np.float32)
        batch = x.shape[0]

        self._context.set_input_shape(self._input_name, x.shape)

        # Compute required buffer sizes
        nbytes_in = x.nbytes
        out_shape = self._context.get_tensor_shape(self._output_name)
        out_shape = tuple(batch if d < 0 else d for d in out_shape)
        h_out = np.empty(out_shape, dtype=np.float32)
        nbytes_out = h_out.nbytes

        # Reuse device buffers; reallocate only when the batch grows beyond
        # the previously allocated capacity.
        if self._d_input is None or nbytes_in > self._d_input_nbytes:
            self._d_input = cuda.mem_alloc(nbytes_in)
            self._d_input_nbytes = nbytes_in
        if self._d_output is None or nbytes_out > self._d_output_nbytes:
            self._d_output = cuda.mem_alloc(nbytes_out)
            self._d_output_nbytes = nbytes_out

        # Create the CUDA stream once per engine instance
        if self._stream is None:
            self._stream = cuda.Stream()

        cuda.memcpy_htod(self._d_input, x)
        self._context.set_tensor_address(self._input_name, int(self._d_input))
        self._context.set_tensor_address(self._output_name, int(self._d_output))

        self._context.execute_async_v3(stream_handle=self._stream.handle)
        self._stream.synchronize()

        cuda.memcpy_dtoh(h_out, self._d_output)

        # Softmax (logits -> probabilities)
        return _softmax(h_out)

    def _ort_infer(self, x: np.ndarray) -> np.ndarray:
        """ONNX Runtime CPU fallback."""
        if not _HAS_ORT:
            raise RuntimeError("Neither TensorRT nor ONNX Runtime is available.")
        # Lazy-load ORT session
        if not hasattr(self, "_ort_session"):
            onnx_path = MODELS_DIR / "lstm.onnx"
            if not onnx_path.exists():
                raise FileNotFoundError(
                    f"ONNX model not found at {onnx_path}. "
                    "Export the model first with `python train.py`."
                )
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 4
            self._ort_session = ort.InferenceSession(
                str(onnx_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self._ort_input_name = self._ort_session.get_inputs()[0].name
            logger.info("ONNX Runtime session created (CPU fallback).")
        logits = self._ort_session.run(
            None, {self._ort_input_name: x.astype(np.float32)}
        )[0]
        return _softmax(logits)


# ---------------------------------------------------------------------------
# PyTorch GPU inference (used when TRT engine is not yet built)
# ---------------------------------------------------------------------------

def pytorch_inference_fn(model) -> callable:
    """
    Return an inference callable wrapping a PyTorch model.

    The callable converts NumPy input to a CUDA tensor, runs inference
    under ``torch.no_grad()``, and returns NumPy probabilities.
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is required for pytorch_inference_fn.")

    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device

    def _infer(x: np.ndarray) -> np.ndarray:
        t = torch.tensor(x, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = model(t)
            proba = F.softmax(logits, dim=-1)
        return proba.cpu().numpy().astype(np.float32)

    return _infer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    x = x - x.max(axis=-1, keepdims=True)
    ex = np.exp(x)
    return (ex / ex.sum(axis=-1, keepdims=True)).astype(np.float32)
