"""DirectML backend for Windows GPU acceleration.

Windows laptops without CUDA/ROCm can still accelerate inference through
DirectML (works on NVIDIA, AMD, and Intel GPUs via DirectX 12). Backed by
``onnxruntime-directml``, which exposes a ``DmlExecutionProvider``.

This backend is Windows-only; on Linux it reports no devices and the
registry skips it automatically.
"""

from __future__ import annotations

import logging
import platform
from typing import Optional

import numpy as np

from nexinfer.backends.base import Backend, BackendCapabilities, DeviceInfo, ModelSpec
from nexinfer.engine.types import DeviceId, DeviceKind

log = logging.getLogger("nexinfer.backends.directml")

try:
    import onnxruntime as ort

    _HAS_ORT = True
except ImportError:  # pragma: no cover
    _HAS_ORT = False

_WIN = platform.system().lower() == "windows"


def _has_directml() -> bool:
    return _HAS_ORT and "DmlExecutionProvider" in ort.get_available_providers()


class DirectMLBackend(Backend):
    name = "directml"
    platform = "windows"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_gpu=True,
            supports_tpu=False,
            supports_quant=False,
            supports_tool_calls=False,
            supports_pipeline_parallel=False,
            supports_tensor_parallel=False,
            extra={"requires": "onnxruntime-directml (Windows)"},
        )

    def detect_devices(self) -> list[DeviceInfo]:
        if not _WIN or not _has_directml():
            return []
        return [
            DeviceInfo("/gpu:directml:0", DeviceKind.GPU_NVIDIA, "microsoft",
                       "DirectML Device (DX12)", 8 * 1024 ** 3, 4.0)
        ]

    def load(self, model_path_or_id: str, spec: ModelSpec, devices: list[DeviceId]) -> None:
        import os
        if not _WIN:
            raise RuntimeError("DirectML backend requires Windows")
        if not _has_directml():
            raise RuntimeError("onnxruntime-directml not available; install onnxruntime-directml on Windows")
        if not os.path.isfile(model_path_or_id):
            raise FileNotFoundError(f"model.onnx not found at {model_path_or_id}")
        self.session = ort.InferenceSession(
            model_path_or_id,
            providers=["DmlExecutionProvider", "CPUExecutionProvider"],
        )
        self.spec = spec

    def prefill(self, req_id: str, input_ids: np.ndarray) -> np.ndarray:
        out = self.session.run(None, {"input_ids": input_ids.reshape(1, -1).astype(np.int64)})
        return np.asarray(out[0])[0, -1:]

    def decode(self, req_ids: list[str], input_ids: np.ndarray) -> np.ndarray:
        out = []
        for tok in input_ids:
            res = self.session.run(None, {"input_ids": np.array([[tok.item()]], dtype=np.int64)})
            out.append(np.asarray(res[0])[0, -1:])
        return np.concatenate(out, axis=0)

    def close(self) -> None:
        self.session = None
