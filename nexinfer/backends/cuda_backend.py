"""CUDA backend (NVIDIA GPUs).

Full driver-backend template for NVIDIA GPUs. Two implementation paths
are supported:

1. **ONNX Runtime GPU** -- ``pip install nexinfer[ort]`` plus the
   ``onnxruntime-gpu`` wheel. Load an exported ``model.onnx`` and run
   it on the GPU through the CUDAExecutionProvider.
2. **Native CUDA kernels** -- for production workloads, integrate the
   ONNX model with a CUDA graph runner (``model.onnx`` + custom
   kernels) or replace this class with a binding to vLLM/TensorRT-LLM
   internals for maximum throughput.

Detects NVIDIA GPUs via ``nvidia-smi`` (no CUDA toolkit required for
detection).
"""

from __future__ import annotations

import logging
import shutil
import subprocess

import numpy as np

from nexinfer.backends.base import Backend, BackendCapabilities, DeviceInfo, ModelSpec
from nexinfer.engine.types import DeviceId, DeviceKind

log = logging.getLogger("nexinfer.backends.cuda")

try:
    import onnxruntime as ort

    _HAS_ORT = True
except ImportError:  # pragma: no cover
    _HAS_ORT = False


def _detect_nvidia_gpus() -> list[tuple[int, str, float]]:
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3.0,
        )
        if r.returncode != 0:
            return []
        out = []
        for i, line in enumerate(r.stdout.strip().splitlines()):
            parts = [p.strip() for p in line.split(",")]
            name = parts[1] if len(parts) > 1 else f"NVIDIA GPU {i}"
            mem_mb = float(parts[2]) if len(parts) > 2 else 0.0
            out.append((i, name, mem_mb))
        return out
    except Exception:
        return []


class CudaBackend(Backend):
    name = "cuda"
    platform = "any"

    def __init__(self) -> None:
        self.session: object | None = None
        self.spec: ModelSpec | None = None

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_gpu=True,
            supports_tpu=False,
            supports_quant=False,
            supports_tool_calls=False,
            supports_pipeline_parallel=True,
            supports_tensor_parallel=True,
            extra={"note": "install onnxruntime-gpu and NVIDIA drivers >= 525"},
        )

    def detect_devices(self) -> list[DeviceInfo]:
        out = []
        for idx, name, mem_mb in _detect_nvidia_gpus():
            out.append(
                DeviceInfo(
                    f"/gpu:nvidia:{idx}",
                    DeviceKind.GPU_NVIDIA,
                    "nvidia",
                    name,
                    int(mem_mb * 1024 * 1024),
                    mem_mb / 1024,
                )
            )
        return out

    def load(self, model_path_or_id: str, spec: ModelSpec, devices: list[DeviceId]) -> None:
        if not _HAS_ORT:
            raise RuntimeError("onnxruntime (with GPU support) not installed")
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError(
                "CUDAExecutionProvider unavailable; install the onnxruntime-gpu wheel "
                "and ensure NVIDIA drivers >= 525 and CUDA 11.8+ are present"
            )
        self.session = ort.InferenceSession(
            model_path_or_id,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
