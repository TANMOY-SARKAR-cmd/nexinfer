"""ROCm backend (AMD GPUs).

Drives AMD discrete GPUs (RX/MI series) through ONNX Runtime's
``ROCMExecutionProvider`` (ROCm >= 6.0) or the MIGraphX/MIOpen path.
Detection uses ``rocm-smi`` or ``lsgpu``. On systems without AMD hardware
or the ``onnxruntime-rocm`` wheel the backend reports zero devices and
is silently skipped by the registry.
"""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
from typing import Optional

import numpy as np

from nexinfer.backends.base import Backend, BackendCapabilities, DeviceInfo, ModelSpec
from nexinfer.engine.types import DeviceId, DeviceKind

log = logging.getLogger("nexinfer.backends.rocm")

try:
    import onnxruntime as ort

    _HAS_ORT = True
except ImportError:  # pragma: no cover
    _HAS_ORT = False


def _detect_amd_gpus() -> list[tuple[int, str, float]]:
    if platform.system() != "Linux":
        return []
    out = []
    # rocm-smi JSON
    if shutil.which("rocm-smi") is not None:
        try:
            import json
            r = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--json"],
                               capture_output=True, text=True, timeout=3.0)
            if r.returncode == 0:
                data = json.loads(r.stdout)
                for key, info in (data.get("system", {}).get("GPU", {}) or {}).items():
                    idx = int("".join(c for c in key if c.isdigit()) or "0")
                    vram_mb = int(info.get("VRAM Total Memory (B)", 0) or 0) / (1024 * 1024)
                    out.append((idx, info.get("Card series", key), vram_mb))
        except Exception:  # noqa: BLE001
            pass
    if not out and shutil.which("lsgpu") is not None:
        try:
            r = subprocess.run(["lsgpu"], capture_output=True, text=True, timeout=3.0)
            if r.returncode == 0 and "AMD" in r.stdout:
                out.append((0, "AMD GPU (lsgpu)", 8192.0))
        except Exception:  # noqa: BLE001
            pass
    return out


class RocmBackend(Backend):
    name = "rocm"
    platform = "linux"

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
            extra={"requires": "onnxruntime-rocm wheel + ROCm >= 6.0"},
        )

    def detect_devices(self) -> list[DeviceInfo]:
        out = []
        for idx, name, mem_mb in _detect_amd_gpus():
            out.append(
                DeviceInfo(f"/gpu:amd:{idx}", DeviceKind.GPU_AMD, "amd",
                           name, int(mem_mb * 1024 * 1024), mem_mb / 1024)
            )
        return out

    def load(self, model_path_or_id: str, spec: ModelSpec, devices: list[DeviceId]) -> None:
        if not _HAS_ORT:
            raise RuntimeError("onnxruntime not installed")
        if "ROCMExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError(
                "ROCMExecutionProvider unavailable; install the onnxruntime-rocm wheel "
                "built against ROCm >= 6.0"
            )
        self.session = ort.InferenceSession(
            model_path_or_id,
            providers=["ROCMExecutionProvider", "CPUExecutionProvider"],
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
