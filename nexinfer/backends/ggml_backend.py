"""GGML (llama.cpp) backend for quantized GGUF models on CPU (+ GPU offload).

Enables running 4-bit-quantized models (Q4_K_M etc.) that fit in the RAM
of low-end laptops. Requires the optional ``llama-cpp-python`` package,
installed via ``pip install nexinfer[ggml]``. When the optional
dependency is missing the backend reports zero devices and is skipped.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

from nexinfer.backends.base import Backend, BackendCapabilities, DeviceInfo, ModelSpec
from nexinfer.engine.types import DeviceId

log = logging.getLogger("nexinfer.backends.ggml")

try:
    from llama_cpp import Llama

    _HAS_LLAMA = True
except ImportError:  # pragma: no cover
    _HAS_LLAMA = False


class GGMLBackend(Backend):
    name = "ggml"
    platform = "any"

    def __init__(self) -> None:
        self.llama: object | None = None
        self.spec: ModelSpec | None = None
        self._logits: dict[str, np.ndarray] = {}

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_gpu=False,
            supports_tpu=False,
            supports_quant=True,  # GGUF q2..q8
            supports_tool_calls=True,  # llama.cpp grammar/tool support
            supports_pipeline_parallel=False,
            supports_tensor_parallel=False,
        )

    def detect_devices(self) -> list[DeviceInfo]:
        if not _HAS_LLAMA:
            return []
        from nexinfer.engine.types import DeviceKind
        devices = [
            DeviceInfo("/cpu:0", DeviceKind.CPU, "generic", "CPU (llama.cpp)", 0, 2.0)
        ]
        # detect NVIDIA via env/nvidia-smi for n_gpu_layers offload
        import shutil
        if shutil.which("nvidia-smi") is not None:
            import subprocess
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=3.0,
                )
                for i, line in enumerate(r.stdout.strip().splitlines()):
                    parts = [p.strip() for p in line.split(",")]
                    devices.append(
                        DeviceInfo(f"/gpu:nvidia:{i}", DeviceKind.GPU_NVIDIA, "nvidia",
                                   parts[1] if len(parts) > 1 else "NVIDIA GPU",
                                   int(float(parts[2]) * 1024 * 1024) if len(parts) > 2 else 0, 4.0)
                    )
            except Exception:  # noqa: BLE001
                pass
        return devices

    def load(self, model_path_or_id: str, spec: ModelSpec, devices: list[DeviceId]) -> None:
        if not _HAS_LLAMA:
            raise RuntimeError("llama-cpp-python is not installed; run `pip install nexinfer[ggml]`")
        self.spec = spec
        gpu_device = next((d for d in devices if d.device_id.startswith("/gpu")), None)
        n_gpu = 999 if gpu_device else 0
        self.llama = Llama(
            model_path=model_path_or_id,
            n_gpu_layers=n_gpu,
            n_ctx=4096,
            n_threads=max(1, (os.cpu_count() or 2) - 1),
            verbose=False,
        )

    def prefill(self, req_id: str, input_ids: np.ndarray) -> np.ndarray:
        assert self.llama is not None
        toks = input_ids.tolist()
        self.llama.eval(toks)
        logits = np.asarray(self.llama._ctx.logits, dtype=np.float32)
        self._logits[req_id] = logits
        return logits[-1:]

    def decode(self, req_ids: list[str], input_ids: np.ndarray) -> np.ndarray:
        assert self.llama is not None
        out = []
        for req_id, tok in zip(req_ids, input_ids):
            self.llama.eval([tok.item()])
            logits = np.asarray(self.llama._ctx.logits, dtype=np.float32)
            out.append(logits[-1:])
        return np.concatenate(out, axis=0)

    def close(self) -> None:
        self.llama = None
        self._logits.clear()
