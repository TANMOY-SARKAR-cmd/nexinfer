"""ONNX Runtime backend.

Runs exported ONNX decoder models with ONNX Runtime execution providers.
Works out of the box on CPU (Windows + Linux), and with the right EP
packages on Intel iGPU/NPU (``onnxruntime-openvino`` on Windows),
NVIDIA GPU (``onnxruntime-gpu``), and WebNN.

Model files: a ``model.onnx`` with inputs ``input_ids`` (1, seq) and
outputs ``logits`` (1, seq, vocab), alongside a KV-cache variant if the
export exposes ``key_values`` inputs/outputs.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

from nexinfer.backends.base import Backend, BackendCapabilities, DeviceInfo, ModelSpec
from nexinfer.engine.types import DeviceId, DeviceKind

log = logging.getLogger("nexinfer.backends.ort")

try:
    import onnxruntime as ort

    _HAS_ORT = True
except ImportError:  # pragma: no cover
    _HAS_ORT = False


# device -> list of execution providers, in preference order
EP_MAP: dict[str, list[str]] = {
    "/cpu": ["CPUExecutionProvider"],
    "/gpu:intel": ["OpenVINOExecutionProvider", "CPUExecutionProvider"],
    "/npu:intel": ["OpenVINOExecutionProvider", "CPUExecutionProvider"],
    "/gpu:nvidia": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "/gpu:amd": ["ROCMExecutionProvider", "CPUExecutionProvider"],
}


class OrtBackend(Backend):
    name = "ort"
    platform = "any"

    def __init__(self) -> None:
        self.session: object | None = None
        self.spec: ModelSpec | None = None
        self._kv_mode: bool = False
        self._kv_cache: dict[str, dict[str, np.ndarray]] = {}

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_gpu=True,
            supports_tpu=False,
            supports_quant=False,
            supports_tool_calls=False,
            supports_pipeline_parallel=False,
            supports_tensor_parallel=False,
            extra={"execution_providers": list(EP_MAP)},
        )

    def detect_devices(self) -> list[DeviceInfo]:
        if not _HAS_ORT:
            return []
        devices = [DeviceInfo("/cpu:0", DeviceKind.CPU, "generic", "CPU (onnxruntime)", 0, 2.5)]
        avail = set(ort.get_available_providers()) if _HAS_ORT else set()
        if "OpenVINOExecutionProvider" in avail:
            devices.append(
                DeviceInfo("/gpu:intel:0", DeviceKind.GPU_INTEL, "intel",
                           "Intel iGPU (OpenVINO)", 4 * 1024 ** 3, 3.0)
            )
        if "CUDAExecutionProvider" in avail:
            devices.append(
                DeviceInfo("/gpu:nvidia:0", DeviceKind.GPU_NVIDIA, "nvidia",
                           "NVIDIA GPU (onnxruntime-gpu)", 8 * 1024 ** 3, 6.0)
            )
        return devices

    def load(self, model_path_or_id: str, spec: ModelSpec, devices: list[DeviceId]) -> None:
        if not _HAS_ORT:
            raise RuntimeError("onnxruntime is not installed; run `pip install nexinfer[ort]`")
        self.spec = spec
        model_file = model_path_or_id if os.path.isfile(model_path_or_id) else None
        if model_file is None and os.path.isdir(model_path_or_id):
            model_file = next((os.path.join(model_path_or_id, f) for f in os.listdir(model_path_or_id)
                               if f.endswith(".onnx")), None)
        if model_file is None:
            raise FileNotFoundError(f"no model.onnx found for {model_path_or_id}")

        # choose providers from the requested devices
        providers = ["CPUExecutionProvider"]
        for dev in devices:
            for prefix, eps in EP_MAP.items():
                if dev.startswith(prefix):
                    providers = eps
                    break
        # filter to actually available EPs
        available = set(ort.get_available_providers())
        providers = [p for p in providers if p in available]
        log.info("ORT providers: %s", providers)

        self.session = ort.InferenceSession(model_file, providers=providers)
        self._kv_mode = any("key_values" in i.name for i in self.session.get_inputs())

    # ------------------------------------------------------------------

    def prefill(self, req_id: str, input_ids: np.ndarray) -> np.ndarray:
        assert self.session is not None
        inputs = {"input_ids": input_ids.reshape(1, -1).astype(np.int64)}
        if self._kv_mode:
            inputs.update(self._past_kv_inputs(req_id, input_ids.size))
        out = self.session.run(None, inputs)
        logits = np.asarray(out[0])
        if self._kv_mode:
            self._capture_present(req_id, out)
        return logits[0, -1:]

    def decode(self, req_ids: list[str], input_ids: np.ndarray) -> np.ndarray:
        assert self.session is not None
        out = []
        for req_id, tok in zip(req_ids, input_ids):
            inputs = {"input_ids": np.array([[tok.item()]], dtype=np.int64)}
            if self._kv_mode:
                inputs.update(self._past_kv_inputs(req_id, None))
            res = self.session.run(None, inputs)
            out.append(np.asarray(res[0])[0, -1:])
            if self._kv_mode:
                self._capture_present(req_id, res)
        return np.concatenate(out, axis=0)

    # ------------------------------------------------------------------

    def _past_kv_inputs(self, req_id: str, seq_len: int | None) -> dict[str, np.ndarray]:
        cache = self._kv_cache.get(req_id, {})
        inputs = {}
        for i in self.session.get_inputs():
            if i.name.endswith("_key_values"):
                idx = int(i.name.rsplit("_", 2)[-2])
                inputs[i.name] = cache.get(idx, np.zeros((1, 0, self.spec.num_kv_heads, self.spec.head_dim), dtype=np.float16))
        return inputs

    def _capture_present(self, req_id: str, outputs: list) -> None:
        cache = self._kv_cache.setdefault(req_id, {})
        out_names = [o.name for o in self.session.get_outputs()]
        for i, name in enumerate(out_names):
            if name.endswith("_present"):
                cache[int(name.rsplit("_", 2)[-2])] = np.asarray(outputs[i])

    def close(self) -> None:
        self.session = None
        self._kv_cache.clear()
