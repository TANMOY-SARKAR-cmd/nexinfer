"""ONNX Runtime backend.

Runs exported ONNX decoder models with ONNX Runtime execution providers.
Works out of the box on CPU (Windows + Linux), and with the right EP
packages on Intel iGPU/NPU (``onnxruntime-openvino`` on Windows),
NVIDIA GPU (``onnxruntime-gpu``), and WebNN.

Two model variants are supported automatically:

1. **Stateless model** — inputs ``input_ids`` (1, seq) -> ``logits`` (1, seq,
   vocab). This is the export shape most toy/demo exporters emit; the
   backend tracks the full token history per request internally so decode
   only recomputes the needed positions.

2. **KV-cache model** — inputs ``input_ids`` (1, 1) + past-key tensors named
   ``key_values_0`` ... and outputs ``logits`` + present tensors named
   ``present_0`` .... Used by proper decoder exports (e.g., from Optimum:
   ``optimum-cli export onnx``). The backend keeps a true per-request KV
   cache and runs O(1) single-token forward passes.

The ``OrtModelBuilder`` in this module can also build minimal decoder test
models programmatically (for CI — no downloads required) via
``OrtModelBuilder.build_demo(path, spec)``.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from nexinfer.backends.base import Backend, BackendCapabilities, DeviceInfo, ModelSpec
from nexinfer.engine.types import DeviceId, DeviceKind

log = logging.getLogger("nexinfer.backends.ort")

try:
    import onnxruntime as ort

    _HAS_ORT = True
except ImportError:  # pragma: no cover
    _HAS_ORT = False

try:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    _HAS_ONNX = True
except ImportError:  # pragma: no cover
    _HAS_ONNX = False


# device -> list of execution providers, in preference order
EP_MAP: dict[str, list[str]] = {
    "/cpu": ["CPUExecutionProvider"],
    "/gpu:intel": ["OpenVINOExecutionProvider", "CPUExecutionProvider"],
    "/npu:intel": ["OpenVINOExecutionProvider", "CPUExecutionProvider"],
    "/gpu:nvidia": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "/gpu:amd": ["ROCMExecutionProvider", "CPUExecutionProvider"],
}


class OrtModelBuilder:
    """Builds minimal decoder-style ONNX models for testing and demos.

    Produces two files next to the model:

    - ``model.onnx`` — stateless export: ``input_ids`` (1, seq) -> ``logits`` (1, seq, vocab)
    - ``model_kv.onnx`` — KV-cache export: ``input_ids`` (1, 1) +
      ``key_values_0``..``key_values_3`` -> ``logits`` (1, 1, vocab) +
      ``present_0_key``/``present_0_value``/``present_1_key``/``present_1_value``

    The tiny model applies learned q/k/v projections and a learned readout,
    so it exercises the same data paths a real export does (it is
    deliberately NOT a faithful LLM — it exists to verify the backend's
    KV management).
    """

    @staticmethod
    def build_demo(model_dir: str, spec: ModelSpec) -> str:
        if not _HAS_ONNX:
            raise RuntimeError("onnx is required to build demo models; `pip install nexinfer[ort]`")
        os.makedirs(model_dir, exist_ok=True)
        OrtModelBuilder._build_stateless(os.path.join(model_dir, "model.onnx"), spec)
        OrtModelBuilder._build_kv_cache(os.path.join(model_dir, "model_kv.onnx"), spec)
        log.info("ORT demo models built in %s", model_dir)
        return model_dir

    @staticmethod
    def _rng(seed: int = 7):
        return np.random.default_rng(seed)

    @staticmethod
    def _build_stateless(path: str, spec: ModelSpec) -> None:
        rng = OrtModelBuilder._rng()
        h = spec.hidden_size

        wte = rng.standard_normal((spec.vocab_size, h)).astype(np.float32) * 0.02
        wq = rng.standard_normal((h, h)).astype(np.float32) * (h**-0.5)
        wout = rng.standard_normal((h, spec.vocab_size)).astype(np.float32) * 0.02
        bias = rng.standard_normal((spec.vocab_size,)).astype(np.float32) * 1e-3

        graph_inputs = [
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, "seq"]),
            helper.make_tensor_value_info("wte", TensorProto.FLOAT, [spec.vocab_size, h]),
            helper.make_tensor_value_info("wq", TensorProto.FLOAT, [h, h]),
            helper.make_tensor_value_info("wout", TensorProto.FLOAT, [h, spec.vocab_size]),
            helper.make_tensor_value_info("bias", TensorProto.FLOAT, [spec.vocab_size]),
        ]
        outputs = [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, "seq", spec.vocab_size])]

        nodes = [
            helper.make_node("Gather", ["wte", "input_ids"], ["emb"]),
            helper.make_node("MatMul", ["emb", "wq"], ["proj"]),
            helper.make_node("MatMul", ["proj", "wout"], ["raw"]),
            helper.make_node("Add", ["raw", "bias"], ["logits"]),
        ]

        graph = helper.make_graph(
            nodes,
            "tiny_stateless",
            graph_inputs,
            outputs,
            initializer=[
                numpy_helper.from_array(wte, "wte"),
                numpy_helper.from_array(wq, "wq"),
                numpy_helper.from_array(wout, "wout"),
                numpy_helper.from_array(bias, "bias"),
            ],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        onnx.checker.check_model(model)
        onnx.save(model, path)

    @staticmethod
    def _build_kv_cache(path: str, spec: ModelSpec) -> None:
        """Two-layer KV-cache decoder demo.

        Past KV inputs are named ``key_values_0``..``key_values_3`` (layer 0
        k/v, layer 1 k/v) and presents ``present_0_key``..``present_1_value``.
        """
        rng = OrtModelBuilder._rng()
        h = spec.hidden_size
        kh = spec.num_kv_heads
        dim = spec.head_dim

        wte = rng.standard_normal((spec.vocab_size, h)).astype(np.float32) * 0.02
        wk = rng.standard_normal((h, kh * dim)).astype(np.float32) * (h**-0.5)
        wout = rng.standard_normal((kh * dim, spec.vocab_size)).astype(np.float32) * 0.02
        bias = rng.standard_normal((spec.vocab_size,)).astype(np.float32) * 1e-3

        graph_inputs = [
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, "seq"]),
            helper.make_tensor_value_info("wk", TensorProto.FLOAT, [h, kh * dim]),
            helper.make_tensor_value_info("wout", TensorProto.FLOAT, [kh * dim, spec.vocab_size]),
            helper.make_tensor_value_info("bias", TensorProto.FLOAT, [spec.vocab_size]),
            helper.make_tensor_value_info("wte", TensorProto.FLOAT, [spec.vocab_size, h]),
            helper.make_tensor_value_info("flat_shape", TensorProto.INT64, [3]),
            helper.make_tensor_value_info("mean_shape", TensorProto.INT64, [2]),
            helper.make_tensor_value_info("neg1", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("big", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("axis1", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("start0", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("end0", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("end1", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("axis2", TensorProto.INT64, [1]),
        ]
        present_outputs = []
        kv_inputs = []
        for i in range(2):
            for kv in ("key", "value"):
                nm = f"{kv}_values_{i}"
                kv_inputs.append(
                    helper.make_tensor_value_info(nm, TensorProto.FLOAT, [1, "past", kh * dim // 2])
                )
                present_outputs.append(
                    helper.make_tensor_value_info(
                        f"present_{i}_{kv}", TensorProto.FLOAT, [1, "new_past", kh * dim // 2]
                    )
                )
        graph_inputs += kv_inputs

        nodes = [
            helper.make_node("Gather", ["wte", "input_ids"], ["emb"]),
            helper.make_node("MatMul", ["emb", "wk"], ["proj"]),
            helper.make_node("Slice", ["proj", "start0", "end0", "axis2"], ["new_0_k"]),
            helper.make_node("Slice", ["proj", "end0", "end1", "axis2"], ["new_0_v"]),
            helper.make_node("Slice", ["proj", "start0", "end0", "axis2"], ["new_1_k"]),
            helper.make_node("Slice", ["proj", "end0", "end1", "axis2"], ["new_1_v"]),
            helper.make_node("Reshape", ["new_0_k", "flat_shape"], ["new_0_kf"]),
            helper.make_node("Reshape", ["new_0_v", "flat_shape"], ["new_0_vf"]),
            helper.make_node("Concat", ["key_values_0", "new_0_kf"], ["present_0_key"], axis=1),
            helper.make_node("Concat", ["value_values_0", "new_0_vf"], ["present_0_value"], axis=1),
            helper.make_node("Reshape", ["new_1_k", "flat_shape"], ["new_1_kf"]),
            helper.make_node("Reshape", ["new_1_v", "flat_shape"], ["new_1_vf"]),
            helper.make_node("Concat", ["key_values_1", "new_1_kf"], ["present_1_key"], axis=1),
            helper.make_node("Concat", ["value_values_1", "new_1_vf"], ["present_1_value"], axis=1),
            helper.make_node("Concat", ["present_0_key", "present_0_value"], ["full_kv"], axis=2),
            helper.make_node("Slice", ["full_kv", "neg1", "big", "axis1"], ["last_k"]),
            helper.make_node("Squeeze", ["last_k", "axis1"], ["last_kf"]),
            helper.make_node("Reshape", ["last_kf", "mean_shape"], ["mean_kf"]),
            helper.make_node("MatMul", ["mean_kf", "wout"], ["raw"]),
            helper.make_node("Add", ["raw", "bias"], ["logits"]),
        ]

        starts0 = numpy_helper.from_array(np.array([0], dtype=np.int64), "start0")
        ends0 = numpy_helper.from_array(np.array([kh * dim // 2], dtype=np.int64), "end0")
        ends1 = numpy_helper.from_array(np.array([kh * dim], dtype=np.int64), "end1")
        axis2 = numpy_helper.from_array(np.array([2], dtype=np.int64), "axis2")
        flat_shape = numpy_helper.from_array(np.array([1, -1, kh * dim // 2], dtype=np.int64), "flat_shape")
        mean_shape = numpy_helper.from_array(np.array([1, kh * dim], dtype=np.int64), "mean_shape")
        neg1 = numpy_helper.from_array(np.array([-1], dtype=np.int64), "neg1")
        big = numpy_helper.from_array(np.array([2**63 - 1], dtype=np.int64), "big")
        axis1 = numpy_helper.from_array(np.array([1], dtype=np.int64), "axis1")

        logits_info = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, "seq", spec.vocab_size])

        graph = helper.make_graph(
            nodes,
            "tiny_kv",
            graph_inputs,
            [logits_info] + present_outputs,
            initializer=[
                numpy_helper.from_array(wk, "wk"),
                numpy_helper.from_array(wout, "wout"),
                numpy_helper.from_array(bias, "bias"),
                numpy_helper.from_array(wte, "wte"),
                starts0,
                ends0,
                ends1,
                axis2,
                flat_shape,
                mean_shape,
                neg1,
                big,
                axis1,
            ],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        onnx.checker.check_model(model)
        onnx.save(model, path)


class OrtBackend(Backend):
    """Functional ONNX Runtime backend with proper KV-cache management."""

    name = "ort"
    platform = "any"

    def __init__(self) -> None:
        self.session: ort.InferenceSession | None = None
        self.spec: ModelSpec | None = None
        self._kv_mode: bool = False
        self._kv_cache: dict[str, dict[str, np.ndarray]] = {}
        self._past_len: dict[str, int] = {}
        # stateless mode history so decode can still be incremental
        self._history: dict[str, list[int]] = {}

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
                DeviceInfo(
                    "/gpu:intel:0", DeviceKind.GPU_INTEL, "intel", "Intel iGPU (OpenVINO)", 4 * 1024**3, 3.0
                )
            )
        if "CUDAExecutionProvider" in avail:
            devices.append(
                DeviceInfo(
                    "/gpu:nvidia:0",
                    DeviceKind.GPU_NVIDIA,
                    "nvidia",
                    "NVIDIA GPU (onnxruntime-gpu)",
                    8 * 1024**3,
                    6.0,
                )
            )
        return devices

    def load(self, model_path_or_id: str, spec: ModelSpec, devices: list[DeviceId]) -> None:
        if not _HAS_ORT:
            raise RuntimeError("onnxruntime is not installed; run `pip install nexinfer[ort]`")
        self.spec = spec
        model_file = self._resolve_model_file(model_path_or_id)
        if model_file is None:
            raise FileNotFoundError(
                f"no model.onnx found for {model_path_or_id}; "
                "export an ONNX decoder (e.g. optimum-cli export onnx) or "
                "place model.onnx / model_kv.onnx in the model directory"
            )

        providers = ["CPUExecutionProvider"]
        for dev in devices:
            for prefix, eps in EP_MAP.items():
                if dev.startswith(prefix):
                    providers = eps
                    break
        available = set(ort.get_available_providers())
        providers = [p for p in providers if p in available]
        log.info("ORT providers: %s", providers)

        self.session = ort.InferenceSession(model_file, providers=providers)
        self._kv_mode = any(i.name.startswith("key_values_") for i in self.session.get_inputs())
        # cache initializer values for graph-input initializers (ORT treats
        # initializers that also appear in graph inputs as overridable)
        self._init_values: dict[str, np.ndarray] = {}
        try:
            g = onnx.load(model_file).graph
            self._init_values = {ini.name: numpy_helper.to_array(ini) for ini in g.initializer}
            log.debug("ORT init values: %s", list(self._init_values))
        except Exception as exc:
            log.warning("could not read initializer values from %s: %s", model_file, exc)
        log.info("ORT model %s loaded in %s mode", model_file, "kv-cache" if self._kv_mode else "stateless")

    @staticmethod
    def _resolve_model_file(model_path_or_id: str) -> str | None:
        if os.path.isfile(model_path_or_id) and model_path_or_id.endswith(".onnx"):
            return model_path_or_id
        if os.path.isdir(model_path_or_id):
            # prefer the KV-cache variant when present
            for name in ("model_kv.onnx", "model.onnx"):
                p = os.path.join(model_path_or_id, name)
                if os.path.isfile(p):
                    return p
            others = [
                os.path.join(model_path_or_id, f) for f in os.listdir(model_path_or_id) if f.endswith(".onnx")
            ]
            return others[0] if others else None
        return None

    # ------------------------------------------------------------------

    def prefill(self, req_id: str, input_ids: np.ndarray) -> np.ndarray:
        assert self.session is not None
        ids = input_ids.reshape(1, -1).astype(np.int64)
        inputs: dict[str, np.ndarray] = {"input_ids": ids}
        if self._kv_mode:
            inputs.update(self._past_kv_inputs(req_id, ids.shape[1]))
            self._past_len[req_id] = ids.shape[1]
        else:
            self._history[req_id] = ids[0].tolist()
        out = self.session.run(None, inputs)
        logits = np.asarray(out[0], dtype=np.float32)
        if self._kv_mode:
            self._capture_present(req_id, out)
        return self._last_token_logits(logits)

    def decode(self, req_ids: list[str], input_ids: np.ndarray) -> np.ndarray:
        assert self.session is not None
        toks = [int(t) for t in np.asarray(input_ids).flatten().tolist()]
        if self._kv_mode:
            out = []
            for req_id, tok in zip(req_ids, toks):
                inputs = {"input_ids": np.array([[tok]], dtype=np.int64)}
                inputs.update(self._past_kv_inputs(req_id, None))
                res = self.session.run(None, inputs)
                out.append(self._last_token_logits(np.asarray(res[0], dtype=np.float32)))
                self._capture_present(req_id, res)
                self._past_len[req_id] = self._past_len.get(req_id, 0) + 1
            return np.vstack(out) if len(out) > 1 else out[0].reshape(1, -1)

        # stateless mode: re-run the full history + new tokens and return
        # the last-position logits (causal by construction)
        out = []
        for req_id, tok in zip(req_ids, toks):
            history = self._history.setdefault(req_id, [])
            history.append(tok)
            ids = np.array([history], dtype=np.int64)
            res = self.session.run(None, {"input_ids": ids})
            out.append(self._last_token_logits(np.asarray(res[0], dtype=np.float32)))
        return np.vstack(out) if len(out) > 1 else out[0].reshape(1, -1)

    @staticmethod
    def _last_token_logits(logits: np.ndarray) -> np.ndarray:
        # accept (1, seq, vocab), (1, vocab) or (seq, vocab) and always
        # return the last-position logits as (1, vocab)
        if logits.ndim == 3:
            return logits[0, -1:].reshape(1, -1)
        if logits.ndim == 2:
            if logits.shape[0] == 1:
                return logits.reshape(1, -1)
            return logits[-1:].reshape(1, -1)
        return logits.reshape(1, -1)

    # ------------------------------------------------------------------

    def _past_kv_inputs(self, req_id: str, seq_len: int | None) -> dict[str, np.ndarray]:
        cache = self._kv_cache.get(req_id, {})
        inputs: dict[str, np.ndarray] = {}
        for i in self.session.get_inputs():
            if i.name in inputs:
                continue
            if i.name.startswith("key_values_"):
                role = "key"
                idx = int(i.name.rsplit("_", 1)[-1])
                arr = cache.get(f"{idx}_{role}")
                if arr is None:
                    arr = np.zeros((1, 0, self.spec.num_kv_heads * self.spec.head_dim // 2), dtype=np.float32)
                inputs[i.name] = arr
            elif i.name.startswith("value_values_"):
                role = "value"
                idx = int(i.name.rsplit("_", 1)[-1])
                arr = cache.get(f"{idx}_{role}")
                if arr is None:
                    arr = np.zeros((1, 0, self.spec.num_kv_heads * self.spec.head_dim // 2), dtype=np.float32)
                inputs[i.name] = arr
            elif i.name in self._init_values:
                inputs[i.name] = self._init_values[i.name]
            else:
                # optional/fallback: zero-fill any remaining graph-input
                # (initializer) based on the declared shape, with dynamic
                # dims set to 0 so they can grow on the first step
                shape = [d if isinstance(d, int) else 1 for d in i.shape]
                dtype = np.float32 if str(i.type) == "tensor(float)" else np.int64
                inputs[i.name] = np.zeros(shape, dtype=dtype)
        return inputs

    def _capture_present(self, req_id: str, outputs: list) -> None:
        cache = self._kv_cache.setdefault(req_id, {})
        out_names = [o.name for o in self.session.get_outputs()]
        for i, name in enumerate(out_names):
            if name.startswith("present_"):
                # present_<layer>_<key|value> -> cache key "<layer>_<key|value>"
                _, layer, role = name.split("_", 2)
                cache[f"{layer}_{role}"] = np.asarray(outputs[i])

    def free(self, req_id: str) -> None:
        """Release per-request KV state (hook for the paged cache)."""
        self._kv_cache.pop(req_id, None)
        self._history.pop(req_id, None)
        self._past_len.pop(req_id, None)

    def close(self) -> None:
        self.session = None
        self._kv_cache.clear()
        self._history.clear()
        self._past_len.clear()
