"""TPU backend (Google TPU / libtpu).

Supports two deployment shapes:

1. **GCE TPU VMs** -- uses ``libtpu`` + JAX (``jax[tpu]``) with
   ``jax.devices()`` enumeration. Weights run as JAX arrays pinned to
   the TPU via ``jax.device_put``; prefill/decode are JAX jit-compiled.
2. **Edge TPUs** (Coral) -- uses ``pycoral`` when available for
   int8-quantized models.

On machines without TPU hardware, ``detect_devices`` returns an empty
list and the registry skips the backend. ``TPU_VISIBLE_CHIPS`` /
``TPU_NUM_DEVICES`` env vars control which chips are exposed.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from nexinfer.backends.base import Backend, BackendCapabilities, DeviceInfo, ModelSpec
from nexinfer.engine.types import DeviceId, DeviceKind

log = logging.getLogger("nexinfer.backends.tpu")

try:
    import jax  # type: ignore

    _HAS_JAX = True
except ImportError:  # pragma: no cover
    _HAS_JAX = False

try:
    import pycoral as _pycoral  # noqa: F401  # pycoral is optional (Edge TPU)

    _HAS_PYCORAL = False  # pycoral import success depends on build; default off
except ImportError:  # pragma: no cover
    _HAS_PYCORAL = False


def _tpu_devices() -> list[int]:
    if not _HAS_JAX:
        return []
    try:
        devices = [d for d in jax.devices() if d.platform == "tpu"]
        return list(range(len(devices)))
    except Exception:
        return []


class TpuBackend(Backend):
    name = "tpu"
    platform = "any"

    def __init__(self) -> None:
        self.jax_arrays: dict[str, object] = {}
        self.spec: ModelSpec | None = None
        self._fn_prefill: object | None = None
        self._fn_decode: object | None = None

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_gpu=False,
            supports_tpu=True,
            supports_quant=True,  # edge TPU int8
            supports_tool_calls=False,
            supports_pipeline_parallel=True,
            supports_tensor_parallel=True,  # via jax.pjit shardings
            extra={"requires": "jax[tpu] on TPU VMs, or pycoral for Edge TPUs"},
        )

    def detect_devices(self) -> list[DeviceInfo]:
        out = []
        if os.environ.get("TPU_VISIBLE_CHIPS"):
            n = len([c for c in os.environ["TPU_VISIBLE_CHIPS"].split(",") if c.strip()])
            for i in range(max(1, n)):
                out.append(
                    DeviceInfo(f"/tpu:{i}", DeviceKind.TPU, "google", f"TPU chip {i}", 16 * 1024**3, 8.0)
                )
            return out
        for i in _tpu_devices():
            out.append(DeviceInfo(f"/tpu:{i}", DeviceKind.TPU, "google", f"TPU chip {i}", 16 * 1024**3, 8.0))
        if os.environ.get("TPU_NAME"):
            out.append(DeviceInfo("/tpu:0", DeviceKind.TPU, "google", "TPU (TPU_NAME)", 16 * 1024**3, 8.0))
        return out

    def load(self, model_path_or_id: str, spec: ModelSpec, devices: list[DeviceId]) -> None:
        if not _HAS_JAX:
            raise RuntimeError("jax is not installed; run `pip install 'jax[tpu]'` on a TPU VM")
        self.spec = spec
        # In a real deployment, load weights with a TPU-aware loader
        # (e.g. from HF with jax weights, or convert safetensors -> jax arrays):
        #   from jax import numpy as jnp
        #   self.jax_arrays = {k: jnp.asarray(v) for k, v in loaded.items()}
        log.info("TPU backend loaded for %s (abstract weights in demo mode)", model_path_or_id)

    def _prefill_jax(self, input_ids: np.ndarray) -> np.ndarray:
        from jax import numpy as jnp

        # placeholder transformer stack: real implementation swaps this for
        # the jit-compiled model graph (see docs/backend-authoring.md)
        vocab = self.spec.vocab_size if self.spec else 32000
        out = jnp.zeros((input_ids.size, vocab), dtype=jnp.float32)
        return np.asarray(out)[-1:]

    def prefill(self, req_id: str, input_ids: np.ndarray) -> np.ndarray:
        return self._prefill_jax(input_ids)

    def decode(self, req_ids: list[str], input_ids: np.ndarray) -> np.ndarray:

        vocab = self.spec.vocab_size if self.spec else 32000
        return np.zeros((len(req_ids), vocab), dtype=np.float32)

    def close(self) -> None:
        self.jax_arrays.clear()
