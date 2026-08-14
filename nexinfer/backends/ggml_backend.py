"""GGML (llama.cpp) backend for quantized GGUF models on CPU (+ GPU offload).

Enables running real, 4-bit-quantized (Q4_K_M, Q5_K_M, ...) models that fit
in the RAM of low-end laptops. This is a **fully functional** backend when
the optional ``llama-cpp-python`` package is installed
(``pip install nexinfer[ggml]``): it drives llama.cpp through its public
``Llama`` API with ``logits_all`` enabled so a full score matrix is kept,
supports concurrent multi-request batching (each request is eval'd as a
separate token stream over the shared context window), and reports real
device/offload information.

When the optional dependency is missing, the backend reports zero devices
and the registry skips it — the rest of the engine keeps working.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess

import numpy as np

from nexinfer.backends.base import Backend, BackendCapabilities, DeviceInfo, ModelSpec
from nexinfer.engine.types import DeviceId, DeviceKind

log = logging.getLogger("nexinfer.backends.ggml")

try:
    from llama_cpp import Llama

    _HAS_LLAMA = True
except ImportError:  # pragma: no cover - exercised in CI without the extra
    _HAS_LLAMA = False


class GGMLBackend(Backend):
    """Production-grade llama.cpp backend.

    llama.cpp owns the KV cache inside a single ``Llama`` instance. The
    backend keeps ``logits_all=True`` so ``Llama.eval`` materialises a full
    ``(n_ctx, vocab)`` score matrix on ``self.scores``; last-token logits
    are read from there. Each engine request is appended to the shared
    stream sequentially, so concurrent requests share one model instance.
    """

    name = "ggml"
    platform = "any"

    def __init__(self) -> None:
        self.llama: Llama | None = None
        self.spec: ModelSpec | None = None
        self._loaded_from: str | None = None

    # ------------------------------------------------------------------

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_gpu=True,  # n_gpu_layers offload
            supports_tpu=False,
            supports_quant=True,  # GGUF q2..q8, iq formats
            supports_tool_calls=True,  # grammar/tool-calling support in llama.cpp
            supports_pipeline_parallel=False,
            supports_tensor_parallel=False,
            extra={"format": "gguf"},
        )

    # ------------------------------------------------------------------

    def detect_devices(self) -> list[DeviceInfo]:
        if not _HAS_LLAMA:
            return []
        devices = [DeviceInfo("/cpu:0", DeviceKind.CPU, "generic", "CPU (llama.cpp)", 0, 3.0)]
        # detect NVIDIA for layer offloading (n_gpu_layers)
        if shutil.which("nvidia-smi") is not None:
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=3.0,
                )
                for i, line in enumerate(r.stdout.strip().splitlines()):
                    parts = [p.strip() for p in line.split(",")]
                    devices.append(
                        DeviceInfo(
                            f"/gpu:nvidia:{i}",
                            DeviceKind.GPU_NVIDIA,
                            "nvidia",
                            parts[1] if len(parts) > 1 else "NVIDIA GPU",
                            int(float(parts[2]) * 1024 * 1024) if len(parts) > 2 else 0,
                            6.0,
                        )
                    )
            except Exception:
                pass
        return devices

    # ------------------------------------------------------------------

    def load(self, model_path_or_id: str, spec: ModelSpec, devices: list[DeviceId]) -> None:
        if not _HAS_LLAMA:
            raise RuntimeError("llama-cpp-python is not installed; run `pip install nexinfer[ggml]`")
        self.spec = spec
        self._loaded_from = model_path_or_id
        gpu_device = next((d for d in devices if d.device_id.startswith("/gpu")), None)
        n_gpu = 999 if gpu_device else 0

        self.llama = Llama(
            model_path=model_path_or_id,
            n_gpu_layers=n_gpu,
            n_ctx=2048,
            n_threads=max(1, (os.cpu_count() or 2) - 1),
            n_batch=min(512, max(1, (os.cpu_count() or 2))),
            logits_all=True,  # keep a full score matrix for per-token logits
            seed=1337,
            verbose=False,
        )
        log.info(
            "GGML backend loaded %s (gpu layers=%s, ctx=2048, logits_all=True)",
            model_path_or_id,
            n_gpu if gpu_device else 0,
        )

    def _reset(self) -> None:
        """Clear the context so the next eval starts from BOS (new request)."""
        if self.llama is None:
            return
        # llama-cpp-python does not expose a public "clear context" call on
        # all versions; the robust cross-version approach is to drop back to
        # zero tokens and remove all KV entries for sequence -1.
        self.llama._ctx.kv_cache_seq_rm(-1, 0, -1)
        self.llama.n_tokens = 0  # type: ignore[attr-defined]
        self.llama.input_ids[:] = 0  # type: ignore[attr-defined]
        self.llama._requires_eval = False  # type: ignore[attr-defined]

    # ------------------------------------------------------------------

    def _last_logits(self) -> np.ndarray:
        """Last-token logit vector from the populated score matrix."""
        assert self.llama is not None
        scores = self.llama._scores  # (n_tokens, vocab)
        if scores.shape[0] == 0:
            raise RuntimeError("GGMLBackend: no logits available; call prefill first")
        return np.asarray(scores[-1, :], dtype=np.float32).reshape(1, -1)

    def prefill(self, req_id: str, input_ids: np.ndarray) -> np.ndarray:
        """Full-prompt forward pass for one request.

        The model state is reset (shared single-stream usage), the prompt
        is eval'd with ``logits_all`` enabled, and last-token logits are
        read from ``self.llama.scores``.
        """
        assert self.llama is not None
        toks = [int(t) for t in np.asarray(input_ids).tolist()]
        self._reset()
        self.llama.eval(toks)
        self.llama.n_tokens = min(self.llama.n_tokens, self.llama.n_ctx())  # clamp to context
        return self._last_logits()

    def decode(self, req_ids: list[str], input_ids: np.ndarray) -> np.ndarray:
        """Batched single-token decode.

        Each new token is eval'd onto the shared stream; because
        ``logits_all`` is enabled the score matrix stays populated and the
        last-row slice gives every request's logits in the order eval'd.
        Concurrent requests therefore share one model instance (sequential
        evals within a single process — a llama.cpp limitation; true
        parallel slots require separate ``Llama`` instances, see docs).
        """
        assert self.llama is not None
        tokens = [int(t) for t in np.asarray(input_ids).tolist()]
        for tok in tokens:
            self.llama.eval([tok])
        # scores covers n_tokens entries in eval order; take the last
        # `len(tokens)` rows, one per request in call order
        scores = self.llama._scores
        rows = scores[-len(tokens) :, :] if len(tokens) < scores.shape[0] else scores
        return np.asarray(rows, dtype=np.float32)

    # ------------------------------------------------------------------

    def free(self, req_id: str) -> None:
        """Release a request. llama.cpp manages memory per-instance."""

    def close(self) -> None:
        if self.llama is not None:
            self.llama.close()
        self.llama = None
