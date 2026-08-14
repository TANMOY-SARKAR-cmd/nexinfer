"""Production-hardening tests for the GGML (llama.cpp) backend.

Phase-3 additions exercised here:

* ``n_ctx`` auto-sizing from GGUF metadata bounded by ``DEFAULT_N_CTX_CAP``
* explicit ``ContextOverflowError`` on prompt overflow (no silent truncation)
* decode-time context-bound check
* BOS guard for empty prompts and out-of-vocab token guard

These tests only run when ``llama-cpp-python`` is importable *and* a demo
GGUF is present (``GGUF_MODEL_PATH`` env var or the pre-populated cache),
matching the skip semantics of ``test_backends_smoke.py``.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("llama_cpp")

from nexinfer.backends.base import DeviceInfo, ModelSpec
from nexinfer.backends.ggml_backend import DEFAULT_N_CTX_CAP, ContextOverflowError, GGMLBackend
from nexinfer.engine.types import DeviceKind
from tests.gguf_fixtures import gguf_path

GGUF = gguf_path() or ""

pytestmark = pytest.mark.skipif(not os.path.exists(GGUF), reason="demo GGUF not available")

SPEC = ModelSpec(
    num_layers=30,
    hidden_size=576,
    num_attention_heads=6,
    num_kv_heads=6,
    head_dim=96,
    vocab_size=49152,
    inter_dim=1536,
)

CPU_DEVICES = [DeviceInfo("/cpu:0", DeviceKind.CPU, "generic", "CPU", 0, 2.5)]


def test_auto_ctx_from_metadata_capped():
    be = GGMLBackend()
    be.load(GGUF, SPEC, CPU_DEVICES)
    try:
        assert 64 <= be.n_ctx <= DEFAULT_N_CTX_CAP
        # metadata reader agrees with what load() picked (unless capped)
        probed = GGMLBackend._probe_model_ctx(GGUF)
        expected = min(probed, DEFAULT_N_CTX_CAP) if probed > 0 else 2048
        assert be.n_ctx == expected
    finally:
        be.close()


def test_explicit_n_ctx_and_cap():
    be = GGMLBackend()
    be.load(GGUF, SPEC, CPU_DEVICES, n_ctx=64)
    try:
        assert be.n_ctx == 64
    finally:
        be.close()

    # a nonsensically large explicit request is bounded by the safety cap
    be2 = GGMLBackend()
    be2.load(GGUF, SPEC, CPU_DEVICES, n_ctx=1_000_000)
    try:
        assert be2.n_ctx == DEFAULT_N_CTX_CAP
    finally:
        be2.close()


def test_prefill_overflow_raises():
    be = GGMLBackend()
    be.load(GGUF, SPEC, CPU_DEVICES, n_ctx=32)
    try:
        with pytest.raises(ContextOverflowError):
            be.prefill("r1", np.array(list(range(64)), dtype=np.int32))
    finally:
        be.close()


def test_decode_overflow_raises():
    # llama.cpp rounds the context up to a 32-token granularity, so the
    # effective window is ``max(requested, 32)`` -- fill it exactly and
    # assert the 33rd token raises ``ContextOverflowError``.
    be = GGMLBackend()
    be.load(GGUF, SPEC, CPU_DEVICES, n_ctx=32)
    try:
        be.prefill("r1", np.array([10, 20], dtype=np.int32))
        for _ in range(30):
            be.decode(["r1"], np.array([5], dtype=np.int32))
        with pytest.raises(ContextOverflowError):
            be.decode(["r1"], np.array([5], dtype=np.int32))
    finally:
        be.close()


def test_bos_guard_and_vocab_guard():
    be = GGMLBackend()
    be.load(GGUF, SPEC, CPU_DEVICES)
    try:
        # empty prompt must never crash -- eval at least a BOS token
        lp = be.prefill("r1", np.array([], dtype=np.int32))
        assert lp.shape == (1, SPEC.vocab_size)

        # wildly out-of-vocab token must map to BOS instead of crashing
        ld = be.decode(["r1"], np.array([99999999], dtype=np.int32))
        assert ld.shape == (1, SPEC.vocab_size)
    finally:
        be.close()


def test_free_isolates_requests():
    be = GGMLBackend()
    be.load(GGUF, SPEC, CPU_DEVICES)
    try:
        be.prefill("r1", np.array([100, 200, 300], dtype=np.int32))
        be.free("r1")
        # after freeing the last request the shared stream is reset; a new
        # prefill on the same id must start clean and still produce logits
        lp = be.prefill("r1", np.array([1, 2], dtype=np.int32))
        assert lp.shape == (1, SPEC.vocab_size)
    finally:
        be.close()
