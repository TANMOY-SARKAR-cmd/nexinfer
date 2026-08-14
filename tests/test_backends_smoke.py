"""Smoke tests for the functional third-party backends (ORT, GGML).

These run only when the optional runtime libraries (``onnxruntime``,
``llama-cpp-python``) are importable; CI jobs that cannot build them
skip the whole file gracefully.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from nexinfer.backends.base import ModelSpec

SPEC = ModelSpec(
    num_layers=2,
    hidden_size=64,
    num_attention_heads=4,
    num_kv_heads=2,
    head_dim=16,
    vocab_size=256,
    inter_dim=128,
)

pytest.importorskip("onnxruntime")
pytest.importorskip("onnxruntime")


def test_ort_kv_and_stateless_modes():
    from nexinfer.backends.ort_backend import OrtBackend, OrtModelBuilder

    with tempfile.TemporaryDirectory() as td:
        OrtModelBuilder.build_demo(td, SPEC)

        be = OrtBackend()
        be.load(td, SPEC, ["/cpu:0"])

        lp = be.prefill("r1", np.array([10, 20, 30], dtype=np.int32))
        assert lp.shape == (1, SPEC.vocab_size)

        ld = be.decode(["r1"], np.array([5], dtype=np.int32))
        assert ld.shape == (1, SPEC.vocab_size)

        # second decode: cache must grow, shapes stay per-request
        ld2 = be.decode(["r1"], np.array([7], dtype=np.int32))
        assert ld2.shape == (1, SPEC.vocab_size)

        be.free("r1")

        # batched decode across two requests
        be.prefill("r1", np.array([10, 20], dtype=np.int32))
        be.prefill("r2", np.array([30, 40, 50], dtype=np.int32))
        batch = be.decode(["r1", "r2"], np.array([5, 9], dtype=np.int32))
        assert batch.shape == (2, SPEC.vocab_size)

        # stateless (full-replay) path
        be2 = OrtBackend()
        be2.load(os.path.join(td, "model.onnx"), SPEC, ["/cpu:0"])
        lp2 = be2.prefill("r1", np.array([10, 20, 30], dtype=np.int32))
        assert lp2.shape == (1, SPEC.vocab_size)
        ld2b = be2.decode(["r1"], np.array([5], dtype=np.int32))
        assert ld2b.shape == (1, SPEC.vocab_size)
        ld3 = be2.decode(["r1"], np.array([7], dtype=np.int32))
        assert ld3.shape == (1, SPEC.vocab_size)


@pytest.mark.skipif(os.environ.get("SKIP_LLAMA_TESTS"), reason="llama model download skipped")
def test_ggml_prefill_decode_batch():
    """Runs a tiny SmolLM2-135M GGUF through the GGML backend.

    The model path is set via ``GGUF_MODEL_PATH``; CI runners that do not
    pre-populate the cache skip this test.
    """
    pytest.importorskip("llama_cpp")
    from nexinfer.backends.ggml_backend import GGMLBackend
    from tests.gguf_fixtures import gguf_path

    gguf = gguf_path() or ""
    if not os.path.exists(gguf):
        pytest.skip("demo GGUF not available")

    # SmolLM2-135M-Instruct: 30 layers, hidden 576, 6 attention heads,
    # vocab 49152, inter dim 1536
    spec = ModelSpec(
        num_layers=30,
        hidden_size=576,
        num_attention_heads=6,
        num_kv_heads=6,
        head_dim=96,
        vocab_size=49152,
        inter_dim=1536,
    )
    from nexinfer.backends.base import DeviceInfo
    from nexinfer.engine.types import DeviceKind

    be = GGMLBackend()
    be.load(
        gguf,
        spec,
        [DeviceInfo("/cpu:0", DeviceKind.CPU, "generic", "CPU", 0, 2.5)],
    )

    lp = be.prefill("r1", np.array([100, 200, 300], dtype=np.int32))
    assert lp.shape == (1, 49152)

    ld = be.decode(["r1"], np.array([400], dtype=np.int32))
    assert ld.shape == (1, 49152)

    # batched decode of two simultaneous requests
    be.prefill("r2", np.array([10, 20], dtype=np.int32))
    batch = be.decode(["r1", "r2"], np.array([500, 600], dtype=np.int32))
    assert batch.shape == (2, 49152)

    be.free("r1")
    be.free("r2")
