"""Abort-on-cancel tests.

Phase-4 additions:
* ``GenerationRequest.abort_flag`` -- external flip stops generation
  with ``finish_reason="abort"``
* ``Engine.register()`` / ``Engine.cancel()`` -- abort registry wired to
  the generation loop
* ``DELETE /v1/chat/completions/<id>`` -- HTTP cancel endpoint
"""

from __future__ import annotations

import json
import threading
import time
from http.server import HTTPServer

import pytest

from nexinfer.backends.base import DeviceInfo, ModelSpec
from nexinfer.backends.cpu_numpy import NumpyBackend
from nexinfer.cli.http_server import HttpServer
from nexinfer.engine.runtime import Engine
from nexinfer.engine.types import GenerationRequest

SPEC = ModelSpec(
    num_layers=4,
    hidden_size=64,
    num_attention_heads=4,
    num_kv_heads=2,
    head_dim=16,
    vocab_size=256,
    inter_dim=128,
)
CPU_DEVICES = [DeviceInfo("/cpu:0", DeviceInfo.kind if False else "/cpu", "generic", "CPU", 0, 1.0)]
# avoid importing DeviceKind everywhere
from nexinfer.engine.types import DeviceKind

CPU_DEVICES = [DeviceInfo("/cpu:0", DeviceKind.CPU, "generic", "CPU", 0, 1.0)]


@pytest.fixture()
def engine() -> Engine:
    be = NumpyBackend()
    be.load("nexinfer-demo-weights", SPEC, CPU_DEVICES)
    eng = Engine()
    eng.bootstrap("nexinfer-demo-weights", SPEC, backend_name="cpu_numpy")
    return eng


def test_abort_flag_stops_generation(engine: Engine) -> None:
    req = engine.register(GenerationRequest(prompt="hello world " * 4, max_tokens=1000))
    abort = req.abort_flag

    def flipper() -> None:
        time.sleep(0.05)
        abort[0] = True

    t = threading.Thread(target=flipper)
    t.start()
    out = engine.generate(req)
    t.join()
    assert out.finish_reason == "abort"
    # registry is cleaned up after completion
    assert req.request_id not in engine._abort_registry


def test_cancel_unknown_request_returns_false(engine: Engine) -> None:
    assert engine.cancel("does-not-exist") is False


def test_cancel_running_request(engine: Engine) -> None:
    req = engine.register(GenerationRequest(prompt="hello world " * 4, max_tokens=1000))

    def canceller() -> None:
        time.sleep(0.05)
        engine.cancel(req.request_id)

    t = threading.Thread(target=canceller)
    t.start()
    out = engine.generate(req)
    t.join()
    assert out.finish_reason == "abort"


def test_streaming_abort_cleans_up(engine: Engine) -> None:
    req = engine.register(GenerationRequest(prompt="hello world " * 4, max_tokens=1000, stream=True))

    def canceller() -> None:
        time.sleep(0.05)
        engine.cancel(req.request_id)

    t = threading.Thread(target=canceller)
    t.start()
    reasons = [tok.finish_reason for tok in engine.generate_stream(req) if tok.finish_reason]
    t.join()
    assert reasons[-1] == "abort"
    assert req.request_id not in engine._abort_registry


@pytest.fixture()
def server(engine: Engine):

    server, th = HttpServer.serve_in_thread(engine, port=0)
    try:
        yield server
    finally:
        server.shutdown()
        th.join(timeout=3)


def test_cancel_endpoint_via_http(engine: Engine, server: HTTPServer) -> None:
    host, port = server.server_address

    def worker() -> None:
        import urllib.request

        payload = json.dumps({"prompt": "hello world " * 4, "max_tokens": 1000}).encode()
        r = urllib.request.Request(f"http://{host}:{port}/v1/completions", data=payload, method="POST")
        r.add_header("Content-Type", "application/json")
        try:
            urllib.request.urlopen(r, timeout=10)
        except Exception:
            pass

    t = threading.Thread(target=worker)
    t.start()
    time.sleep(0.05)

    import urllib.request

    body = json.dumps({"request_id": "missing"}).encode()
    r = urllib.request.Request(
        f"http://{host}:{port}/v1/chat/completions/missing", data=body, method="DELETE"
    )
    r.add_header("Content-Type", "application/json")
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(r)
    assert exc_info.value.code == 404
    t.join(timeout=5)
