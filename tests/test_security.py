"""Tests for the HTTP security, observability, and status surface.

Covers ``nexinfer/services/security.py`` (API-key gate, rate limiter, input
policy, tool-schema sanitisation), the ``/v1/status`` endpoint, CORS, JSON
logging, and per-subsystem latency spans against the real ``HttpServer``.
All tests use an in-thread server backed by a ``FakeEngine`` so no
dependencies beyond the stdlib are required.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

# ---------------------------------------------------------------------------
# Fake engine + server bootstrap helpers
# ---------------------------------------------------------------------------


class _FakeStatus:
    def __init__(self) -> None:
        self.model = "nexinfer-demo-weights"
        self.backend_names = ["cpu_numpy"]


class FakeEngine:
    """Minimal engine stand-in for HTTP-layer tests."""

    def __init__(self) -> None:
        self.status = _FakeStatus()
        self.scheduler = None
        self._generator = object()  # marks the engine as bootstrapped
        self._calls: list[str] = []

    def register(self, req):  # noqa: ANN001 - mirrors Engine.register
        return req

    def generate(self, req):  # noqa: ANN001 - matches Engine.generate shape loosely
        self._calls.append(req.request_id)
        from nexinfer.engine.types import TokenOutput

        return TokenOutput(
            text="ok",
            token_id=3,
            finish_reason="stop",
            usage={"prompt_tokens": 4, "completion_tokens": 1},
        )


from http.server import HTTPServer as _HTTPServer


def _server(engine: FakeEngine, api_key=None, rate_limit=0) -> tuple[int, _HTTPServer]:  # noqa: ANN001
    from nexinfer.cli.http_server import _Handler
    from nexinfer.services.security import ApiKeyAuth, InputPolicy, RateLimiter

    _Handler.engine = engine
    _Handler.auth = ApiKeyAuth.from_env_or_flag(api_key)
    _Handler.limiter = RateLimiter(limit=rate_limit)
    _Handler.policy = InputPolicy()
    from socketserver import ThreadingMixIn

    class Threaded(ThreadingMixIn, _HTTPServer):  # noqa: ANN201
        daemon_threads = True
        allow_reuse_address = True

    server = Threaded(("127.0.0.1", 0), _Handler)
    _Handler.start_time = __import__("time").time()
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server.server_address[1], server  # type: ignore[return-value]


def _post(port, path, body, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


# ---------------------------------------------------------------------------
# ApiKeyAuth
# ---------------------------------------------------------------------------


def test_api_key_auth_gate_open_by_default():
    from nexinfer.services.security import ApiKeyAuth

    auth = ApiKeyAuth()
    assert not auth.enabled
    assert auth.check(None)
    assert auth.check("anything")


def test_api_key_auth_gate_blocks_without_token():
    from nexinfer.services.security import ApiKeyAuth

    auth = ApiKeyAuth("secret-key")
    assert auth.enabled
    assert not auth.check(None)
    assert not auth.check("")
    assert not auth.check("Basic dXNlcjpwYXNz")  # wrong scheme
    assert not auth.check("Bearer wrong-key")
    assert auth.check("Bearer secret-key")


def test_api_key_auth_generate_and_env():
    from nexinfer.services.security import ApiKeyAuth

    key = ApiKeyAuth.generate_key()
    assert len(key) == 64
    auth = ApiKeyAuth.from_env_or_flag(key)
    assert auth.enabled and auth.check(f"Bearer {key}")
    with pytest.MonkeyPatch.context() as m:
        m.setenv("NEXINFER_API_KEY", f"  {key}  ")
        assert ApiKeyAuth.from_env_or_flag().check(f"Bearer {key}")


# ---------------------------------------------------------------------------
# RateLimiter
# ---------------------------------------------------------------------------


def test_rate_limiter_allows_then_rejects():
    from nexinfer.services.security import RateLimiter

    rl = RateLimiter(limit=2)
    assert rl.enabled
    assert rl.allow("client-a")
    assert rl.allow("client-a")
    assert not rl.allow("client-a")
    assert rl.allow("client-b")  # per-client bucket


def test_rate_limiter_disabled_and_cleanup():
    from nexinfer.services.security import RateLimiter

    rl = RateLimiter(limit=0)
    assert not rl.enabled
    for _ in range(100):
        assert rl.allow("x")
    rl.cleanup()


# ---------------------------------------------------------------------------
# InputPolicy
# ---------------------------------------------------------------------------


def test_input_policy_prompt_validation():
    from nexinfer.services.security import InputPolicy

    p = InputPolicy()
    assert p.validate_prompt("") is not None
    assert p.validate_prompt("   ") is not None
    assert p.validate_prompt("\x00bad") is not None
    assert p.validate_prompt("hello") is None
    assert p.validate_prompt(123) is not None
    assert p.validate_prompt("x" * (p.max_prompt_chars + 1)) is not None


def test_input_policy_messages_and_sampling():
    from nexinfer.services.security import InputPolicy

    p = InputPolicy()
    assert p.validate_messages([]) is not None
    assert p.validate_messages([{"role": "user", "content": "hi"}]) is None
    assert p.validate_messages([{"role": "hacker", "content": "x"}]) is not None
    assert p.validate_max_tokens(0) is not None
    assert p.validate_max_tokens(1 << 30) is not None
    assert p.validate_max_tokens(64) is None
    assert p.validate_sampling(2.0, 0.5, 20) is None
    assert p.validate_sampling(10.0, 0.5, 20) is not None
    assert p.validate_stop_sequences(["a", "b"]) is None
    assert p.validate_stop_sequences(["", "b"]) is not None


def test_input_policy_tool_schema_sanitisation():
    from nexinfer.services.security import InputPolicy

    safe, err = InputPolicy.sanitize_tool_schemas(
        [{"type": "function", "function": {"name": "add", "parameters": {"n": 3, "tags": ["m", None]}}}]
    )
    assert err is None
    assert safe[0]["function"]["name"] == "add"
    unsafe, err2 = InputPolicy.sanitize_tool_schemas([{"fn": object()}])
    assert err2 is not None


def test_is_safe_trace_id():
    from nexinfer.services.security import is_safe_trace_id

    assert is_safe_trace_id("abc-123_X")
    assert not is_safe_trace_id("abc\ninject")
    assert not is_safe_trace_id("")


# ---------------------------------------------------------------------------
# HTTP server: auth, rate limit, validation, CORS
# ---------------------------------------------------------------------------


def test_auth_required_401_and_accepted_200():
    engine = FakeEngine()
    port, server = _server(engine, api_key="secret-key", rate_limit=10)
    try:
        body = {"messages": [{"role": "user", "content": "hi"}]}
        status, _ = _post(port, "/v1/chat/completions", body)
        assert status == 401
        status, _ = _post(port, "/v1/chat/completions", body, {"Authorization": "Bearer wrong"})
        assert status == 401
        status, data = _post(port, "/v1/chat/completions", body, {"Authorization": "Bearer secret-key"})
        assert status == 200
        assert data["choices"][0]["finish_reason"] == "stop"
        assert "chatcmpl-" in data["id"]
    finally:
        server.shutdown()


def test_open_server_no_auth_needed():
    engine = FakeEngine()
    port, server = _server(engine)
    try:
        status, data = _post(port, "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
        assert status == 200
        assert data["object"] == "chat.completion"
    finally:
        server.shutdown()


def test_rate_limit_429():
    engine = FakeEngine()
    port, server = _server(engine, rate_limit=10)
    try:
        headers = {"Authorization": "Bearer x"}  # any value passes open gate
        body = {"messages": [{"role": "user", "content": "hi"}]}
        for _ in range(10):
            status, _ = _post(port, "/v1/chat/completions", body, headers)
            assert status == 200
        status, data = _post(port, "/v1/chat/completions", body, headers)
        assert status == 429
    finally:
        server.shutdown()


def test_input_validation_errors():
    engine = FakeEngine()
    port, server = _server(engine)
    try:
        cases = [
            ({"messages": []}, "messages"),
            ({"messages": [{"role": "x"}]}, "role"),
            ({"messages": [{"role": "user", "content": "hi"}], "max_tokens": -1}, "max_tokens"),
            ({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1 << 30}, "max_tokens"),
            ({"messages": [{"role": "user", "content": "hi"}], "tools": ["not-an-object"]}, "tools"),
        ]
        for body, expected in cases:
            status, data = _post(port, "/v1/chat/completions", body)
            assert status == 400, f"case {body}: expected 400 for {expected}"
            assert expected in data["error"]["message"].lower()
    finally:
        server.shutdown()


def test_bad_request_body_rejected():
    """A raw body that is not a JSON object should get a 400, not a 500."""
    engine = FakeEngine()
    port, server = _server(engine)
    try:
        # send invalid JSON bytes directly (urllib path cannot carry an object)
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=b"{not valid json",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                json.loads(resp.read())
            # unexpected 200 is still a failure (body was junk)
            assert resp.status != 200
        except urllib.error.HTTPError as err:
            assert err.code == 400
    finally:
        server.shutdown()


def test_status_endpoint():
    engine = FakeEngine()
    port, server = _server(engine)
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/status")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        assert data["status"] == "ready"
        assert data["model"] == "nexinfer-demo-weights"
        assert "cpu_numpy" in data["backends"]
        assert data["uptime_seconds"] >= 0
        assert isinstance(data["subsystems"], dict)
    finally:
        server.shutdown()


def test_cors_headers_present():
    engine = FakeEngine()
    port, server = _server(engine)
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", method="OPTIONS")
        with urllib.request.urlopen(req, timeout=10) as resp:
            headers = dict(resp.headers)
        assert headers.get("Access-Control-Allow-Origin") in ("*", "http://localhost")
        assert "POST" in headers.get("Access-Control-Allow-Methods", "")
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# Structured logging + spans
# ---------------------------------------------------------------------------


def test_json_logging_emits_structured_line(capfd):
    from nexinfer.services.logging_config import configure_logging, new_trace_id
    from nexinfer.services.security import is_safe_trace_id

    assert is_safe_trace_id(new_trace_id())
    logger = configure_logging(fmt="json")
    logger.info("phase3 structured log check", extra={"trace_id": "a1b2c3d4", "tokens_in": 5})
    captured = capfd.readouterr().err
    assert captured, "expected JSON log output on stderr"
    line = next(l for l in captured.splitlines() if "phase3 structured log check" in l)
    payload = json.loads(line)
    assert payload["trace_id"] == "a1b2c3d4"
    assert payload["tokens_in"] == 5
    assert payload["level"] == "INFO"


def test_span_timer_feeds_metrics(caplog):
    import time

    from nexinfer.services.logging_config import SpanTimer, configure_logging
    from nexinfer.services.metrics import METRICS

    before_total = METRICS._subsystem_sums.get("prefill", 0.0)
    configure_logging(fmt="json", level=__import__("logging").DEBUG)
    import logging

    with (
        caplog.at_level(logging.DEBUG, logger="nexinfer.services.logging"),
        SpanTimer("prefill", METRICS, trace_id="t1") as span,
    ):
        time.sleep(0.02)
    assert METRICS._subsystem_sums.get("prefill", 0.0) >= before_total + 0.01
    assert span.elapsed > 0.01
    assert any("span prefill" in rec.message for rec in caplog.records)
