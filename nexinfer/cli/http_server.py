"""OpenAI-compatible HTTP server for NexusInfer.

Endpoints::

    GET  /v1/models                        -- list the loaded model
    POST /v1/chat/completions              -- standard chat completion
    POST /v1/completions                   -- legacy completions
    GET  /v1/status                        -- engine health + queue depth + uptime
    GET  /health                           -- health check
    GET  /metrics                          -- Prometheus exposition

Security & hygiene (Phase-3 hardening):

* **API key auth** (optional)   -- ``Authorization: Bearer <key>``. No key
  configured means the gate is open (dev mode); set ``--api-key`` or
  ``NEXINFER_API_KEY`` to lock the server down.
* **Rate limiting** (optional)  -- token bucket per client IP, configured
  with ``--rate-limit`` requests/minute; 0 = unlimited.
* **Input validation**          -- prompt length, ``max_tokens`` caps,
  message/role checks, stop-sequence limits, and sanitisation of the
  function-call ``tools`` array to safe primitives only.
* **CORS headers**              -- added to every response so browser
  clients can talk to the server directly.
* **Structured observability**  -- ``--log-format json`` emits one JSON
  object per log line with trace ids and per-subsystem latency spans.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any

from nexinfer.services.security import ApiKeyAuth, InputPolicy, RateLimiter

log = logging.getLogger("nexinfer.http")


class _Handler(BaseHTTPRequestHandler):
    engine = None  # set by HttpServer
    auth: ApiKeyAuth = ApiKeyAuth()
    limiter: RateLimiter = RateLimiter()
    policy: InputPolicy = InputPolicy()
    start_time: float = 0.0
    cors_origin: str = "*"  # "*" = any origin; set to a concrete origin to restrict
    tokenizer = None  # optional HF tokenizer for accurate token accounting
    template = None  # default chat-template name

    def version_string(self) -> str:
        return "nexinfer/0.4.0"

    def log_message(self, format, *args):
        log.info(format, *args)

    def _send_json(self, code: int, obj: Any) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._add_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _add_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", self.cors_origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Max-Age", "600")

    def do_OPTIONS(self) -> None:
        """Preflight so browsers can POST from another origin."""
        self.send_response(204)
        self._add_cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _read_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        # reject megabyte-scale bodies early (the engine does not need them)
        if length > 8 * 1024 * 1024:
            raise ValueError("request body exceeds 8 MiB limit")
        return json.loads(self.rfile.read(length)) if length else {}

    def _client_id(self) -> str:
        return self.client_address[0] or "unknown"

    # ------------------------------------------------------------------

    def do_DELETE(self) -> None:
        """Cancel an in-flight generation: ``DELETE /v1/chat/completions/<id>``.

        Flips the request's abort flag; the generation loop stops at the
        next decode step with ``finish_reason="abort"`` and the in-flight
        call returns promptly.
        """
        if not self.auth.check(self.headers.get("Authorization")):
            self._send_json(401, {"error": {"message": "invalid or missing API key"}})
            return
        prefix = "/v1/chat/completions/"
        if self.path.startswith(prefix):
            request_id = self.path[len(prefix) :].split("?")[0]
            if self.engine is not None and self.engine.cancel(request_id):
                log.info("request %s cancelled by client %s", request_id, self._client_id())
                self._send_json(200, {"id": request_id, "object": "chat.completion", "cancelled": True})
            else:
                self._send_json(404, {"error": {"message": "request not found or already finished"}})
            return
        self._send_json(404, {"error": {"message": "not found"}})

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        elif self.path == "/metrics":
            self._send_metrics()
        elif self.path == "/v1/models":
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.engine.status.model
                            if self.engine and self.engine.status
                            else "unknown",
                            "object": "model",
                            "created": 0,
                            "owned_by": "nexinfer",
                            "backend": self.engine.status.backend_names
                            if self.engine and self.engine.status
                            else [],
                        }
                    ],
                },
            )
        elif self.path == "/v1/status":
            self._send_status()
        else:
            self._send_json(404, {"error": {"message": "not found"}})

    def _send_metrics(self) -> None:
        from nexinfer.services.metrics import METRICS

        text = METRICS.prometheus_text()
        body = text.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._add_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _send_status(self) -> None:
        """Structured engine status: health, backend info, queue depth,
        uptime, and the subsystem latency breakdown."""
        from nexinfer.services.metrics import METRICS

        ready = self.engine is not None and self.engine._generator is not None
        status = self.engine.status if self.engine is not None else None
        uptime = time.time() - (self.start_time or time.time())
        payload: dict[str, Any] = {
            "status": "ready" if ready else "initialising",
            "model": getattr(status, "model", "unknown"),
            "backends": getattr(status, "backend_names", []) or [],
            "uptime_seconds": round(uptime, 1),
            "queue_depth": getattr(self.engine, "queue_depth", 0) or 0,
            "scheduler_running": (
                getattr(self.engine.scheduler, "num_running", 0)
                if self.engine and hasattr(self.engine, "scheduler")
                else 0
            ),
            "subsystems": {k: v for k, v in METRICS._subsystem_sums.items()},
        }
        self._send_json(200, payload)

    def do_POST(self) -> None:
        if self.path not in ("/v1/chat/completions", "/v1/completions"):
            self._send_json(404, {"error": {"message": "not found"}})
            return

        # ---- security gate -------------------------------------------------
        if not self.auth.check(self.headers.get("Authorization")):
            log.warning("401 auth denied client=%s path=%s", self._client_id(), self.path)
            self._send_json(401, {"error": {"message": "invalid or missing API key"}})
            return
        if not self.limiter.allow(self._client_id()):
            self._send_json(429, {"error": {"message": "rate limit exceeded"}})
            return

        if self.engine is None or self.engine._generator is None:
            self._send_json(503, {"error": {"message": "engine not ready"}})
            return

        # ---- parse + validate body -----------------------------------------
        try:
            body = self._read_body()
        except Exception as exc:
            self._send_json(400, {"error": {"message": f"bad request: {exc}"}})
            return
        if not isinstance(body, dict):
            self._send_json(400, {"error": {"message": "body must be a JSON object"}})
            return

        trace_id = str(uuid.uuid4().hex[:12])
        client = self._client_id()

        if self.path == "/v1/chat/completions":
            messages = body.get("messages", [])
            err = self.policy.validate_messages(messages)
            if err:
                self._send_json(400, {"error": {"message": f"messages: {err}"}})
                return
            tools = body.get("tools") or []
            tool_schemas, err = InputPolicy.sanitize_tool_schemas(tools)
            if err:
                self._send_json(400, {"error": {"message": f"tools: {err}"}})
                return
            tool_schemas = [t.get("function", t) for t in tool_schemas]
            template_name = body.get("template") or type(self).template or "minimal"
            from nexinfer.engine.chat_template import ChatTemplate

            prompt = ChatTemplate(template_name).apply(messages)
        else:
            prompt = body.get("prompt", "")
            tool_schemas = []

        err = self.policy.validate_prompt(prompt)
        if err:
            self._send_json(400, {"error": {"message": f"prompt: {err}"}})
            return
        max_tokens = body.get("max_tokens") or 128
        err = self.policy.validate_max_tokens(int(max_tokens))
        if err:
            self._send_json(400, {"error": {"message": f"max_tokens: {err}"}})
            return
        err = self.policy.validate_sampling(
            body.get("temperature", 0.8), body.get("top_p", 0.9), body.get("top_k", 50)
        )
        if err:
            self._send_json(400, {"error": {"message": err}})
            return
        err = self.policy.validate_stop_sequences(body.get("stop") or [])
        if err:
            self._send_json(400, {"error": {"message": f"stop: {err}"}})
            return

        # ---- build request --------------------------------------------------
        from nexinfer.engine.types import GenerationRequest

        req = self.engine.register(
            GenerationRequest(
                prompt=prompt,
                max_tokens=int(max_tokens),
                temperature=float(body.get("temperature", 0.8)),
                top_p=float(body.get("top_p", 0.9)),
                top_k=int(body.get("top_k", 50)),
                stop_sequences=body.get("stop") or [],
                tools=tool_schemas,
                tool_choice=str(body.get("tool_choice", "auto")),
                agent_id=body.get("agent_id"),
                skill=body.get("skill"),
            )
        )
        completion_id = f"chatcmpl-{req.request_id}"

        # ---- queue-wait span ------------------------------------------------
        from nexinfer.services.metrics import METRICS

        METRICS.record_request()
        METRICS.set_queue_depth(getattr(self.engine, "queue_depth", 0) or 0)
        tok = getattr(self, "tokenizer", None)
        input_tokens = len(tok.encode(prompt)) if tok else 0
        METRICS.record_tokens(input_tokens=input_tokens)

        t0 = time.perf_counter()
        from nexinfer.services.logging_config import SpanTimer

        with SpanTimer("total", METRICS, trace_id=trace_id):
            try:
                out = self.engine.generate(req)
            except Exception as exc:
                METRICS.record_request(failed=True)
                log.error("request %s from %s failed: %s", trace_id, client, exc)
                self._send_json(500, {"error": {"message": f"generation failed: {exc}"}})
                return
        METRICS.record_tokens(output_tokens=len(out.text.split()))
        log.info(
            "request %s client=%s tokens_in=%d tokens_out=%d %.2fs",
            trace_id,
            client,
            input_tokens,
            len(out.text.split()),
            time.perf_counter() - t0,
        )

        # ---- response -------------------------------------------------------
        resp: dict[str, Any] = {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.engine.status.model if self.engine.status else "unknown",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": out.text,
                        **(
                            {
                                "tool_calls": [
                                    {
                                        "id": f"call-{i}",
                                        "type": "function",
                                        "function": {
                                            "name": tc["name"],
                                            "arguments": json.dumps(tc["arguments"]),
                                        },
                                    }
                                    for i, tc in enumerate(out.tool_calls)
                                ]
                            }
                            if out.tool_calls
                            else {}
                        ),
                    },
                    "finish_reason": "tool_calls" if out.tool_calls else out.finish_reason,
                }
            ],
            "usage": {**out.usage, "total_tokens": sum(out.usage.values())},
        }
        self._send_json(200, resp)


class _ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class HttpServer:
    """OpenAI-compatible HTTP front end for the engine.

    Security and observability knobs (all optional -- unset == dev mode):

    * ``api_key``       -- Bearer token gate (also ``NEXINFER_API_KEY``)
    * ``rate_limit``    -- requests per minute per client IP (0 = off)
    * ``log_format``    -- ``"text"`` | ``"json"``
    * ``cors_origin``   -- ``Access-Control-Allow-Origin`` value
    """

    def __init__(
        self,
        engine: Any,
        tokenizer: Any = None,
        api_key: str | None = None,
        rate_limit: int = 0,
        log_format: str = "text",
        cors_origin: str = "*",
        max_prompt_chars: int | None = None,
        max_tokens_cap: int | None = None,
    ) -> None:
        self.engine = engine
        self.tokenizer = tokenizer
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.log_format = log_format
        self.cors_origin = cors_origin
        self.max_prompt_chars = max_prompt_chars
        self.max_tokens_cap = max_tokens_cap

    def run(self, host: str, port: int) -> None:
        from nexinfer.services.logging_config import configure_logging

        configure_logging(fmt=self.log_format)
        _Handler.engine = self.engine
        _Handler.tokenizer = self.tokenizer
        _Handler.template = getattr(self, "template", None)
        auth = ApiKeyAuth.from_env_or_flag(self.api_key)
        _Handler.auth = auth
        _Handler.limiter = RateLimiter(limit=self.rate_limit)
        _Handler.policy = InputPolicy(
            max_prompt_chars=self.max_prompt_chars,
            max_tokens_cap=self.max_tokens_cap,
        )
        _Handler.start_time = time.time()
        _Handler.cors_origin = self.cors_origin
        if auth.enabled:
            log.info("HTTP auth enabled (no key -> 401)")
        if _Handler.limiter.enabled:
            log.info("rate limit: %d req/min per client", self.rate_limit)
        server = _ThreadedHTTPServer((host, port), _Handler)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.shutdown()

    # -- test helpers (avoid binding a real socket) ---------------------------

    @classmethod
    def serve_in_thread(cls, engine: Any, **kw: Any) -> tuple[HTTPServer, threading.Thread]:
        """Spin up the server on a random loopback port in a daemon thread.
        Returns ``(server, thread)``; ``server.server_address[1]`` is the port."""
        server = _ThreadedHTTPServer(("127.0.0.1", 0), _Handler)
        _Handler.engine = engine
        _Handler.tokenizer = kw.get("tokenizer")
        _Handler.template = kw.get("template")
        _Handler.auth = ApiKeyAuth.from_env_or_flag(kw.get("api_key"))
        _Handler.limiter = RateLimiter(limit=kw.get("rate_limit", 0))
        _Handler.policy = InputPolicy()
        _Handler.start_time = time.time()
        _Handler.cors_origin = kw.get("cors_origin", "*")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread
