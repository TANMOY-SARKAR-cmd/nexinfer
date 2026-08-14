"""OpenAI-compatible HTTP server for NexusInfer.

Endpoints::

    GET  /v1/models                        -- list the loaded model
    POST /v1/chat/completions              -- standard chat completion
    POST /v1/completions                   -- legacy completions
    GET  /health                           -- health check

Accepts function-calling ``tools`` arrays and routes them through the
engine tool registry (web_fetch, memory_*, mcp:* tools).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any

log = logging.getLogger("nexinfer.http")


class _Handler(BaseHTTPRequestHandler):
    engine = None  # set by HttpServer

    def log_message(self, format, *args):
        log.info(format, *args)

    def _send_json(self, code: int, obj: Any) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length)) if length else {}

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        elif self.path == "/metrics":
            self._send_metrics()
        elif self.path == "/v1/models":
            model = self.engine.status.model if self.engine and self.engine.status else "unknown"
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": model,
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
        else:
            self._send_json(404, {"error": {"message": "not found"}})

    def _send_metrics(self) -> None:
        from nexinfer.services.metrics import METRICS

        text = METRICS.prometheus_text()
        body = text.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path not in ("/v1/chat/completions", "/v1/completions"):
            self._send_json(404, {"error": {"message": "not found"}})
            return
        if self.engine is None or self.engine._generator is None:
            self._send_json(503, {"error": {"message": "engine not ready"}})
            return
        try:
            body = self._read_body()
        except Exception as exc:
            self._send_json(400, {"error": {"message": str(exc)}})
            return

        if self.path == "/v1/chat/completions":
            messages = body.get("messages", [])
            tools = body.get("tools") or []
            tool_schemas = [t.get("function", t) for t in tools]
            template_name = body.get("template") or self._template or "minimal"
            from nexinfer.engine.chat_template import ChatTemplate

            prompt = ChatTemplate(template_name).apply(messages)
        else:
            prompt = body.get("prompt", "")
            tool_schemas = []

        from nexinfer.engine.types import GenerationRequest

        req = GenerationRequest(
            prompt=prompt or " ",
            max_tokens=body.get("max_tokens") or 128,
            temperature=body.get("temperature", 0.8),
            top_p=body.get("top_p", 0.9),
            stop_sequences=body.get("stop") or [],
            tools=tool_schemas,
            agent_id=body.get("agent_id"),
            skill=body.get("skill"),
        )
        import time as _mt

        from nexinfer.services.metrics import METRICS

        METRICS.record_request()
        try:
            METRICS.set_queue_depth(getattr(self.engine, "queue_depth", 0) or 0)
        except Exception:
            pass
        input_tokens = len(self.tokenizer.encode(prompt)) if self.tokenizer else 0
        METRICS.record_tokens(input_tokens=input_tokens)
        t0 = _mt.monotonic()
        try:
            out = self.engine.generate(req)
        except Exception as exc:
            METRICS.record_request(failed=True)
            self._send_json(500, {"error": {"message": str(exc)}})
            return
        METRICS.record_latency(_mt.monotonic() - t0)
        METRICS.record_tokens(output_tokens=len(out.text.split()))
        resp: dict[str, Any] = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
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
    def __init__(self, engine: Any, tokenizer: Any = None) -> None:
        self.engine = engine
        self.tokenizer = tokenizer

    @property
    def _template(self):
        return getattr(self, "template", None)

    def run(self, host: str, port: int) -> None:
        _Handler.engine = self.engine
        server = _ThreadedHTTPServer((host, port), _Handler)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.shutdown()
