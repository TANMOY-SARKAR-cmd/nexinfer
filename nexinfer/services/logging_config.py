"""Structured observability for NexusInfer.

Two complementary facilities live here:

1. **Structured JSON logging** -- optional ``--log-format json`` mode that
   replaces the default formatter with a JSON emitter. Every record
   carries a machine-parseable envelope (timestamp, logger, level,
   message, module, line, plus any ``extra`` fields such as
   ``trace_id``). Non-JSON mode is unchanged and stays human-readable.

2. **Latency spans** -- ``SpanTimer`` measures per-subsystem latency
   (prefill / decode / queue wait / tool call / transport hop) and feeds
   both the JSON log record (emitted at span end) and the
   ``MetricsCollector`` (cumulative per-subsystem totals + histograms).
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from typing import Any, Self

log = logging.getLogger("nexinfer.services.logging")

SUBSYSTEMS = ("prefill", "decode", "queue_wait", "tool_call", "transport", "total")


class JsonFormatter(logging.Formatter):
    """One structured JSON object per log record."""

    def format(self, record: logging.Record) -> str:  # type: ignore[override]
        payload: dict[str, Any] = {
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "line": record.lineno,
            "msg": record.getMessage(),
        }
        # merge structured extras (trace_id, subsystem, latency_ms, ...)
        for key in ("trace_id", "client", "subsystem", "latency_ms", "tokens_in", "tokens_out"):
            if key in record.__dict__:
                payload[key] = record.__dict__[key]
        if record.exc_info and record.exc_info[0] is not None:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(
    fmt: str = "text",
    level: int = logging.INFO,
    logger_name: str = "nexinfer",
) -> logging.Logger:
    """Configure the ``nexinfer`` logger tree.

    * ``fmt="text"`` (default) -- stdlib ``basicConfig`` style, human-readable
    * ``fmt="json"``           -- one JSON object per line (pipe to jq / Loki)
    """
    root = logging.getLogger(logger_name)
    root.setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            JsonFormatter()
            if fmt.lower() == "json"
            else logging.Formatter("%(asctime)s %(levelname)-7s %(name)s %(message)s")
        )
        root.addHandler(handler)
    return root


# ---------------------------------------------------------------------------
# Latency spans
# ---------------------------------------------------------------------------


class SpanTimer:
    """Lightweight per-subsystem latency measurement.

    Usage::

        with SpanTimer("prefill", metrics=METRICS) as span:
            logits = backend.prefill(req_id, ids)

    On exit the span records its elapsed time into the metrics collector
    (per-subsystem total + histogram bucket) and fires a structured log
    line at DEBUG level (visible in ``--log-format json`` mode).
    """

    def __init__(
        self,
        subsystem: str,
        metrics: Any = None,  # nexinfer.services.metrics.MetricsCollector
        trace_id: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.subsystem = subsystem
        self.metrics = metrics
        self.trace_id = trace_id
        self.extra = extra or {}
        self.start: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> Self:
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self.start
        if self.metrics is not None and self.subsystem in SUBSYSTEMS:
            self.metrics.record_latency(self.elapsed, subsystem=self.subsystem)
        log.debug(
            "span %s %.3fs",
            self.subsystem,
            self.elapsed,
            extra={
                "trace_id": self.trace_id,
                "subsystem": self.subsystem,
                "latency_ms": round(self.elapsed * 1000, 2),
                **self.extra,
            },
        )


def new_trace_id() -> str:
    """Generate a log-friendly trace id (short hex)."""
    return uuid.uuid4().hex[:12]
