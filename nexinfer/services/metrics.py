"""Prometheus-style metrics collector for the inference engine.

A single ``MetricsCollector`` instance is shared across the process. It
tracks request counts, token throughput, latency histograms, paged KV
block usage and scheduler queue depth, and can render the classic
Prometheus text exposition format for ``GET /metrics``.
"""

from __future__ import annotations

import logging
import math
import threading
import time

log = logging.getLogger("nexinfer.services.metrics")

HISTOGRAM_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


def _escape_help(text: str) -> str:
    return text.replace("\\", r"\\").replace("\n", r"\n")


class MetricsCollector:
    """Thread-safe process-wide metrics store."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests_total = 0
        self.requests_failed = 0
        self.tokens_input_total = 0
        self.tokens_output_total = 0
        self.inference_seconds_total = 0.0
        self._latency_samples: list[float] = []
        self.paged_blocks_total = 0
        self.paged_blocks_free = 0
        self.queue_depth = 0
        self.num_requests_running = 0
        self.start_time = time.time()
        self._bucket_counts: list[int] = [0] * (len(HISTOGRAM_BUCKETS) + 1)
        self._latency_sum = 0.0
        # per-subsystem breakdown: prefill / decode / queue_wait / tool_call / transport / total
        self._subsystem_sums: dict[str, float] = {}
        self._subsystem_counts: dict[str, int] = {}

    # -- counters ----------------------------------------------------------

    def record_request(self, *, failed: bool = False) -> None:
        with self._lock:
            self.requests_total += 1
            if failed:
                self.requests_failed += 1

    def record_tokens(self, *, input_tokens: int = 0, output_tokens: int = 0) -> None:
        with self._lock:
            self.tokens_input_total += input_tokens
            self.tokens_output_total += output_tokens

    def record_latency(self, seconds: float, subsystem: str = "total") -> None:
        """Record a latency sample. ``subsystem`` is one of
        ``prefill/decode/queue_wait/tool_call/transport/total`` and feeds
        the per-subsystem breakdown (totals + histogram) exposed on
        ``/metrics`` and in JSON log spans."""
        with self._lock:
            self.inference_seconds_total += seconds
            self._latency_sum += seconds
            self._latency_samples.append(seconds)
            # cap the sample history to bound memory
            if len(self._latency_samples) > 100_000:
                self._latency_samples = self._latency_samples[-50_000:]
            # bucket counters
            bucket = len(HISTOGRAM_BUCKETS)
            for idx, bound in enumerate(HISTOGRAM_BUCKETS):
                if seconds <= bound:
                    bucket = idx
                    break
            for i in range(bucket, len(self._bucket_counts)):
                self._bucket_counts[i] += 1
            # per-subsystem accumulation
            self._subsystem_sums[subsystem] = self._subsystem_sums.get(subsystem, 0.0) + seconds
            self._subsystem_counts[subsystem] = self._subsystem_counts.get(subsystem, 0) + 1

    def set_queue_depth(self, depth: int) -> None:
        with self._lock:
            self.queue_depth = depth

    def set_running(self, n: int) -> None:
        with self._lock:
            self.num_requests_running = n

    def set_block_usage(self, total: int, free: int) -> None:
        with self._lock:
            self.paged_blocks_total = total
            self.paged_blocks_free = free

    # -- readouts ----------------------------------------------------------

    def latency_quantile(self, q: float) -> float:
        with self._lock:
            if not self._latency_samples:
                return 0.0
            s = sorted(self._latency_samples)
            rank = max(0, min(len(s) - 1, int(math.ceil(q * len(s))) - 1))
            return s[rank]

    # -- exposition --------------------------------------------------------

    def prometheus_text(self) -> str:
        """Render the Prometheus text exposition format."""
        lines: list[str] = []

        def _counter(name: str, value: float, help_text: str) -> None:
            lines.append(f"# HELP {name} {_escape_help(help_text)}")
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value:.6f}" if isinstance(value, float) else f"{name} {value}")

        def _gauge(name: str, value: float, help_text: str) -> None:
            lines.append(f"# HELP {name} {_escape_help(help_text)}")
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")

        def _histogram(name: str) -> None:
            lines.append(f"# HELP {name} inference request latency in seconds")
            lines.append(f"# TYPE {name} histogram")
            with self._lock:
                buckets = list(self._bucket_counts)
                latency_sum = self._latency_sum
                count = buckets[-1] if buckets else 0
            for bound, cum in zip(HISTOGRAM_BUCKETS, buckets):
                lines.append(f'{name}_bucket{{le="{bound:g}"}} {cum}')
            lines.append(f'{name}_bucket{{le="+Inf"}} {count}')
            lines.append(f"{name}_sum {latency_sum:.6f}")
            lines.append(f"{name}_count {count}")

        _counter(
            "nexinfer_requests_total", self.requests_total, "Total number of generation requests processed"
        )
        _counter(
            "nexinfer_requests_failed_total",
            self.requests_failed,
            "Total number of failed generation requests",
        )
        _counter(
            "nexinfer_tokens_input_total",
            self.tokens_input_total,
            "Total number of input (prompt) tokens processed",
        )
        _counter(
            "nexinfer_tokens_output_total",
            self.tokens_output_total,
            "Total number of generated output tokens",
        )
        _counter(
            "nexinfer_inference_seconds_total",
            self.inference_seconds_total,
            "Total wall-clock seconds spent in generation",
        )
        _histogram("nexinfer_request_latency_seconds")
        with self._lock:
            subsystems = dict(self._subsystem_sums)
            subsystem_counts = dict(self._subsystem_counts)
        for sub, total in subsystems.items():
            _counter(
                f'nexinfer_subsystem_seconds_total{{subsystem="{sub}"}}',
                total,
                f"Total seconds spent in the {sub} subsystem",
            )
            _counter(
                f'nexinfer_subsystem_total{{subsystem="{sub}"}}',
                subsystem_counts.get(sub, 0),
                f"Total number of {sub} spans measured",
            )
        _gauge("nexinfer_queue_depth", self.queue_depth, "Number of requests waiting in the scheduler queue")
        _gauge(
            "nexinfer_requests_running",
            self.num_requests_running,
            "Number of requests currently being generated",
        )
        with self._lock:
            total, free = self.paged_blocks_total, self.paged_blocks_free
        if total > 0:
            used = total - free
            _gauge("nexinfer_kv_blocks_total", total, "Total number of paged KV cache blocks")
            _gauge("nexinfer_kv_blocks_used", used, "Number of paged KV cache blocks currently allocated")
        uptime = time.time() - self.start_time
        _gauge("nexinfer_process_uptime_seconds", uptime, "Seconds since the metrics collector was created")
        return "\n".join(lines) + "\n"

    @classmethod
    def instance(cls) -> MetricsCollector:
        """Process-wide singleton accessor."""
        if getattr(cls, "_instance", None) is None:
            cls._instance = cls()
        return cls._instance


METRICS = MetricsCollector.instance()
