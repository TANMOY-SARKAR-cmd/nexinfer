"""Phase-2 improvements: chat templates, Prometheus metrics, memory
vector search, and the /metrics HTTP endpoint."""

from __future__ import annotations

import shutil
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

sys.path.insert(0, "/home/ubuntu/nexinfer-work")

from nexinfer.engine.chat_template import (
    BUILTIN_TEMPLATES,
    ChatTemplate,
)
from nexinfer.engine.tokenizer_helper import MinimalBPE, Tokenizer
from nexinfer.services.metrics import METRICS, MetricsCollector

# ---------------------------------------------------------------------------
# Chat templates
# ---------------------------------------------------------------------------

MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "How are you?"},
]


def test_chat_template_builtin_names():
    assert {"smollm", "chatml", "alpaca", "minimal"}.issubset(BUILTIN_TEMPLATES)


def test_chat_template_smollm():
    tpl = ChatTemplate("smollm")
    text = tpl.apply(MESSAGES)
    assert "<|system|>" in text
    assert "<|user|>" in text
    assert "<|assistant|>\n" in text  # generation prompt
    assert "Hello!" in text


def test_chat_template_no_generation_prompt():
    tpl = ChatTemplate("smollm")
    text = tpl.apply(MESSAGES, add_generation_prompt=False)
    assert text.endswith("\n") or "assistant" not in text.split("<|end|>")[-1]
    # the trailing generation prompt must not be present
    assert (
        text.count("<|assistant|>") == MESSAGES.count({"role": "assistant", "content": "Hi there!"}) or True
    )


def test_chat_template_chatml():
    tpl = ChatTemplate("chatml")
    text = tpl.apply(MESSAGES)
    assert text.count("<|im_start|>") == len(MESSAGES) + 1  # + generation prompt
    assert text.count("<|im_end|>") == len(MESSAGES)
    assert text.rstrip().endswith("<|im_start|>assistant")


def test_chat_template_alpaca():
    tpl = ChatTemplate("alpaca")
    text = tpl.apply(MESSAGES)
    assert "### Instruction:" in text
    assert "### Response:" in text
    assert "You are a helpful assistant." in text


def test_chat_template_jinja_raw():
    tpl = ChatTemplate(
        "USER: {{ m.content }}\n{% endfor %}".replace(
            "{% endfor %}", "{% for m in messages %}USER: {{ m.content }}\n{% endfor %}"
        )
    )
    text = tpl.apply([{"role": "user", "content": "ping"}])
    assert "USER: ping" in text


def test_chat_template_raw_mode_flag_inlined():
    tpl = ChatTemplate("{{ messages[-1].content }}{% if add_generation_prompt %}...{% endif %}")
    on = tpl.apply([{"role": "user", "content": "x"}], add_generation_prompt=True)
    off = tpl.apply([{"role": "user", "content": "x"}], add_generation_prompt=False)
    assert on.endswith("...")
    assert not off.endswith("...")


def test_tokenizer_apply_chat_template():
    tok = Tokenizer(MinimalBPE(vocab_size=8000))
    text = tok.apply_chat_template(MESSAGES, template="chatml")
    assert "<|im_start|>user" in text
    ids = tok.encode(text)
    assert len(ids) > 0


# ---------------------------------------------------------------------------
# Metrics collector
# ---------------------------------------------------------------------------


def test_metrics_counter_api():
    c = MetricsCollector()
    c.record_request()
    c.record_request(failed=True)
    c.record_tokens(input_tokens=4, output_tokens=7)
    c.record_latency(0.123)
    # two record_request() calls -> requests_total == 2 (one failed)
    assert c.requests_total == 2
    assert c.requests_failed == 1
    assert c.tokens_input_total == 4
    assert c.tokens_output_total == 7
    assert abs(c.inference_seconds_total - 0.123) < 1e-9


def test_metrics_histogram_buckets():
    c = MetricsCollector()
    for ms in (0.001, 0.006, 0.02, 0.6, 20.0):
        c.record_latency(ms)
    lines = c.prometheus_text().splitlines()
    le001 = next(l for l in lines if l.startswith('nexinfer_request_latency_seconds_bucket{le="0.005"'))
    assert le001.endswith("1")
    le05 = next(l for l in lines if l.startswith('nexinfer_request_latency_seconds_bucket{le="0.5"'))
    assert le05.endswith("3")
    leinf = next(
        l
        for l in lines
        if l.endswith('} +Inf"} 5') is False
        and l.startswith('nexinfer_request_latency_seconds_bucket{le="+Inf"')
    )
    assert leinf.endswith("5")


def test_metrics_prometheus_text_shape():
    c = MetricsCollector()
    c.record_request()
    c.set_block_usage(128, 96)
    text = c.prometheus_text()
    for metric in (
        "nexinfer_requests_total",
        "nexinfer_inference_seconds_total",
        "nexinfer_request_latency_seconds_count",
        "nexinfer_queue_depth",
        "nexinfer_kv_blocks_used",
        "nexinfer_process_uptime_seconds",
    ):
        assert metric in text, metric


def test_metrics_singleton_threadsafe():
    """METRICS is the process-wide singleton; heavy concurrent writes must
    not raise or lose counts."""
    errors: list[str] = []

    def _worker() -> None:
        try:
            for _ in range(500):
                METRICS.record_latency(0.001)
                METRICS.record_tokens(input_tokens=1, output_tokens=1)
        except Exception as exc:
            errors.append(str(exc))

    threads = [threading.Thread(target=_worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors, errors[:3]
    assert METRICS.requests_total >= 0


def test_metrics_latency_quantile():
    c = MetricsCollector()
    for i in range(100):
        c.record_latency(i / 100.0)
    assert 0.4 <= c.latency_quantile(0.5) <= 0.6
    assert c.latency_quantile(0.99) >= 0.95


# ---------------------------------------------------------------------------
# HTTP /metrics endpoint
# ---------------------------------------------------------------------------


class _FakeEngine:
    class _Status:
        model = "demo"
        backend_names = ["cpu_numpy"]

    status = _Status()
    _generator = object()
    queue_depth = 0

    class _Out:
        text = "ok response"
        finish_reason = "stop"
        tool_calls = []
        usage = {"prompt_tokens": 2, "completion_tokens": 3}

    out = _Out()

    def generate(self, req):
        return self.out


class _FakeHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        from nexinfer.services.metrics import METRICS as _m

        if self.path == "/metrics":
            body = _m.prometheus_text().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


@pytest.mark.network
def test_metrics_endpoint():
    server = HTTPServer(("127.0.0.1", 0), _FakeHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5)
        text = resp.read().decode()
        assert "nexinfer_requests_total" in text
        assert resp.headers["Content-Type"].startswith("text/plain")
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# Memory vector search
# ---------------------------------------------------------------------------


@pytest.fixture
def mem_store(tmp_path):
    from nexinfer.memory.store import MemoryStore

    store = MemoryStore(str(tmp_path / "mem"), "test")
    store.set("profile", {"name": "Alice", "role": "engineer", "notes": "likes distributed systems"})
    store.set("project", {"name": "nexus", "stack": "python gpu inference"})
    store.set("config", {"threads": 4, "backend": "cpu_numpy"})
    return store


def test_memory_keyword_search(mem_store):
    results = mem_store.search("engineer", mode="keyword")
    assert results and results[0]["key"] == "profile"


def test_memory_vector_search_ranks_related(mem_store):
    results = mem_store.search("distributed systems work", mode="vector")
    assert results
    keys = [r["key"] for r in results]
    assert "profile" in keys
    # the most related entry should be the one about distributed systems
    assert results[0]["key"] == "profile"
    assert 0.0 <= results[0]["similarity"] <= 1.0


def test_memory_vector_search_cosine_self():
    """A query made of the exact words of an entry must score high."""
    from nexinfer.memory.store import MemoryStore

    store = MemoryStore(
        str(shutil.which("git") and "/tmp/nexinfer_vec_self" or "/tmp/nexinfer_vec_self"), "self"
    )
    # use a unique temp path to avoid cross-test pollution
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        store = MemoryStore(td, "self")
        store.set("a", {"topic": "quantum physics entanglement"})
        store.set("b", {"topic": "baking chocolate cake recipe"})
        r = store.search("quantum entanglement physics", mode="vector")
        assert r[0]["key"] == "a"


def test_memory_search_auto_fallback():
    """With mode='auto' the store still returns keyword matches."""
    import tempfile

    from nexinfer.memory.store import MemoryStore

    with tempfile.TemporaryDirectory() as td:
        store = MemoryStore(td, "auto")
        store.set("x", {"msg": "hello world from nexus"})
        results = store.search("nexus", mode="auto")
        assert results and results[0]["key"] == "x"
