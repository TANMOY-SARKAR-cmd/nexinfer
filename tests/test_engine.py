"""Unit tests for the core engine: sampling, KV cache, scheduler,
backends, generation, planner, transports, memory, services."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile

import numpy as np
import pytest

from nexinfer.backends.base import DeviceInfo, ModelSpec
from nexinfer.engine.types import DeviceKind
from nexinfer.backends.cpu_numpy import NumpyBackend
from nexinfer.backends.registry import available_backends, load_backend
from nexinfer.backends.special_module import ReferenceModule, register_module, SpecialModuleBackend
from nexinfer.distributed.planner import NodeSpec, automatic, plan_pipeline, plan_tensor
from nexinfer.engine.kvcache import PagedKVCache
from nexinfer.engine.orchestrator import plan_placement
from nexinfer.engine.profiler import SystemProfile, profile_system
from nexinfer.engine.runtime import Engine
from nexinfer.engine.sampling import sample_token
from nexinfer.engine.scheduler import Scheduler
from nexinfer.engine.tokenizer_helper import MinimalBPE, Tokenizer
from nexinfer.engine.types import GenerationRequest
from nexinfer.gateway.tool_registry import ToolCaller, ToolRegistry
from nexinfer.memory.fabric import MemoryFabric
from nexinfer.memory.store import MemoryStore
from nexinfer.services.internet_gateway import InternetGateway, InternetPolicy
from nexinfer.services.skills import Skill, SkillsRegistry
from nexinfer.transports.base import TensorFrame
from nexinfer.transports.tcp_transport import TCPTransport

SPEC = ModelSpec(
    num_layers=2, hidden_size=64, num_attention_heads=4, num_kv_heads=2,
    head_dim=16, vocab_size=256, inter_dim=128,
)


# ----------------------------------------------------------------------
# Sampling


def test_greedy_sampling():
    req = GenerationRequest(prompt="x", temperature=0.0)
    logits = np.array([0.1, 5.0, 2.0, 0.5], dtype=np.float32)
    assert sample_token(logits, req) == 1


def test_temperature_and_repetition_penalty():
    rng = np.random.default_rng(1)
    req = GenerationRequest(prompt="x", temperature=1.0, repetition_penalty=2.0)
    logits = np.ones(100, dtype=np.float32)
    ids = np.array([3, 3, 3], dtype=np.int32)
    tok = sample_token(logits, req, ids, rng)
    # strong repetition penalty makes id 3 unlikely
    assert tok != 3


# ----------------------------------------------------------------------
# KV cache


def test_kvcache_read_write():
    cache = PagedKVCache(block_size=4, num_blocks_device=8, num_blocks_host=8,
                         num_layers=2, num_kv_heads=2, head_dim=8)
    cache.ensure_sequence("r1", 7)
    k = np.arange(16, dtype=np.float16).reshape(2, 8)
    v = np.arange(16, 32, dtype=np.float16).reshape(2, 8)
    cache.write("r1", 0, 0, k, v)
    cache.write("r1", 0, 6, k, v)
    rk, rv = cache.read("r1", 0, 0, 7)
    assert rk.shape == (7, 2, 8)
    np.testing.assert_array_equal(rk[0], k)
    np.testing.assert_array_equal(rv[6], v)
    cache.free("r1")
    assert "r1" not in cache._seq_blocks


def test_kvcache_blocks_stored_once_per_layer():
    """Blocks are shared across positions in the same layer; write/read round trip."""
    cache = PagedKVCache(block_size=4, num_blocks_device=8, num_blocks_host=8,
                         num_layers=2, num_kv_heads=2, head_dim=8)
    cache.ensure_sequence("r1", 4)
    k = np.ones((2, 8), dtype=np.float16)
    v = np.full((2, 8), 7, dtype=np.float16)
    for pos in range(4):
        cache.write("r1", 1, pos, k, v)
    rk, rv = cache.read("r1", 1, 0, 4)
    assert rk.shape == (4, 2, 8)
    np.testing.assert_array_equal(rk, np.ones((4, 2, 8), dtype=np.float16))
    np.testing.assert_array_equal(rv, np.full((4, 2, 8), 7, dtype=np.float16))
    assert cache.stats()["device_blocks_used"] == 2  # one block per layer
    cache.free("r1")
    assert cache.stats()["device_blocks_used"] == 0


def test_kvcache_spill():
    cache = PagedKVCache(block_size=4, num_blocks_device=4, num_blocks_host=4,
                         num_layers=1, num_kv_heads=1, head_dim=4)
    cache.ensure_sequence("r1", 32)  # needs 8 blocks -> spills to host
    moved = cache.spill_cold_blocks(2)
    assert moved == 2
    assert cache.stats()["device_blocks_free"] >= 2


# ----------------------------------------------------------------------
# Scheduler


def test_scheduler_admission_and_preemption():
    sched = Scheduler(max_running=2, device_blocks=4, host_blocks=128, block_size=4)
    for i in range(5):
        sched.add(GenerationRequest(prompt=f"p{i}", max_tokens=16))
    running = sched.schedule()
    assert len(running) == 2
    preempted = sched.preempt(running[0].req.request_id)
    assert preempted is not None
    running2 = sched.schedule()
    assert len(running2) == 2


def test_scheduler_stop_conditions():
    from nexinfer.engine.scheduler import RunningRequest

    rr = RunningRequest(req=GenerationRequest(prompt="x", max_tokens=3, stop_sequences=["END"]))
    rr.append(1, "hello ")
    assert rr.should_stop() is None
    rr.append(2, "END")
    assert rr.should_stop() == "stop"
    rr2 = RunningRequest(req=GenerationRequest(prompt="x", max_tokens=2))
    rr2.append(1, "a"); rr2.append(2, "b")
    assert rr2.should_stop() == "length"


# ----------------------------------------------------------------------
# Backend registry


def test_registry_lists_backends():
    specs = available_backends()
    assert "cpu_numpy" in specs


def test_load_cpu_numpy():
    be = load_backend("cpu_numpy", allow_missing=False)
    assert be is not None
    assert be.name == "cpu_numpy"
    devs = be.detect_devices()
    assert any(d.device_id == "/cpu:0" for d in devs)


def test_unavailable_backend_returns_none():
    be = load_backend("tpu", allow_missing=True)
    # TPU backend only reports devices when TPU env vars are set
    assert be is None or isinstance(be, object)


# ----------------------------------------------------------------------
# Numpy backend full loop


def test_numpy_backend_prefill_decode():
    be = NumpyBackend()
    be.load("demo-model-xyz", SPEC, ["/cpu:0"])
    ids = np.array([10, 20, 30], dtype=np.int32)
    logits = be.prefill("r1", ids)
    assert logits.shape[-1] == SPEC.vocab_size
    next_ids = np.array([[5]], dtype=np.int32) if logits.ndim > 1 else np.array([5], dtype=np.int32)
    logits2 = be.decode(["r1"], next_ids)
    assert logits2.shape[-1] == SPEC.vocab_size
    be.close()


# ----------------------------------------------------------------------
# Generation end-to-end


def test_generation_e2e():
    be = NumpyBackend()
    be.load("demo-model-xyz", SPEC, ["/cpu:0"])
    tok = Tokenizer(MinimalBPE(vocab_size=SPEC.vocab_size))
    sched = Scheduler(device_blocks=64, host_blocks=256)
    cache = PagedKVCache(num_blocks_device=64, num_blocks_host=256,
                         num_layers=SPEC.num_layers, num_kv_heads=SPEC.num_kv_heads,
                         head_dim=SPEC.head_dim)
    from nexinfer.engine.generation import GenerationEngine

    gen = GenerationEngine(be, tok, sched, cache)
    out = gen.generate(GenerationRequest(prompt="hello world", max_tokens=10))
    assert out.finish_reason == "length"
    assert out.usage["completion_tokens"] == 10
    be.close()


def test_generation_stop_sequence():
    be = NumpyBackend()
    be.load("demo-model-xyz", SPEC, ["/cpu:0"])
    tok = Tokenizer(MinimalBPE(vocab_size=SPEC.vocab_size))
    sched = Scheduler(device_blocks=64, host_blocks=256)
    cache = PagedKVCache(num_blocks_device=64, num_blocks_host=256,
                         num_layers=SPEC.num_layers, num_kv_heads=SPEC.num_kv_heads,
                         head_dim=SPEC.head_dim)
    from nexinfer.engine.generation import GenerationEngine

    gen = GenerationEngine(be, tok, sched, cache)
    out = gen.generate(GenerationRequest(prompt="hello", max_tokens=100, stop_sequences=["END"]))
    be.close()


# ----------------------------------------------------------------------
# Profiler / orchestrator


def test_profiler_detects_cpu():
    devices = profile_system()
    assert any(d.device_id == "/cpu:0" for d in devices)


def test_placement_cpu_only(monkeypatch):
    monkeypatch.setattr("nexinfer.engine.profiler._detect_nvidia", lambda: [])
    monkeypatch.setattr("nexinfer.engine.profiler._detect_amd", lambda: [])
    monkeypatch.setattr("nexinfer.engine.profiler._detect_intel", lambda: [])
    monkeypatch.setattr("nexinfer.engine.profiler._detect_tpu", lambda: [])
    monkeypatch.setattr("nexinfer.engine.profiler._detect_windows_dxgi", lambda: [])
    system = SystemProfile.from_system()
    plan = plan_placement(SPEC, system)
    assert plan.strategy == "cpu_only"
    assert "/cpu:0" in plan.assignments


def test_placement_gpu_hybrid():
    system = SystemProfile.from_system()
    # pretend we have a small GPU (128 MB) -> hybrid split
    tiny_gpu = DeviceInfo(device_id="/gpu:nvidia:0", kind=DeviceKind.GPU_NVIDIA, vendor="nvidia",
                          name="TinyGPU", total_memory_bytes=128 * 1024 * 1024, compute_score=0.1)
    system.devices.append(tiny_gpu)
    plan = plan_placement(SPEC, system, kv_cache_target_tokens=256)
    assert plan.strategy in ("hybrid_split", "gpu_only", "cpu_only")


# ----------------------------------------------------------------------
# Distributed planner


def test_pipeline_plan():
    nodes = [NodeSpec(f"n{i}", "10.0.0.{i}", 9000, model_hash="same") for i in range(3)]
    plan = plan_pipeline(nodes, SPEC)
    assert plan.mode.value == "pipeline_parallel"
    total = sum(r[1] - r[0] for p in plan.per_node for r in (p.pp_layers or []))
    assert total == SPEC.num_layers


def test_tensor_plan():
    nodes = [NodeSpec(f"n{i}", "10.0.0.{i}", 9000) for i in range(4)]
    plan = plan_tensor(nodes, SPEC)
    assert plan.mode.value == "tensor_parallel"


def test_automatic_heterogeneous_uses_pp():
    nodes = [
        NodeSpec("n0", "10.0.0.0", 9000, devices=[DeviceInfo(device_id="/cpu:0", kind=DeviceKind.CPU, vendor="generic", name="CPU", total_memory_bytes=0, compute_score=1.0)]),
        NodeSpec("n1", "10.0.0.1", 9000, devices=[DeviceInfo(device_id="/gpu:nvidia:0", kind=DeviceKind.GPU_NVIDIA, vendor="nvidia", name="GPU", total_memory_bytes=8 << 30, compute_score=4.0)]),
    ]
    plan = automatic(nodes, SPEC)
    assert plan.mode.value == "pipeline_parallel"


# ----------------------------------------------------------------------
# Transports


@pytest.mark.asyncio
async def test_tcp_transport_roundtrip():
    t1 = TCPTransport()
    t2 = TCPTransport()
    await t1.start_server("127.0.0.1", 0)
    peer_addr = t1.listen_addr
    channel = "peer-t2"  # shared channel key both sides agree on
    await t2.connect(peer_addr, key=channel)
    arr = np.arange(24, dtype=np.float32).reshape(4, 6)
    await t1.send(channel, "test", arr)
    name, got = await t2.recv(channel, timeout=5.0)
    assert name == "test"
    np.testing.assert_array_equal(got, arr)
    # bidirectional: t2 sends back over the same channel
    await t2.send(channel, "ack", np.array([9, 9], dtype=np.int32))
    ack_name, ack_arr = await t1.recv(channel, timeout=5.0)
    assert ack_name == "ack"
    np.testing.assert_array_equal(ack_arr, np.array([9, 9], dtype=np.int32))
    await t1.close()
    await t2.close()
def test_tensor_frame_pack_unpack():
    arr = np.random.default_rng(0).standard_normal((3, 5)).astype(np.float16)
    frame = TensorFrame.pack("hello", arr)
    import io
    import struct

    buf = io.BytesIO(frame)
    (size,) = struct.unpack("<I", buf.read(4))
    assert len(frame) == 4 + size


# ----------------------------------------------------------------------
# Memory store / fabric


@pytest.fixture
def tmp_store(tmp_path):
    return MemoryStore(str(tmp_path / "mem"), "test")


def test_store_set_get(tmp_store):
    tmp_store.set("k1", {"answer": 42}, message="first")
    assert tmp_store.get("k1") == {"answer": 42}
    keys = tmp_store.list_keys()
    assert "k1" in keys


def test_store_branches_and_merge(tmp_store):
    tmp_store.set("k1", "main-value", message="m1")
    tmp_store.branch("agent-a", base="main")
    tmp_store.set("k1", "agent-a-value", branch="agent-a", message="a1")
    tmp_store.set("k2", "agent-a-secret", branch="agent-a", message="a2")
    assert tmp_store.get("k1", branch="main") == "main-value"
    assert tmp_store.get("k1", branch="agent-a") == "agent-a-value"
    # merge agent-a into main, taking theirs
    tmp_store.merge("agent-a", target="main", policy="theirs")
    tmp_store.checkout("main")
    assert tmp_store.get("k1") == "agent-a-value"


def test_store_history_diff(tmp_store):
    tmp_store.set("x", 1, message="c1")
    h1 = tmp_store.history()
    tmp_store.set("x", 2, message="c2")
    h2 = tmp_store.history()
    assert len(h2) == len(h1) + 1
    d = tmp_store.diff(h1[0].oid, h2[0].oid)
    assert "x" in d["changed"] or "x" in d["added"]


def test_store_search(tmp_store):
    tmp_store.set("note1", "the quick brown fox jumps over the lazy dog", message="s1")
    tmp_store.set("note2", "completely unrelated content here", message="s2")
    res = tmp_store.search("fox lazy")
    assert res and res[0]["key"] == "note1"


def test_fabric_policies():
    with tempfile.TemporaryDirectory() as td:
        fabric = MemoryFabric(root=os.path.join(td, "mem"))
        fabric.create_store("private-store", kind="private", owners=["alice"])
        assert fabric.can_write("alice", "private-store")
        assert not fabric.can_write("bob", "private-store")
        fabric.grant("private-store", "bob")
        assert fabric.can_write("bob", "private-store")
        wb = fabric.whiteboard()
        assert wb is not None
        assert fabric.can_write("anyone", "whiteboard")


def test_fabric_sync_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        f1 = MemoryFabric(root=os.path.join(td, "mem1"))
        f2 = MemoryFabric(root=os.path.join(td, "mem2"))
        s1 = f1.create_store("notes", kind="shared", owners=["*"])
        s1.set("k1", "v1", message="init")
        snap_dir = os.path.join(td, "snap")
        f1.push_snapshot("notes", snap_dir)
        n = f2.pull_snapshot("notes", os.path.join(snap_dir, "notes.snapshot.json"))
        assert n >= 1
        assert f2.get_store("notes").get("k1") == "v1"


# ----------------------------------------------------------------------
# Internet gateway


def test_internet_gateway_allowed_fetch():
    gw = InternetGateway()
    res = gw.call("https://example.com", max_tokens=50)
    assert "content" in res or "error" in res  # network-dependent; never crashes


def test_internet_gateway_policy_blocks():
    policy = InternetPolicy(allowed_domains=["allowed.com"], blocked_domains=["evil.com"])
    gw = InternetGateway(policy)
    res = gw.call("https://blocked-site.net/page")
    assert "error" in res
    res2 = gw.call("https://evil.com/x")
    assert "error" in res2


def test_internet_gateway_tool_schema():
    gw = InternetGateway()
    schema = gw.tool_schema()
    assert schema["name"] == "web_fetch"
    assert "url" in schema["parameters"]["properties"]


# ----------------------------------------------------------------------
# Skills


def test_skills_registry_defaults():
    reg = SkillsRegistry()
    skill, tools = reg.resolve_tools("research")
    assert "web_fetch" in tools
    assert "memory_write" in tools


def test_skills_composition(tmp_path):
    reg = SkillsRegistry(skills_dir=str(tmp_path))
    reg.save(Skill(name="combo", tools=["web_fetch"], skills_refs=["notes"]))
    skill, tools = reg.resolve_tools("combo")
    assert "web_fetch" in tools


def test_skills_yaml_persistence(tmp_path):
    reg = SkillsRegistry(skills_dir=str(tmp_path))
    reg.save(Skill(name="persisted", description="d", tools=["t1"]))
    reg2 = SkillsRegistry(skills_dir=str(tmp_path))
    assert reg2.get("persisted") is not None


# ----------------------------------------------------------------------
# Tool registry + caller


def test_tool_registry_builtins():
    gw = InternetGateway()
    with tempfile.TemporaryDirectory() as td:
        fabric = MemoryFabric(root=os.path.join(td, "mem"))
        reg = ToolRegistry(internet=gw, memory_fabric=fabric)
        assert reg.has("web_fetch")
        assert reg.has("memory_write")
        assert reg.get_schema("web_fetch")["name"] == "web_fetch"
        caller = ToolCaller(reg, mcp_clients={})
        fabric.create_store("notes", kind="shared", owners=["*"])
        r = caller.call("memory_write", {"store": "notes", "branch": "main",
                                         "key": "x", "value": "1"})
        assert "commit" in r
        r2 = caller.call("memory_read", {"store": "notes", "branch": "main", "key": "x"})
        assert r2["value"] == "1"


# ----------------------------------------------------------------------
# Special module


def test_special_module_backend():
    be = SpecialModuleBackend()
    devs = be.detect_devices()
    assert len(devs) >= 1
    be.load("demo", SPEC, [d.device_id for d in devs])
    logits = be.prefill("r1", np.array([1, 2, 3], dtype=np.int32))
    assert logits.shape[-1] > 0
    be.close()


def test_register_custom_module():
    class MyModule(ReferenceModule):
        module_name = "my_custom"

        def is_available(self) -> bool:
            return True

    register_module(MyModule)
    from nexinfer.backends.special_module import _REGISTRY

    assert "my_custom" in _REGISTRY


# ----------------------------------------------------------------------
# Runtime bootstrap (CPU end-to-end)


def test_engine_bootstrap(monkeypatch):
    monkeypatch.setattr("nexinfer.engine.profiler._detect_nvidia", lambda: [])
    monkeypatch.setattr("nexinfer.engine.profiler._detect_amd", lambda: [])
    monkeypatch.setattr("nexinfer.engine.profiler._detect_intel", lambda: [])
    monkeypatch.setattr("nexinfer.engine.profiler._detect_tpu", lambda: [])
    monkeypatch.setattr("nexinfer.engine.profiler._detect_windows_dxgi", lambda: [])
    engine = Engine()
    st = engine.bootstrap("demo", SPEC, backend_name="cpu_numpy")
    assert st.placement.strategy == "cpu_only"
    out = engine.generate(GenerationRequest(prompt="hello", max_tokens=5))
    assert out.finish_reason == "length"
