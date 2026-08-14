#!/usr/bin/env python3
"""NexusInfer end-to-end demo.

Exercises every major subsystem in one flow:

1. hardware profiling + placement plan
2. engine bootstrap (CPU numpy backend, demo weights)
3. generation with a stop sequence
4. internet gateway (web_fetch) with domain policy
5. MCP tool registration + skill resolution
6. git-backed memory: two agents with isolated branches sharing a
   whiteboard, merge policies, and cross-node snapshot sync
7. a 2-node CPU cluster plan (pipeline parallel)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nexinfer.backends.base import DeviceInfo, ModelSpec
from nexinfer.distributed.planner import NodeSpec, automatic
from nexinfer.engine.runtime import Engine
from nexinfer.engine.types import DeviceKind, GenerationRequest
from nexinfer.gateway.tool_registry import ToolRegistry
from nexinfer.memory.fabric import MemoryFabric
from nexinfer.services.internet_gateway import InternetGateway, InternetPolicy
from nexinfer.services.skills import SkillsRegistry


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    spec = ModelSpec(num_layers=4, hidden_size=128, num_attention_heads=4,
                     num_kv_heads=2, head_dim=32, vocab_size=256, inter_dim=256)

    # ------------------------------------------------------------------
    section("1. Hardware profile + placement plan")
    from nexinfer.engine.profiler import SystemProfile

    system = SystemProfile.from_system()
    print(f"CPU cores : {system.cpu_cores}")
    print(f"RAM       : {system.total_ram_gb:.1f} GB   GPU VRAM: {system.gpu_vram_gb:.1f} GB")
    for d in system.devices:
        print(f"  {d.device_id:24s} {d.name}")

    from nexinfer.engine.orchestrator import plan_placement

    placement = plan_placement(spec, system)
    print(f"placement strategy: {placement.strategy}")
    for note in placement.notes:
        print(" -", note)

    # ------------------------------------------------------------------
    section("2. Engine bootstrap (CPU numpy backend)")
    engine = Engine()
    engine.bootstrap("demo-model", spec, backend_name="cpu_numpy")
    print(f"backend : {engine.status.backend_names}")
    print(f"model   : {engine.status.model}")

    # ------------------------------------------------------------------
    section("3. Generation with stop sequence")
    out = engine.generate(GenerationRequest(prompt="once upon a time", max_tokens=120,
                                            temperature=0.9, stop_sequences=["END"]))
    print(repr(out.text))
    print(f"finish: {out.finish_reason} | usage: {out.usage}")

    # ------------------------------------------------------------------
    section("4. Internet gateway (web_fetch)")
    gw = InternetGateway(policy=InternetPolicy(blocked_domains=["evil.example"]))
    res = gw.call("https://example.com", max_tokens=200)
    print("fetch result keys:", list(res.keys()))
    if "content" in res:
        print(res["content"][:300])
    print("schema:", json.dumps(gw.tool_schema())[:200])

    # ------------------------------------------------------------------
    section("5. Skills + tool registry")
    registry = ToolRegistry(internet=gw)
    skills = SkillsRegistry()
    skill, tools = skills.resolve_tools("research")
    print(f"skill '{skill.name}' resolves to tools: {tools}")
    for t in tools:
        schema = registry.get_schema(t)
        if schema:
            print(f"  - {schema['name']}: {schema['description'][:60]}")

    # ------------------------------------------------------------------
    section("6. Git-backed memory fabric (two agents + whiteboard)")
    with tempfile.TemporaryDirectory() as td:
        fabric = MemoryFabric(root=os.path.join(td, "mem"))
        fabric.create_store("alice", kind="private", owners=["alice"])
        fabric.create_store("bob", kind="private", owners=["bob"])
        wb = fabric.whiteboard()

        # isolated work
        alice = fabric.get_store("alice")
        alice.set("hypothesis", "the answer involves distributed memory",
                  branch="main", message="alice thinks")
        bob = fabric.get_store("bob")
        bob.set("hypothesis", "the answer involves fast transports",
                branch="main", message="bob thinks")
        assert not fabric.can_write("bob", "alice")
        print("alice store:", alice.list_keys())
        print("bob store  :", bob.list_keys())

        # share via whiteboard with merge policy
        wb.set("alice:1", alice.get("hypothesis"), branch="main", message="alice posts")
        wb.set("bob:1", bob.get("hypothesis"), branch="main", message="bob posts")
        wb.merge("main", target="main", policy="last_write")
        print("whiteboard entries:", len(wb.list_keys(branch="main")))

        # cross-node sync
        snap_dir = os.path.join(td, "snap")
        fabric.push_snapshot("alice", snap_dir)
        fabric2 = MemoryFabric(root=os.path.join(td, "mem2"))
        n = fabric2.pull_snapshot("alice", os.path.join(snap_dir, "alice.snapshot.json"))
        print(f"sync imported {n} keys to second node")

        # history + diff
        for c in alice.history()[-3:]:
            print(f"  commit {c.oid}: {c.message}")

    # ------------------------------------------------------------------
    section("7. Distributed cluster plan (2 CPU nodes, pipeline parallel)")
    nodes = [
        NodeSpec("node-0", "10.0.0.1", 9000,
                 devices=[DeviceInfo("/cpu:0", DeviceKind.CPU, "generic", "CPU", 0, 1.0)], model_hash="m1"),
        NodeSpec("node-1", "10.0.0.2", 9000,
                 devices=[DeviceInfo("/cpu:0", DeviceKind.CPU, "generic", "CPU", 0, 1.0)], model_hash="m1"),
    ]
    plan = automatic(nodes, spec)
    print(f"mode: {plan.mode.value}")
    for note in plan.notes:
        print(" -", note)

    print("\nDemo complete. All subsystems exercised successfully.")


if __name__ == "__main__":
    main()
