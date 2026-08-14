"""NexusInfer CLI.

Commands::

    nexinfer run        -- run a prompt against a model (local)
    nexinfer serve      -- start the HTTP server (OpenAI-compatible)
    nexinfer chat       -- interactive chat with skills
    nexinfer profile    -- show detected hardware + placement plan
    nexinfer cluster    -- start coordinator or worker node
    nexinfer mcp        -- manage MCP servers (add/list/tools/call)
    nexinfer memory     -- manage git-backed memory stores
    nexinfer backends   -- list available backends and detected devices
    nexinfer web        -- test the internet gateway
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import Any

from nexinfer.engine.types import GenerationRequest


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nexinfer", description="NexusInfer: cross-platform distributed inference engine")
    p.add_argument("--verbose", "-v", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    # run
    pr = sub.add_parser("run", help="run a prompt against a model")
    pr.add_argument("--model", required=True, help="model dir, npz, or HF repo id")
    pr.add_argument("--prompt", required=True)
    pr.add_argument("--max-tokens", type=int, default=64)
    pr.add_argument("--backend", default=None, help="force a backend name")
    pr.add_argument("--temperature", type=float, default=0.8)
    pr.add_argument("--stop", action="append", default=[])
    pr.add_argument("--layers", type=int, default=12, help="model layers (planning/demo models)")
    pr.add_argument("--hidden", type=int, default=768, help="hidden size (planning/demo models)")

    # serve
    ps = sub.add_parser("serve", help="start the HTTP server")
    ps.add_argument("--model", required=True)
    ps.add_argument("--host", default="0.0.0.0")
    ps.add_argument("--port", type=int, default=8000)
    ps.add_argument("--backend", default=None)
    ps.add_argument("--layers", type=int, default=12)
    ps.add_argument("--hidden", type=int, default=768)

    # chat
    pc = sub.add_parser("chat", help="interactive chat")
    pc.add_argument("--model", required=True)
    pc.add_argument("--skill", default="default")
    pc.add_argument("--agent", default=None)
    pc.add_argument("--layers", type=int, default=12)
    pc.add_argument("--hidden", type=int, default=768)

    # profile
    pp = sub.add_parser("profile", help="detect hardware and show placement plan")
    pp.add_argument("--benchmark", action="store_true")
    pp.add_argument("--layers", type=int, default=12, help="model layers (for planning)")
    pp.add_argument("--hidden", type=int, default=768)

    # cluster
    pcl = sub.add_parser("cluster", help="start coordinator or worker")
    pcl.add_argument("role", choices=["coordinator", "worker"])
    pcl.add_argument("--node-id", required=True)
    pcl.add_argument("--host", default="0.0.0.0")
    pcl.add_argument("--port", type=int, default=9000)
    pcl.add_argument("--transport", default="tcp", choices=["tcp", "grpc", "webrtc", "rdma"])
    pcl.add_argument("--peers", nargs="*", default=[])
    pcl.add_argument("--layers", type=int, default=12)
    pcl.add_argument("--hidden", type=int, default=768)

    # mcp
    pm = sub.add_parser("mcp", help="manage MCP servers")
    pms = pm.add_subparsers(dest="mcp_cmd", required=True)
    pma = pms.add_parser("add", help="add an MCP server")
    pma.add_argument("name")
    pma.add_argument("--command")
    pma.add_argument("--args", nargs="*")
    pma.add_argument("--url", help="SSE/streamable-HTTP server URL")
    pma.add_argument("--env", nargs="*", default=[])
    pml = pms.add_parser("list", help="list configured MCP servers")
    pmt = pms.add_parser("tools", help="list tools of a server")
    pmt.add_argument("name")

    # memory
    pmn = sub.add_parser("memory", help="manage git-backed memory stores")
    pmns = pmn.add_subparsers(dest="memory_cmd", required=True)
    c = pmns.add_parser("create", help="create a store")
    c.add_argument("name")
    c.add_argument("--kind", choices=["private", "shared", "grant"], default="private")
    c.add_argument("--owners", nargs="*", default=["default"])
    g = pmns.add_parser("grant", help="grant agent write access")
    g.add_argument("store")
    g.add_argument("agent")
    w = pmns.add_parser("write", help="write a key to a branch")
    w.add_argument("store")
    w.add_argument("key")
    w.add_argument("value")
    w.add_argument("--branch", default="main")
    w.add_argument("--message", default="cli update")
    r = pmns.add_parser("read", help="read a key")
    r.add_argument("store")
    r.add_argument("key")
    r.add_argument("--branch", default="main")
    ls = pmns.add_parser("list", help="list keys in a branch")
    ls.add_argument("store")
    ls.add_argument("--branch", default="main")
    h = pmns.add_parser("history", help="show commit history")
    h.add_argument("store")
    h.add_argument("--branch", default="main")
    mg = pmns.add_parser("merge", help="merge source branch into target")
    mg.add_argument("store")
    mg.add_argument("source")
    mg.add_argument("--target", default="main")
    mg.add_argument("--policy", choices=["ours", "theirs", "last_write"], default="theirs")
    sy = pmns.add_parser("sync", help="push/pull store snapshots between nodes")
    sy.add_argument("direction", choices=["push", "pull"])
    sy.add_argument("store")
    sy.add_argument("path", help="dir (push) or snapshot file (pull)")

    # backends
    pb = sub.add_parser("backends", help="list backends and detected devices")

    # web
    pw = sub.add_parser("web", help="test the internet gateway")
    pw.add_argument("url")
    pw.add_argument("--max-tokens", type=int, default=1000)
    return p


# ----------------------------------------------------------------------
# Command implementations


def _cmd_run(args: argparse.Namespace) -> None:
    from nexinfer.backends.base import ModelSpec
    from nexinfer.engine.runtime import Engine

    engine = Engine()
    spec = _default_spec(args)
    engine.bootstrap(args.model, spec, backend_name=args.backend)
    req = GenerationRequest(prompt=args.prompt, max_tokens=args.max_tokens,
                            temperature=args.temperature, stop_sequences=args.stop)
    out = engine.generate(req)
    print(out.text)
    print(f"\n[{out.finish_reason} | {out.usage}]")


def _cmd_serve(args: argparse.Namespace) -> None:
    from nexinfer.backends.base import ModelSpec
    from nexinfer.engine.runtime import Engine
    from nexinfer.cli.http_server import HttpServer

    engine = Engine()
    spec = _default_spec(args)
    engine.bootstrap(args.model, spec, backend_name=args.backend)
    server = HttpServer(engine)
    print(f"NexusInfer server on http://{args.host}:{args.port} (OpenAI-compatible /v1/chat/completions)")
    server.run(args.host, args.port)


def _cmd_chat(args: argparse.Namespace) -> None:
    from nexinfer.backends.base import ModelSpec
    from nexinfer.engine.runtime import Engine
    from nexinfer.services.skills import SkillsRegistry

    engine = Engine()
    spec = _default_spec(args)
    engine.bootstrap(args.model, spec)
    skills = SkillsRegistry()
    skill, tools = skills.resolve_tools(args.skill)
    print(f"chat (skill={skill.name}, tools={tools}). Type 'quit' to exit.")
    history = ""
    while True:
        try:
            line = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line.strip().lower() in ("quit", "exit"):
            break
        if not line.strip():
            continue
        req = GenerationRequest(prompt=line.strip(), max_tokens=256, agent_id=args.agent,
                                tools=[engine.backend.capabilities() and {} for _ in []])
        out = engine.generate(req)
        print(out.text)
        if out.tool_calls:
            for tc in out.tool_calls:
                print(f"[tool call] {tc['name']} {json.dumps(tc['arguments'])}")
        history += out.text + "\n"


def _cmd_profile(args: argparse.Namespace) -> None:
    from nexinfer.backends.base import ModelSpec
    from nexinfer.engine.orchestrator import plan_placement
    from nexinfer.engine.profiler import SystemProfile

    system = SystemProfile.from_system(benchmark=args.benchmark)
    spec = ModelSpec(
        num_layers=args.layers, hidden_size=args.hidden, num_attention_heads=8,
        num_kv_heads=4, head_dim=args.hidden // 8, vocab_size=32000, inter_dim=args.hidden * 4,
    )
    placement = plan_placement(spec, system)
    print(f"CPU cores : {system.cpu_cores}")
    print(f"RAM       : {system.total_ram_gb:.1f} GB")
    print(f"GPU VRAM  : {system.gpu_vram_gb:.1f} GB")
    print("Devices:")
    for d in system.devices:
        print(f"  {d.device_id:24s} {d.name:40s} {d.total_memory_bytes / 1024 ** 3:7.1f} GB")
    print(f"\nPlacement strategy: {placement.strategy}")
    for dev, ranges in placement.assignments.items():
        print(f"  {dev}: layers {ranges}")
    print(f"KV-cache: device={placement.kv_cache_device}, device_blocks={placement.kv_cache_blocks_device}, "
          f"host_blocks={placement.kv_cache_blocks_host}")
    for note in placement.notes:
        print(" -", note)


def _cmd_cluster(args: argparse.Namespace) -> None:
    from nexinfer.backends.base import ModelSpec
    from nexinfer.backends.cpu_numpy import NumpyBackend
    from nexinfer.distributed.coordinator import Coordinator
    from nexinfer.distributed.worker import Worker
    from nexinfer.transports.registry import make_transport

    spec = ModelSpec(num_layers=args.layers, hidden_size=args.hidden, num_attention_heads=8,
                     num_kv_heads=4, head_dim=args.hidden // 8, vocab_size=32000, inter_dim=args.hidden * 4)
    transport = make_transport(args.transport)
    if args.role == "coordinator":
        coord = Coordinator(args.node_id, spec, transport, args.host, args.port, manual_peers=args.peers)
        port = asyncio.run(coord.start())
        print(f"coordinator {args.node_id} listening on port {port}")
        try:
            asyncio.get_event_loop().run_forever()
        except KeyboardInterrupt:
            asyncio.run(coord.close())
    else:
        backend = NumpyBackend()
        worker = Worker(args.node_id, backend, spec, None, transport, rank=1,
                        control_host=args.host, control_port=args.port)
        asyncio.run(worker.start())


def _cmd_mcp(args: argparse.Namespace) -> None:
    from nexinfer.cli.config import McpConfig

    cfg = McpConfig()
    if args.mcp_cmd == "add":
        config: dict[str, Any] = {}
        if args.url:
            config = {"url": args.url}
        else:
            config = {"command": args.command or "npx", "args": args.args or [],
                      "env": dict(kv.split("=", 1) for kv in args.env if "=" in kv)}
        cfg.add(args.name, config)
        print(f"added MCP server {args.name!r}")
    elif args.mcp_cmd == "list":
        for name, conf in cfg.list_all().items():
            print(f"{name}: {json.dumps(conf)}")
    elif args.mcp_cmd == "tools":
        print("use `nexinfer serve` with an MCP-enabled config to connect servers (SDK required)")


def _cmd_memory(args: argparse.Namespace) -> None:
    from nexinfer.memory.fabric import MemoryFabric

    fabric = MemoryFabric()
    if args.memory_cmd == "create":
        fabric.create_store(args.name, kind=args.kind, owners=args.owners)
        print(f"created store {args.name!r} ({args.kind})")
    elif args.memory_cmd == "grant":
        fabric.grant(args.store, args.agent)
        print(f"granted {args.agent!r} on {args.store!r}")
    elif args.memory_cmd == "write":
        store = fabric.get_store(args.store) or fabric.create_store(args.store)
        cid = store.set(args.key, args.value, branch=args.branch, message=args.message)
        print(f"committed {cid}")
    elif args.memory_cmd == "read":
        store = fabric.get_store(args.store)
        val = store.get(args.key, branch=args.branch) if store else None
        print(json.dumps(val, indent=2))
    elif args.memory_cmd == "list":
        store = fabric.get_store(args.store)
        print(json.dumps(store.list_keys(branch=args.branch) if store else [], indent=2))
    elif args.memory_cmd == "history":
        store = fabric.get_store(args.store)
        if store:
            for c in store.history(branch=args.branch):
                print(f"{c.oid}  {c.message}")
    elif args.memory_cmd == "merge":
        store = fabric.get_store(args.store)
        cid = store.merge(args.source, target=args.target, policy=args.policy) if store else ""
        print(f"merged -> {cid}")
    elif args.memory_cmd == "sync":
        if args.direction == "push":
            path = fabric.push_snapshot(args.store, args.path)
            print(f"snapshot written: {path}")
        else:
            n = fabric.pull_snapshot(args.store, args.path)
            print(f"imported {n} keys")


def _cmd_backends(args: argparse.Namespace) -> None:  # pylint: disable=unused-argument
    from nexinfer.backends.registry import available_backends, load_backend
    from nexinfer.engine.profiler import SystemProfile

    print("Registered backends:")
    for name, spec in sorted(available_backends().items()):
        be = load_backend(name, allow_missing=True)
        status = "OK" if be is not None else "unavailable"
        devices = ", ".join(d.device_id for d in (be.detect_devices() if be else []))
        print(f"  {name:18s} {status:12s} {devices}")
    print("\nDetected system devices:")
    system = SystemProfile.from_system()
    for d in system.devices:
        print(f"  {d.device_id:24s} {d.name}")


def _cmd_web(args: argparse.Namespace) -> None:
    from nexinfer.services.internet_gateway import InternetGateway

    gw = InternetGateway()
    result = gw.call(args.url, max_tokens=args.max_tokens)
    print(json.dumps(result, indent=2)[:2000])


def _default_spec(args: argparse.Namespace) -> "ModelSpec":
    from nexinfer.backends.base import ModelSpec

    hidden = getattr(args, "hidden", 768) or 768
    return ModelSpec(
        num_layers=getattr(args, "layers", 12) or 12,
        hidden_size=hidden,
        num_attention_heads=8, num_kv_heads=4, head_dim=hidden // 8,
        vocab_size=32000, inter_dim=hidden * 4,
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    dispatch = {
        "run": _cmd_run, "serve": _cmd_serve, "chat": _cmd_chat,
        "profile": _cmd_profile, "cluster": _cmd_cluster, "mcp": _cmd_mcp,
        "memory": _cmd_memory, "backends": _cmd_backends, "web": _cmd_web,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
