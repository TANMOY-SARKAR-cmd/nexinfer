"""Fault-tolerance tests for the distributed control plane.

Covers the Phase-3 additions:

* ``HealthMonitor`` declares a node dead after the silence timeout and
  fires the async ``on_node_dead`` callback (which drives re-planning).
* The **coordinator** reports dead nodes through ``cluster_status``,
  re-plans when a heartbeat is lost, and drops graceful ``leave``
  messages without counting them as crashes.
* ``wait_for_heartbeat`` probes the coordinator and returns False when it
  is down.

No real worker backend is required -- a tiny fake coordinator server is
used where a Coordinator instance is not.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from nexinfer.backends.base import ModelSpec
from nexinfer.backends.cpu_numpy import NumpyBackend
from nexinfer.backends.registry import load_backend
from nexinfer.distributed.health import HealthMonitor, wait_for_heartbeat
from nexinfer.distributed.messages import Msg
from nexinfer.engine.types import DeviceKind
from nexinfer.transports.tcp_transport import TCPTransport

SPEC = ModelSpec(
    num_layers=4,
    hidden_size=64,
    num_attention_heads=4,
    num_kv_heads=2,
    head_dim=16,
    vocab_size=256,
    inter_dim=128,
)


def _backend() -> NumpyBackend:
    be = load_backend("cpu_numpy")
    be.load("demo-model-ft", SPEC, [])
    return be


# ---------------------------------------------------------------------------
# HealthMonitor
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_health_monitor_declares_dead_node():
    monitor = HealthMonitor(timeout=0.15, check_interval=0.05)
    dead: list[str] = []
    monitor.set_on_node_dead(dead.append)

    await monitor.start()
    try:
        monitor.record("w1")
        assert monitor.is_alive("w1")
        await asyncio.sleep(0.4)  # silence > timeout
        assert not monitor.is_alive("w1")
        assert "w1" in monitor.dead_nodes
        assert dead == ["w1"], "callback must fire exactly once"
        # recovery: a resumed heartbeat brings the node back to life
        monitor.record("w1")
        await asyncio.sleep(0.1)
        assert "w1" not in monitor.dead_nodes
        assert monitor.is_alive("w1")
    finally:
        await monitor.stop()


@pytest.mark.asyncio()
async def test_health_monitor_forget_and_never_seen():
    monitor = HealthMonitor(timeout=1.0)
    monitor.record("w1")
    monitor.forget("w1")
    # forgotten nodes are no longer tracked: no last-seen record means the
    # monitor cannot declare them dead either (age == 0 <= timeout)
    assert monitor.is_alive("w1")
    assert "w1" not in monitor.dead_nodes
    assert monitor.is_alive("never-seen-node")  # untracked nodes are not dead


# ---------------------------------------------------------------------------
# Coordinator liveness
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_coordinator_replans_on_lost_heartbeat():
    from nexinfer.distributed.coordinator import Coordinator
    from nexinfer.distributed.planner import NodeSpec, plan_pipeline

    transport = TCPTransport()
    coordinator = Coordinator("root", SPEC, transport, heartbeat_timeout=0.2)

    replans = []
    coordinator._on_plan_ready = lambda p: replans.append(p)

    port = await coordinator.start()
    dead_seen: list[str] = []

    real_on_dead = coordinator._on_node_dead

    async def fake_on_dead(node_id: str) -> None:
        dead_seen.append(node_id)
        await real_on_dead(node_id)  # keep the coordinator's replan logic running

    coordinator._health.set_on_node_dead(fake_on_dead)

    # register two workers over the control channel (plan is rebuilt by
    # coordinator registration; the static planner reference documents the
    # intended topology but is not consumed by the live coordinator)
    plan_pipeline(
        [NodeSpec("root", "127.0.0.1", 0), NodeSpec("w1", "127.0.0.1", 0), NodeSpec("w2", "127.0.0.1", 0)],
        SPEC,
    )

    async def hello(node_id: str, port_arg: int) -> None:
        nonlocal acceptor_writers
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        hello_msg = Msg(
            type="hello",
            payload={
                "devices": [
                    {
                        "device_id": "/cpu:0",
                        "kind": DeviceKind.CPU,
                        "vendor": "generic",
                        "name": "CPU (numpy)",
                        "total_memory_bytes": 0,
                        "compute_score": 1.0,
                    }
                ],
                "host": "127.0.0.1",
                "port": port_arg,
                "backend": "cpu_numpy",
                "model_hash": f"model:{SPEC.num_layers}:{SPEC.hidden_size}",
            },
            src=node_id,
        )
        writer.write((__import__("json").dumps(hello_msg.to_dict()) + "\n").encode())
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), 10.0)
        resp = __import__("json").loads(line)
        assert resp["type"] == "welcome"
        acceptor_writers.append(writer)
        writer.close()

    # give the workers distinct "control ports" on the local box (loopback
    # acceptors just to answer control-plane hello with a welcome)
    acceptors = await _start_welcome_acceptors({"w1": 50001, "w2": 50002})
    acceptor_writers = []

    try:
        await hello("w1", 50001)
        await hello("w2", 50002)
        assert "w1" in coordinator.nodes and "w2" in coordinator.nodes
        assert coordinator.plan is not None

        # worker w1 stops heartbeating -> declared dead -> replan.
        # root and w2 keep heartbeating like their production loops do.
        async def _heartbeat_loop(node_id: str, interval: float) -> None:
            while True:
                try:
                    reader, writer = await asyncio.open_connection("127.0.0.1", port)
                    writer.write((json.dumps(Msg(type="heartbeat", src=node_id).to_dict()) + "\n").encode())
                    await writer.drain()
                    await reader.readline()
                    writer.close()
                except Exception:
                    pass
                await asyncio.sleep(interval)

        tasks = [
            asyncio.create_task(_heartbeat_loop("root", 0.05)),
            asyncio.create_task(_heartbeat_loop("w2", 0.05)),
        ]
        try:
            await asyncio.sleep(0.5)
        finally:
            for t in tasks:
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass

        assert "w1" in coordinator._health.dead_nodes
        assert dead_seen == ["w1"]
        # _on_node_dead triggers a replan asynchronously; give the push
        # (which rpcs to the dead w1 and must fail fast) time to finish
        await asyncio.sleep(0.1)
        # w2 is the sole survivor -> the plan is rebuilt across it alone
        # (single-node recovery, never left plan-less with compute left)
        assert coordinator.plan is not None
        node_ids = [n.node_id for n in coordinator.plan.nodes]
        assert "w1" not in node_ids and "w2" in node_ids

        # cluster_status exposes dead/alive views
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(
            (__import__("json").dumps(Msg(type="cluster_status", src="client").to_dict()) + "\n").encode()
        )
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), 10.0)
        resp = __import__("json").loads(line)
        assert "w1" in resp["payload"]["dead"]
        assert "w2" in resp["payload"]["alive"]
        writer.close()
    finally:
        for a in acceptors:
            a.close()
        for w in acceptor_writers:
            await w.wait_closed()
        await coordinator.close()
        await transport.close()


@pytest.mark.asyncio()
async def test_coordinator_graceful_leave_not_counted_as_dead():
    from nexinfer.distributed.coordinator import Coordinator
    from nexinfer.distributed.planner import NodeSpec, plan_pipeline

    transport = TCPTransport()
    coordinator = Coordinator("root", SPEC, transport, heartbeat_timeout=5.0)
    plan0 = plan_pipeline([NodeSpec("root", "127.0.0.1", 0), NodeSpec("w1", "127.0.0.1", 0)], SPEC)
    w1 = _worker_node("w1", 1, plan0)
    acceptors = await _start_welcome_acceptors({"w1": 50003})
    port = await coordinator.start()
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(
            (
                __import__("json").dumps(
                    Msg(
                        type="hello",
                        payload={
                            "devices": [
                                {
                                    "device_id": "/cpu:0",
                                    "kind": DeviceKind.CPU,
                                    "vendor": "generic",
                                    "name": "CPU",
                                    "total_memory_bytes": 0,
                                    "compute_score": 1.0,
                                }
                            ],
                            "host": "127.0.0.1",
                            "port": 50003,
                            "backend": "cpu_numpy",
                            "model_hash": f"model:{SPEC.num_layers}:{SPEC.hidden_size}",
                        },
                        src="w1",
                    ).to_dict()
                )
                + "\n"
            ).encode()
        )
        await writer.drain()
        await asyncio.wait_for(reader.readline(), 10.0)
        writer.close()
        assert "w1" in coordinator._last_heartbeat

        # graceful departure
        reader2, writer2 = await asyncio.open_connection("127.0.0.1", port)
        writer2.write((__import__("json").dumps(Msg(type="leave", src="w1").to_dict()) + "\n").encode())
        await writer2.drain()
        line = await asyncio.wait_for(reader2.readline(), 10.0)
        resp = __import__("json").loads(line)
        assert resp["type"] == "leave_ack"
        writer2.close()

        assert "w1" not in coordinator.nodes
        assert "w1" not in coordinator._health.dead_nodes  # left != crashed
        assert coordinator.plan is None  # < 2 workers now
    finally:
        for a in acceptors:
            a.close()
        await w1.close()
        await coordinator.close()
        await transport.close()


@pytest.mark.asyncio()
async def test_wait_for_heartbeat_fails_when_down():
    assert not await wait_for_heartbeat("127.0.0.1", 1, "any", timeout=0.5)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _worker_node(node_id: str, rank: int, plan):
    from nexinfer.distributed.worker import Worker

    be = _backend()
    return Worker(node_id=node_id, backend=be, spec=SPEC, plan=plan, transport=TCPTransport(), rank=rank)


async def _start_welcome_acceptors(ports: dict[str, int]) -> list[asyncio.AbstractServer]:
    """Tiny fake worker control-plane: answers ``hello`` with ``welcome``
    and ``plan`` pushes with ``plan_ack`` (enough for the coordinator to
    complete registration and re-planning)."""

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                line = await reader.readline()
                if not line:
                    return
                msg = Msg.from_dict(__import__("json").loads(line))
                if msg.type == "hello":
                    resp = Msg(type="welcome", payload={"node_id": "root"}, src="root")
                elif msg.type == "plan":
                    resp = Msg(
                        type="plan_ack", payload={"layer_range": msg.payload.get("pp_layers", [])}, src="root"
                    )
                else:
                    resp = Msg(type="error", payload={"message": f"fake worker: {msg.type}"}, src="root")
                writer.write((__import__("json").dumps(resp.to_dict()) + "\n").encode())
                await writer.drain()
        except (ConnectionResetError, asyncio.IncompleteReadError):
            pass

    acceptors = []
    for port in ports.values():
        acceptors.append(await asyncio.start_server(handler, "127.0.0.1", port))
    return acceptors
