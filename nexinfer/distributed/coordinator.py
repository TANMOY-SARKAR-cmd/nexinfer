"""Cluster coordinator.

The coordinator (rank-0 node, elected per cluster) is responsible for:

* peer registration and heartbeat liveness,
* local mDNS service announcement (zeroconf) so nodes find each other
  on a LAN without config,
* pushing the ``ClusterPlan`` to workers,
* resplitting the model when nodes join/leave,
* collecting the final logits from the pipeline/TP root and returning
  them to the generation engine.

Manual peer config is also supported (``--peers``), which is required
for machines on different networks; NAT traversal then uses the
WebRTC transport.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Callable

from nexinfer.backends.base import DeviceInfo, ModelSpec
from nexinfer.distributed.messages import Msg
from nexinfer.distributed.planner import ClusterPlan, NodeSpec, automatic
from nexinfer.engine.profiler import SystemProfile
from nexinfer.transports.base import Transport

log = logging.getLogger("nexinfer.distributed.coordinator")

SERVICE_TYPE = "_nexinfer._tcp.local."


class Coordinator:
    def __init__(
        self,
        node_id: str,
        spec: ModelSpec,
        transport: Transport,
        host: str = "0.0.0.0",
        port: int = 0,
        manual_peers: list[str] | None = None,
    ) -> None:
        self.node_id = node_id
        self.spec = spec
        self.transport = transport
        self.host = host
        self.port = port
        self.manual_peers = manual_peers or []
        self.nodes: dict[str, NodeSpec] = {}
        self.plan: ClusterPlan | None = None
        self._server: asyncio.AbstractServer | None = None
        self._last_heartbeat: dict[str, float] = {}
        self._on_plan_ready: Callable | None = None

    # ------------------------------------------------------------------

    async def start(self) -> int:
        srv = await asyncio.start_server(self._handle_control, self.host, self.port)
        self._server = srv
        self.port = srv.sockets[0].getsockname()[1]
        self.nodes[self.node_id] = NodeSpec(
            node_id=self.node_id, host=self.host, port=self.port,
            devices=SystemProfile.from_system().devices, model_hash=f"model:{self.spec.num_layers}:{self.spec.hidden_size}",
        )
        await self.transport.start_server(self.host, self.port + 1000, on_peer=self._peer_connected)
        await self._announce_mdns()
        log.info("coordinator %s on port %d", self.node_id, self.port)
        return self.port

    async def _announce_mdns(self) -> None:
        try:
            from zeroconf import Zeroconf, ServiceInfo
            import socket

            zc = Zeroconf()
            info = ServiceInfo(
                SERVICE_TYPE,
                f"nexinfer-{self.node_id}.{SERVICE_TYPE}",
                addresses=[socket.inet_aton("127.0.0.1")],
                port=self.port,
                properties={"node_id": self.node_id},
            )
            zc.register_service(info)
        except Exception as exc:  # noqa: BLE001
            log.debug("mDNS announce failed (non-fatal): %s", exc)

    def _peer_connected(self, peer: str) -> None:
        log.info("peer connected: %s", peer)

    async def _handle_control(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                line = await reader.readline()
                if not line:
                    return
                msg = Msg.from_dict(json.loads(line))
                resp = await self._dispatch(msg)
                writer.write((json.dumps(resp.to_dict()) + "\n").encode())
                await writer.drain()
        except (ConnectionResetError, asyncio.IncompleteReadError):
            pass

    async def _dispatch(self, msg: Msg) -> Msg:
        if msg.type == "hello":
            devices = [
                DeviceInfo(
                    device_id=d.get("device_id", "/cpu:0"),
                    kind=d.get("kind", "/cpu"),
                    vendor=d.get("vendor", "generic"),
                    name=d.get("name", "unknown"),
                    total_memory_bytes=d.get("total_memory_bytes", 0),
                    compute_score=d.get("compute_score", 1.0),
                )
                for d in msg.payload.get("devices", [])
            ]
            spec = msg.payload.get("spec", {})
            node = NodeSpec(
                node_id=msg.src or msg.payload.get("node_id", "unknown"),
                host=msg.payload.get("host", "unknown"),
                port=msg.payload.get("port", 0),
                devices=devices,
                backend_name=msg.payload.get("backend", ""),
                model_hash=msg.payload.get("model_hash", ""),
            )
            self.nodes[node.node_id] = node
            self._last_heartbeat[node.node_id] = time.time()
            await self._replan()
            return Msg(type="welcome", payload={"node_id": self.node_id}, src=self.node_id)
        if msg.type == "heartbeat":
            self._last_heartbeat[msg.src] = time.time()
            return Msg(type="heartbeat_ack", src=self.node_id)
        return Msg(type="error", payload={"message": f"unknown {msg.type}"}, src=self.node_id)

    # ------------------------------------------------------------------

    async def _replan(self) -> None:
        """Recompute the cluster plan whenever membership changes."""
        members = [n for n, t in self._last_heartbeat.items()
                   if time.time() - t < 30.0] or list(self.nodes)
        nodes = [self.nodes[n] for n in members]
        if len(nodes) < 2:
            self.plan = None
            return
        self.plan = automatic(nodes, self.spec)
        log.info("cluster plan (%s, %d nodes): %s", self.plan.mode.value, len(nodes), "; ".join(self.plan.notes))
        await self._push_plan()
        if self._on_plan_ready:
            self._on_plan_ready(self.plan)

    async def _push_plan(self) -> None:
        if self.plan is None:
            return
        for node, pplan in zip(self.plan.nodes, self.plan.per_node):
            payload = {"pp_layers": pplan.pp_layers or [], "tp_slices": pplan.tp_slices or [],
                       "world_size": pplan.world_size, "rank": pplan.rank}
            await self._rpc(node, Msg(type="plan", payload=payload, src=self.node_id))

    async def _rpc(self, node: NodeSpec, msg: Msg, timeout: float = 10.0) -> Msg:
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(node.host, node.port), timeout
            )
            writer.write((json.dumps(msg.to_dict()) + "\n").encode())
            await writer.drain()
            line = await asyncio.wait_for(reader.readline(), timeout)
            writer.close()
            return Msg.from_dict(json.loads(line)) if line else Msg(type="error")
        except Exception as exc:  # noqa: BLE001
            log.warning("rpc to %s failed: %s", node.node_id, exc)
            return Msg(type="error", payload={"message": str(exc)}, src=self.node_id)

    async def close(self) -> None:
        if self._server:
            self._server.close()
        await self.transport.close()
