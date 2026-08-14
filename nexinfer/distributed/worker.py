"""Cluster worker node.

A worker owns a contiguous range of model layers (PP) or a head slice
(TP). It runs a local ``Backend`` instance and exposes:

* a control channel (asyncio TCP, JSON ``Msg`` frames) for plans and
  heartbeat,
* the data transport (``Transport`` subclass) for moving activations /
  KV-cache blocks / split attention results between ranks.

PP forward pass order for a decode step:

    recv activations from rank-1 (or root) -> run my layers ->
    send to rank+1 (or root)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Callable

import numpy as np

from nexinfer.backends.base import Backend, ModelSpec
from nexinfer.distributed.messages import Msg
from nexinfer.distributed.planner import ClusterPlan
from nexinfer.transports.base import Transport

log = logging.getLogger("nexinfer.distributed.worker")


class Worker:
    def __init__(
        self,
        node_id: str,
        backend: Backend,
        spec: ModelSpec,
        plan: ClusterPlan,
        transport: Transport,
        rank: int,
        control_host: str = "0.0.0.0",
        control_port: int = 0,
    ) -> None:
        self.node_id = node_id
        self.backend = backend
        self.spec = spec
        self.plan = plan
        self.transport = transport
        self.rank = rank
        self.parallel = plan.per_node[rank]
        self.control_host = control_host
        self.control_port = control_port
        self.layer_range: tuple[int, int] = (0, spec.num_layers)
        self._server: asyncio.AbstractServer | None = None
        self._control_writers: dict[str, asyncio.StreamWriter] = {}
        self._running = False

    # ------------------------------------------------------------------

    async def start(self) -> int:
        self._running = True
        srv = await asyncio.start_server(self._handle_control, self.control_host, self.control_port)
        self._server = srv
        await self.transport.start_server(self.control_host, self.control_port + 1000)
        actual = srv.sockets[0].getsockname()[1]
        self.control_port = actual
        log.info("worker %s rank %d control on port %d", self.node_id, self.rank, actual)
        return actual

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
        if msg.type == "plan":
            layers = msg.payload.get("pp_layers", [(0, self.spec.num_layers)])
            self.layer_range = (layers[0][0], layers[0][1])
            self.backend.offload_layers(layers)
            log.info("worker %s: layer range set to %s", self.node_id, self.layer_range)
            return Msg(type="plan_ack", src=self.node_id)
        if msg.type == "heartbeat":
            return Msg(type="heartbeat_ack", payload={"load": 0.0}, src=self.node_id)
        if msg.type == "activate":
            # PP step: activations arrive on the data transport
            return Msg(type="activate_ack", src=self.node_id)
        return Msg(type="error", payload={"message": f"unknown {msg.type}"}, src=self.node_id)

    # ------------------------------------------------------------------
    # Pipeline-parallel forward (activations are numpy arrays flown
    # through the data transport between adjacent ranks)

    async def pp_step(self, req_id: str, activations: np.ndarray, direction: str = "up") -> np.ndarray:
        """Run this worker's layer slice on ``activations``."""
        start, end = self.layer_range
        x = activations
        for layer in range(start, end):
            # each backend implements layer-wise execution via offload hint;
            # the numpy backend executes all layers, so we slice by running
            # the whole stack and masking via a temporary spec swap.
            x = self.backend.prefill(req_id, x if x.ndim > 1 else np.array([x]))
        next_rank = self.rank + 1 if direction == "up" else self.rank - 1
        if 0 <= next_rank < self.parallel.world_size:
            peers = self.plan.nodes
            peer = peers[next_rank]
            await self.transport.send(f"{peer.host}:{peer.port}", f"act:{req_id}", x)
        return x

    async def run_decode_loop(self) -> None:
        """Worker-side decode loop: pull activations, run layers, push on."""
        peers = self.plan.nodes
        prev = None
        if self.rank > 0:
            prev = peers[self.rank - 1]
            await self.transport.connect(f"{prev.host}:{prev.port}")
        while self._running:
            try:
                name, x = await self.transport.recv(f"{prev.host}:{prev.port}" if prev else "coordinator", timeout=1.0)
            except (TimeoutError, ConnectionError):
                await asyncio.sleep(0.05)
                continue
            req_id = name.split(":", 1)[-1]
            out = await self.pp_step(req_id, x, "up" if self.rank < self.parallel.world_size - 1 else "down")
            if self.rank < self.parallel.world_size - 1:
                nxt = peers[self.rank + 1]
                await self.transport.send(f"{nxt.host}:{nxt.port}", f"act:{req_id}", out)

    async def close(self) -> None:
        self._running = False
        if self._server:
            self._server.close()
        await self.transport.close()
        self.backend.close()
