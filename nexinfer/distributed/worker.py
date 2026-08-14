"""Cluster worker node.

A worker owns a contiguous range of model layers (**pipeline
parallelism**) or a head slice (tensor parallelism). It runs a local
``Backend`` instance and exposes:

* a control channel (asyncio TCP, JSON ``Msg`` frames) for plans and
  heartbeat,
* the data transport (``Transport`` subclass) for moving activations /
  KV-cache blocks / split attention results between ranks.

Pipeline-parallel forward pass order for a decode step (ring over the
transport, channels keyed per edge so each side reads from the key it
agreed on during the HELLO handshake):

    rank-0 embeds & runs layers 0..k  -> send activations to rank-1
    rank-1 runs layers k..j           -> send to rank-2  ...
    last rank runs final layers -> lm head -> send logits to coordinator

Intermediate ranks never see token ids or the vocabulary -- only
``hidden_size`` activation vectors cross the wire, which is what makes
pipeline parallelism backend-agnostic (the numpy backend honours the
assigned slice via ``offload_layers``; GGML/ORT run whole models locally
and are therefore root/leaf candidates).
"""

from __future__ import annotations

import asyncio
import json
import logging

import numpy as np

from nexinfer.backends.base import Backend, ModelSpec
from nexinfer.distributed.health import reconnect_loop
from nexinfer.distributed.messages import Msg
from nexinfer.distributed.planner import ClusterPlan, NodeSpec
from nexinfer.engine.types import ParallelMode
from nexinfer.transports.base import Transport

log = logging.getLogger("nexinfer.distributed.worker")


def _channel(rank: int, direction: str = "up") -> str:
    """Stable per-edge channel key: ``pp:rA->rB``."""
    if direction == "up":
        return f"pp:{rank}->{rank + 1}"
    return f"pp:{rank - 1}->{rank}"


def _data_peer(node: NodeSpec) -> str:
    """Address of a node's DATA transport (control port + 1000)."""
    return f"{node.host}:{node.port + 1000}"


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
        self._result_writer: asyncio.StreamWriter | None = None
        self._results: list[np.ndarray] = []
        self._running = False
        # coordinator contact info (set before ``register()`` / ``run_reconnect_loop()``)
        self.coordinator_host: str = "127.0.0.1"
        self.coordinator_port: int = 0

    # ------------------------------------------------------------------

    async def start(self) -> int:
        # remember the event loop this worker runs on so that clients
        # (DistributedEngine) can schedule ``generate`` on the same loop
        # the transport server tasks use -- cross-loop queue traffic
        # deadlocks because queue feeds and consumers wake each other
        # through the same loop
        self.loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        self._running = True
        srv = await asyncio.start_server(self._handle_control, self.control_host, self.control_port)
        self._server = srv
        # the data transport listens adjacent to the control socket; the
        # coordinator learns about it through the ``data_port`` payload
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

    async def _register(self, node_id: str) -> bool:
        """Send ``hello`` to the coordinator and return True on ``welcome``.

        Used both for the initial registration and by
        ``run_reconnect_loop`` when the coordinator restarts."""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.coordinator_host, self.coordinator_port), timeout=5.0
            )
            try:
                msg = Msg(
                    type="hello",
                    payload={
                        "node_id": node_id,
                        "host": self.coordinator_host,
                        "port": self.control_port,
                        "devices": [d.to_dict() for d in self.backend.detect_devices()],
                        "backend": self.backend.name,
                        "model_hash": getattr(
                            self.spec, "model_hash", f"model:{self.spec.num_layers}:{self.spec.hidden_size}"
                        ),
                        "spec": {
                            "num_layers": self.spec.num_layers,
                            "hidden_size": self.spec.hidden_size,
                        },
                    },
                    src=node_id,
                )
                writer.write((json.dumps(msg.to_dict()) + "\n").encode())
                await writer.drain()
                line = await asyncio.wait_for(reader.readline(), timeout=10.0)
                if not line:
                    return False
                resp = Msg.from_dict(json.loads(line))
                log.info("worker %s: register with coordinator -> %s", node_id, resp.type)
                return resp.type == "welcome"
            finally:
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass
        except Exception as exc:
            log.warning("worker %s: coordinator unreachable: %s", node_id, exc)
            return False

    async def run_reconnect_loop(self) -> None:
        """Fault tolerance: when the coordinator crashes or the network
        partition heals, keep retrying with exponential backoff until the
        worker is told to shut down (``self._running``) or registration
        succeeds.

        Note: ``register()`` should be called once before starting this
        loop; ``run_reconnect_loop`` handles the *recovery* path."""
        await reconnect_loop(
            self.node_id,
            self._register,
            self.coordinator_host,
            self.coordinator_port,
            running_predicate=lambda: self._running,
        )

    async def _dispatch(self, msg: Msg) -> Msg:
        if msg.type == "plan":
            payload = msg.payload
            pp_layers = payload.get("pp_layers") or [(0, self.spec.num_layers)]
            self.layer_range = (pp_layers[0][0], pp_layers[0][1])
            self.backend.offload_layers(pp_layers)
            # rebuild the full plan so data-plane addresses (host:port) are
            # correct for every rank, not just the local one
            nodes = [NodeSpec(n["node_id"], n["host"], n["port"]) for n in payload.get("nodes", [])]
            self.plan = ClusterPlan(
                mode=self.plan.mode if self.plan is not None else ParallelMode.PIPELINE,
                nodes=nodes,
                per_node=self.parallel if self.parallel is not None else [],
            )
            log.info("worker %s: layer range set to %s", self.node_id, self.layer_range)
            return Msg(type="plan_ack", payload={"layer_range": list(self.layer_range)}, src=self.node_id)
        if msg.type == "hello":
            # workers also send a hello-style register message so the
            # coordinator knows their data-port (control + 1000)
            return Msg(
                type="welcome",
                payload={"node_id": self.node_id, "data_port": self.control_port + 1000},
                src=self.node_id,
            )
        if msg.type == "heartbeat":
            return Msg(type="heartbeat_ack", payload={"load": 0.0}, src=self.node_id)
        if msg.type == "activate":
            # PP step triggered by the coordinator (prefill/decode). The
            # actual activation tensor is pushed through the data transport
            # under the edge channel key; the worker pulls it in
            # ``run_decode_loop``. This handler just acknowledges setup.
            return Msg(type="activate_ack", src=self.node_id)
        if msg.type == "prefill_request":
            req_id = msg.payload.get("req_id", "req")
            tok_ids = np.array(msg.payload.get("input_ids", []), dtype=np.int32)
            # rank-0 embeds and runs its slice; intermediates receive
            # activations from rank-0 through the pipeline loop.
            x = self.backend.prefill(req_id, tok_ids)
            if self.rank == self.parallel.world_size - 1:
                # final rank: return logits to the coordinator directly
                return Msg(
                    type="prefill_response",
                    payload={"req_id": req_id, "logits": x.tolist()},
                    src=self.node_id,
                )
            return Msg(type="activate", payload={"req_id": req_id, "kind": "prefill"}, src=self.node_id)
        return Msg(type="error", payload={"message": f"unknown {msg.type}"}, src=self.node_id)

    # ------------------------------------------------------------------
    # Pipeline-parallel loop.
    #
    # Per edge there is ONE channel key both sides agree on:
    # rank k sends with key ``pp:k->k+1`` and rank k+1 recv's with the
    # SAME key. The transport's HELLO handshake installs the queue under
    # the key the sender announces, so rank k+1 must connect first (the
    # coordinator wires both sides in ``run_decode_loop`` below).

    async def _forward_edge(self, req_id: str, x: np.ndarray) -> None:
        """Push activations (or final logits) to the next rank's DATA port."""
        nxt = self.plan.nodes[self.rank + 1]
        key = _channel(self.rank, "up")
        peer = _data_peer(nxt)
        await self.transport.connect(peer, key=key)
        await self.transport.send(peer, key, x)

    async def run_decode_loop(self) -> None:
        """Worker-side pipeline loop: pull activations from the previous
        rank, run this slice, push the output to the next rank (or back
        to the coordinator from the final rank)."""
        peers = self.plan.nodes
        ws = self.parallel.world_size
        if self.rank > 0:
            prev = peers[self.rank - 1]
            key = _channel(self.rank - 1, "up")
            peer = _data_peer(prev)
            log.info("[%s rank %d] connect edge peer %s key %s", self.node_id, self.rank, peer, key)
            await self.transport.connect(peer, key=key)
            # pull exactly once per incoming frame keyed by our edge
            while self._running:
                try:
                    name, x = await self.transport.recv(key, timeout=1.0)
                    log.info("[%s rank %d] recv frame %s shape %s", self.node_id, self.rank, name, x.shape)
                except (TimeoutError, ConnectionError):
                    await asyncio.sleep(0.05)
                    continue
                req_id = name.split(":", 1)[-1] if ":" in name else name
                try:
                    out = self.backend.prefill(req_id, x)  # activations -> activations/logits
                except Exception as exc:
                    log.error("[%s rank %d] prefill failed: %s", self.node_id, self.rank, exc)
                    continue
                out = np.asarray(out, dtype=np.float32)
                if self.rank == ws - 1:
                    # final rank: activations -> logits; sample-ready logits
                    # live on the LAST position of the replayed sequence
                    if out.ndim > 1 and out.shape[0] > 1:
                        out = out[-1]
                    res_key = f"result:{self.node_id}"
                    # push logits to rank-0's DATA transport (its control
                    # port + 1000) where it is listening on the result key
                    root_node = peers[0]
                    root_data_peer = f"{root_node.host}:{root_node.port + 1000}"
                    await self.transport.connect(root_data_peer, key=res_key)
                    log.info(
                        "[%s rank %d] final push to %s key %s shape %s",
                        self.node_id,
                        self.rank,
                        root_data_peer,
                        res_key,
                        out.shape,
                    )
                    await self.transport.send(root_data_peer, res_key, out)
                    log.info("[%s rank %d] final push sent", self.node_id, self.rank)
                    self._results.append(out)
                else:
                    # intermediate rank: forward only the last-row hidden
                    # state (causal attention makes it sufficient)
                    if out.ndim > 1 and out.shape[0] > 1:
                        out = out[-1]
                    out = out.reshape(-1)
                    await self._forward_edge(req_id, out)

    # ------------------------------------------------------------------
    # Distributed generation driver (root rank side).
    #
    # ``generate`` walks the pipeline ring explicitly for both prefill
    # and every decode step: token ids -> embed+layers (rank-0) ->
    # activations hop rank-by-rank -> logits land at the coordinator,
    # where greedy sampling decides the next token and the loop repeats.

    async def generate(
        self,
        req_id: str,
        input_ids: list[int],
        max_tokens: int = 16,
        timeout: float = 60.0,
    ) -> list[np.ndarray]:
        """Run distributed generation from the root (rank-0) worker and
        return the sequence of sampled token logits, in token order.

        Only rank-0 may call this; the pipeline ring is walked per step.
        ``input_ids`` are embedded by rank-0, activations hop through
        intermediate ranks, and the final rank pushes logits back to the
        coordinator under ``result:{node_id}`` -- ``generate`` reads them
        from ``self.transport`` under that same key (the coordinator
        never interferes because the key is worker-owned here)."""
        peers = self.plan.nodes
        ws = self.parallel.world_size
        sampled_logits = []
        # pre-wire the result key so final-rank pushes have a queue to land in.
        # The final rank pushes logits back on key ``result:<final-node>``.
        # rank-0 hosts the mailbox: it ensures its own data transport is
        # listening and self-connects to it under the result key so local
        # recv works from the same socket pair the final rank connects to.
        res_key = f"result:{peers[ws - 1].node_id}"
        # the final rank pushes logits to rank-0's DATA transport; rank-0's
        # data transport is already listening (started in ``start``), so the
        # final rank's connection installs the server-side receive queue
        # under the result key and ``recv`` picks it up via polling
        if not self.transport.listen_addr:
            await self.transport.start_server("127.0.0.1", 0)
        x = np.array(input_ids, dtype=np.int32)
        # PP "replay" semantics: because intermediate ranks only cache the
        # KV for their own layer slice, root re-embeds the full context
        # every step and ships the complete activation matrix through the
        # ring; every rank recomputes attention from scratch on the full
        # sequence and hands on only its last-row hidden state.
        hidden = self.spec.hidden_size
        # seed the replay buffer with the post-slice hidden state of every
        # input token (each row = root-slice output for one position)
        first_out = self.backend.prefill(req_id, x)
        first_out = np.asarray(first_out, dtype=np.float32)
        embeds: list[np.ndarray] = list(first_out.reshape(-1, hidden))
        # first decode step uses the last-token logits of the prompt; the
        # token id is derived from them so subsequent steps always have a
        # valid ``tok_id`` to feed back through the root slice
        tok_id = int(np.argmax(np.asarray(first_out, dtype=np.float32)[-1].reshape(-1)))
        for step in range(max_tokens):
            if step == 0:
                out = first_out
            else:
                # the new token must also pass through the root slice so
                # that the replay matrix stays homogeneous (all rows are
                # post-root-slice hidden states)
                new_hidden = self.backend.prefill(req_id, np.array([[tok_id]], dtype=np.int32))
                new_hidden = np.asarray(new_hidden, dtype=np.float32).reshape(-1)[-hidden:]
                embeds.append(new_hidden)
                out = self.backend.prefill(req_id, np.stack(embeds, axis=0))
            cur = out
            # forward the activations through the ring; intermediate ranks
            # recv from the edge they listen on, run their slice, forward
            # (handled by run_decode_loop on non-root workers); root walks
            # the same contract by recv/send on each edge.
            for hop in range(ws - 1):
                if hop > 0:
                    key = _channel(hop - 1, "up")
                    prev = peers[hop]
                    p_peer = _data_peer(prev)
                    await self.transport.connect(p_peer, key=key)
                    _name, cur = await self.transport.recv(key, timeout=timeout)
                    cur = np.asarray(cur, dtype=np.float32)
                if hop < ws - 1:
                    nxt = peers[hop + 1]
                    n_key = _channel(hop, "up")
                    n_peer = _data_peer(nxt)
                    await self.transport.connect(n_peer, key=n_key)
                    await self.transport.send(n_peer, n_key, cur)
            # rank-0 receives the logits the final rank pushed back on
            # the result key; ws==1 means logits came straight from local
            if ws > 1:
                log.info(
                    "[gen %s] root recv waiting on %s; queues=%s",
                    req_id,
                    res_key,
                    list(self.transport._queues),
                )
                _name, logits = await self.transport.recv(res_key, timeout=timeout)
                log.info("[gen %s] root recv got %s shape %s", req_id, _name, logits.shape)
                logits = np.asarray(logits, dtype=np.float32)
            else:
                logits = np.asarray(out, dtype=np.float32)
            # last-row logits: prefill returns full-sequence outputs; the
            # sampled token always corresponds to the last position
            if logits.ndim > 1 and logits.shape[0] > 1:
                logits = logits[-1]
            if logits.ndim == 0:
                logits = logits.reshape(1)
            sampled_logits.append(logits)
            tok_id = int(np.argmax(logits))
        return sampled_logits

    # ------------------------------------------------------------------

    async def close(self) -> None:
        self._running = False
        if self._server:
            self._server.close()
        await self.transport.close()
        self.backend.close()
