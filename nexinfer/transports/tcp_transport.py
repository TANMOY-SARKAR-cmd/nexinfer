"""TCP transport: zero-dependency tensor movement over asyncio streams.

Peer channels are addressed by a stable ``peer key`` negotiated with a
``HELLO`` handshake: each outbound ``connect(host:port, key=...)`` sends its
key to the server, which installs the receive queue under the same key. In
the opposite direction, the server accepts the connection, receives the
HELLO frame, and installs the sender's queue under that key — so ``send``
and ``recv`` for a given key always use the SAME underlying socket, and
either side may address the other by the key both agreed on.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable

import numpy as np

from nexinfer.transports.base import TensorFrame, Transport

log = logging.getLogger("nexinfer.transports.tcp")


class TCPTransport(Transport):
    name = "tcp"

    def __init__(self) -> None:
        self._writers: dict[str, asyncio.StreamWriter] = {}
        self._queues: dict[str, asyncio.Queue] = {}
        self._server: asyncio.AbstractServer | None = None
        self._on_peer: Callable | None = None
        self._serve_task: asyncio.Task | None = None
        self._listen_addr: str = ""

    @property
    def listen_addr(self) -> str:
        """'host:port' this transport is accepting connections on (empty if none)."""
        return self._listen_addr

    async def start_server(self, host: str, port: int, on_peer: Callable | None = None) -> None:
        self._on_peer = on_peer

        async def _handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            key = "unknown"
            try:
                # handshake: first frame names the sender's channel key
                name, _ = await TensorFrame.read_frame(reader)
                key = name if name.startswith("HELLO:") else ""
                if key:
                    key = key.split(":", 1)[1]
                    log.debug("tcp: handshake key %s from %s", key, writer.get_extra_info("peername"))
                else:
                    # legacy/unnamed sender: address by socket peername
                    peer = writer.get_extra_info("peername")
                    key = f"{peer[0]}:{peer[1]}" if peer else "unknown"
                q: asyncio.Queue = asyncio.Queue()
                self._queues[key] = q
                # the local side may also *send* to this connection by key:
                self._writers[key] = writer
                if self._on_peer:
                    self._on_peer(key)
                while True:
                    n, arr = await TensorFrame.read_frame(reader)
                    await q.put((n, arr))
            except asyncio.IncompleteReadError:
                log.debug("tcp: peer channel %s disconnected", key)
            finally:
                self._queues.pop(key, None)
                self._writers.pop(key, None)
                writer.close()

        self._server = await asyncio.start_server(_handler, host, port)
        self._serve_task = asyncio.create_task(self._server.serve_forever())
        sock = self._server.sockets[0]
        addr = sock.getsockname()
        self._listen_addr = f"{addr[0]}:{addr[1]}"
        log.info("tcp transport listening on %s:%d", host, addr[1])

    async def connect(self, peer: str, key: str | None = None) -> None:
        """Connect to a listening peer; ``key`` is the shared channel id.

        If ``key`` is omitted, ``self.listen_addr`` is used (works when the
        caller has a server running — i.e. in a full mesh).
        """
        if peer in self._writers and key in self._writers:
            return
        resolved_key = key or self.listen_addr or peer
        if resolved_key in self._writers:
            return
        host, _, port = peer.rpartition(":")
        reader, writer = await asyncio.open_connection(host, int(port))
        # handshake: announce our channel key so the server installs the
        # receive queue under the SAME key we will use for recv/send
        writer.write(TensorFrame.pack(f"HELLO:{resolved_key}", np.zeros(0, dtype=np.uint8)))
        await writer.drain()
        q: asyncio.Queue = asyncio.Queue()
        self._queues[resolved_key] = q
        self._writers[resolved_key] = writer
        if self._on_peer:
            self._on_peer(resolved_key)

        async def _read() -> None:
            try:
                while True:
                    n, arr = await TensorFrame.read_frame(reader)
                    await q.put((n, arr))
            except asyncio.IncompleteReadError:
                log.debug("tcp: channel %s closed", resolved_key)
            finally:
                self._writers.pop(resolved_key, None)

        asyncio.create_task(_read(), name=f"tcp-read-{resolved_key}")
        log.debug("tcp: connected to %s with key %s", peer, resolved_key)

    async def send(self, peer: str, name: str, arr: np.ndarray) -> None:
        writer = self._writers.get(peer)
        if writer is None:
            # ``peer`` must be an address when no prior channel exists
            if ":" not in peer or peer.count(":") != 1:
                # remote channel may still be completing its HELLO handshake;
                # poll briefly before giving up
                for _ in range(50):
                    writer = self._writers.get(peer)
                    if writer is not None:
                        break
                    await asyncio.sleep(0.05)
                if writer is None:
                    raise ConnectionError(
                        f"no channel for {peer!r}; call connect(addr, key=peer) on both sides first"
                    )
                frame = TensorFrame.pack(name, arr)
                writer.write(frame)
                await writer.drain()
                return
            await self.connect(peer, key=peer)
            writer = self._writers[peer]
        frame = TensorFrame.pack(name, arr)
        writer.write(frame)
        await writer.drain()

    async def recv(self, peer: str, timeout: float = 30.0) -> tuple[str, np.ndarray]:
        q = self._queues.get(peer)
        if q is None:
            raise ConnectionError(f"no queue for peer {peer}; connect() first or wait for inbound")
        return await asyncio.wait_for(q.get(), timeout)

    async def close(self) -> None:
        if self._serve_task:
            self._serve_task.cancel()
        if self._server:
            self._server.close()
        for w in self._writers.values():
            w.close()
        self._writers.clear()
        self._queues.clear()
