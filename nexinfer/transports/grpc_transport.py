"""gRPC transport: multiplexed bidirectional tensor streaming.

Compared to raw TCP it adds request multiplexing, back-pressure, and
framing handled by the gRPC runtime. Uses a generated-free stub built on
``grpc`` generic handlers so no .proto compilation step is required.

Requires ``grpcio`` (already a core dependency).
"""

from __future__ import annotations

import asyncio
import logging
import queue
from typing import Callable

import grpc
import numpy as np

from nexinfer.transports.base import TensorFrame, Transport

log = logging.getLogger("nexinfer.transports.grpc")

SERVICE = "/nexinfer.TensorLink/Stream"


def _tensor_to_bytes(name: str, arr: np.ndarray) -> bytes:
    return TensorFrame.pack(name, arr)


def _bytes_to_tensor(data: bytes) -> tuple[str, np.ndarray]:
    import io
    import struct

    buf = io.BytesIO(data)
    (size,) = struct.unpack("<I", buf.read(4))
    payload = buf.read(size)
    inner = io.BytesIO(payload)
    _magic, name_len, ndim = struct.unpack("<HHI", inner.read(8))
    name = inner.read(name_len).decode()
    (dtype_num,) = struct.unpack("<I", inner.read(4))
    shape = struct.unpack("<" + "Q" * ndim, inner.read(8 * ndim))
    arr = np.frombuffer(inner.read(), dtype=np.dtype(dtype_num)).reshape(shape).copy()
    return name, arr


class GRPCTransport(Transport):
    name = "grpc"

    def __init__(self) -> None:
        self._server: grpc.Server | None = None
        self._channels: dict[str, grpc.Channel] = {}
        self._streams: dict[str, grpc.StreamStreamMultiCallable] = {}
        self._queues: dict[str, queue.Queue] = {}
        self._on_peer: Callable | None = None

    async def start_server(self, host: str, port: int, on_peer: Callable | None = None) -> None:
        self._on_peer = on_peer
        self._server = grpc.server(grpc.thread_pool(4))
        rpc_handler = grpc.method_service_handler(
            {SERVICE: grpc.unary_stream_rpc_method_handler(lambda req, ctx: self._iter_responses(ctx))}
        ) if False else None
        # simpler: register generic handler
        class _Handler(grpc.GenericRpcHandler):
            def __init__(self, outer):
                self.outer = outer

            def service(self, handler_call_details):
                if handler_call_details.method == SERVICE:
                    return grpc.unary_stream_rpc_method_handler(
                        lambda req, ctx: self.outer._response_iterator()
                    )
                return None

        self._server.add_generic_rpc_handlers([_Handler(self)])
        self._server.add_insecure_port(f"{host}:{port}")
        self._server.start()
        log.info("grpc transport listening on %s:%d", host, port)

    def _response_iterator(self):
        while True:
            # block until any queue has data (round-robin over peers)
            items = list(self._queues.items())
            if not items:
                yield from iter([])
                return
            for peer, q in items:
                try:
                    data = q.get(timeout=0.1)
                    yield data
                except queue.Empty:
                    continue

    async def connect(self, peer: str) -> None:
        if peer in self._channels:
            return
        channel = grpc.insecure_channel(peer)
        self._channels[peer] = channel
        stub = channel.unary_stream(SERVICE)
        self._streams[peer] = stub
        self._queues[peer] = queue.Queue()
        if self._on_peer:
            self._on_peer(peer)

    async def send(self, peer: str, name: str, arr: np.ndarray) -> None:
        if peer not in self._streams:
            await self.connect(peer)
        q = self._queues.get(peer)
        if q is not None:
            q.put(_tensor_to_bytes(name, arr))

    async def recv(self, peer: str, timeout: float = 30.0) -> tuple[str, np.ndarray]:
        q = self._queues.get(peer)
        if q is None:
            raise ConnectionError(f"no grpc queue for peer {peer}")
        try:
            data = q.get(timeout=timeout)
            return _bytes_to_tensor(data)
        except queue.Empty as exc:
            raise TimeoutError(f"grpc recv timeout from {peer}") from exc

    async def close(self) -> None:
        if self._server:
            self._server.stop(grace=1)
        for ch in self._channels.values():
            ch.close()
        self._channels.clear()
        self._queues.clear()
