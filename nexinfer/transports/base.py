"""Transport abstraction for distributed tensor/KV movement.

A ``Transport`` moves numpy arrays between nodes. NexusInfer ships four
transports and lets administrators pick per-link:

* ``TCPTransport`` -- zero-dependency, works on any LAN/WAN (default)
* ``GRPCTransport`` -- multiplexed streams, lower latency on stable LANs
* ``WebRTCTransport`` -- P2P through NAT/firewalls via STUN signaling
* ``RDMATransport`` -- kernel-bypass when RoCE/InfiniBand fabric detected

All transports serialize tensors as length-prefixed (dtype, shape, bytes)
frames over asyncio streams (TCP) or data channels (WebRTC).
"""

from __future__ import annotations

import asyncio
import io
import logging
import struct
from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

log = logging.getLogger("nexinfer.transports")


class TensorFrame:
    """Length-prefixed tensor serialization."""

    MAGIC = 0x4E58  # "NX"

    @staticmethod
    def pack(name: str, arr: np.ndarray) -> bytes:
        name_b = name.encode("utf-8")
        header = struct.pack("<HHI", TensorFrame.MAGIC, len(name_b), arr.ndim)
        parts = [header, name_b]
        dt = arr.dtype.str.encode("ascii")
        parts.append(struct.pack("<I", len(dt)))
        parts.append(dt)
        parts.append(struct.pack("<" + "Q" * arr.ndim, *arr.shape))
        parts.append(arr.tobytes())
        buf = io.BytesIO()
        payload = b"".join(parts)
        buf.write(struct.pack("<I", len(payload)))
        buf.write(payload)
        return buf.getvalue()

    @staticmethod
    async def read_frame(reader: asyncio.StreamReader) -> tuple[str, np.ndarray]:
        size_data = await reader.readexactly(4)
        (size,) = struct.unpack("<I", size_data)
        payload = await reader.readexactly(size)
        buf = io.BytesIO(payload)
        magic, name_len, ndim = struct.unpack("<HHI", buf.read(8))
        if magic != TensorFrame.MAGIC:
            raise ValueError(f"bad frame magic {magic:#x}")
        name = buf.read(name_len).decode("utf-8")
        (dt_len,) = struct.unpack("<I", buf.read(4))
        dt = buf.read(dt_len).decode("ascii")
        shape = struct.unpack("<" + "Q" * ndim, buf.read(8 * ndim))
        data = buf.read()
        arr = np.frombuffer(data, dtype=np.dtype(dt)).reshape(shape)
        return name, arr.copy()


class Transport(ABC):
    name: str = "base"

    @abstractmethod
    async def send(self, peer: str, name: str, arr: np.ndarray) -> None: ...

    @abstractmethod
    async def recv(self, peer: str, timeout: float = 30.0) -> tuple[str, np.ndarray]: ...

    @abstractmethod
    async def start_server(self, host: str, port: int, on_peer: Callable | None = None) -> None: ...

    @abstractmethod
    async def connect(self, peer: str) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    @staticmethod
    def available_transports() -> list[str]:
        return ["tcp", "grpc", "webrtc", "rdma"]
