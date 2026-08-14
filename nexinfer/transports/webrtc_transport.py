"""WebRTC transport: P2P tensor movement through NATs/firewalls.

Uses ``aiortc`` data channels. Nodes exchange SDP offers/answers through a
small signaling server (bundled, see ``nexinfer.distributed.signaling``)
or manually pasted offer/answer strings, then stream tensors over
unordered binary data channels with chunking.

Requires ``pip install nexinfer[webrtc]``; when ``aiortc`` is missing the
transport reports itself unavailable.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Callable

import numpy as np

from nexinfer.transports.base import TensorFrame, Transport

log = logging.getLogger("nexinfer.transports.webrtc")

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, RTCDataChannel  # type: ignore
    from aiortc.contrib.signaling import BYE  # type: ignore

    _HAS_AIORTC = True
except ImportError:  # pragma: no cover
    _HAS_AIORTC = False
    RTCPeerConnection = None
    RTCSessionDescription = None

CHUNK = 64 * 1024  # data channel message cap


class WebRTCTransport(Transport):
    name = "webrtc"

    def __init__(self) -> None:
        self._pcs: dict[str, RTCPeerConnection] = {}
        self._channels: dict[str, RTCDataChannel] = {}
        self._queues: dict[str, asyncio.Queue] = {}
        self._on_peer: Callable | None = None

    @classmethod
    def available(cls) -> bool:
        return _HAS_AIORTC

    def _new_pc(self) -> RTCPeerConnection:
        pc = RTCPeerConnection()
        pc.addTransceiver = None  # unused
        return pc

    async def start_server(self, host: str, port: int, on_peer: Callable | None = None) -> None:
        """port here is the signaling helper port; the transport itself is P2P."""
        self._on_peer = on_peer
        self._signaling_port = int(port)
        log.info("webrtc transport ready (signaling on port %d)", port)

    async def offer_for(self, peer_id: str) -> str:
        """Create an SDP offer for ``peer_id``; return JSON to send out of band."""
        if not self.available():
            raise RuntimeError("aiortc not installed; pip install nexinfer[webrtc]")
        pc = RTCPeerConnection()
        self._pcs[peer_id] = pc
        channel = pc.createDataChannel(f"tensors-{peer_id}")
        self._setup_channel(peer_id, channel)
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        return pc.localDescription.sdp

    async def accept_answer(self, peer_id: str, answer_sdp: str) -> None:
        pc = self._pcs[peer_id]
        await pc.setRemoteDescription(RTCSessionDescription(sdp=answer_sdp, type="answer"))

    async def answer_offer(self, peer_id: str, offer_sdp: str) -> str:
        if not self.available():
            raise RuntimeError("aiortc not installed")
        pc = RTCPeerConnection()
        self._pcs[peer_id] = pc

        @pc.on("datachannel")
        def _on_dc(channel: RTCDataChannel) -> None:
            self._setup_channel(peer_id, channel)

        await pc.setRemoteDescription(RTCSessionDescription(sdp=offer_sdp, type="offer"))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        return pc.localDescription.sdp

    def _setup_channel(self, peer_id: str, channel: RTCDataChannel) -> None:
        self._channels[peer_id] = channel
        self._queues.setdefault(peer_id, asyncio.Queue())
        if self._on_peer:
            self._on_peer(peer_id)

        buffer = bytearray()

        @channel.on("message")
        def _on_msg(data: bytes | str) -> None:
            if isinstance(data, str):
                return
            buffer.extend(data)
            if len(buffer) >= 4:
                (size,) = __import__("struct").unpack("<I", buffer[:4])
                if len(buffer) >= 4 + size:
                    payload = bytes(buffer[4 : 4 + size])
                    del buffer[: 4 + size]
                    try:
                        buf = __import__("io").BytesIO(payload)
                        import struct as _st

                        _magic, name_len, ndim = _st.unpack("<HHI", buf.read(8))
                        name = buf.read(name_len).decode()
                        (dtype_num,) = _st.unpack("<I", buf.read(4))
                        shape = _st.unpack("<" + "Q" * ndim, buf.read(8 * ndim))
                        arr = np.frombuffer(buf.read(), dtype=np.dtype(dtype_num)).reshape(shape).copy()
                        self._queues[peer_id].put_nowait((name, arr))
                    except Exception as exc:  # noqa: BLE001
                        log.warning("webrtc frame decode failed: %s", exc)

    async def send(self, peer: str, name: str, arr: np.ndarray) -> None:
        channel = self._channels.get(peer)
        if channel is None:
            raise ConnectionError(f"no data channel to peer {peer}")
        frame = TensorFrame.pack(name, arr)
        for i in range(0, len(frame), CHUNK):
            channel.send(frame[i : i + CHUNK])
        await asyncio.sleep(0)

    async def recv(self, peer: str, timeout: float = 30.0) -> tuple[str, np.ndarray]:
        q = self._queues.get(peer)
        if q is None:
            raise ConnectionError(f"no channel for peer {peer}")
        return await asyncio.wait_for(q.get(), timeout)

    async def connect(self, peer: str) -> None:
        # connection is established via offer/answer dance; this is a no-op guard
        return None

    async def close(self) -> None:
        for pc in self._pcs.values():
            await pc.close()
        self._pcs.clear()
        self._channels.clear()
        self._queues.clear()
