"""Transport factory: pick the best available transport by name."""

from __future__ import annotations

from nexinfer.transports.base import Transport


def make_transport(name: str) -> Transport:
    """Create a transport by name. Falls back to TCP for unavailable ones."""
    name = name.lower()
    if name == "tcp":
        from nexinfer.transports.tcp_transport import TCPTransport

        return TCPTransport()
    if name == "grpc":
        from nexinfer.transports.grpc_transport import GRPCTransport

        return GRPCTransport()
    if name == "webrtc":
        from nexinfer.transports.webrtc_transport import WebRTCTransport

        t = WebRTCTransport()
        if not t.available():
            raise RuntimeError("WebRTC transport unavailable; install nexinfer[webrtc]")
        return t
    if name == "rdma":
        from nexinfer.transports.rdma_transport import RDMATransport

        return RDMATransport()
    raise ValueError(f"unknown transport {name!r}; choose from tcp|grpc|webrtc|rdma")
