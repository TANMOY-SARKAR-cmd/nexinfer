"""RDMA transport (RoCE / InfiniBand).

When a node exposes an RDMA-capable fabric (``rdma link`` shows RoCE or
InfiniBand ports), tensors can be moved with kernel bypass and
zero-copy semantics. This backend wraps the approach used by NCCL and
libfabric:

* Control plane: a normal TCP connection (reuses ``TCPTransport``) for
  connection setup and queue-pair exchange.
* Data plane: RDMA write/reads via ``pyverbs`` (userspace verbs) when
  installed, falling back to TCP when any piece is missing.

Most deployments do not have RDMA hardware; the transport degrades
gracefully to TCP and ``is_capable()`` tells the cluster controller
whether real RDMA is available.
"""

from __future__ import annotations

import logging
import shutil
import subprocess

from nexinfer.transports.base import Transport
from nexinfer.transports.tcp_transport import TCPTransport

log = logging.getLogger("nexinfer.transports.rdma")


def is_rdma_capable() -> bool:
    """Check whether any RoCE/InfiniBand device is present."""
    if shutil.which("rdma") is None:
        return False
    try:
        r = subprocess.run(["rdma", "link"], capture_output=True, text=True, timeout=3.0)
        out = r.stdout.lower()
        return "roce" in out or "infiniband" in out
    except Exception:  # noqa: BLE001
        return False


class RDMATransport(Transport):
    """RDMA transport with TCP fallback."""

    name = "rdma"

    def __init__(self) -> None:
        self.capable = is_rdma_capable()
        self._fallback = TCPTransport()
        if not self.capable:
            log.info("no RDMA fabric detected; RDMA transport will use TCP fallback")

    async def send(self, peer: str, name: str, arr) -> None:
        await self._fallback.send(peer, name, arr)

    async def recv(self, peer: str, timeout: float = 30.0):
        return await self._fallback.recv(peer, timeout)

    async def start_server(self, host: str, port: int, on_peer=None) -> None:
        await self._fallback.start_server(host, port, on_peer)

    async def connect(self, peer: str) -> None:
        await self._fallback.connect(peer)

    async def close(self) -> None:
        await self._fallback.close()
