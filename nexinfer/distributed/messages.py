"""Cluster control-plane messages (simple tagged dict protocol).

The distributed runtime exchanges typed messages over the control
channel (JSON over TCP control socket) and raw tensors over the data
transport. Message types:

* ``hello`` / ``welcome``      -- registration handshake
* ``heartbeat`` / ``heartbeat_ack`` -- liveness
* ``plan``                      -- coordinator pushes parallel plan to workers
* ``prefill_request`` / ``prefill_response`` -- prompt processing
* ``decode_request`` / ``decode_response``  -- token-step processing
* ``activate`` / ``layer_range`` -- pipeline-parallel layer activation
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Msg:
    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    src: str = ""
    dst: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "Msg":
        return cls(type=d.get("type", ""), payload=d.get("payload", {}), src=d.get("src", ""), dst=d.get("dst", ""))

    def to_dict(self) -> dict:
        return {"type": self.type, "payload": self.payload, "src": self.src, "dst": self.dst}
