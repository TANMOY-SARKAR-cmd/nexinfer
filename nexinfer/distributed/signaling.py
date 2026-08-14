"""Bundled SDP signaling server for the WebRTC transport.

A tiny asyncio TCP relay that pairs two nodes: each posts its SDP offer
under a session id, and the peer fetches it and posts the answer. Nodes
can also simply paste the SDP strings manually (``--offer`` /
``--answer`` CLI flags); the signaling server only automates it.

Run: ``python -m nexinfer.distributed.signaling --port 8900``
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

log = logging.getLogger("nexinfer.signaling")


class SignalingServer:
    def __init__(self) -> None:
        self.sessions: dict[str, dict] = {}

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            data = await reader.read(65536)
            req = json.loads(data)
            kind = req.get("type")
            sid = req.get("session")
            resp: dict = {}
            if kind == "post_offer":
                self.sessions.setdefault(sid, {})["offer"] = req.get("sdp")
                resp = {"status": "ok"}
            elif kind == "post_answer":
                self.sessions.setdefault(sid, {})["answer"] = req.get("sdp")
                resp = {"status": "ok"}
            elif kind == "get_offer":
                offer = self.sessions.get(sid, {}).get("offer")
                resp = {"status": "ok", "sdp": offer}
            elif kind == "get_answer":
                answer = self.sessions.get(sid, {}).get("answer")
                resp = {"status": "ok", "sdp": answer}
            else:
                resp = {"status": "error", "message": f"unknown type {kind}"}
            writer.write(json.dumps(resp).encode() + b"\n")
            await writer.drain()
        except Exception as exc:
            log.warning("signaling error: %s", exc)
        finally:
            writer.close()

    async def serve(self, host: str, port: int) -> None:
        server = await asyncio.start_server(self.handle, host, port)
        log.info("signaling server on %s:%d", host, port)
        async with server:
            await server.serve_forever()


async def fetch_peer_sdp(
    signaling_host: str, signaling_port: int, session: str, role: str, local_sdp: str | None = None
) -> str | None:
    """High-level helper: post local SDP and wait for the peer's SDP."""
    reader, writer = await asyncio.open_connection(signaling_host, signaling_port)
    if local_sdp:
        writer.write(
            json.dumps({"type": f"post_{role}", "session": session, "sdp": local_sdp}).encode() + b"\n"
        )
        await writer.drain()
        await reader.readline()
    peer = None
    other = "answer" if role == "offer" else "offer"
    for _ in range(60):
        writer.write(json.dumps({"type": f"get_{other}", "session": session}).encode() + b"\n")
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=5.0)
        if line:
            data = json.loads(line)
            if data.get("sdp"):
                peer = data["sdp"]
                break
        await asyncio.sleep(2.0)
    writer.close()
    return peer


def main() -> None:
    parser = argparse.ArgumentParser(description="NexusInfer SDP signaling server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8900)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    asyncio.run(SignalingServer().serve(args.host, args.port))


if __name__ == "__main__":
    main()
