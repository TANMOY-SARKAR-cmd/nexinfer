"""MCP server: expose NexusInfer as an MCP server to other agents.

External hosts (Claude, Cursor, custom agents) can drive NexusInfer
through standard MCP tools::

    generate        -- run a generation request
    list_models     -- engine status + available backends
    memory_read     -- read the git-backed memory store
    memory_write    -- commit a value to an agent's branch

Transport: stdio (default) or SSE via ``mcp.server.sse``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

log = logging.getLogger("nexinfer.gateway.mcp_server")

try:
    from mcp.server.fastmcp import FastMCP

    _HAS_MCP = True
except ImportError:  # pragma: no cover
    _HAS_MCP = False
    FastMCP = None


class NexusMcpServer:
    """Wraps a running Engine + MemoryFabric as an MCP server."""

    def __init__(self, engine: Any, memory_fabric: Any, skills_registry: Any) -> None:
        if not _HAS_MCP:
            raise RuntimeError("mcp SDK missing; run `pip install nexinfer[mcp]`")
        self.mcp = FastMCP("NexusInfer")
        self.engine = engine
        self.memory = memory_fabric
        self.skills = skills_registry
        self._register_tools()

    def _register_tools(self) -> None:
        engine = self.engine
        memory = self.memory
        skills = self.skills

        @self.mcp.tool()
        def generate(prompt: str, max_tokens: int = 128, model: str = "",
                     skill: str = "default", temperature: float = 0.8) -> str:
            """Run an inference request against the NexusInfer engine."""
            from nexinfer.engine.types import GenerationRequest

            req = GenerationRequest(
                prompt=prompt, max_tokens=max_tokens, temperature=temperature, skill=skill
            )
            if engine._generator is None:
                return json.dumps({"error": "engine not bootstrapped"})
            tok = engine.generate(req)
            return json.dumps({
                "text": tok.text,
                "finish_reason": tok.finish_reason,
                "usage": tok.usage,
                "tool_calls": tok.tool_calls,
            })

        @self.mcp.tool()
        def list_models() -> str:
            """Show engine status: model, backend, device placement."""
            if engine.status is None:
                return json.dumps({"error": "engine not bootstrapped"})
            st = engine.status
            return json.dumps({
                "model": st.model,
                "backends": st.backend_names,
                "strategy": st.placement.strategy,
                "notes": st.placement.notes,
                "devices": [
                    {"id": d.device_id, "kind": d.kind.value, "name": d.name}
                    for d in st.profile.devices
                ],
            })

        @self.mcp.tool()
        def memory_read(store: str, branch: str, key: str | None = None) -> str:
            """Read a value (or list keys) from the git-backed memory store."""
            store_obj = memory.get_store(store)
            if store_obj is None:
                return json.dumps({"error": f"store {store!r} not found"})
            if key:
                val = store_obj.get(key, branch=branch)
                return json.dumps({"store": store, "branch": branch, "key": key, "value": val})
            return json.dumps({"keys": store_obj.list_keys(branch=branch)})

        @self.mcp.tool()
        def memory_write(store: str, branch: str, key: str, value: str,
                         message: str = "agent update") -> str:
            """Commit a value into a git-backed memory branch."""
            store_obj = memory.get_store(store)
            if store_obj is None:
                store_obj = memory.create_store(store)
            cid = store_obj.set(key, value, branch=branch, message=message)
            return json.dumps({"store": store, "branch": branch, "key": key, "commit": cid})

        @self.mcp.tool()
        def whiteboard_read() -> str:
            """Read the shared multi-agent whiteboard."""
            wb = memory.whiteboard()
            return json.dumps({"entries": wb.list_keys(branch="main"), "data": wb.get("entries", branch="main")})

        @self.mcp.tool()
        def whiteboard_write(agent_id: str, entry: str) -> str:
            """Post an entry to the shared whiteboard (last-write-wins merge)."""
            wb = memory.whiteboard()
            cid = wb.set(f"{agent_id}:{len(wb.list_keys(branch='main')) + 1}", entry,
                         branch="main", message=f"whiteboard post by {agent_id}")
            return json.dumps({"commit": cid})

    async def run_stdio(self) -> None:
        await self.mcp.run_stdio_async() if hasattr(self.mcp, "run_stdio_async") else await self.mcp.run_async("stdio")

    async def run_sse(self, host: str = "0.0.0.0", port: int = 8999) -> None:
        await self.mcp.run_sse_async(host=host, port=port)
