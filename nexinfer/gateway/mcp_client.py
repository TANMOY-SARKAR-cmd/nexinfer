"""MCP gateway client: connects NexusInfer to MCP servers.

Wraps the official ``mcp`` Python SDK so any MCP server (stdio or
SSE/streamable-HTTP) can expose its tools to the inference engine.
Discovered MCP tools are mapped to the engine tool registry and become
callable during generation alongside ``web_fetch`` and memory tools.

Requires ``pip install nexinfer[mcp]``. When the SDK is missing,
``McpGatewayClient.available()`` returns False and the CLI degrades
gracefully.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("nexinfer.gateway.mcp")

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client

    _HAS_MCP = True
except ImportError:  # pragma: no cover
    _HAS_MCP = False
    ClientSession = None


@dataclass
class McpTool:
    server: str
    name: str
    description: str
    input_schema: dict[str, Any]

    def to_engine_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema if self.input_schema else {"type": "object", "properties": {}},
        }


class McpGatewayClient:
    """Lifecycle-managed client for one MCP server."""

    def __init__(self, server_name: str, config: dict[str, Any]) -> None:
        self.server_name = server_name
        self.config = config  # {command, args, env} | {url}
        self.session: ClientSession | None = None
        self._tools: list[McpTool] = []
        self._exit_stack: AsyncExitStack | None = None
        self._read_task: asyncio.Task | None = None

    @staticmethod
    def available() -> bool:
        return _HAS_MCP

    # ------------------------------------------------------------------

    async def connect(self) -> list[McpTool]:
        if not self.available():
            raise RuntimeError("mcp SDK not installed; run `pip install nexinfer[mcp]`")
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()
        try:
            if "url" in self.config:
                streams = await self._exit_stack.enter_async_context(sse_client(self.config["url"]))
            else:
                cmd = self.config.get("command", "npx")
                args = self.config.get("args", [])
                env = {**os.environ, **(self.config.get("env") or {})}
                params = StdioServerParameters(command=cmd, args=args, env=env)
                streams = await self._exit_stack.enter_async_context(stdio_client(params))
            read, write = streams
            self.session = await self._exit_stack.enter_async_context(ClientSession(read, write))
            await self.session.initialize()
            result = await self.session.list_tools()
            self._tools = [
                McpTool(
                    server=self.server_name,
                    name=t.name,
                    description=t.description or "",
                    input_schema=(t.inputSchema or {}),
                )
                for t in result.tools
            ]
            log.info("MCP server %s: %d tools", self.server_name, len(self._tools))
        except Exception:
            await self.disconnect()
            raise
        return self._tools

    @property
    def tools(self) -> list[McpTool]:
        return self._tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.session is None:
            raise RuntimeError(f"not connected to MCP server {self.server_name}")
        result = await self.session.call_tool(name, arguments)
        out = []
        for block in result.content:
            if hasattr(block, "text"):
                out.append(block.text)
            else:
                out.append(json.dumps(block.dict() if hasattr(block, "dict") else str(block)))
        return {"server": self.server_name, "name": name, "content": "\n".join(out)}

    async def disconnect(self) -> None:
        if self._exit_stack:
            try:
                await self._exit_stack.__aexit__(None, None, None)
            except Exception:
                pass
        self.session = None
        self._exit_stack = None
