"""Tool registry: the single registry every caller (generation loop,
skills, MCP bridge) consults to resolve tool names to callables and
schemas.

Built-in tool families:

* ``web_fetch``            -- InternetGateway
* ``memory_*``             -- MemoryFabric operations
* ``mcp:<server>:<tool>``  -- MCP server tools
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from nexinfer.services.internet_gateway import InternetGateway

log = logging.getLogger("nexinfer.gateway.tools")


class ToolRegistry:
    def __init__(self, internet: InternetGateway | None = None, memory_fabric: Any = None) -> None:
        self.internet = internet
        self.memory = memory_fabric
        self._callables: dict[str, Callable] = {}
        self._schemas: dict[str, dict[str, Any]] = {}
        self._register_builtins()

    # ------------------------------------------------------------------

    def _register_builtins(self) -> None:
        if self.internet:
            self.register("web_fetch", self.internet.call, self.internet.tool_schema())
        if self.memory:
            fab = self.memory

            def _mem_write(store: str, branch: str, key: str, value: str, message: str = "update") -> dict:
                store_obj = fab.get_store(store) or fab.create_store(store)
                return {"commit": store_obj.set(key, value, branch=branch, message=message)}

            def _mem_read(store: str, branch: str, key: str) -> dict:
                store_obj = fab.get_store(store)
                return {"value": store_obj.get(key, branch=branch) if store_obj else None}

            def _mem_search(store: str, query: str, branch: str = "main", top_k: int = 5) -> dict:
                store_obj = fab.get_store(store)
                if store_obj is None:
                    return {"results": []}
                return {"results": store_obj.search(query, top_k=top_k)}

            def _mem_branch(store: str, branch: str, base: str = "main") -> dict:
                store_obj = fab.get_store(store) or fab.create_store(store)
                store_obj.branch(branch, base=base)
                return {"branch": branch, "base": base}

            self.register("memory_write", _mem_write, {
                "name": "memory_write",
                "description": "Commit a value into a git-backed memory branch",
                "parameters": {"type": "object", "properties": {
                    "store": {"type": "string"}, "branch": {"type": "string"},
                    "key": {"type": "string"}, "value": {"type": "string"},
                    "message": {"type": "string"},
                }, "required": ["store", "branch", "key", "value"]},
            })
            self.register("memory_read", _mem_read, {
                "name": "memory_read",
                "description": "Read a key from a memory branch",
                "parameters": {"type": "object", "properties": {
                    "store": {"type": "string"}, "branch": {"type": "string"}, "key": {"type": "string"},
                }, "required": ["store", "branch", "key"]},
            })
            self.register("memory_search", _mem_search, {
                "name": "memory_search",
                "description": "Semantic search over committed memory entries",
                "parameters": {"type": "object", "properties": {
                    "store": {"type": "string"}, "query": {"type": "string"},
                    "branch": {"type": "string"}, "top_k": {"type": "integer"},
                }, "required": ["store", "query"]},
            })
            self.register("memory_branch", _mem_branch, {
                "name": "memory_branch",
                "description": "Create an isolated memory branch from a base",
                "parameters": {"type": "object", "properties": {
                    "store": {"type": "string"}, "branch": {"type": "string"}, "base": {"type": "string"},
                }, "required": ["store", "branch"]},
            })

    def register(self, name: str, fn: Callable, schema: dict[str, Any]) -> None:
        self._callables[name] = fn
        self._schemas[name] = schema

    def register_mcp_tools(self, server_name: str, mcp_tools: list[Any]) -> None:
        from nexinfer.gateway.mcp_client import McpGatewayClient

        for t in mcp_tools:
            if isinstance(t, McpGatewayClient):
                for tool in t.tools:
                    name = f"mcp:{server_name}:{tool.name}"
                    self._callables[name] = lambda args, _t=tool, _c=t: None  # placeholder
                    self._schemas[name] = tool.to_engine_schema()
        # proper async calling happens in ToolCaller

    def get_schema(self, name: str) -> dict[str, Any] | None:
        return self._schemas.get(name)

    def has(self, name: str) -> bool:
        return name in self._callables

    def names(self) -> list[str]:
        return list(self._callables)


class ToolCaller:
    """Resolves tool-call dicts (from the generation loop) to results.

    Handles synchronous built-ins directly and routes
    ``mcp:<server>:<tool>`` through the matching ``McpGatewayClient``.
    """

    def __init__(self, registry: ToolRegistry, mcp_clients: dict[str, Any]) -> None:
        self.registry = registry
        self.mcp_clients = mcp_clients  # server_name -> McpGatewayClient

    def call(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        fn = self.registry._callables.get(tool_name)
        if fn is not None and not tool_name.startswith("mcp:"):
            try:
                return fn(**arguments)
            except Exception as exc:  # noqa: BLE001
                return {"error": str(exc), "tool": tool_name}
        if tool_name.startswith("mcp:"):
            _, server, tool = tool_name.split(":", 2)
            client = self.mcp_clients.get(server)
            if client is None:
                return {"error": f"MCP server {server!r} not connected", "tool": tool_name}
            try:
                import asyncio
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop and loop.is_running():
                    from concurrent.futures import Future
                    future = asyncio.run_coroutine_threadsafe(client.call_tool(tool, arguments), loop)
                    return future.result(timeout=30)
                return asyncio.run(client.call_tool(tool, arguments))
            except Exception as exc:  # noqa: BLE001
                return {"error": str(exc), "tool": tool_name}
        return {"error": f"unknown tool {tool_name!r}", "tool": tool_name}
