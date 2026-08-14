"""Persistent config store (~/.nexinfer/config.json) for MCP servers
and engine settings."""

from __future__ import annotations

import json
import os
from typing import Any

CONFIG_DIR = os.path.expanduser("~/.nexinfer")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")


class McpConfig:
    def __init__(self) -> None:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        self._data: dict[str, Any] = {}
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH) as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}

    def _save(self) -> None:
        with open(CONFIG_PATH, "w") as f:
            json.dump(self._data, f, indent=2)

    def add(self, name: str, config: dict[str, Any]) -> None:
        self._data.setdefault("mcp_servers", {})[name] = config
        self._save()

    def remove(self, name: str) -> None:
        self._data.get("mcp_servers", {}).pop(name, None)
        self._save()

    def list_all(self) -> dict[str, Any]:
        return self._data.get("mcp_servers", {})

    def get(self, name: str) -> dict[str, Any] | None:
        return self._data.get("mcp_servers", {}).get(name)
