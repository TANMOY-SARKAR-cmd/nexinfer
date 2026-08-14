"""Memory fabric: multi-agent memory with isolated + shared spaces.

The fabric owns many ``MemoryStore`` repos:

* **isolated stores** -- one per agent (or per model), private branches;
  other agents cannot see them unless they have an explicit grant.
* **shared stores** -- e.g. the ``whiteboard`` where many agents read
  and publish; merges follow a configured policy per store.
* **access policies** -- ``private | shared | grant:<agent>`` per store.

Network sync: ``push_snapshot`` / ``pull_snapshot`` exchange the whole
store state with another node (used by ``nexinfer memory sync``).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from nexinfer.memory.store import MemoryStore

log = logging.getLogger("nexinfer.memory.fabric")

DEFAULT_MEMORY_ROOT = os.path.expanduser("~/.nexinfer/memory")


@dataclass
class StorePolicy:
    store: str
    kind: str  # private | shared | grant
    owners: list[str]  # agent ids with write access (kind=private)
    merge_policy: str = "theirs"  # ours | theirs | last_write


class MemoryFabric:
    def __init__(self, root: str = DEFAULT_MEMORY_ROOT) -> None:
        self.root = root
        os.makedirs(self.root, exist_ok=True)
        self._stores: dict[str, MemoryStore] = {}
        self._policies: dict[str, StorePolicy] = {}
        self._load_existing()

    # ------------------------------------------------------------------

    def _load_existing(self) -> None:
        policy_file = os.path.join(self.root, "policies.json")
        if os.path.exists(policy_file):
            try:
                with open(policy_file) as f:
                    for d in json.load(f):
                        self._policies[d["store"]] = StorePolicy(**d)
            except Exception:  # noqa: BLE001
                pass
        for name in os.listdir(self.root):
            store_dir = os.path.join(self.root, name)
            if os.path.isdir(store_dir) and os.path.isdir(os.path.join(store_dir, ".git")):
                self._stores[name] = MemoryStore(store_dir, name)

    def _persist_policies(self) -> None:
        with open(os.path.join(self.root, "policies.json"), "w") as f:
            json.dump([
                {"store": p.store, "kind": p.kind, "owners": p.owners, "merge_policy": p.merge_policy}
                for p in self._policies.values()
            ], f, indent=2)

    # ------------------------------------------------------------------

    def create_store(self, name: str, kind: str = "private",
                     owners: list[str] | None = None, merge_policy: str = "theirs") -> MemoryStore:
        if name in self._stores:
            return self._stores[name]
        store = MemoryStore(os.path.join(self.root, name), name)
        self._stores[name] = store
        self._policies[name] = StorePolicy(store=name, kind=kind,
                                           owners=owners or ["default"],
                                           merge_policy=merge_policy)
        self._persist_policies()
        return store

    def get_store(self, name: str) -> MemoryStore | None:
        return self._stores.get(name)

    def whiteboard(self) -> MemoryStore:
        wb = self.get_store("whiteboard")
        if wb is None:
            wb = self.create_store("whiteboard", kind="shared",
                                   owners=["*"], merge_policy="last_write")
        return wb

    def can_write(self, agent_id: str, store_name: str) -> bool:
        policy = self._policies.get(store_name)
        if policy is None:
            return False
        if policy.kind == "shared":
            return True
        if policy.kind == "private":
            return agent_id in policy.owners or "*" in policy.owners
        if policy.kind == "grant":
            return agent_id in policy.owners
        return False

    def grant(self, store_name: str, agent_id: str) -> None:
        policy = self._policies.get(store_name)
        if policy and agent_id not in policy.owners:
            policy.owners.append(agent_id)
            self._persist_policies()

    def revoke(self, store_name: str, agent_id: str) -> None:
        policy = self._policies.get(store_name)
        if policy and agent_id in policy.owners:
            policy.owners.remove(agent_id)
            self._persist_policies()

    def list_stores(self) -> list[str]:
        return list(self._stores)

    # ------------------------------------------------------------------
    # Network sync between nodes (same-model agents across machines)

    def push_snapshot(self, store_name: str, target_dir: str) -> str:
        """Write a store snapshot JSON into ``target_dir``. Returns path."""
        store = self._stores.get(store_name)
        if store is None:
            raise KeyError(f"store {store_name!r} not found")
        os.makedirs(target_dir, exist_ok=True)
        path = os.path.join(target_dir, f"{store_name}.snapshot.json")
        with open(path, "w") as f:
            json.dump(store.export_snapshot(), f, indent=2)
        return path

    def pull_snapshot(self, store_name: str, source_path: str,
                      policy: str | None = None) -> int:
        """Import a snapshot from another node."""
        with open(source_path) as f:
            snapshot = json.load(f)
        store = self.get_store(store_name)
        if store is None:
            pol = self._policies.get(store_name)
            store = self.create_store(store_name, kind=pol.kind if pol else "shared",
                                      owners=pol.owners if pol else ["*"],
                                      merge_policy=policy or (pol.merge_policy if pol else "theirs"))
        else:
            store.merge_policy = policy or store.merge_policy if hasattr(store, "merge_policy") else policy
        return store.import_snapshot(snapshot)

    def close(self) -> None:
        return None
