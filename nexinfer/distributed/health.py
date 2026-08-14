"""Distributed fault-tolerance helpers.

This module adds the two missing pieces of the cluster control plane:

* ``HealthMonitor`` -- run by the **coordinator**; periodically pings every
  registered node. When a node's heartbeat goes silent longer than
  ``timeout``, it is declared dead and the cluster plan is recomputed
  across the surviving nodes (a crashed worker's layers get redistributed
  automatically).

* ``reconnect_loop`` -- run by a **worker**; if the coordinator is
  unreachable (crash / restart / network partition), the worker keeps
  trying with exponential backoff and re-registers as soon as the
  coordinator comes back, picking up the pushed plan again.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time

from nexinfer.distributed.messages import Msg

log = logging.getLogger("nexinfer.distributed.health")

DEFAULT_HEARTBEAT_TIMEOUT_S = 30.0
DEFAULT_CHECK_INTERVAL_S = 5.0


class HealthMonitor:
    """Periodically probe registered nodes and declare them dead when they
    stop heartbeating.

    The monitor does **not** own the plan itself -- it only knows *who is
    alive* and calls an async ``on_node_dead`` callback so the coordinator
    can replan. A node that recovers within the timeout never triggers a
    replan (its heartbeat simply resumes).
    """

    def __init__(
        self,
        timeout: float = DEFAULT_HEARTBEAT_TIMEOUT_S,
        check_interval: float = DEFAULT_CHECK_INTERVAL_S,
    ) -> None:
        self.timeout = timeout
        # check often enough to notice a silence promptly: the interval is at
        # most a third of the timeout (default stays 5 s for production
        # timeouts like 30 s, but a sub-second test timeout still works)
        self.check_interval = min(check_interval, max(0.02, timeout / 3.0))
        self._last_seen: dict[str, float] = {}
        self._dead: set[str] = set()
        self._on_node_dead = None
        self._task: asyncio.Task | None = None
        self._running = False

    @property
    def dead_nodes(self) -> set[str]:
        return set(self._dead)

    def record(self, node_id: str) -> None:
        """Call on every heartbeat/welcome arrival."""
        self._last_seen[node_id] = time.time()
        self._dead.discard(node_id)

    def forget(self, node_id: str) -> None:
        self._last_seen.pop(node_id, None)
        self._dead.discard(node_id)

    def set_on_node_dead(self, callback) -> None:
        self._on_node_dead = callback

    def _age(self, node_id: str) -> float:
        seen = self._last_seen.get(node_id)
        return 0.0 if seen is None else time.time() - seen

    def is_alive(self, node_id: str) -> bool:
        return self._age(node_id) <= self.timeout

    async def _tick(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self.check_interval)
                ages = {nid: round(time.time() - t, 3) for nid, t in self._last_seen.items()}
                log.debug("health tick ages=%s timeout=%s", ages, self.timeout)
                for node_id in list(self._last_seen):
                    if node_id not in self._dead and self._age(node_id) > self.timeout:
                        self._dead.add(node_id)
                        log.warning(
                            "node %s declared DEAD (no heartbeat for %.0fs)", node_id, self._age(node_id)
                        )
                        if self._on_node_dead is not None:
                            try:
                                result = self._on_node_dead(node_id)
                                if asyncio.iscoroutine(result):
                                    await result
                            except Exception as exc:  # never let the monitor die
                                log.error("on_node_dead callback failed: %s", exc)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                log.error("health monitor tick failed: %s", exc)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._tick())

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None


async def reconnect_loop(
    node_id: str,
    register_fn,  # async (node_id) -> bool -- returns True on success
    host: str,
    port: int,
    *,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: float = 0.3,
    running_predicate=None,  # () -> bool; False stops the loop (worker close)
    log_fn=None,
) -> None:
    """Worker-side reconnection loop with exponential backoff.

    Call this from the worker after its coordinator connection drops:

        await reconnect_loop(self.node_id, self._register,
                             "coordinator.example.com", 9000,
                             running_predicate=lambda: self._running)

    Delays grow as 1s -> 2s -> 4s -> ... capped at ``max_delay`` with
    optional random jitter so a mass-reboot storm does not hit the
    coordinator simultaneously.
    """
    delay = initial_delay
    while True:
        if running_predicate is not None and not running_predicate():
            return
        try:
            ok = await register_fn(node_id)
        except Exception:
            ok = False
            log_fn = log_fn or log.error
        if ok:
            log.info("node %s re-registered with coordinator %s:%d", node_id, host, port)
            return
        log.warning("node %s: coordinator %s:%d unreachable, retry in %.1fs", node_id, host, port, delay)
        await asyncio.sleep(delay)
        delay = min(max_delay, delay * 2)
        if jitter:
            delay *= 1.0 + random.random() * jitter


async def wait_for_heartbeat(
    coordinator_host: str,
    coordinator_port: int,
    node_id: str,
    *,
    timeout: float = 10.0,
) -> bool:
    """Send one ``heartbeat`` to the coordinator and return True on ack.

    Used both by workers (liveness probe before an expensive operation)
    and by tests (simulating coordinator ping from a client)."""
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(coordinator_host, coordinator_port), timeout
        )
        try:
            msg = Msg(type="heartbeat", src=node_id)
            writer.write((msg.to_json() + "\n").encode())
            await writer.drain()
            line = await asyncio.wait_for(reader.readline(), timeout)
            if not line:
                return False
            resp = Msg.from_json(line)
            return resp.type == "heartbeat_ack"
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
    except Exception:
        return False
