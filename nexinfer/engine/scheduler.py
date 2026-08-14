"""Continuous batching scheduler.

vLLM-style continuous scheduler: requests are admitted immediately when
blocks are free (no fixed batch boundaries), preempted when memory is
scarce, and prioritized by waiting time. The scheduler yields a batch of
(RunningRequest) objects each step that the backend executes together.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from nexinfer.engine.types import GenerationRequest


class ReqStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    SWAPPED = "swapped"


@dataclass
class RunningRequest:
    req: GenerationRequest
    status: ReqStatus = ReqStatus.WAITING
    generated_ids: list[int] = field(default_factory=list)
    generated_text: str = ""
    waiting_since: float = field(default_factory=time.time)
    blocks_needed: int = 0
    stopped: bool = False
    tool_calls: list[dict] = field(default_factory=list)

    @property
    def input_ids(self) -> np.ndarray:
        return np.array(self.generated_ids, dtype=np.int32)

    def append(self, token_id: int, text: str) -> None:
        self.generated_ids.append(token_id)
        self.generated_text += text

    def should_stop(self) -> str | None:
        """Return finish reason if a stop condition is met, else None."""
        if self.req.max_tokens > 0 and len(self.generated_ids) >= self.req.max_tokens:
            return "length"
        for stop in self.req.stop_sequences:
            if stop and stop in self.generated_text:
                return "stop"
        return None


class Scheduler:
    """Continuous batching scheduler with simple priority admission."""

    def __init__(
        self,
        max_running: int = 16,
        max_waiting: int = 256,
        device_blocks: int = 1024,
        host_blocks: int = 4096,
        block_size: int = 16,
        preempt_mode: str = "recompute",  # recompute | swap
    ) -> None:
        self.max_running = max_running
        self.max_waiting = max_waiting
        self.total_blocks = device_blocks + host_blocks
        self.block_size = block_size
        self.preempt_mode = preempt_mode
        self.waiting: deque[RunningRequest] = deque()
        self.running: dict[str, RunningRequest] = {}
        self._used_blocks = 0

    # ------------------------------------------------------------------
    # Admission
    # ------------------------------------------------------------------

    def add(self, req: GenerationRequest) -> RunningRequest | None:
        if len(self.waiting) + len(self.running) >= self.max_waiting:
            return None
        rr = RunningRequest(req=req, blocks_needed=0)
        self.waiting.append(rr)
        return rr

    def _blocks_needed_for(self, rr: RunningRequest, future_len: int) -> int:
        total = len(rr.generated_ids) + future_len
        return (total + self.block_size - 1) // self.block_size

    def schedule(self, max_new_blocks: int | None = None) -> list[RunningRequest]:
        """Admit waiting requests and return the current running set."""
        free = self.total_blocks - self._used_blocks
        for rr in list(self.waiting):
            if len(self.running) >= self.max_running:
                break
            need = self._blocks_needed_for(rr, rr.req.max_tokens)
            if need <= free:
                self.waiting.popleft()
                rr.status = ReqStatus.RUNNING
                self.running[rr.req.request_id] = rr
                self._used_blocks += need
        return list(self.running.values())

    def finish(self, req_id: str) -> RunningRequest | None:
        rr = self.running.pop(req_id, None)
        if rr is not None:
            rr.status = ReqStatus.FINISHED
            need = self._blocks_needed_for(rr, 0)
            self._used_blocks = max(0, self._used_blocks - need)
        return rr

    def preempt(self, req_id: str) -> RunningRequest | None:
        """Preempt the longest-running request to free blocks."""
        if not self.running:
            return None
        victim = max(self.running.values(), key=lambda r: time.time() - r.waiting_since)
        rr = self.running.pop(victim.req.request_id)
        need = self._blocks_needed_for(rr, 0)
        self._used_blocks = max(0, self._used_blocks - need)
        if self.preempt_mode == "recompute":
            rr.generated_ids = []
            rr.generated_text = ""
        rr.status = ReqStatus.SWAPPED
        self.waiting.appendleft(rr)
        return rr

    def request(self, req_id: str) -> RunningRequest | None:
        return self.running.get(req_id) or next((r for r in self.waiting if r.req.request_id == req_id), None)

    @property
    def num_waiting(self) -> int:
        """Requests queued but not yet admitted."""
        return len(self.waiting)

    def num_running(self) -> int:
        return len(self.running)

    @property
    def utilization(self) -> float:
        return self._used_blocks / max(1, self.total_blocks)
