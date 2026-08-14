"""Core shared types for NexusInfer.

Defines device naming, generation requests, sampling parameters, and the
data contracts used across the engine, backends, and distributed layers.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DeviceKind(str, Enum):
    CPU = "/cpu"
    GPU_NVIDIA = "/gpu:nvidia"
    GPU_AMD = "/gpu:amd"
    GPU_INTEL = "/gpu:intel"  # iGPU
    TPU = "/tpu"
    NPU_INTEL = "/npu:intel"
    MODULE_CUSTOM = "/module:custom"


# vendor-neutral device identifiers, e.g. "/gpu:nvidia:0", "/tpu:0", "/module:custom:0"
DeviceId = str


def device_id(kind: DeviceKind, index: int = 0, vendor: str | None = None) -> DeviceId:
    base = kind.value if kind != DeviceKind.GPU_INTEL else kind.value
    if vendor:
        return f"{base}:{vendor}:{index}"
    return f"{base}:{index}"


@dataclass
class GenerationRequest:
    """A single prompt-to-text generation request."""

    prompt: str
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    max_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.05
    stop_sequences: list[str] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)  # function-calling schema
    tool_choice: str = "auto"  # auto | required | none
    tools_only: bool = False  # if true, only return tool calls, no text
    agent_id: str | None = None  # binds the request to an agent / memory branch
    skill: str | None = None  # skills bundle name
    stream: bool = False
    # abort-on-cancel: flip to ``True`` externally (e.g. an HTTP client
    # disconnecting or an explicit cancel endpoint); the generation loop
    # checks it each decode step and stops with ``finish_reason="abort"``
    abort_flag: list[bool] = field(default_factory=lambda: [False])


@dataclass
class TokenOutput:
    text: str
    token_id: int | None = None
    finish_reason: str | None = None  # stop | length | tool_call | error
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)


class ParallelMode(str, Enum):
    TENSOR = "tensor_parallel"
    PIPELINE = "pipeline_parallel"


@dataclass
class ParallelPlan:
    mode: ParallelMode
    world_size: int
    rank: int
    # tensor parallel: slice indices per rank; pipeline: layer ranges per rank
    tp_slices: list[tuple[int, int]] | None = None
    pp_layers: list[tuple[int, int]] | None = None

    @property
    def is_root(self) -> bool:
        return self.rank == 0


@dataclass
class MemoryRef:
    """Reference to a location in the memory fabric."""

    store: str  # repo id
    branch: str
    key: str | None = None

    def __str__(self) -> str:
        key_part = f"/{self.key}" if self.key else ""
        return f"{self.store}:{self.branch}{key_part}"
