"""Abstract backend interface for NexusInfer.

Every accelerator/vendor backend (CPU numpy, GGML, ONNX Runtime, CUDA,
ROCm, DirectML, Metal, libtpu, custom special module) implements this
interface and registers itself as an entry-point plugin under
`nexinfer.backends` so the engine can load drivers without any code
changes to the core.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nexinfer.engine.types import DeviceId, GenerationRequest, TokenOutput, device_id, DeviceKind


@dataclass
class DeviceInfo:
    device_id: DeviceId
    kind: DeviceKind
    vendor: str
    name: str
    total_memory_bytes: int
    compute_score: float  # normalized throughput score from auto-benchmark


@dataclass
class ModelSpec:
    """Common model spec shared between backends for placement planning."""

    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    inter_dim: int = 0
    dtype: str = "float16"
    quant: str | None = None  # e.g. "q4_k_m"

    def bytes_per_layer(self) -> int:
        d = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1, "int4": 0.5}[self.dtype]
        base = (self.hidden_size ** 2) * 4  # attention + gate/fc weight matrices
        return int(base * d) + self.inter_dim * self.hidden_size * d


@dataclass
class BackendCapabilities:
    supports_gpu: bool = False
    supports_tpu: bool = False
    supports_quant: bool = False
    supports_tool_calls: bool = False
    supports_pipeline_parallel: bool = False
    supports_tensor_parallel: bool = False
    min_python: str = "3.11"
    extra: dict[str, Any] = field(default_factory=dict)


class Backend(ABC):
    """Base class all inference backends must implement."""

    name: str = "base"
    platform: str = "any"  # any | linux | windows | darwin

    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """Report what this backend can do."""
        ...

    @abstractmethod
    def detect_devices(self) -> list[DeviceInfo]:
        """Discover devices this backend can drive (may be empty)."""
        ...

    @abstractmethod
    def load(self, model_path_or_id: str, spec: ModelSpec, devices: list[DeviceId]) -> None:
        """Load model weights onto the requested device placement."""
        ...

    @abstractmethod
    def prefill(self, req_id: str, input_ids: np.ndarray) -> np.ndarray:
        """Run a full-prompt forward pass; return next-token logits."""
        ...

    @abstractmethod
    def decode(self, req_ids: list[str], input_ids: np.ndarray) -> np.ndarray:
        """Batched single-token decode; return logits (len(req_ids), vocab)."""
        ...

    def offload_layers(
        self, layer_ranges: list[tuple[int, int]] | None
    ) -> None:  # pylint: disable=unused-argument
        """Move weight layers between devices (used by orchestrator)."""
        return None

    def benchmark(self, device: DeviceId, seconds: float = 2.0) -> float:
        """Return a normalized throughput score for the device."""
        return 1.0

    def close(self) -> None:
        return None


def default_device_map(devices: list[DeviceInfo]) -> list[DeviceId]:
    return [d.device_id for d in devices]
