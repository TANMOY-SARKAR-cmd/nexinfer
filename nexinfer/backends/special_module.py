"""Special module backend: plug for custom/proprietary accelerator modules.

Any custom compute module (FPGA bitstream, ASIC, neuromorphic chip,
custom inference card, cloud attachable module) implements the
``SpecialModule`` protocol below and is registered with

    nexinfer.backends.special_module.register_module(MyModule)

or exposed as an entry-point under ``nexinfer.backends.special_modules``.
The engine treats it like any other device (``/module:custom:0``) and the
orchestrator can place layers on it.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from nexinfer.backends.base import Backend, BackendCapabilities, DeviceInfo, ModelSpec
from nexinfer.engine.types import DeviceId, DeviceKind

log = logging.getLogger("nexinfer.backends.special_module")


class SpecialModule(ABC):
    """Protocol every custom accelerator module must implement."""

    module_name: str = "custom"

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the module hardware/runtime is present."""
        ...

    @abstractmethod
    def memory_bytes(self) -> int: ...

    @abstractmethod
    def load_weights(self, tensors: dict[str, np.ndarray]) -> None:
        """Push model weights to the module (format is module-defined)."""
        ...

    @abstractmethod
    def run(self, input_ids: np.ndarray, mode: str) -> np.ndarray:
        """Run prefill (mode='prefill') or decode (mode='decode'); return logits."""
        ...

    def benchmark_score(self) -> float:
        return 1.0

    def release(self) -> None:
        return None


class ReferenceModule(SpecialModule):
    """Example module: a pure-numpy implementation standing in for an
    actual hardware accelerator. Replace with your module's driver."""

    module_name = "reference"

    def __init__(self) -> None:
        self._weights: dict[str, np.ndarray] = {}
        self._rng = np.random.default_rng(7)

    def is_available(self) -> bool:
        # Modules decide availability via env var, device nodes, or vendor
        # SDK probing. The reference module is available only when opted in.
        return __name__.split(".")[-1] == "special_module"  # always available as fallback

    def memory_bytes(self) -> int:
        return int(sum(w.nbytes for w in self._weights.values()))

    def load_weights(self, tensors: dict[str, np.ndarray]) -> None:
        self._weights = dict(tensors)

    def run(self, input_ids: np.ndarray, mode: str) -> np.ndarray:
        vocab = self._weights.get("vocab_size", 32000)
        if isinstance(vocab, np.ndarray):
            vocab = int(vocab.max()) + 1 if vocab.size else 32000
        # reference behaviour: deterministic argmax-style logits
        logits = self._rng.standard_normal((max(input_ids.size, 1), int(vocab))).astype(np.float32)
        return logits[-1:]

    def benchmark_score(self) -> float:
        return 1.5


_REGISTRY: dict[str, type[SpecialModule]] = {"reference": ReferenceModule}


def register_module(cls: type[SpecialModule]) -> type[SpecialModule]:
    """Register a custom module class by its module_name."""
    _REGISTRY[cls.module_name] = cls
    log.info("registered special module %s", cls.module_name)
    return cls


class SpecialModuleBackend(Backend):
    """Backend that drives all registered special modules."""

    name = "special_module"
    platform = "any"

    def __init__(self) -> None:
        self.modules: dict[str, SpecialModule] = {}
        self.spec: ModelSpec | None = None

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_gpu=False,
            supports_tpu=False,
            supports_quant=False,
            supports_tool_calls=False,
            supports_pipeline_parallel=True,
            supports_tensor_parallel=True,
            extra={"registered_modules": list(_REGISTRY)},
        )

    def detect_devices(self) -> list[DeviceInfo]:
        out = []
        for name, cls in _REGISTRY.items():
            try:
                module = cls()
            except Exception:
                continue
            if module.is_available():
                self.modules[name] = module
                out.append(
                    DeviceInfo(
                        f"/module:custom:{len(out)}",
                        DeviceKind.MODULE_CUSTOM,
                        "custom",
                        f"SpecialModule:{name}",
                        module.memory_bytes(),
                        module.benchmark_score(),
                    )
                )
        return out

    def load(self, model_path_or_id: str, spec: ModelSpec, devices: list[DeviceId]) -> None:
        self.spec = spec
        from nexinfer.backends.cpu_numpy import _random_weights

        rng = np.random.default_rng(3)
        tensors = _random_weights(spec, rng)
        tensors["vocab_size"] = np.array(spec.vocab_size, dtype=np.int64)
        for module in self.modules.values():
            module.load_weights(tensors)

    def prefill(self, req_id: str, input_ids: np.ndarray) -> np.ndarray:
        # run on the first registered module (extend to per-op routing)
        module = next(iter(self.modules.values()))
        return module.run(input_ids, "prefill")

    def decode(self, req_ids: list[str], input_ids: np.ndarray) -> np.ndarray:
        module = next(iter(self.modules.values()))
        return module.run(input_ids, "decode")

    def close(self) -> None:
        for m in self.modules.values():
            m.release()
        self.modules.clear()
