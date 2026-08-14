"""High-level engine runtime.

One-stop entry point used by the CLI and the server: profile the
machine, pick backends, plan placement, load the model, and expose
``generate`` / ``generate_stream``.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from dataclasses import dataclass

from nexinfer.backends.base import Backend, ModelSpec
from nexinfer.backends.registry import available_backends, load_backend
from nexinfer.engine.generation import GenerationEngine
from nexinfer.engine.kvcache import PagedKVCache
from nexinfer.engine.orchestrator import PlacementPlan, plan_placement
from nexinfer.engine.profiler import SystemProfile
from nexinfer.engine.scheduler import Scheduler
from nexinfer.engine.tokenizer_helper import Tokenizer
from nexinfer.engine.types import GenerationRequest, TokenOutput

log = logging.getLogger("nexinfer.runtime")


@dataclass
class EngineStatus:
    profile: SystemProfile
    placement: PlacementPlan
    backend_names: list[str]
    model: str


class Engine:
    """Main runtime: profiler + orchestrator + backend + generation."""

    def __init__(self) -> None:
        self.backend: Backend | None = None
        self.tokenizer: Tokenizer | None = None
        self._generator: GenerationEngine | None = None
        self.status: EngineStatus | None = None
        self.scheduler: Scheduler | None = None
        # abort registry: request id -> the abort-flag list shared with the
        # ``GenerationRequest`` it was created with. An HTTP cancel call
        # flips the flag while the generation loop is running (abort-on-cancel).
        self._abort_registry: dict[str, list[bool]] = {}

    # EngineStatus carries ``profile`` and ``placement`` for the
    # ``nexinfer profile`` command; the scheduler is also exposed so the
    # HTTP layer can report queue depth without digging into the engine.

    # ------------------------------------------------------------------

    def bootstrap(
        self,
        model: str,
        spec: ModelSpec,
        backend_name: str | None = None,
        kv_tokens: int = 2048,
        benchmark: bool = False,
    ) -> EngineStatus:
        system = SystemProfile.from_system(benchmark=benchmark)
        placement = plan_placement(spec, system, kv_cache_target_tokens=kv_tokens)
        backends = self._resolve_backends(backend_name, system)
        backend = backends[0]
        log.info(
            "placement: %s | backend: %s | strategy: %s",
            placement.strategy,
            backend.name,
            placement.strategy,
        )
        try:
            backend.load(model, spec, [placement.root_device])
        except FileNotFoundError:
            # real weights unavailable (CI/demo mode) -> load random demo weights
            log.warning("weights not found for %s; loading random demo weights", model)
            backend.load("nexinfer-demo-weights", spec, [placement.root_device])
        tok = self._fallback_tokenizer() if backend_name == "cpu_numpy" else Tokenizer.load(model)
        sched = Scheduler(
            device_blocks=max(placement.kv_cache_blocks_device, 64),
            host_blocks=max(placement.kv_cache_blocks_host, 256),
        )
        cache = PagedKVCache(
            num_blocks_device=max(placement.kv_cache_blocks_device, 64),
            num_blocks_host=max(placement.kv_cache_blocks_host, 256),
            num_layers=spec.num_layers,
            num_kv_heads=spec.num_kv_heads,
            head_dim=spec.head_dim,
        )
        self.backend = backend
        self.tokenizer = tok
        self.scheduler = sched
        self._generator = GenerationEngine(backend, tok, sched, cache)
        self.status = EngineStatus(
            profile=system,
            placement=placement,
            backend_names=[b.name for b in backends],
            model=model,
        )
        return self.status

    def register(self, req: GenerationRequest) -> GenerationRequest:
        """Register ``req`` for abort-on-cancel and return it for chaining.

        Flipping the flag via ``engine.cancel(req.request_id)`` stops the
        running generation with ``finish_reason="abort"``.
        """
        self._abort_registry[req.request_id] = req.abort_flag
        return req

    def cancel(self, request_id: str) -> bool:
        """Cancel an in-flight generation by id. Idempotent; ``False`` when
        the request is unknown or already finished."""
        flag = self._abort_registry.get(request_id)
        if flag is None:
            return False
        flag[0] = True
        return True

    def generate(self, req: GenerationRequest) -> TokenOutput:
        self._require()
        try:
            return self._generator.generate(req)
        finally:
            self._abort_registry.pop(req.request_id, None)

    def generate_stream(self, req: GenerationRequest) -> Generator[TokenOutput, None, None]:
        self._require()
        try:
            yield from self._generator.generate_stream(req)
        finally:
            self._abort_registry.pop(req.request_id, None)

    @property
    def queue_depth(self) -> int:
        """Waiting-queue depth exposed to the HTTP ``/v1/status`` endpoint."""
        return self.scheduler.num_waiting if self.scheduler is not None else 0

    # ------------------------------------------------------------------

    def _require(self) -> None:
        if self._generator is None:
            raise RuntimeError("engine not bootstrapped; call bootstrap() first")

    def _resolve_backends(self, prefer: str | None, system: SystemProfile) -> list[Backend]:
        if prefer:
            be = load_backend(prefer, allow_missing=True)
            if be is None:
                raise RuntimeError(
                    f"backend {prefer!r} unavailable (dependencies missing); "
                    f"available: {sorted(available_backends())}"
                )
            return [be]
        from nexinfer.backends.cpu_numpy import NumpyBackend

        # prefer accelerated backends matching detected devices, fallback to CPU numpy
        gpus = [d for d in system.devices if d.device_id.startswith("/gpu")]
        if gpus:
            for name in ("cuda", "rocm", "directml", "ort"):
                be = load_backend(name, allow_missing=True)
                if be is not None and be.detect_devices():
                    return [be]
        tpus = [d for d in system.devices if d.device_id.startswith("/tpu")]
        if tpus:
            be = load_backend("tpu", allow_missing=True)
            if be is not None:
                return [be]
        return [NumpyBackend()]

    @staticmethod
    def _fallback_tokenizer() -> Tokenizer:
        from nexinfer.engine.tokenizer_helper import MinimalBPE

        return Tokenizer(MinimalBPE())
