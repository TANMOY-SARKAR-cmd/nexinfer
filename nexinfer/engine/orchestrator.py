"""Device placement orchestrator.

Plans how a model's layers and KV-cache are spread across the detected
CPU/GPU/TPU/NPU devices. Designed for low-end laptops:

* If total VRAM >= model size          -> whole model on fastest GPU
* If VRAM covers a prefix of layers    -> place those layers on GPU, rest on CPU
* Otherwise                            -> split large matmul operators between
                                          CPU and any weak GPU (iGPU) per-layer

The plan is expressed as ``PlacementPlan`` objects that backends consume
via ``offload_layers`` / ``load``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from nexinfer.backends.base import Backend, ModelSpec
from nexinfer.engine.profiler import SystemProfile
from nexinfer.engine.types import DeviceId

log = logging.getLogger("nexinfer.orchestrator")

HEADROOM = 0.85  # keep 15% of device memory free for KV cache / OS


@dataclass
class PlacementPlan:
    """Per-device allocation of model layers plus KV-cache capacity."""

    assignments: dict[DeviceId, list[tuple[int, int]]] = field(default_factory=dict)  # device -> layer ranges
    kv_cache_device: DeviceId | None = None
    kv_cache_blocks_device: int = 0
    kv_cache_blocks_host: int = 0
    strategy: str = ""  # gpu_only | hybrid_split | cpu_only
    notes: list[str] = field(default_factory=list)

    @property
    def root_device(self) -> DeviceId:
        for dev, ranges in self.assignments.items():
            if ranges and ranges[0][0] == 0:
                return dev
        return next(iter(self.assignments), "/cpu:0")


def _model_bytes(spec: ModelSpec, quant_factor: float | None = None) -> float:
    bpl = spec.bytes_per_layer()
    total = bpl * spec.num_layers
    if quant_factor:
        total *= quant_factor
    return float(total)


def _quant_factor(quant: str | None) -> float | None:
    if quant is None:
        return None
    q = quant.lower()
    if "q4" in q:
        return 0.5
    if "q3" in q:
        return 0.4
    if "q8" in q:
        return 0.75
    if "q2" in q:
        return 0.3
    return None


def plan_placement(
    spec: ModelSpec,
    system: SystemProfile,
    kv_cache_target_tokens: int = 2048,
    block_size: int = 16,
) -> PlacementPlan:
    """Compute where each model layer should live."""
    plan = PlacementPlan()
    model_bytes = _model_bytes(spec, _quant_factor(spec.quant))
    model_gb = model_bytes / 1024**3
    kv_blocks_needed = (kv_cache_target_tokens + block_size - 1) // block_size

    gpus = sorted(
        [d for d in system.devices if d.kind.value.startswith("/gpu")],
        key=lambda d: d.compute_score,
        reverse=True,
    )
    tpus = [d for d in system.devices if d.kind.value.startswith("/tpu")]

    if not gpus and not tpus:
        plan.assignments["/cpu:0"] = [(0, spec.num_layers)]
        plan.kv_cache_device = None
        plan.kv_cache_blocks_device = 0
        plan.kv_cache_blocks_host = max(kv_blocks_needed, 256)
        plan.strategy = "cpu_only"
        plan.notes.append(f"no accelerator found; full model ({model_gb:.1f} GB) on CPU")
        return plan

    best = gpus[0] if gpus else tpus[0]
    vram = best.total_memory_bytes * HEADROOM
    if vram >= model_bytes:
        plan.assignments[best.device_id] = [(0, spec.num_layers)]
        _leftover = (vram - model_bytes) / 1e9
        plan.kv_cache_device = best.device_id
        plan.kv_cache_blocks_device = kv_blocks_needed
        plan.kv_cache_blocks_host = 0
        plan.strategy = "gpu_only"
        plan.notes.append(f"model ({model_gb:.1f} GB) fits entirely on {best.device_id}")
        return plan

    # hybrid: fit as many layers as possible on the GPU, rest on CPU
    bpl = model_bytes / spec.num_layers
    gpu_layers = min(spec.num_layers, int(vram // bpl))
    if gpu_layers > 0:
        plan.assignments[best.device_id] = [(0, gpu_layers)]
        plan.assignments["/cpu:0"] = [(gpu_layers, spec.num_layers)]
        plan.kv_cache_device = best.device_id
        plan.kv_cache_blocks_device = kv_blocks_needed // 2
        plan.kv_cache_blocks_host = max(kv_blocks_needed - plan.kv_cache_blocks_device, 64)
        plan.strategy = "hybrid_split"
        plan.notes.append(f"{gpu_layers}/{spec.num_layers} layers on {best.device_id}, rest on CPU")
        if len(gpus) > 1:
            plan.notes.append(f"{len(gpus) - 1} additional GPU(s) available for distributed expansion")
    else:
        plan.assignments["/cpu:0"] = [(0, spec.num_layers)]
        plan.kv_cache_blocks_host = kv_blocks_needed
        plan.strategy = "cpu_only"
        plan.notes.append("VRAM too small for even one layer; CPU-only with weak-GPU assist")

    # TPU note
    if tpus:
        plan.notes.append(f"TPU detected ({', '.join(t.device_id for t in tpus)}); load with tpu backend")
    return plan


def select_backends(system: SystemProfile, prefer: str | None = None) -> list[Backend]:
    """Pick the best loadable backends for the detected devices."""
    from nexinfer.backends.registry import detect_all_backends

    backends = detect_all_backends()
    if prefer:
        backends = [b for b in backends if b.name == prefer] + [b for b in backends if b.name != prefer]
    return backends
