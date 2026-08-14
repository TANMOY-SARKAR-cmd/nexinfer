"""Distributed parallel planner.

Given a set of nodes (each advertising its devices and memory) and a
model spec, produces a ``ClusterPlan`` that assigns:

* **Pipeline parallelism (PP)** -- contiguous layer ranges to nodes.
  Works with ANY backend because it only moves activations between
  layer boundaries. Default strategy when nodes are heterogeneous.
* **Tensor parallelism (TP)** -- splits attention heads / FFN columns
  across ranks *within* a homogeneous group of nodes (same device kind).

Constraint (per user requirement): all nodes must run the SAME model
architecture (same spec, compatible quantization). Plans are validated
against that constraint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nexinfer.backends.base import DeviceInfo, ModelSpec
from nexinfer.engine.types import ParallelMode, ParallelPlan


@dataclass
class NodeSpec:
    node_id: str
    host: str
    port: int
    devices: list[DeviceInfo] = field(default_factory=list)
    backend_name: str = ""
    model_hash: str = ""  # hash of weights/arch to enforce same-model constraint


@dataclass
class ClusterPlan:
    mode: ParallelMode
    nodes: list[NodeSpec]
    per_node: list[ParallelPlan] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def for_rank(self, rank: int) -> ParallelPlan:
        return self.per_node[rank]


def validate_same_model(nodes: list[NodeSpec], spec: ModelSpec) -> list[str]:
    """All nodes must advertise the same architecture hash. Returns warnings."""
    warnings = []
    hashes = {n.model_hash for n in nodes if n.model_hash}
    if len(hashes) > 1:
        warnings.append("nodes report different model hashes; distributed inference will be incorrect")
    return warnings


def plan_pipeline(nodes: list[NodeSpec], spec: ModelSpec) -> ClusterPlan:
    """Split model layers evenly across nodes (PP)."""
    n = len(nodes)
    per = spec.num_layers // n
    remainder = spec.num_layers % n
    plan = ClusterPlan(mode=ParallelMode.PIPELINE, nodes=nodes)
    start = 0
    for i, node in enumerate(nodes):
        count = per + (1 if i < remainder else 0)
        plan.per_node.append(
            ParallelPlan(
                mode=ParallelMode.PIPELINE,
                world_size=n,
                rank=i,
                pp_layers=[(start, start + count)],
            )
        )
        plan.notes.append(f"node {node.node_id}: layers {start}-{start + count}")
        start += count
    return plan


def plan_tensor(nodes: list[NodeSpec], spec: ModelSpec) -> ClusterPlan:
    """Split attention heads + FFN columns across ranks (TP)."""
    n = len(nodes)
    if spec.num_attention_heads % n != 0:
        raise ValueError(
            f"tensor parallel size {n} must divide num_attention_heads {spec.num_attention_heads}"
        )
    heads_per = spec.num_attention_heads // n
    plan = ClusterPlan(mode=ParallelMode.TENSOR, nodes=nodes)
    for i, node in enumerate(nodes):
        plan.per_node.append(
            ParallelPlan(
                mode=ParallelMode.TENSOR,
                world_size=n,
                rank=i,
                tp_slices=[(i * heads_per, (i + 1) * heads_per)],
            )
        )
        plan.notes.append(f"node {node.node_id}: attention heads {i * heads_per}-{(i + 1) * heads_per}")
    return plan


def automatic(nodes: list[NodeSpec], spec: ModelSpec) -> ClusterPlan:
    """Pick PP for heterogeneous clusters, TP for homogeneous ones."""
    kinds = {tuple(sorted(d.kind.value for d in n.devices)) for n in nodes}
    warnings = validate_same_model(nodes, spec)
    if len(kinds) == 1 and spec.num_attention_heads % len(nodes) == 0:
        plan = plan_tensor(nodes, spec)
    else:
        plan = plan_pipeline(nodes, spec)
    plan.notes.extend(warnings)
    return plan
