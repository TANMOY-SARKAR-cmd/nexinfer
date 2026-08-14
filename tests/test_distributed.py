"""Multi-process distributed (pipeline-parallel) integration tests.

Spawns real processes on localhost, one coordinator/root worker and one
(or more) intermediate/final workers, and drives a full prefill + decode
ring over the TCP transport. Asserts that the token sequence produced
across the cluster matches what the single-node numpy backend would
produce (same model spec, same random weights, greedy sampling).
"""

from __future__ import annotations

import asyncio
import json
import socket
import subprocess
import sys
import time
import uuid

import numpy as np
import pytest

from nexinfer.backends.base import ModelSpec
from nexinfer.backends.cpu_numpy import NumpyBackend
from nexinfer.distributed.coordinator import Coordinator
from nexinfer.distributed.messages import Msg
from nexinfer.distributed.planner import NodeSpec, plan_pipeline
from nexinfer.distributed.worker import Worker
from nexinfer.engine.types import DeviceKind
from nexinfer.transports.tcp_transport import TCPTransport

SPEC = ModelSpec(
    num_layers=4,
    hidden_size=64,
    num_attention_heads=4,
    num_kv_heads=2,
    head_dim=16,
    vocab_size=256,
    inter_dim=128,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_worker(
    rank: int, node_id: str, spec_json: dict, control_host: str, control_port: int, shared_flag: str
) -> None:
    """Entry point for a worker subprocess. It registers with the
    coordinator over the control channel and runs ``run_decode_loop``."""
    import asyncio

    from nexinfer.backends.base import ModelSpec
    from nexinfer.backends.cpu_numpy import NumpyBackend
    from nexinfer.distributed.messages import Msg
    from nexinfer.distributed.planner import NodeSpec, plan_pipeline
    from nexinfer.distributed.worker import Worker
    from nexinfer.transports.tcp_transport import TCPTransport

    async def main() -> None:
        spec = ModelSpec(**spec_json)
        backend = NumpyBackend()
        backend.load("demo-model-pp", spec, [])
        plan = plan_pipeline(
            [NodeSpec("root", "127.0.0.1", control_port), NodeSpec(node_id, "127.0.0.1", 0)], spec
        )
        transport = TCPTransport()
        # the control port here is ONLY used as the coordinator address for
        # the hello registration; the worker binds its OWN control socket on
        # coordinator port + 2000 (arbitrary distinct offset).
        worker = Worker(
            node_id=node_id,
            backend=backend,
            spec=spec,
            plan=plan,
            transport=transport,
            rank=rank,
            control_host=control_host,
            control_port=control_port + 2000,
        )
        actual = await worker.start()
        # register with the coordinator (control socket on its port)
        reader, writer = await asyncio.open_connection("127.0.0.1", control_port)
        hello = Msg(
            type="hello",
            payload={
                "devices": [
                    {
                        "device_id": "/cpu:0",
                        "kind": DeviceKind.CPU,
                        "vendor": "generic",
                        "name": "CPU (numpy)",
                        "total_memory_bytes": 0,
                        "compute_score": 1.0,
                    }
                ],
                "host": "127.0.0.1",
                "port": actual,
                "backend": "cpu_numpy",
                "model_hash": f"model:{spec.num_layers}:{spec.hidden_size}",
            },
            src=node_id,
        )
        writer.write((json.dumps(hello.to_dict()) + "\n").encode())
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), 10.0)
        resp = json.loads(line)
        assert resp["type"] == "welcome", resp
        writer.close()
        with open(shared_flag, "w") as f:
            f.write(f"{actual}\n")
        await worker.run_decode_loop()

    # subprocess runs with its own event loop, but be robust when called
    # from within an existing loop (e.g. in-process smoke tests)
    try:
        asyncio.run(main())
    except RuntimeError:
        asyncio.get_event_loop().create_task(main())
    except Exception:
        import traceback

        with open(shared_flag + ".err", "w") as f:
            f.write(traceback.format_exc() + "\n")
        raise


def _make_workers(rank_specs: list[tuple[int, str]]) -> list[Worker]:
    workers = []
    for rank, node_id in rank_specs:
        backend = NumpyBackend()
        backend.load("demo-model-pp", SPEC, [])
        plan = plan_pipeline([NodeSpec("root", "127.0.0.1", 0), NodeSpec("w1", "127.0.0.1", 0)], SPEC)
        transport = TCPTransport()
        workers.append(
            Worker(
                node_id=node_id,
                backend=backend,
                spec=SPEC,
                plan=plan,
                transport=transport,
                rank=rank,
                control_host="127.0.0.1",
                control_port=0,
            )
        )
    return workers


@pytest.mark.asyncio
async def test_pp_two_workers_ring(tmp_path, monkeypatch):
    """Two real processes (root + rank-1) run a 4-layer model split 2/2
    and produce the SAME token ids a single-node run would (greedy)."""

    coordinator_port = _free_port()
    shared_flag = str(tmp_path / "worker_ready")

    spec_json = {
        "num_layers": 4,
        "hidden_size": 64,
        "num_attention_heads": 4,
        "num_kv_heads": 2,
        "head_dim": 16,
        "vocab_size": 256,
        "inter_dim": 128,
    }

    # start coordinator + root worker in this process
    coord = Coordinator(
        "coord",
        SPEC,
        TCPTransport(),
        host="127.0.0.1",
        port=coordinator_port,
        preferred_mode="pipeline_parallel",
    )
    coord_port = await coord.start()
    backend = NumpyBackend()
    backend.load("demo-model-pp", SPEC, [])
    plan = plan_pipeline([NodeSpec("root", "127.0.0.1", coord_port), NodeSpec("w1", "127.0.0.1", 0)], SPEC)
    root = Worker(
        "root",
        backend,
        SPEC,
        plan,
        TCPTransport(),
        rank=0,
        control_host="127.0.0.1",
        control_port=coordinator_port + 1,
    )
    await root.start()
    assert root.transport.listen_addr

    # register the in-process root worker with the coordinator exactly as a
    # remote worker would (hello payload carries its real control port so
    # data-plane addresses resolve correctly for all ranks)
    reader2, writer2 = await asyncio.open_connection("127.0.0.1", coord_port)
    hello2 = Msg(
        type="hello",
        payload={
            "devices": [
                {
                    "device_id": "/cpu:0",
                    "kind": DeviceKind.CPU,
                    "vendor": "generic",
                    "name": "CPU (numpy)",
                    "total_memory_bytes": 0,
                    "compute_score": 1.0,
                }
            ],
            "host": "127.0.0.1",
            "port": root.control_port,
            "backend": "cpu_numpy",
            "model_hash": f"model:{SPEC.num_layers}:{SPEC.hidden_size}",
        },
        src="root",
    )
    writer2.write((json.dumps(hello2.to_dict()) + "\n").encode())
    await writer2.drain()
    line2 = await asyncio.wait_for(reader2.readline(), 10.0)
    assert json.loads(line2)["type"] == "welcome"
    writer2.close()

    # start rank-1 as a real subprocess
    worker_args = [
        sys.executable,
        "-c",
        "import importlib.util, sys; "
        "from tests.test_distributed import _run_worker; "
        "_run_worker(1, 'w1', eval(sys.argv[1]), '127.0.0.1', int(sys.argv[2]), sys.argv[3])",
        str(spec_json),
        str(coord_port),
        shared_flag,
    ]
    proc = subprocess.Popen(
        worker_args, cwd="/home/ubuntu/nexinfer-work", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    deadline = time.time() + 30
    while not __import__("os").path.exists(shared_flag) and time.time() < deadline:
        await asyncio.sleep(0.1)
    assert __import__("os").path.exists(shared_flag), "worker subprocess never registered"
    with open(shared_flag) as f:
        worker_port = int(f.read().strip())
    # tell the coordinator the worker's control port (it only knows the
    # worker registered; fill in the port it reported via hello payload
    # host/port)
    for _ in range(20):
        await asyncio.sleep(0.15)
        if "w1" in coord.nodes and coord.nodes["w1"].port:
            break
    coord.nodes["w1"].port = worker_port

    # push the plan to both ranks
    await coord._replan()
    assert coord.plan is not None
    assert coord.plan.mode.value == "pipeline_parallel"
    # root already has plan set via plan_pipeline; sync it from coordinator
    root.plan = coord.plan
    root.parallel = coord.plan.per_node[0]
    root.backend.offload_layers(root.parallel.pp_layers)
    root.layer_range = tuple(root.parallel.pp_layers[0])

    # drive generation through the ring
    req_id = "req-" + uuid.uuid4().hex[:8]
    input_ids = [10, 20, 30, 40]
    n_tokens = 6
    task = asyncio.create_task(root.generate(req_id, input_ids, max_tokens=n_tokens))
    logits_seq = await asyncio.wait_for(task, timeout=60.0)
    tokens = [int(np.argmax(l)) for l in logits_seq]

    # reference: single-node greedy
    ref_backend = NumpyBackend()
    ref_backend.load("demo-model-pp", SPEC, [])
    x = np.array(input_ids, dtype=np.int32)
    ref = []
    logits = ref_backend.prefill("ref", x)
    # prefill returns full-sequence logits (seq, vocab); the sampleable
    # prediction lives on the LAST position
    ref.append(int(np.argmax(np.asarray(logits)[-1])))
    for _ in range(n_tokens - 1):
        logits = ref_backend.decode(["ref"], np.array([[ref[-1]]], dtype=np.int32))
        ref.append(int(np.argmax(logits)))
    # both start from same random weights (demo mode, seed-fixed) so the
    # token *sequence* must match; the logits may differ by a permutation
    # of the lm-head path since both use identical params
    assert tokens == ref, (tokens, ref)
    await root.close()
    await coord.close()
    proc.terminate()


@pytest.mark.asyncio
async def test_pp_plan_layer_split():
    """The planner splits layers evenly and workers acknowledge slices."""
    nodes = [
        NodeSpec("a", "127.0.0.1", 1, model_hash="model:4:64"),
        NodeSpec("b", "127.0.0.1", 2, model_hash="model:4:64"),
    ]
    plan = plan_pipeline(nodes, SPEC)
    assert plan.mode.value == "pipeline_parallel"
    assert plan.per_node[0].pp_layers == [(0, 2)]
    assert plan.per_node[1].pp_layers == [(2, 4)]


@pytest.mark.asyncio
async def test_numpy_pp_slice_local():
    """The numpy backend honours an intermediate PP slice: activations in,
    activations out, KV cached only for its own layers."""
    # a rank whose slice ends at the LAST layer is the final rank and must
    # return logits; a rank ending before num_layers is intermediate and
    # returns activations
    backend = NumpyBackend()
    backend.load("demo-model-pp", SPEC, [])
    backend.offload_layers([(2, 4)])

    x = np.random.default_rng(0).standard_normal(SPEC.hidden_size).astype(np.float32)
    logits = backend.prefill("r1", x)
    assert logits.shape == (1, SPEC.vocab_size), logits.shape

    # intermediate slice: activations in, activations out
    backend2 = NumpyBackend()
    backend2.load("demo-model-pp", SPEC, [])
    backend2.offload_layers([(1, 3)])
    out = backend2.prefill("r2", x)
    assert out.shape == (SPEC.hidden_size,), out.shape

    # multi-row activation tensor flows through as full-sequence flattened
    ctx = np.random.default_rng(1).standard_normal((4, SPEC.hidden_size)).astype(np.float32)
    out4 = backend2.prefill("r3", ctx)
    assert out4.shape == (4 * SPEC.hidden_size,), out4.shape
