"""Generation-parity integration test for ``DistributedEngine``.

Spins up a real coordinator and two real worker processes (a 4-layer
model split 2/2) and drives ``DistributedEngine`` (the client-facing
synchronous API) through the TCP data transport. The token ids produced
by the distributed run must match a reference local (single-process)
numpy-backend run token-for-token -- that is the parity guarantee:
activations must survive the wire.

The asyncio event loop is driven from a daemon thread so that in-process
servers (coordinator control/data sockets, root worker) keep accepting
connections while the main test thread performs synchronous work such as
spawning the worker subprocess and polling for its ready flag.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import threading
import time
import uuid
from http.server import HTTPServer

import numpy as np
import pytest

from nexinfer.backends.base import DeviceKind, ModelSpec
from nexinfer.backends.cpu_numpy import NumpyBackend
from nexinfer.distributed.coordinator import Coordinator
from nexinfer.distributed.engine import DistributedEngine
from nexinfer.distributed.messages import Msg
from nexinfer.distributed.planner import NodeSpec, plan_pipeline
from nexinfer.distributed.worker import Worker
from nexinfer.engine.tokenizer_helper import MinimalBPE, Tokenizer
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
SPEC_JSON = {
    "num_layers": 4,
    "hidden_size": 64,
    "num_attention_heads": 4,
    "num_kv_heads": 2,
    "head_dim": 16,
    "vocab_size": 256,
    "inter_dim": 128,
}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker_entry(rank: int, node_id: str, spec_json: dict, coordinator_port: int, ready_path: str) -> None:
    """Subprocess entry (mirrors ``test_distributed._run_worker``): load
    the backend, register with the coordinator, run the data loop."""
    import asyncio

    async def main() -> None:
        from nexinfer.backends.cpu_numpy import NumpyBackend
        from nexinfer.distributed.planner import NodeSpec, plan_pipeline
        from nexinfer.distributed.worker import Worker
        from nexinfer.transports.tcp_transport import TCPTransport

        spec = ModelSpec(**spec_json)
        backend = NumpyBackend()
        backend.load("demo-model-pp", spec, [])
        plan = plan_pipeline(
            [NodeSpec("root", "127.0.0.1", coordinator_port), NodeSpec(node_id, "127.0.0.1", 0)],
            spec,
        )
        worker = Worker(
            node_id=node_id,
            backend=backend,
            spec=spec,
            plan=plan,
            transport=TCPTransport(),
            rank=rank,
            control_host="127.0.0.1",
            control_port=_free_port(),
        )
        actual = await worker.start()
        reader, writer = await asyncio.open_connection("127.0.0.1", coordinator_port)
        hello = Msg(
            type="hello",
            payload={
                "node_id": node_id,
                "host": "127.0.0.1",
                "port": actual,
                "devices": [
                    {
                        "device_id": "/cpu:0",
                        # serialize the DeviceKind enum member; ``to_dict``
                        # converts it to its string value for JSON transport
                        "kind": DeviceKind.CPU,
                        "vendor": "generic",
                        "name": "CPU (numpy)",
                        "total_memory_bytes": 0,
                        "compute_score": 1.0,
                    }
                ],
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
        with open(ready_path, "w") as f:
            f.write(f"{actual}\n")
        await worker.run_decode_loop()

    try:
        asyncio.run(main())
    except RuntimeError:
        asyncio.get_event_loop().create_task(main())
    except Exception:
        import traceback

        with open(ready_path + ".err", "w") as f:
            f.write(traceback.format_exc() + "\n")
        raise


def _run_worker_subprocess(
    rank: int, node_id: str, coordinator_port: int, ready_path: str
) -> subprocess.Popen:
    log_file = open(ready_path + ".log", "w")  # noqa: SIM115 (fd handed to Popen; must outlive the child)
    try:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import sys; "
                "from tests.test_distributed_parity import _worker_entry; "
                "_worker_entry(int(sys.argv[1]), sys.argv[2], eval(sys.argv[3]), int(sys.argv[4]), sys.argv[5])",
                str(rank),
                node_id,
                str(SPEC_JSON),
                str(coordinator_port),
                ready_path,
            ],
            cwd="/home/ubuntu/nexinfer-work",
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    except Exception:
        log_file.close()
        raise
    return proc


@pytest.mark.network
def test_distributed_generation_parity() -> None:
    """Token ids from a 2-worker distributed run must match the local
    numpy reference run token-for-token (greedy decoding)."""
    coordinator_port = _free_port()
    ready_path = f"/tmp/nexinfer-parity-{uuid.uuid4().hex[:8]}.ready"
    for path in (ready_path, ready_path + ".err"):
        if os.path.exists(path):
            os.remove(path)

    loop = asyncio.new_event_loop()
    # Drive the event loop from a daemon thread so the in-process
    # coordinator/worker servers keep accepting connections while the
    # main thread does synchronous work (subprocess spawn, polling).
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()
    try:
        # --- coordinator + rank-0 worker in this process ---
        coord = Coordinator(
            "coord",
            SPEC,
            TCPTransport(),
            host="127.0.0.1",
            port=coordinator_port,
            preferred_mode="pipeline_parallel",
        )
        coord_port = asyncio.run_coroutine_threadsafe(coord.start(), loop).result()
        backend = NumpyBackend()
        backend.load("demo-model-pp", SPEC, [])
        plan = plan_pipeline(
            [NodeSpec("root", "127.0.0.1", coord_port), NodeSpec("w1", "127.0.0.1", 0)], SPEC
        )
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
        asyncio.run_coroutine_threadsafe(root.start(), loop).result()
        assert root.transport.listen_addr

        # register the in-process root with the coordinator
        async def _root_hello() -> bool:
            reader2, writer2 = await asyncio.open_connection("127.0.0.1", coord_port)
            hello2 = Msg(
                type="hello",
                payload={
                    "devices": [
                        {
                            "device_id": "/cpu:0",
                            "kind": DeviceKind.CPU.value,
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
            ok = json.loads(line2)["type"] == "welcome"
            writer2.close()
            return ok

        assert asyncio.run_coroutine_threadsafe(_root_hello(), loop).result()

        # --- rank-1 as a real subprocess ---
        proc = _run_worker_subprocess(1, "w1", coord_port, ready_path)
        try:
            deadline = time.time() + 30
            while not os.path.exists(ready_path) and time.time() < deadline:
                time.sleep(0.1)
            assert os.path.exists(ready_path), "worker subprocess never registered"
            with open(ready_path) as f:
                worker_port = int(f.read().strip())
            # reconcile the coordinator's node record with the real control port
            for _ in range(20):
                time.sleep(0.15)
                if "w1" in coord.nodes and coord.nodes["w1"].port:
                    break
            coord.nodes["w1"].port = worker_port
            asyncio.run_coroutine_threadsafe(coord._replan(), loop).result()
            assert coord.plan is not None
            root.plan = coord.plan
            root.parallel = coord.plan.per_node[0]
            root.backend.offload_layers(root.parallel.pp_layers)
            root.layer_range = tuple(root.parallel.pp_layers[0])

            # --- client-facing distributed engine run ---
            tokenizer = Tokenizer(MinimalBPE(vocab_size=SPEC.vocab_size))
            engine = DistributedEngine(rank0_worker=root, tokenizer=tokenizer)
            _harness_http_server(engine)
            tokens = list(engine.generate_stream(GenerationRequest(prompt="hello world test", max_tokens=8)))
            dist_ids = [t.token_id for t in tokens]

            # --- reference: single-node greedy ---
            ref_backend = NumpyBackend()
            ref_backend.load("demo-model-pp", SPEC, [])
            input_ids = list(tokenizer.encode("hello world test"))
            logits = ref_backend.prefill("ref", np.array(input_ids, dtype=np.int32))
            ref = [int(np.argmax(np.asarray(logits)[-1].reshape(-1)))]
            for _ in range(7):
                logits = ref_backend.decode(["ref"], np.array([[ref[-1]]], dtype=np.int32))
                ref.append(int(np.argmax(np.asarray(logits).reshape(-1))))
            assert dist_ids == ref, (dist_ids, ref)
        finally:
            asyncio.run_coroutine_threadsafe(root.close(), loop).result()
            asyncio.run_coroutine_threadsafe(coord.close(), loop).result()
            proc.terminate()
            proc.wait(timeout=10)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=5)
        for path in (ready_path, ready_path + ".log", ready_path + ".err"):
            if os.path.exists(path):
                os.remove(path)


def _harness_http_server(engine: DistributedEngine) -> None:
    """Cheap in-thread HTTP harness so the engine is reachable the same
    way a real HTTP server would wire it (class-level handler state)."""
    from socketserver import ThreadingMixIn

    from nexinfer.cli.http_server import _Handler

    _Handler.engine = engine
    _Handler.auth = None
    _Handler.limiter = None
    _Handler.policy = None

    class Threaded(ThreadingMixIn, HTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    server = Threaded(("127.0.0.1", 0), _Handler)
    # ``shutdown()`` blocks forever unless ``serve_forever()`` is actually
    # running on another thread, so run the server briefly on a daemon
    # thread to exercise the class-level handler wiring without leaking
    # threads (test teardown kills them).
    import threading

    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    server.shutdown()


from nexinfer.engine.types import GenerationRequest  # noqa: E402
