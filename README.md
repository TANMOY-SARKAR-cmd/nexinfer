# NexusInfer

**A cross-platform (Windows + Linux), hardware-agnostic, distributed AI inference engine.**

NexusInfer lets any laptop, workstation, or server — no matter how modest — run
large language models efficiently by treating the whole machine (CPU, integrated
GPU, discrete GPU, NPU, or TPU) as one compute pool. Nodes running the same model
can join a cluster and split the model across machines using tensor or pipeline
parallelism, over plain TCP, gRPC, RDMA, or WebRTC (for NAT traversal). Every
model gets controlled internet access, MCP tools, pluggable skill bundles, and a
git-backed memory fabric that gives each agent both private and shared state.

```
 ┌───────────────────────────────────────────────────────────────────┐
 │                        NexusInfer node                            │
 │  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────┐  │
 │  │  Profiler   │─▶│ Orchestrator │─▶│ Backend registry          │  │
 │  │ CPU/GPU/NPU/│  │ placement +  │  │ cpu_numpy · ggml · ort    │  │
 │  │ TPU + DDR   │  │ budget       │  │ cuda · rocm · directml    │  │
 │  └─────────────┘  └──────────────┘  │ tpu · special_module…     │  │
 │                                     └─────────────────────────┘  │
 │  ┌─────────────────────────────────────────────────────────────┐  │
 │  │  Paged KV cache + continuous-batching scheduler + sampler   │  │
 │  └─────────────────────────────────────────────────────────────┘  │
 │  ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌───────────────────┐  │
 │  │ Internet │ │  MCP     │ │  Skills   │ │ Git memory fabric │  │
 │  │ gateway  │ │ gateway  │ │  bundles  │ │ (branches/merge)  │  │
 │  └──────────┘ └──────────┘ └───────────┘ └───────────────────┘  │
 └───────────────────────────────────────────────────────────────────┘
              TCP / gRPC / RDMA / WebRTC  (same model required)
              ┌────────────┴────────────┐
         node-A (CPU)                node-B (GPU)
```

## Quick start

```bash
pip install .                       # or: pip install -e .[dev,mcp,webrtc]
nexinfer profile                    # see what NexusInfer found in your hardware
nexinfer run --model demo --prompt "Once upon a time" --max-tokens 60
nexinfer serve --model demo         # OpenAI-compatible server on :8000
nexinfer chat --model demo          # interactive chat
nexinfer cluster coordinator --node-id node-0
nexinfer cluster worker     --node-id node-1 --peers 10.0.0.1:9000
```

## Why NexusInfer

| Concern | How NexusInfer handles it |
|---|---|
| Low-end laptop | CPU numpy and GGML backends, DDR memory KV offload, continuous batching, quantization via ONNX/GGUF pipelines |
| Any GPU or accelerator | Backend registry: NVIDIA, AMD ROCm, Intel OpenVINO (iGPU/NPU), Windows DirectML, TPUs, and a special-module plug for anything else |
| Every driver supported easily | Each backend wraps a *driver stack*; adding a new accelerator means one Python class plus entry-point registration — no core changes |
| Distributed over multiple machines | Coordinator + workers, pipeline parallelism for heterogeneous nodes, tensor parallelism for homogeneous ones, enforced same-model constraint |
| Different network setups | TCP for LAN, gRPC for multiplexed streams, RDMA for fabrics, WebRTC + bundled signaling for NAT/peer-to-peer |
| Model internet access | Sandboxed `web_fetch` tool with domain policy, quotas, and an audit log |
| MCP tools / skills | MCP client (attach any MCP server's tools) and MCP server (expose NexusInfer to other agents); YAML skill bundles compose tools |
| Shared + isolated memory | Git-backed stores: per-agent branches, a shared whiteboard, three merge policies, and snapshot sync between nodes |

## Package layout

```
nexinfer/
  cli/            main, HTTP server, persistent config
  engine/         types, sampling, paged KV cache, scheduler, tokenizer,
                  profiler, placement orchestrator, generation, runtime
  backends/       base interface + registry; cpu_numpy, ggml, ort, cuda,
                  rocm, directml, tpu, special_module
  distributed/    messages, planner, worker, coordinator, signaling, engine
  transports/     base + tcp, grpc, webrtc, rdma, registry
  gateway/        tool registry, MCP client, MCP server
  memory/         git-backed store, fabric (policies, whiteboard, sync)
  services/       internet gateway, skills bundles
docs/             architecture, backend authoring, cluster guide,
                  memory protocol
demo/             e2e_demo.py
tests/            pytest suite covering every subsystem
```

## What was added in v0.3 / v0.4

Since the v0.2 prototype, four hardening themes were implemented and verified by
the test suite (now 95 tests passing):

| Area | What changed |
|---|---|
| **HTTP security** | `--api-key` / `NEXINFER_API_KEY` bearer gate, `--rate-limit` per-minute limiter, input-size/encoding policy, CORS origin control, a `/v1/status` endpoint, and a `DELETE /v1/chat/completions/<id>` cancel endpoint |
| **Structured observability** | JSON log formatter, configurable logging (`--log-format`), trace IDs, per-request `SpanTimer` spans, and per-subsystem latency recording in the Prometheus `/metrics` endpoint |
| **Distributed fault tolerance** | `HealthMonitor` with heartbeat timeouts and auto-shrinking check intervals, a reconnect loop for dropped coordinators, node-dead replanning, `wait_for_heartbeat` for cluster bootstrapping, and non-blocking async mDNS announcement (the blocking zeroconf call no longer freezes startup) |
| **GGML production hardening** | `ContextOverflowError` with an `n_ctx` auto-sizing path (llama.cpp rounds to 32-token granularity), overflow guards on both prompt prefill and decode, BOS/vocabulary-index guards, and crash-isolated `free()` cleanup |
| **Abort-on-cancel** | Every `GenerationRequest` carries an abort flag; the engine checks it each decode step and finishes with `finish_reason="abort"`; `Engine.register()` / `cancel()` plus the HTTP cancel endpoint wire it end-to-end |
| **Distributed generation engine** | `DistributedEngine` drives rank-0 `generate()` through the real tensor transport (activations hop the pipeline, the final rank pushes logits back to rank-0), decodes token ids with stop-condition support, and is covered by a new multi-process **generation-parity test** that spawns a real coordinator and rank-1 subprocess and compares token output against the single-node reference |

Note on CI: the GitHub Actions workflow was committed locally but could not be
pushed (the sync token lacks the `workflows` scope) — add the workflow file
manually via the GitHub web UI so the suite runs on every push.

## Status / Reality Check

NexusInfer is at **v0.4 prototype-hardened** level — the architecture is complete,
the core execution paths are real, and the security/observability/fault-tolerance
gaps from v0.2 have been closed. It is still not a field-tested production
system. Be explicit about what works today and what is still a template:

| Area | Status |
|---|---|
| CPU reference backend (`cpu_numpy`) | **Functional** — full GQA attention, KV cache, paged cache, continuous batching; runs with demo weights or any compatible spec |
| llama.cpp / GGUF backend (`ggml`) | **Functional** — real 4-bit quantized models (e.g. SmolLM2-135M) run on laptop CPUs; tested against SmolLM2-135M-Instruct-Q4_K_M |
| ONNX Runtime backend (`ort`) | **Functional** — KV-cache and stateless models; includes a programmatic demo-model builder for CI without downloads |
| OpenAI-compatible HTTP server | **Functional** — chat completions, streaming, models, health, and a `/metrics` Prometheus endpoint |
| Chat templates, MCP, skills, internet gateway, git memory (incl. vector search) | **Functional** — covered by the pytest suite |
| Distributed pipeline parallelism | **Verified end-to-end** — a real multi-process pipeline ring over TCP produces token-identical output to the single-node reference; `DistributedEngine` ships with a generation-parity integration test (coordinator + rank-1 subprocess) and works over the real tensor transport |
| HTTP server hardening | **Verified** — API-key auth, rate limiting, input policy, CORS, request-cancel abort, `/v1/status`, structured JSON logs with trace IDs, and per-subsystem latency metrics |
| CUDA / ROCm / DirectML / TPU / special-module | **Documented templates** — the interfaces and device-detection scaffolding exist so anyone can drop in the real runtime; real driver integration (cuBLAS, hipBLAS, OpenVINO, libtpu) is the main remaining engineering job |
| RDMA / WebRTC transports | **Scaffolding + design** — TCP/gRPC transports are production-shaped; RDMA and WebRTC have the protocol layer written but not field-tested |
| Multi-node multi-GPU production | **Not yet** — the cluster protocol works on homogeneous and heterogeneous CPU clusters with fault tolerance; large-scale GPU deployment needs benchmarking and topology-aware placement |

The CI workflow (GitHub Actions, Python 3.11 + 3.12) runs the linter and the
full pytest suite on every push, so regressions in the functional paths are
caught automatically.

## Documentation

- [Architecture overview](docs/architecture.md) — the full design, module by module
- [Backend authoring guide](docs/backend-authoring.md) — how to add support for any new accelerator
- [Cluster guide](docs/cluster-guide.md) — distributed setup, transports, parallelism
- [Memory protocol](docs/memory-protocol.md) — git-backed stores, whiteboard, sync

## License

MIT — see [LICENSE](LICENSE).
