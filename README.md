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

## Documentation

- [Architecture overview](docs/architecture.md) — the full design, module by module
- [Backend authoring guide](docs/backend-authoring.md) — how to add support for any new accelerator
- [Cluster guide](docs/cluster-guide.md) — distributed setup, transports, parallelism
- [Memory protocol](docs/memory-protocol.md) — git-backed stores, whiteboard, sync

## License

MIT — see [LICENSE](LICENSE).
