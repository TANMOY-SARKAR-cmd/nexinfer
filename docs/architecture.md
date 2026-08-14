# NexusInfer Architecture

This document walks through the design of NexusInfer, explaining how each module
fits together and the reasoning behind key decisions.

## 1. Design goals

NexusInfer was built around five requirements that together shape every
module boundary.

1. **Run anywhere a Python process can.** A single codebase for Windows and
   Linux, with hardware discovery rather than configuration.
2. **Use the whole machine.** CPU + DDR, iGPU, dGPU, NPU, and TPU are treated
   as candidates in one placement decision, not as separate runtimes.
3. **Add hardware without touching the core.** Every accelerator is a backend
   plugin behind one stable interface (`Backend`) discovered through Python
   entry points.
4. **Distribute freely.** Any set of nodes running the *same model* can form a
   cluster; the planner chooses tensor parallelism for homogeneous nodes and
   pipeline parallelism for heterogeneous ones, and transports span LAN,
   fabric, and NAT-traversed links.
5. **Give models agency.** Internet access, MCP tools, skill bundles, and a
   git-backed memory fabric are first-class services consumed through the
   same tool-registry path the generation loop uses for function calling.

## 2. Request path

A generation request flows through five stages:

```
prompt → tokenizer → scheduler (admit/preempt) → KV cache write
     → backend.forward (only layers assigned to this node)
     → sampler (greedy / temperature / top-p / penalties / stopping)
     → tool-call resolution (if tools enabled) → next token
```

The scheduler (`engine/scheduler.py`) implements continuous batching: new
requests are admitted as long as their projected KV-cache blocks fit in the
device/host block budget; the oldest running request is preempted if not.
Stopping is checked per token against `max_tokens`, `stop_sequences`, and
tool-call generation.

The paged KV cache (`engine/kvcache.py`) mirrors vLLM's block design: fixed
`block_size` pages allocated per sequence, with automatic spill of cold
blocks from device memory to host RAM when the device budget is exhausted —
this is what lets large-context sessions run on a low-end laptop.

## 3. Hardware discovery and placement

`SystemProfile.from_system()` (engine/profiler.py) enumerates devices on both
platforms:

| Detected device | Detection method |
|---|---|
| CPU | `os.cpu_count`, platform memory counters |
| NVIDIA GPU | `nvidia-smi` / NVML (Linux and Windows) |
| AMD GPU | ROCm `rocm-smi`, fallback Windows DXGI |
| Intel GPU / NPU | Windows DXGI for iGPU; `oneapi-ls`/sysfs for NPU |
| TPU | `TPU_VISIBLE_DEVICES` / TPU environment variables |
| Custom modules | each `ReferenceModule` reports via its `detect_devices()` |

`plan_placement()` (engine/orchestrator.py) then produces a
`Placement` decision: **cpu_only**, **gpu_only**, or **hybrid_split**, plus a
per-device layer assignment and a KV-cache device/host block split that keeps
85% of device memory in reserve so the machine stays usable.

## 4. Backend registry

All backends implement the small interface in `backends/base.py`:

```python
class Backend:
    name: str
    capabilities() -> dict          # quantization, fp8, offload…
    detect_devices() -> list[DeviceInfo]
    load(model, spec, device_ids)   # load weights / offload layers
    prefill(req_id, input_ids) -> logits
    decode(req_ids, next_ids) -> logits
    offload_layers(ranges)          # used by pipeline parallelism
    close()
```

Shipped backends and the driver stack each wraps:

| Backend | Driver stack | Notes |
|---|---|---|
| `cpu_numpy` | NumPy | reference implementation, always available |
| `ggml` | llama.cpp / llama-cpp-python | GGUF, CPU+GPU hybrid inference |
| `ort` | ONNX Runtime | CPU, DirectML, CUDA, ROCm, OpenVINO (iGPU/NPU) providers |
| `directml` | DirectML provider | Windows GPU acceleration |
| `cuda` | PyTorch CUDA kernels | template for custom kernels |
| `rocm` | PyTorch ROCm | AMD discrete GPUs |
| `tpu` | libtpu / jax | TPUs (Google Cloud or USB Accelerator) |
| `special_module` | `ReferenceModule` plug | anything else; see the backend-authoring guide |

Registry discovery (`backends/registry.py`) uses `pkg_resources`/`importlib`
entry points so a pip-installed backend package registers itself automatically.
`load_backend()` gracefully returns `None` when optional dependencies are
missing, letting the engine fall back to the next best backend.

## 5. Distributed runtime

The cluster has a **coordinator** (rank 0) and any number of **workers**:

- The coordinator registers nodes (manual `--peers`, LAN mDNS announcement,
  or WebRTC signaling), keeps heartbeat liveness, and pushes a
  `ClusterPlan` produced by `distributed/planner.py` whenever membership
  changes.
- Workers run a contiguous layer range (PP) or an attention-head slice
  (TP). Pipeline-parallel activations flow over the data transport between
  adjacent ranks; the root rank returns logits to the sampler.
- The **same-model constraint**: every node advertises a `model_hash`
  (architecture + quantization fingerprint). `validate_same_model()` warns
  when hashes differ, because mixing incompatible weights produces silently
  wrong output.

Transport options (`transports/`):

| Transport | When to use |
|---|---|
| `tcp` | default, simplest; works everywhere |
| `grpc` | multiplexed streams with back-pressure; busy clusters |
| `webrtc` | nodes behind NAT / on different networks; requires the
  bundled SDP signaling server (`python -m nexinfer.distributed.signaling`) |
| `rdma` | RoCE/InfiniBand fabrics; zero-copy paths; degrades to TCP
  automatically when no fabric is present |

## 6. Services layer

**Internet gateway** (`services/internet_gateway.py`) exposes a `web_fetch`
tool with an allow/block domain policy, scheme restrictions, redirect and
size limits, per-agent quotas, and a full audit log. Fetches run in a
dedicated thread pool so a slow host never stalls generation.

**Skills** (`services/skills.py`) are YAML bundles (`name`, `tools`,
`skills_refs`, `system_hint`) stored under `~/.nexinfer/skills/`.
Resolution flattens references, so a `research` skill can compose `notes`.

**MCP gateway** (`gateway/`):

- `mcp_client.py` connects to any MCP server (stdio or SSE/HTTP) and maps its
  tools to registry names `mcp:<server>:<tool>`.
- `mcp_server.py` exposes NexusInfer itself as an MCP server (`generate`,
  `list_models`, `memory_read/write`, `whiteboard_*`), so other agents and
  editors can drive it.
- `tool_registry.py` is the single lookup both the generation loop and the
  HTTP/MCP layers consult; `ToolCaller` routes synchronous built-ins and
  async MCP calls uniformly.

## 7. Memory fabric

`memory/store.py` makes every memory store a real git repository under
`~/.nexinfer/memory/<name>/`, with key files as `data/<branch>/<key>.json`
and every write as a commit. This gives free, battle-tested properties:

- **isolation** — each agent works on its own branch; other agents simply
  cannot see it.
- **sharing** — the `whiteboard` store is a shared branch where agents post
  entries.
- **merge policies** — `ours`, `theirs`, or `last_write`, chosen per store.
- **history and undo** — `history()`, `diff()`, `revert()`, branching and
  merging are literal git operations.
- **network sync** — `push_snapshot()`/`pull_snapshot()` exchange a store's
  full state between nodes of the same cluster, so distributed agents share
  memory even without a shared filesystem.

`memory/fabric.py` enforces access policies (`private`, `shared`, `grant`)
per store and owns the whiteboard singleton.

## 8. HTTP server

`nexinfer serve` exposes an OpenAI-compatible surface: `GET /v1/models`,
`POST /v1/chat/completions`, `POST /v1/completions`, `GET /health`. The chat
endpoint accepts `tools` arrays that are resolved through the tool registry,
so function-calling clients can drive web_fetch, memory, and MCP tools
directly.

## 9. What was deliberately left extensible

- **Model weights**: the shipped numpy backend loads demo weights; real
  weights flow through the ggml/ort backends (GGUF/ONNX) or a `--weights`
  loader you add per format.
- **Embeddings for memory search**: `store.search()` is currently keyword
  based; plug a small embedding model in via the same backend registry to
  upgrade it to vector search.
- **More transports** (QUIC, Unix domain sockets) and **more merge
  policies** are one subclass away.
