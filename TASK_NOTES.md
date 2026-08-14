# NexusInfer Task Notes (internal)

## Task
Build "NexusInfer": cross-platform (Win/Linux), hardware-agnostic, distributed inference engine.
User's plan APPROVED as-is (plan at /home/ubuntu/plan.md). Deliver: CLI+server, core engine
(paged KV cache, continuous batching), backend registry (cpu_numpy, ggml, ort, cuda, rocm,
directml, tpu, special_module), resource orchestrator, distributed runtime (coordinator/worker,
TP/PP, tcp/grpc/webrtc/rdma transports), internet gateway, MCP gateway (client+server),
skills bundles, git-like memory fabric, tests, docs, e2e demo.

## Repo location
/home/ubuntu/nexinfer — pyproject.toml (entry point: nexinfer.cli.main:main), package nexinfer/
subdirs: cli, engine, backends, distributed, transports, gateway, memory, services; tests/,
docs/, examples/, demo/.

## Status (phases 1-6 DONE)
- Phase 2: engine types, sampling, kvcache, scheduler, tokenizer_helper (HF + MinimalBPE),
  profiler, orchestrator, generation, runtime.
- Phase 3: backends base.py, registry.py, cpu_numpy, ggml_backend, ort_backend, directml,
  cuda, rocm, tpu, special_module.
- Phase 4: distributed messages, planner, worker, coordinator (mDNS via zeroconf),
  engine.py, signaling.py (SDP), transports (base, tcp, webrtc, grpc, rdma, registry).
- Phase 5: services/internet_gateway, services/skills, gateway/mcp_client, mcp_server,
  tool_registry, cli/config.
- Phase 6: memory/store.py (git-subprocess based, branches, merge policies, search,
  export/import snapshots), memory/fabric.py (policies private/shared/grant, whiteboard,
  push/pull_snapshot).
- Phase 7 (current): cli/main.py, cli/http_server.py, tests/test_engine.py (comprehensive),
  demo/e2e_demo.py all written. REMAINING: docs/ (README.md, architecture.md,
  backend-authoring.md, cluster-guide.md, memory-protocol.md), run tests, run demo, verify, deliver.
- Phase 8: integration verification, possible GitHub repo creation (gh repo create nexinfer --private),
  final delivery message.

## Phase 7/8 current debugging state (tests) — UPDATE 3
FIXED so far: kvcache allocate-on-write + interleaved (2*block_size) blocks, ensure_sequence sentinel init, MemoryStore .keep sentinel + _commit_all, Tokenizer MinimalBPE encode fallback, cpu_numpy _layer n_new/causal-mask fix, decode (1,256) working, merge policy overlay via _overlay_branch, search multi-word TF scoring, sampling top_k clipping (take last-position logits in generation.py), runtime bootstrap demo-weights fallback + MinimalBPE tokenizer for cpu_numpy, TCPTransport HELLO handshake channel keys + listen_addr property.
REMAINING fixes needed:
- tests/test_engine.py test_tcp_transport_roundtrip: update to new API — t1.start_server then t2.connect(f"127.0.0.1:{port}", key=f"127.0.0.1:{port}") using t1.listen_addr as key; send/recv by that key; also test send() auto-connect uses peer as key (send works as-is if connect(peer, key=peer)).
- tests/test_engine.py test_generation_e2e/stop_sequence: prompt encode now works (MinimalBPE fallback), vocab 256 vs top_k=50 — check GenerationRequest defaults (top_k=50 > vocab 256 fine; earlier index -50 was logits shape (3,256) 2D — FIXED by last-position slice). e2e test: GenerationRequest prompt "hello world" may encode to 0 tokens with word regex? MinimalBPE encode "hello world" -> [id, id]. max_tokens=10 finish length. Should pass now.
- test_engine_bootstrap: model="demo" dir, numpy backend -> FileNotFoundError caught -> "nexinfer-demo-weights" random; tokenizer fallback OK (backend_name=="cpu_numpy"). Should pass now.
- e2e_demo.py + demo/ dir: consider creating demo weights .npz so bootstrap works without fallback; optional.
- After tests pass: run demo/e2e_demo.py, run CLI (nexinfer --help, nexinfer profile, nexinfer backends, nexinfer memory commands), then create private GitHub repo TANMOY-SARKAR-cmd/nexinfer? (user has repos TANMOY-SARKAR-cmd/PassMen, status-page, weather_web_app — naming maybe 'nexinfer' or similar), deliver.

## Phase 7/8 current debugging state (tests) — UPDATE 2
Remaining 5 failures (test ids + fixes):
- test_numpy_backend_prefill_decode: "ValueError: cannot reshape array of size 64 into shape (4,4,16)" — in decode loop past_k shape. SPEC: check num_layers=4, num_kv_heads=4, head_dim=16? past_k stored per layer as (seq, kv_heads, dim); new_k reshape (1, kv_heads, dim) should work. size 64 = 4*4*4? k from a @ wk has shape (seq, kv_heads*head_dim=64). reshape(1, 4, 16) expects 64 items but a@wk for seq=1 gives (1,64)=64 items -> reshape(1,4,16)=64 OK... error says size 64 into (4,4,16)=256. So spec has kv_heads=4? seq 4? Actually prefill stores k (4,4,16) 256; decode seq_len - past_k.shape[0] = total_len-past? total_len = k_all[i].shape[0]+1 = 4+1=5; new_k reshape(1,4,16)=64 OK. Hmm reshape target (4,4,16) means spec.num_kv_heads or head_dim mismatch: maybe num_kv_heads=16? Then past_k (4,16,16)=1024? Whatever — likely my reshape uses wrong params: use spec.num_kv_heads for NEW tokens: (1, num_kv_heads, head_dim). Check actual SPEC in test file. Maybe spec.num_kv_heads differs from stored shape? No, _layer reshapes past with same spec.
- test_generation_e2e: tokenizer_helper encode uses self._hf.encode(text).ids (hf tokenizer API) but MinimalBPE fallback has its own encode — 'list' object has no attribute 'ids' means Tokenizer wrapper calls .ids on the fallback which returns list. FIX: in encode(), if isinstance(self._hf, MinimalBPE): return self._hf.encode(text). See tokenizer_helper.py line 72.
- test_engine_bootstrap: bootstrap("demo",...) — runtime passes model name to backend.load; "demo" is a dir w/o npz. FIX runtime: if backend_name=="cpu_numpy" use "demo-model-xyz" id or fallback when FileNotFoundError. Simplest: in runtime._fallback, pass "demo-model-xyz" as model id for numpy backend.
- test_store_branches_and_merge: after merge theirs into main, checkout main and get("k1") expects agent-a-value — merge theirs didn't apply. Check store.merge implementation: maybe merge target/source args reversed or git merge theirs strategy. FIX merge impl.
- test_store_search: returns [] — check search impl (keyword scoring over JSON values; maybe searches only main branch? note1 set on main branch by tmp_store.set -> should match "fox lazy". Check search logic).
- test_tcp_transport_roundtrip: fixed (frame dtype string); rerun shows PASSED.

## Phase 7/8 current debugging state (tests) — older
Failing tests and fixes-in-progress:
- MemoryStore init: git won't commit in an empty tree with --allow-empty in some git versions? Actually error is "git commit -q -m init memory store failed" with empty stderr — need `_git(... "commit", "--allow-empty-tree", ...)` for git 2.35+ OR ensure data/.keep exists. FIX: write data/main/.keep file before add/commit (most reliable).
- test_placement_gpu_hybrid: DeviceKind.GPU -> use DeviceKind.GPU_NVIDIA. automatic() planner uses d.kind.value strings like "/cpu", "/gpu:nvidia".
- cpu_numpy decode: past_k is (seq,h,dim) 3D but new k is 2D (seq*hidden) — need reshape new k/v in decode loop like prefill does: k.reshape(1, num_kv_heads, head_dim) before concat.
- test_tcp_transport_roundtrip: frame fixed (dtype string); recv timeout remains — likely because t1._queues for peer doesn't exist until send's connect? t1.send creates writer but _on_peer callback queues are only created in the inbound handler; outbound send must also register its own queue (check tcp_transport send() logic; recv queue created in connect() but inbound frames from THIS direction go to the READER loop created only for INBOUND connections. Outbound read loop exists per code above. Hmm read loop spawns in connect. Maybe q.get key mismatch: _queues keyed by peer string; t2 connects to t1, but t1 never receives because t1's send() writes to t2's inbound queue — verify send() writes to writer and the READER on the far side. If t2.recv times out, far side reader may have died. Check tcp_transport.py _handler + connect read loops.)
- e2e generation tests: "demo" fails because demo/ dir exists (isdir) but has no .npz. FIX tests: use model id "demo-model-xyz" (non-file, non-dir) which triggers random-weight demo mode, or pass SPEC and let numpy backend generate random weights.
- kvcache interleaved storage: write uses arr[off] for k and arr[block_size+off] for v; tests' shape assertions need (num_kv_heads, head_dim) k/v inputs — tests pass k,v shape (2,8) but read returns (7,2,8). read shape OK; assertion mismatch was k shape (2,8) vs (8,). Test passes k (8,) single-head? Actually num_kv_heads=2, head_dim=8; write expects arr[off]=k which copies (8,) into (2,8) row -> broadcast. read returns (7,2,8). Test compares rk[0] (2,8) with k (8,). FIX test: k/v should be shape (num_kv_heads, head_dim)=(2,8). Same for test_kvcache_blocks_stored_once_per_layer.
- scheduler test: fixed block size 4, max_tokens 16 -> 4 blocks each, device+host=132 blocks, fine.
- git: ensure data/main/.keep to allow first commit of empty dir.

## Key facts from research (for docs)
- llama.cpp wins on hardware breadth / CPU+GPU hybrid inference (bizon-tech.com blog, May 2026).
- vLLM parallelism: TP (Megatron-LM style), PP, DP; docs.vllm.ai serving/parallelism_scaling.
- ONNX Runtime EPs: OpenVINO (Intel CPU/iGPU/NPU), CUDA, ROCm, DirectML (Windows EP components,
  support.microsoft.com 2026/01).
- MCP: open standard (Anthropic 2024-11), servers/clients/hosts roles (Databricks blog).
- Git-based agent memory: GCC (arxiv 2508.00031), Letta context repositories, Reddit "2 years
  building agent memory ended up using Git".
- Multi-agent memory: blackboard/whiteboard pattern (greennode.ai, mem0 blog 2026-07).

## Known design choices
- Device ids: /cpu:0, /gpu:nvidia:i, /gpu:amd:i, /gpu:intel:i, /tpu:i, /npu:intel:i, /module:custom:i.
- Placement strategies: cpu_only | gpu_only | hybrid_split. HEADROOM 0.85.
- Distributed constraint: same model (model_hash check) across nodes.
- Memory root: ~/.nexinfer/memory/<store>/.git; branches under data/<branch>/.json files.
- Skills dir: ~/.nexinfer/skills/*.yml; defaults: default, research, coding, memory.
- MCP config: ~/.nexinfer/config.json mcp_servers.
- Signaling server: python -m nexinfer.distributed.signaling --port 8900.
- HTTP server: OpenAI-compatible /v1/chat/completions, /v1/models, /v1/completions, /health.
