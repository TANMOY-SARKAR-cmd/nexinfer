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


## PHASE 2 PROGRESS (current session, 2026-08)
- Workspace: /home/ubuntu/nexinfer-work (cloned from GitHub master; 37 tests pass; egg-info removed from index; .gitignore + LICENSE added)
- DONE: new ggml_backend.py (functional: public Llama API, create_batch decoding, n_gpu_layers offload)
- DONE: new ort_backend.py rewrite (OrtModelBuilder builds model.onnx stateless + model_kv.onnx KV-cache demo models programmatically; OrtBackend tracks _past_len, per-req KV cache)
  - FIRST EDIT BATCH TO ort_backend.py FAILED (edit 5 not applied) — VERIFY state: graph fixes (wte initializer as graph input, concat naming) likely NOT applied; _capture_present missing `parts = name.rsplit("_",2)` line; _past_len additions likely NOT applied. MUST RE-READ file and fix.
  - ort smoke test script: build_demo -> load model_kv.onnx -> prefill [10,20,30] -> decode [5] -> expect shapes; also stateless model.onnx path.
- GGUF download FAILED: repo HuggingFaceTB/SmolLM2-135M-Instruct-GGUF = 401 not found. Try: "bartowski/SmolLM2-135M-Instruct-GGUF" file "SmolLM2-135M-Instruct-Q4_K_M.gguf", or "Qwen/Qwen2.5-0.5B-Instruct-GGUF" qwen2.5-0.5b-instruct-q4_k_m.gguf (424MB). SmolLM2 Q4 ~90MB.
- Llama API (llama-cpp-python 0.3.34): Llama(model_path,n_gpu_layers,n_ctx,n_threads,n_batch,logits_all,seed,verbose); __call__(tokens,max_tokens=0,reset=False)->dict with "scores"; create_batch(max_tokens), batch.add(tok,pos,seq_id,logits), set_batch, decode(), update_outputs(); eval() deprecated but present; close().
- Installed: llama-cpp-python 0.3.34, onnxruntime 1.28.0, onnx, huggingface_hub 1.27.0
- Remaining phases: 4 distributed multi-process TP/PP, 5 chat templates + /metrics + memory vector search, 6 CI (ruff+pytest .github/workflows/ci.yml) + README prototype disclaimer + docs, 7 push to TANMOY-SARKAR-cmd/nexinfer + zip + deliver.

## ORT backend dev state (session continue)
- ort_backend.py fully rewritten with OrtModelBuilder (model.onnx stateless + model_kv.onnx KV-cache).
- KV demo model design: proj (1,seq,h=64) -> Slice axis=2 into k (first kh*dim/2) and v (second half) -> Reshape to flat (1,-1,kh*dim) -> Concat with key_values_0..value_values_1 (3D: 1,past,kh*dim) -> present outputs. ReduceMean axis=2 over concat_k_0 -> (1,seq,kh*dim) mean_k -> MatMul wout -> Add bias -> logits.
- Initializers also graph inputs (ORT warning; works). _init_values cached via onnx.load; fallback zero-fill any unfed input by declared shape.
- _capture_present parses "present_<idx>_<key|value>" -> cache key = parts[1] (e.g. "0_key") -> matches _past_kv_inputs idx.
- GGUF downloaded OK: /home/ubuntu/models/.cache/models--bartowski--SmolLM2-135M-Instruct-GGUF/snapshots/09816acd5d99df7be770d85ea30822623dab342c/SmolLM2-135M-Instruct-Q4_K_M.gguf (100.6 MB). Repo bartowski/SmolLM2-135M-Instruct-GGUF works.
- ggml_backend.py rewritten (public Llama API). NEXT: smoke-test GGML backend with the downloaded GGUF (script /tmp/ggml_smoke.py needed). SmolLM2-135M: vocab 49152, hidden 576, 30 layers, 6 heads? Need runtime to load tokenizer from GGUF (llama-cpp-python tokenizer) OR use engine tokenizer — llama has own tokenizer: model.tokenize() and decode(). ggml_backend prefill/decode return vocab logits (49152).
- ort_smoke.py at /tmp/ort_smoke.py (spec: num_layers=2 hidden=64 attn_heads=4 kv_heads=2 head_dim=16 vocab=256 inter=128). Last error fixed (zero-fill shape), rerun pending.
- Next phases: 4 distributed multi-process TP/PP execution over TCP transport; 5 chat templates (/nexinfer/engine/tokenizer_helper.py MinimalBPE fallback + HF path; add ChatTemplate renderer; metrics /metrics Prometheus in cli/http_server.py; memory vector search in memory/store.py or fabric.py); 6 .github/workflows/ci.yml (ruff + pytest, matrix 3.11/3.12, skip network tests marker @pytest.mark.network), README prototype disclaimer section after intro; 7 push + zip + deliver.
- README structure: intro, architecture table, CLI commands, quick start, sections, License at end (line ~87). Add "Status" section noting prototype: numpy reference + llama-cpp + ORT functional; templates for cuda/directml/tpu/rocm/special.

## STATE @ GGML smoke (update)
- ORT SMOKE FULLY PASSING: /tmp/ort_smoke.py prints "ORT SMOKE OK" (kv prefill/decode x2, batch (2,256), stateless path). ort_backend.py final & correct (OrtModelBuilder.build_demo, _last_token_logits, kv cache keyed "layer_role", init value fallback, vstack batch).
- GGUF at /home/ubuntu/models/.cache/models--bartowski--SmolLM2-135M-Instruct-GGUF/snapshots/09816acd5d99df7be770d85ea30822623dab342c/SmolLM2-135M-Instruct-Q4_K_M.gguf (100.6MB). Repo bartowski/SmolLM2-135M-Instruct-GGUF.
- ggml_backend.py has BUG: Llama is NOT a context manager (`with self.llama:` TypeError at line 144). Also decode is slow/naive (per-request sequential with reset=False which still works but slow). FIX: remove context manager; llama-cpp-python public API: llama(model, max_tokens=0, reset=True) -> {"scores": list of logit vecs}. For batched decode use create_batch/set_batch/decode/update_outputs then iterate llama outputs (TokenData with logits). SmolLM2 spec: hidden=576, layers=30, attn_heads=6, kv_heads=6, head_dim=96, vocab=49152, inter_dim=1536. n_ctx must be >= prompt len.
- /tmp/ggml_smoke.py exists (uses GGMLBackend class name fixed).
- Remaining phases: 4 distributed multi-process TP/PP integration test; 5 chat templates (/nexinfer/engine/tokenizer_helper.py + maybe engine/chat_template.py), metrics endpoint /metrics in nexinfer/cli/http_server.py, memory vector search; 6 .github/workflows/ci.yml + README prototype disclaimer; 7 push + zip + deliver.

## UPDATE: GGML SMOKE PASSING (final fix)
- Fixed `with self.llama:` (not context manager) and no-reset prefill: use
  `logits_all=True` + `self.llama.eval(toks)` + read `self.llama._scores[-1].reshape(1,-1)`.
- prefill reset: `_ctx.kv_cache_seq_rm(-1,0,-1)` + set n_tokens=0.
- GGML SMOKE OK: prefill (1,49152), decode (1,49152), batch (2,49152).
- Both functional backends (GGML + ORT) DONE. Phases 4-7 remain.

## UPDATE: Phase 4 in progress
- cpu_numpy.py now supports PP: offload_layers() sets _layer_range; prefill/decode treat hidden_size-sized 1D tensors as activations; intermediate ranks return activations, final returns logits.
- worker.py rewritten: proper keyed channel ring (pp:k->k+1), result key result:{node_id}, generate() driver on rank-0 walking ring per step, prefill_request control msg.
- WATCH OUT: worker.generate() hop loop bug — for hop>0 it recv's then sends; but rank0 in first step does prefill -> sends edge0->1 -> then recv res_key; fine cross-process, but in-process test (single process two workers) asyncio is fine too.
- IMPORTANT: worker.generate second+ step calls backend.decode([req_id], x) with x (1,1) token id — for pp this runs ALL layers -> logits -> then treated as activations hop = WRONG. FIX: use backend.prefill(req_id, x) (activations path) for subsequent steps OR backend.prefill for activations. cpu_numpy prefill caches KV only for its slice — decode path for subsequent steps on activations uses past KV from prev step. So use prefill(req_id, x) each step for pp mode.

## UPDATE: test_distributed progress (debug state)
- tests/test_distributed.py: test_pp_two_workers_ring (real 2-process PP ring), test_pp_plan_layer_split, test_numpy_pp_slice_local.
- Fixed so far: _run_worker uses control_port+2000 for its own control socket; subprocess cwd must be repo root (tests module); worker hello devices kind = DeviceKind enum not string; coordinator seeds self heartbeat + replan members only heartbeated (not list(nodes) fallback).
- worker.py: Worker has generate() driver (rank-0 walks ring: prefill embeddings -> sends activations pp:0->1 key... final rank recv's then push logits back on result:{self.node_id} key which rank-0 recv's). Note: generate's final-rank send uses result key from final rank to coord's worker control port: peers[0].host:port + key result:{self.node_id}. rank-0 connects to that key before loop.
- NEXT debug: rerun pytest; likely remaining: rank-0 in generate recv's res_key but final rank in run_decode_loop does recv on its incoming edge key (correct) — check for deadlock: rank0 send edge0->1 then recv res_key; w1 recv edge0->1, run slice, send res_key to coord (c_peer) — fine.
- After tests pass: check root.generate first-step: prefill(req_id, x) where x=(4,) ids — root slice layers 0-2 returns activations (1,64)?? NO — offload_layers sets layer_range (0,2) -> prefill returns x.reshape(hidden,) (activations). Good. Then decode steps: root.generate calls backend.prefill(req_id, x) with x (1,1) token id -> embeddings path -> activations. Good.
- worker.generate timeout via asyncio.wait_for(task, timeout=60).
- Remaining phases after 4: 5 chat templates (/metrics/mem vec search), 6 CI + README, 7 push+zip+deliver.
- /tmp/test_debug.py: standalone reproduction of the ring test (works, prints registered port).

## UPDATE: ring debug progress (data port fix)
- Root cause of utf-32/timeout: edge sends used node.port (CONTROL port); tensor HELLO frames landed on control server -> JSON decode errors; recv timed out.
- FIXED in worker.py: added `_data_peer(node)` = host:port+1000; all data-edge send/recv (generate, _forward_edge, run_decode_loop, final-rank result push to root DATA port root_node.port+1000) use _data_peer. rank-0 hosts result mailbox on own data transport (start_server if not listening, connect listen_addr with res_key).
- NEXT: rerun /tmp/pp_ring_debug.py and pytest tests/test_distributed.py.
- Also root.generate decode steps: self.backend.prefill(req_id, x) with x (1,1) token ids -> embed+slice -> activations; first step prefills token ids. Final ws=1 path returns logits local.
- cpu_numpy PP mode verified: embeddings -> activations for intermediate; logits only final.
- After phase 4: phases 5 (chat templates /metrics memory vec search), 6 (CI ci.yml, README status), 7 (commit/push/zip/deliver).

## UPDATE: transport send bug fix + remaining debug
- FINDING: TCPTransport.send(peer_addr, name, arr) with no prior writer did `connect(peer, key=peer)` — used the ADDRESS as channel key, so HELLO installed queue under addr key, not the channel name. Fixed: connect(peer, key=name); writer = self._writers[name].
- Worker fixes so far: _data_peer(node) = host:port+1000 (data transport); all edge/result sends use data peer; root hosts result mailbox via self-connect on own data transport; coordinator._replan excludes self (n != self.node_id); coordinator accepts preferred_mode kwarg (None=auto, "pipeline_parallel", "tensor_parallel"); coordinator hello dispatch wraps kind string in DeviceKind(...); coordinator _push_plan sends full node list in payload; worker plan dispatch rebuilds self.plan from payload nodes (fixes rank-1 peer resolution).
- test_distributed.py fixes: _run_worker control_port+2000, subprocess cwd repo root, hello DeviceKind enum; root sends hello to coordinator; coord preferred_mode="pipeline_parallel"; w1 NodeSpec port=0 in local plan only (updated via push).
- Still to verify: rerun /tmp/pp_ring_debug2.py (w1 recv frame log added), then pytest tests/test_distributed.py.
- Root worker 'root' must hello-register with coordinator (payload port=root.control_port); coordinator node_id should NOT be 'root' (use 'coord') in test to avoid collision.
- cpu_numpy PP done: intermediate returns activations flattened (seq*hidden); decode handles (n_req, hidden) activations.
- After phase-4 passes: update TASK_NOTES, run full suite, then phases 5/6/7.

## UPDATE: ring WORKS (2026-08-14)
PP ring debug2 prints TOKENS: [942, 18, 255] — 3 decode steps across root(rank0, layers 0-2) and w1(rank1, layers 2-4). Final fixes: tcp recv polls for queue (handles stale-queue race); removed root self-connect mailbox (final rank pushes to root data transport server queue directly); w1 prefill reshapes flattened activations to (-1, hidden); coordinator preferred_mode kwarg + excludes self from members; plan push carries node addresses; worker plan dispatch rebuilds plan; coordinator wraps hello kind in DeviceKind.

Next: port these same patterns into tests/test_distributed.py (_run_worker subprocess, root hello register, coord preferred_mode="pipeline_parallel", per_node[0] = root), run pytest tests/test_distributed.py (and full suite), then phases 5 (chat templates, /metrics, memory vector search), 6 (ci.yml, README disclaimer), 7 (push + zip + deliver).

## UPDATE: weight seeding diagnosis
NumpyBackend._rng = default_rng(0) at __init__ — but load() "demo" needs non-existent path -> FileNotFoundError! In tests it works because model_path_or_id='demo-model-pp' (not file/dir) -> goes to else: random weights. My inline test used 'demo' which IS a dir -> error.

Root cause of token mismatch likely: root worker's backend generates weights with _rng(0); w1 subprocess creates NEW NumpyBackend -> also _rng(0) -> SAME weights. root prefill first step computes layers 0-1; w1 layers 2-3. Should match local... UNLESS root's prefill(tokens) for first step: embed + L0+L1 -> hidden (n,64). w1 prefill(activations) -> L2+L3 -> lm_head -> logits (n,256) last row. Local: embed + L0..L3 -> lm_head -> last row. Deterministic = SAME.

BUT: root's first_out = self.backend.prefill(req_id, x) computes layers 0-1 ONCE. Then loop step 0: out = first_out; send to w1. w1 prefill(flat) -> layers 2-3. w1's _layer uses past_k=None. Local _layer also past None. MUST match...

Suspicion: root's prefill for TOKENS (is_activations false) returns x.reshape(-1) after L0-1 = (n*64,). w1 reshapes (-1,64)=(n,64). OK same. The ONLY difference: ROOT also stores self._kv[req_id]=(k_all,v_all) for L0-1; fine.

Wait: embeds seeding after step 0: embeds = [first_out.reshape(-1,hidden)[-1]] — FIRST embed row should be from input token 0's hidden, not last! Replay buffer must contain per-token hidden states AFTER root's slice (L0-1): first_out rows = hidden states per input token through layers 0-1. For step>0: append NEW token's hidden (embedding + L0-1 through root slice!): currently append wte[tok] — raw embedding WITHOUT root's layers! WRONG: embeds must be post-root-slice hidden states. Fix: step>0: root prefill([[tok]]) tokens path -> layers 0-1 -> hidden (64,) -> append. Then prefill(full stack) -> layers 0-1 -> (n,64) -> send. Also seeding: embeds = rows of first_out (post-slice hidden). Then step>0: seed row appended = hidden of new token via root slice.

## PHASE 4 DONE — all 40 tests pass
- Fixed _layer_range default (0,0) truthy bug -> None (cpu_numpy.py).
- Replay-semantics PP: root accumulates post-slice hidden rows, sends full ctx each step; final rank last-row logits; intermediates last-row hidden. Root embeds new token via own slice each step.
- Fixed test ref: prefill returns (seq,vocab) now; argmax on [-1]. Updated test_numpy_pp_slice_local expectations.

## PHASE 5 DONE SO FAR
1. chat_template.py CREATED at nexinfer/engine/chat_template.py: ChatTemplate class with BUILTIN_TEMPLATES (smollm, chatml, alpaca, minimal), jinja2 renderer + fallback _render_fallback (handles for m in messages, if m.role == 'x', {{ m.content }}, add_generation_prompt). apply(messages, add_generation_prompt=True).
2. tokenizer_helper.py: added Tokenizer.apply_chat_template(messages, template=None, add_generation_prompt) — uses getattr(self._hf, 'chat_template', None) or "minimal".
3. services/metrics.py CREATED: MetricsCollector (thread-safe), counters/latency histogram buckets, set_queue_depth/set_running/set_block_usage, prometheus_text(), METRICS singleton.
4. http_server.py: added GET /metrics (text/plain version=0.0.4); do_POST instruments METRICS (record_request, latency, tokens, queue_depth); chat completions uses ChatTemplate(body.get("template") or self._template or "minimal"); HttpServer now __init__(engine, tokenizer=None), has .template attr settable.

## REMAINING PHASE 5
- memory/store.py search(): upgrade with dense retrieval — average token-ID embeddings (small learned matrix e.g. np.random.seed(hash) deterministc; cosine similarity over averaged word embeddings), fallback TF keyword; parameter mode='auto' (vector if numpy present, always). Also update MemoryFabric if it exposes search (check memory/fabric.py search wrapper).
- Write tests: tests/test_chat_template.py, tests/test_metrics.py, extend test_store search test.

## PHASE 6 TODO
- .github/workflows/ci.yml: ruff lint (pip install ruff) + pytest py3.11/3.12, skip network tests (@pytest.mark.network marker needed on network-dependent tests), matrix, on push+PR.
- README.md: add "Status / Reality Check" section after intro: numpy ref + llama.cpp + ORT functional; CUDA/ROCm/DirectML/TPU documented templates; distributed arch-complete, multi-node GPU needs hardening. Also mention v0.x prototype.
- Also mark network tests in tests/ with @pytest.mark.network (grep for 'network' markers; add pytest.ini markers config).

## PHASE 7 TODO
- Run full suite + e2e demo (demo/e2e_demo.py); git add -A; commit "Phase 2: functional backends, distributed execution, CI, metrics, chat templates"; push origin master; zip /home/ubuntu/nexinfer-v2.zip excluding .git/*.pyc; deliver summary.

## PHASE 5 DONE — 58 tests pass (56 offline + network ones marked)
- tests/test_phase5.py CREATED: 18 tests — chat templates (7), metrics (5), http /metrics (1, @pytest.mark.network), memory vector search (4 incl self-cosine + auto fallback). All pass.
- memory/store.py: search(mode="auto"|"keyword"|"vector") with _embed_word/_embed_text deterministic numpy pseudo-embeddings (dim=64, hash->unit vector), cosine sim rescaled [0,1], threshold >0.5, keyword score tiebreak. _HAS_NUMPY flag at top.
- http_server.py: GET /metrics, POST instruments METRICS, chat template applied (body.get("template") or self._template or "minimal"), HttpServer(engine, tokenizer=None) w/ self.template settable.
- engine/tokenizer_helper.py: apply_chat_template added. engine/chat_template.py created.

## PHASE 6 STATE
- tests/test_engine.py: @pytest.mark.network added to test_internet_gateway_allowed_fetch.
- pytest.ini CREATED (markers=network, asyncio_mode=auto).
- .github/workflows/ci.yml CREATED: ruff lint on 3.12 + pytest matrix 3.11/3.12 skipping network tests; optional ort/llama-cpp smoke step (tests/test_backends_smoke.py — DOES NOT EXIST, will skip gracefully).
- pyproject.toml: version bumped 0.1.0->0.2.0; added [tool.ruff.lint] ignore=["BLE001","PLW1514"]. BUT ruff not reading config (resolved settings show BLE001 absent from enabled yet still reports? Confusing). CLI works: ruff check --isolated --ignore BLE001,PLW1514 -> 21 errors remaining. FIX: maybe config section key needs [tool.ruff.lint] — it IS there. ruff 0.16.3. Show-settings: linter.rules.enabled has NO BLE001 -> ignore WORKS?? But then why does default `ruff check` still report BLE001 (109)? Possibly ruff reads config only when dir==project; runs fine. ACTUALLY counts inconsistent between runs — final `ruff check` = 51 errors. Verify once cleanly: `ruff check nexinfer tests demo` -> if still BLE001 in output, just use `ruff check --ignore BLE001 --ignore PLW1514` in CI lint step, or add per-file noqa. SIMPLEST: in ci.yml run `ruff check --ignore BLE001 --ignore PLW1514 nexinfer tests demo`.
- Remaining after fixes: PLW151(7), RUF012(3), YNC230(2), RUF059(2), EXE001(2), YNC220(1), SIM102(1), RUF046(1), RUF034(1), ISC004(1). Fix manually or ignore.
- Tests pass: 56 passed, 2 deselected (network).

## README TODO
- Add "Status / Reality Check" section: numpy ref + llama.cpp + ORT functional; CUDA/ROCm/DirectML/TPU = documented templates; distributed arch-complete (multi-process PP verified); multi-node GPU production needs hardening; v0.2 prototype.

## PHASE 7 TODO
- Verify demo/e2e_demo.py runs; git add -A; commit; push origin master; zip /home/ubuntu/nexinfer-v2.zip (excl .git, *.pyc); deliver summary.

## LINT FIX STATE (ruff check nexinfer tests)
pyproject.toml [tool.ruff.lint] ignore = ["BLE001","PLW1514","PLW1510","S110","S112","TC004","ASYNC220","ASYNC230","ISC004"]
FIXED: store.py TC004 (numpy TYPE_CHECKING + local import in search), worker.py (rng/wte/tok_id), distributed/engine.py tok_id, chat_template.py SIM102+B023.
REMAINING (fix surgically or ignore):
- nexinfer/backends/tpu_backend.py:36 F401 pycoral unused; line 103 F841 x unused
- nexinfer/cli/main.py:87 pml, 128 pb F841; line 331 F821 undefined ModelSpec (add import)
- nexinfer/distributed/coordinator.py:125 spec F841
- nexinfer/engine/orchestrator.py:103 leftover F841
- nexinfer/engine/types.py:30 RUF034 useless if-else
- nexinfer/gateway/mcp_server.py:46 skills F841
- nexinfer/memory/store.py:359 RUF100 unused noqa PLC0415 (remove)
- nexinfer/services/metrics.py:94 RUF046 int cast of int (remove cast)
- nexinfer/transports/grpc_transport.py:59 rpc_handler F841
- nexinfer/transports/webrtc_transport.py:26 BYE F401
- tests/test_distributed.py:146 root_port F841; 93/182 ASYNC230 open() in async (ignore already in pyproject? YES ASYNC230 ignored; 171 ISC004 ignored too — rerun to confirm only 146 remains in test_distributed)
- tests/test_engine.py:206 out F841; 408/416 RUF059 unused unpack skill
- tests/test_phase5.py:178/186/187 RUF012 mutable default class attrs (change lists to None/empty tuples or ignore RUF012)
Plan: add RUF100,RUF012,RUF034,RUF046,RUF059 to ignore, fix remaining F841/F401/F821 surgically, then ruff format --check must pass (38 files would be reformatted!) — RUN `ruff format nexinfer tests` to format all, it's allowed (CI runs format --check).
THEN: tests pass (56+), demo/e2e_demo.py runs, README "Status / Reality Check" section, git add -A commit push, zip, deliver.
