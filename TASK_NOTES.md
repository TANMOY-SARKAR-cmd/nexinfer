
### ISSUE: GGUF model file NOT FOUND in /home/ubuntu (context said /home/ubuntu/models/.cache/.../SmolLM2-135M-Instruct-Q4_K_M.gguf 100.6MB but glob finds nothing — may have been cleared or path different). test_ggml_hardening and test_backends_smoke will SKIP if absent. test_backends_smoke.py test_ggml_smoke also needs it. Need to re-download with huggingface_hub if tests should run for real. Previous session: "Downloaded SmolLM2-135M Q4_K_M GGUF model (100.6 MB)" — so it existed before; maybe in ~/models or /home/ubuntu/models. Check /home/ubuntu/ listing.
FIX OPTIONS: (a) huggingface_hub snapshot_download("bartowski/SmolLM2-135M-Instruct-GGUF", local_dir="/home/ubuntu/models") — takes ~1-2 min; (b) accept skip and note in summary. Prefer (a) for real validation.

### GGUF restore plan
Download script: python3 /tmp/download_model.py (huggingface_hub snapshot, saves to /home/ubuntu/models/bartowski/SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-Q4_K_M.gguf). The test GGUF path is ~/models/.cache/models--bartowski--SmolLM2-135M-Instruct-GGUF/snapshots/.../SmolLM2-135M-Instruct-Q4_K_M.gguf — hf_hub_download with local_dir layout differs; simplest: run download, then copy/link file to test path, or set GGUF_MODEL_PATH env. Better: after download, update test_ggml_hardening GGUF constant to use the downloaded path (os.path.expanduser("~/models/bartowski/...")). ALSO update test_backends_smoke.py GGUF path constant the same way if it has one. Then run pytest.

### GGUF path fix progress
Created tests/gguf_fixtures.py with gguf_path() (env -> ~/models/bartowski/SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-Q4_K_M.gguf -> legacy snapshot path). Updated test_ggml_hardening.py to use it (GGUF = gguf_path() or ""). Remaining: update test_backends_smoke.py similarly (replace os.environ.get block at lines 80-87 with from tests.gguf_fixtures import gguf_path; gguf = gguf_path() or ""; keep skip). Then python3 /tmp/download_model.py to fetch the model (~1.5 min, ~100MB). Then lint + pytest.
NOTE: ruff may flag absolute import "from tests.gguf_fixtures" in test files — pytest tests/ rootdir; tests package needs __init__.py? tests dir may lack __init__.py (previous pytest ran fine with test imports like from nexinfer.*). Absolute import from tests.gguf_fixtures inside tests/ works if pytest rootdir is nexinfer-work (it is; pytest.ini there). Check pytest.ini rootdir. If ruff F401 on os in test_backends_smoke after change, remove import os.

### Progress: tests/__init__.py created (empty docstring). test_backends_smoke + test_ggml_hardening updated to use gguf_fixtures.gguf_path().
Next: (1) python3 /tmp/download_model.py (download ~100MB GGUF to ~/models/bartowski/...). (2) ruff check nexinfer tests && ruff format --check nexinfer tests (format if fail). (3) python3 -m pytest tests -q. (4) README Phase 3 + version 0.3.0. (5) git commit+push. (6) zip + deliver.

### Executing now: python3 /tmp/download_model.py — wait ~2 min for 100MB.
