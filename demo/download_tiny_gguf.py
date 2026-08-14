#!/usr/bin/env python3
"""Download a tiny GGUF model for local testing (optional, not part of the package)."""
from __future__ import annotations

import os

from huggingface_hub import hf_hub_download

REPO = "HuggingFaceTB/SmolLM2-135M-Instruct-GGUF"
FILE = "smollm2-135m-instruct-q4_k_m.gguf"
OUT = os.path.expanduser("~/models")

os.makedirs(OUT, exist_ok=True)
path = hf_hub_download(repo_id=REPO, filename=FILE, cache_dir=os.path.join(OUT, ".cache"))
print(f"downloaded: {path} ({os.path.getsize(path) / 1024 ** 2:.1f} MB)")
