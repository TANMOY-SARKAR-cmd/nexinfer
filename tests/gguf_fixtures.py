"""Shared helper that resolves the SmolLM2-135M-Instruct Q4_K_M GGUF path.

Looks at ``GGUF_MODEL_PATH`` first, then at the layout produced by
``huggingface_hub.hf_hub_download(local_dir=~/models)`` (Phase 3 test
machines), and finally at the pre-populated snapshot cache used by
earlier CI runs. Tests ``pytest.skip`` when no copy is found.
"""

from __future__ import annotations

import os


def gguf_path() -> str | None:
    """Return an existing GGUF path or ``None``."""
    env = os.environ.get("GGUF_MODEL_PATH")
    if env and os.path.exists(env):
        return env
    home = os.path.expanduser("~")
    candidates = [
        # Phase-3 layout: downloaded by hf_hub_download(local_dir=~/models)
        os.path.join(
            home, "models", "bartowski", "SmolLM2-135M-Instruct-GGUF", "SmolLM2-135M-Instruct-Q4_K_M.gguf"
        ),
        # Legacy pre-populated snapshot cache layout
        os.path.join(
            home,
            "models",
            ".cache",
            "models--bartowski--SmolLM2-135M-Instruct-GGUF",
            "snapshots",
            "09816acd5d99df7be770d85ea30822623dab342c",
            "SmolLM2-135M-Instruct-Q4_K_M.gguf",
        ),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None
