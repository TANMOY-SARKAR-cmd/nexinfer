"""CPU numpy reference backend.

A minimal transformer decoder implemented with plain NumPy -- the
portable reference that runs on *any* Windows/Linux machine with no
drivers at all. Used for development, tests, low-end fallback, and as
the contract example every other backend must satisfy.

Weights are either generated (demo mode) or loaded from a
``model.npz``/safetensors file containing ``wte``, ``blk.{i}.attn_wq``,
``blk.{i}.attn_wk``, ``blk.{i}.attn_wv``, ``blk.{i}.attn_wo``,
``blk.{i}.ffn_w1``, ``blk.{i}.ffn_w2``, ``blk.{i}.attn_norm``,
``blk.{i}.ffn_norm`` (GPT-2 style, swappable per backend).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

from nexinfer.backends.base import (
    Backend,
    BackendCapabilities,
    DeviceInfo,
    ModelSpec,
)
from nexinfer.engine.types import DeviceId, DeviceKind

log = logging.getLogger("nexinfer.backends.cpu_numpy")


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


class NumpyBackend(Backend):
    """Portable CPU-only transformer decoder backend (numpy)."""

    name = "cpu_numpy"
    platform = "any"

    def __init__(self) -> None:
        self.spec: ModelSpec | None = None
        self.params: dict[str, np.ndarray] = {}
        self._kv: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}
        self._rng = np.random.default_rng(0)

    # ------------------------------------------------------------------
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_gpu=False,
            supports_tpu=False,
            supports_quant=False,
            supports_tool_calls=False,
            supports_pipeline_parallel=True,
            supports_tensor_parallel=False,
        )

    def detect_devices(self) -> list[DeviceInfo]:
        return [
            DeviceInfo(
                device_id="/cpu:0",
                kind=DeviceKind.CPU,
                vendor="generic",
                name="CPU (numpy)",
                total_memory_bytes=0,
                compute_score=1.0,
            )
        ]

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, model_path_or_id: str, spec: ModelSpec, devices: list[DeviceId]) -> None:
        self.spec = spec
        if os.path.isfile(model_path_or_id):
            self.params = _load_npz(model_path_or_id, spec)
        elif os.path.isdir(model_path_or_id):
            npz = next((f for f in os.listdir(model_path_or_id) if f.endswith(".npz")), None)
            if npz:
                self.params = _load_npz(os.path.join(model_path_or_id, npz), spec)
            else:
                raise FileNotFoundError(f"no .npz weights in {model_path_or_id}")
        else:
            log.info("demo mode: generating random weights for model id %s", model_path_or_id)
            self.params = _random_weights(spec, self._rng)

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def prefill(self, req_id: str, input_ids: np.ndarray) -> np.ndarray:
        spec = self.spec
        assert spec is not None, "backend not loaded"
        k_all, v_all = [], []
        x = _embed(input_ids, self.params["wte"], spec.vocab_size)
        for i in range(spec.num_layers):
            x, k, v = self._layer(x, i, input_ids.size, spec)
            k_all.append(k)
            v_all.append(v)
        self._kv[req_id] = (k_all, v_all)
        return self._lm_head(x)

    def decode(self, req_ids: list[str], input_ids: np.ndarray) -> np.ndarray:
        """input_ids shape (n_req, 1); returns logits (n_req, vocab)."""
        spec = self.spec
        assert spec is not None, "backend not loaded"
        outs = []
        for req_id, tok in zip(req_ids, input_ids):
            k_all, v_all = self._kv[req_id]
            x = _embed(np.array([tok.item()]), self.params["wte"], spec.vocab_size)
            x = x.reshape(1, spec.hidden_size)
            new_ks, new_vs = [], []
            for i in range(spec.num_layers):
                total_len = k_all[i].shape[0] + 1
                x, k, v = self._layer(x, i, total_len, spec, past_k=k_all[i], past_v=v_all[i])
                new_ks.append(k)
                new_vs.append(v)
            self._kv[req_id] = (new_ks, new_vs)
            outs.append(self._lm_head(x))
        return np.concatenate(outs, axis=0)

    def close(self) -> None:
        self.params.clear()
        self._kv.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _lm_head(self, x: np.ndarray) -> np.ndarray:
        spec = self.spec
        x = _layernorm(x, self.params["ln_f"])
        return x @ self.params["wte"].T * (spec.hidden_size ** -0.5)

    def _layer(
        self,
        x: np.ndarray,
        layer: int,
        seq_len: int,
        spec: ModelSpec,
        past_k: np.ndarray | None = None,
        past_v: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = self.params
        # attention
        a = _layernorm(x, p[f"blk.{layer}.attn_norm"])
        q = a @ p[f"blk.{layer}.attn_wq"]
        k = a @ p[f"blk.{layer}.attn_wk"]
        v = a @ p[f"blk.{layer}.attn_wv"]
        scale = spec.head_dim ** -0.5
        n_new = seq_len - past_k.shape[0] if past_k is not None else seq_len
        if past_k is not None:
            new_k = k.reshape(n_new, spec.num_kv_heads, spec.head_dim)
            new_v = v.reshape(n_new, spec.num_kv_heads, spec.head_dim)
            k = np.concatenate([past_k, new_k], axis=0)
            v = np.concatenate([past_v, new_v], axis=0)
            q = q.reshape(n_new, spec.num_attention_heads, spec.head_dim)
            seq_len = k.shape[0]
        else:
            # multi-query-ish broadcast: (seq, heads, dim)
            k = k.reshape(k.shape[0], spec.num_kv_heads, spec.head_dim)
            v = v.reshape(v.shape[0], spec.num_kv_heads, spec.head_dim)
            q = q.reshape(q.shape[0], spec.num_attention_heads, spec.head_dim)
        # GQA attention
        repeats = spec.num_attention_heads // spec.num_kv_heads
        k_r = np.repeat(k, repeats, axis=1) if repeats > 1 else k
        v_r = np.repeat(v, repeats, axis=1) if repeats > 1 else v
        attn = (q.transpose(1, 0, 2) @ k_r.transpose(1, 2, 0)) * scale
        # causal mask: rows = query positions (n_new), cols = full context (seq_len)
        # queries start at offset (seq_len - n_new) in the history
        q_off = seq_len - n_new
        mask = np.triu(np.full((n_new, seq_len), -1e9), k=q_off + 1)
        attn = _softmax(attn + mask[None, :, :])
        out = (attn @ v_r.transpose(1, 0, 2)).transpose(1, 0, 2).reshape(n_new, spec.hidden_size)
        x = x + out @ p[f"blk.{layer}.attn_wo"]
        # FFN (swish-gated)
        f = _layernorm(x, p[f"blk.{layer}.ffn_norm"])
        gate = _silu(f @ p[f"blk.{layer}.ffn_w1"])
        up = f @ p[f"blk.{layer}.ffn_w3"] if f"blk.{layer}.ffn_w3" in p else up_identity(f, p, layer)
        x = x + (gate * up) @ p[f"blk.{layer}.ffn_w2"]
        return x, k, v


def up_identity(f: np.ndarray, p: dict, layer: int) -> np.ndarray:
    return f @ p[f"blk.{layer}.ffn_w1"]


def _embed(ids: np.ndarray, wte: np.ndarray, vocab_size: int) -> np.ndarray:
    ids = np.clip(ids, 0, vocab_size - 1)
    return wte[ids]


def _layernorm(x: np.ndarray, w: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return w * (x - mean) / np.sqrt(var + eps)


def _load_npz(path: str, spec: ModelSpec) -> dict[str, np.ndarray]:
    data = dict(np.load(path))
    # ensure every expected key exists; pad missing keys with small random
    rng = np.random.default_rng(42)
    for key in _expected_keys(spec):
        if key not in data:
            data[key] = rng.standard_normal(_key_shape(key, spec)).astype(np.float32) * 0.02
    return data


def _expected_keys(spec: ModelSpec) -> list[str]:
    keys = ["wte", "ln_f"]
    for i in range(spec.num_layers):
        keys += [
            f"blk.{i}.attn_norm",
            f"blk.{i}.attn_wq",
            f"blk.{i}.attn_wk",
            f"blk.{i}.attn_wv",
            f"blk.{i}.attn_wo",
            f"blk.{i}.ffn_norm",
            f"blk.{i}.ffn_w1",
            f"blk.{i}.ffn_w2",
            f"blk.{i}.ffn_w3",
        ]
    return keys


def _key_shape(key: str, spec: ModelSpec) -> tuple[int, ...]:
    h, kh, d = spec.hidden_size, spec.num_kv_heads, spec.head_dim
    if key in ("wte", "ln_f"):
        return (spec.vocab_size, h) if key == "wte" else (h,)
    if "attn_wq" in key:
        return (h, spec.num_attention_heads * d)
    if "attn_wk" in key:
        return (h, kh * d)
    if "attn_wv" in key:
        return (h, kh * d)
    if "attn_wo" in key:
        return (spec.num_attention_heads * d, h)
    if "ffn_w1" in key:
        return (h, spec.inter_dim or h * 4)
    if "ffn_w2" in key:
        return ((spec.inter_dim or h * 4), h)
    if "ffn_w3" in key:
        return (h, spec.inter_dim or h * 4)
    return (h,)


def _random_weights(spec: ModelSpec, rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {k: rng.standard_normal(_key_shape(k, spec)).astype(np.float32) * 0.02 for k in _expected_keys(spec)}
