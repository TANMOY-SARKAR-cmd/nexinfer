"""Paged KV-cache with block allocation and CPU/VRAM spill.

Inspired by vLLM's PagedAttention: the KV-cache is split into fixed-size
blocks. Each request holds a sequence of block indices instead of a
contiguous buffer, so memory is never wasted on tail padding and can be
compacted, spilled to host RAM, or reassembled for offloading.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np


@dataclass
class CacheBlock:
    index: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    seq_len: int
    tier: str  # "device" (fast: GPU/TPU) or "host" (CPU RAM) or "disk"


class PagedKVCache:
    """Paged attention key/value cache.

    Layout: blocks[layer][2 (k,v)][block_idx] -> np.ndarray of shape
    (block_size, num_kv_heads, head_dim).

    A `tier` system allows hot blocks to live on device memory and cold
    blocks to spill to host RAM (or disk via `disk_blocks`).
    """

    def __init__(
        self,
        block_size: int = 16,
        num_blocks_device: int = 1024,
        num_blocks_host: int = 4096,
        num_layers: int = 12,
        num_kv_heads: int = 4,
        head_dim: int = 64,
        dtype: np.dtype = np.float16,
    ) -> None:
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = np.dtype(dtype)
        self._lock = threading.Lock()

        # free pools per tier
        self._free_device: list[int] = list(range(num_blocks_device))
        self._free_host: list[int] = list(range(num_blocks_host))
        self._next_block = num_blocks_device + num_blocks_host

        # allocated storage: tier -> block_idx -> array
        self._storage_device: dict[int, np.ndarray] = {}
        self._storage_host: dict[int, np.ndarray] = {}

        # request_id -> list of (layer-block arrays refs handled via indices)
        # For simplicity each sequence maps to block indices per layer:
        self._seq_blocks: dict[str, list[list[int]]] = {}  # req -> per-layer block idx list

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def _alloc_block(self, prefer_tier: str = "device") -> tuple[int, str]:
        with self._lock:
            if prefer_tier == "device" and self._free_device:
                idx = self._free_device.pop()
                return idx, "device"
            if self._free_host:
                idx = self._free_host.pop()
                return idx, "host"
            # growth with compaction warning: allocate beyond pools lazily
            idx = self._next_block
            self._next_block += 1
            return idx, "host"

    def ensure_sequence(self, req_id: str, seq_len: int, prefer_tier: str = "device") -> list[list[int]]:
        """Ensure block coverage for `seq_len` tokens across all layers."""
        n_blocks = (seq_len + self.block_size - 1) // self.block_size
        layers = self.num_layers
        if req_id not in self._seq_blocks:
            self._seq_blocks[req_id] = [[-1] * n_blocks for _ in range(layers)]

        seq = self._seq_blocks[req_id]
        # allocate a real block for every sentinel entry (initial coverage and
        # extension when the sequence grows)
        for layer_blocks in seq:
            while len(layer_blocks) < n_blocks:
                layer_blocks.append(-1)
            for i in range(n_blocks):
                if layer_blocks[i] == -1:
                    idx, tier = self._alloc_block(prefer_tier)
                    layer_blocks[i] = idx
                    storage = self._storage_device if tier == "device" else self._storage_host
                    storage[idx] = np.zeros(
                        (2 * self.block_size, self.num_kv_heads, self.head_dim), dtype=self.dtype
                    )
        return seq

    def get_block_array(self, layer: int, block_idx: int) -> np.ndarray:
        if block_idx in self._storage_device:
            return self._storage_device[block_idx]
        return self._storage_host[block_idx]

    # ------------------------------------------------------------------
    # Access helpers
    # ------------------------------------------------------------------

    def _ensure_block(self, req_id: str, layer: int, b: int) -> None:
        """Allocate the block for index ``b`` on ``layer`` if it is still a sentinel."""
        seq = self._seq_blocks[req_id]
        while len(seq[layer]) <= b:
            idx, tier = self._alloc_block("device")
            seq[layer].append(idx)
            storage = self._storage_device if tier == "device" else self._storage_host
            storage[idx] = np.zeros((2 * self.block_size, self.num_kv_heads, self.head_dim), dtype=self.dtype)

    def write(self, req_id: str, layer: int, pos: int, k: np.ndarray, v: np.ndarray) -> None:
        """Write k/v for a single position (pos) into the right block."""
        blocks = self._seq_blocks[req_id][layer]
        b = pos // self.block_size
        off = pos % self.block_size
        if blocks[b] == -1:
            self._ensure_block(req_id, layer, b)
            blocks = self._seq_blocks[req_id][layer]
        arr = self.get_block_array(layer, blocks[b])
        arr[off] = k
        arr[self.block_size + off] = v  # interleave k,v in the same block storage

    def read(self, req_id: str, layer: int, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
        """Read k/v for [start, end) as contiguous arrays."""
        blocks = self._seq_blocks[req_id][layer]
        k_out, v_out = [], []
        for pos in range(start, end):
            b = pos // self.block_size
            off = pos % self.block_size
            if blocks[b] == -1:
                self._ensure_block(req_id, layer, b)
                blocks = self._seq_blocks[req_id][layer]
            arr = self.get_block_array(layer, blocks[b])
            k_out.append(arr[off])
            v_out.append(arr[self.block_size + off])
        return np.stack(k_out), np.stack(v_out)

    def free(self, req_id: str) -> None:
        with self._lock:
            if req_id not in self._seq_blocks:
                return
            for layer_blocks in self._seq_blocks.pop(req_id):
                for idx in layer_blocks:
                    if idx in self._storage_device:
                        del self._storage_device[idx]
                        self._free_device.append(idx)
                    elif idx in self._storage_host:
                        del self._storage_host[idx]
                        self._free_host.append(idx)

    # ------------------------------------------------------------------
    # Stats / spill helpers
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, int]:
        return {
            "device_blocks_used": len(self._storage_device),
            "device_blocks_free": len(self._free_device),
            "host_blocks_used": len(self._storage_host),
            "host_blocks_free": len(self._free_host),
        }

    def spill_cold_blocks(self, n: int) -> int:
        """Move n least-recently-used device blocks to host tier. Returns count moved."""
        # naive LRU: spill first allocated blocks
        with self._lock:
            moved = 0
            for idx in list(self._storage_device.keys()):
                if moved >= n:
                    break
                arr = self._storage_device.pop(idx)
                self._storage_host[idx] = arr.astype(np.float16, copy=False)
                self._free_device.append(idx)
                moved += 1
            return moved
