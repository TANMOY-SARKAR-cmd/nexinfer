"""Token sampling strategies (greedy, temperature, top-k, top-p)."""

from __future__ import annotations

import numpy as np

from nexinfer.engine.types import GenerationRequest


def apply_repetition_penalty(logits: np.ndarray, input_ids: np.ndarray, penalty: float) -> np.ndarray:
    """Apply repetition penalty to logits (Penman & Chen style)."""
    if penalty == 1.0 or input_ids.size == 0:
        return logits
    logits = logits.copy()
    unique = np.unique(input_ids)
    sign = np.sign(logits[unique])
    sign[sign == 0] = 1.0
    logits[unique] -= sign * (penalty - 1.0) * penalty
    return logits


def sample_token(
    logits: np.ndarray,
    params: GenerationRequest,
    input_ids: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> int:
    """Sample a single token id from logits using the request's sampling params."""
    rng = rng or np.random.default_rng()
    logits = np.asarray(logits, dtype=np.float32)
    if params.repetition_penalty != 1.0 and input_ids is not None:
        logits = apply_repetition_penalty(logits, np.asarray(input_ids), params.repetition_penalty)

    if params.temperature <= 0.0 or params.top_k == 1:
        return int(np.argmax(logits))

    logits = logits / max(params.temperature, 1e-6)

    # top-k
    if params.top_k > 0 and params.top_k < logits.size:
        threshold = np.sort(logits)[-params.top_k]
        logits = np.where(logits < threshold, -np.inf, logits)

    probs = np.exp(logits - np.max(logits))
    probs /= probs.sum()

    # top-p (nucleus)
    if params.top_p < 1.0:
        order = np.argsort(-probs)
        sorted_probs = probs[order]
        cumulative = np.cumsum(sorted_probs)
        # keep the first token that pushes cumsum >= top_p
        keep_mask = cumulative <= params.top_p
        keep_mask[np.argmax(~keep_mask)] = True  # include boundary token
        allowed = order[keep_mask]
        new_probs = np.zeros_like(probs)
        new_probs[allowed] = probs[allowed]
        new_probs /= new_probs.sum()
        probs = new_probs

    return int(rng.choice(probs.size, p=probs))
