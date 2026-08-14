"""Generation engine: wires tokenizer + scheduler + backend + sampler.

Produces tokens step by step (auto-regressive loop) with continuous
batching, stop conditions, and function-calling (tool) output parsing.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Generator

import numpy as np

from nexinfer.backends.base import Backend
from nexinfer.engine.kvcache import PagedKVCache
from nexinfer.engine.sampling import sample_token
from nexinfer.engine.scheduler import Scheduler
from nexinfer.engine.tokenizer_helper import Tokenizer
from nexinfer.engine.types import GenerationRequest, TokenOutput

log = logging.getLogger("nexinfer.generation")

# minimal function-call extraction: look for <function=Name>{json}</function
TOOL_CALL_RE = re.compile(
    r"<function=([A-Za-z0-9_]+)>(.*?)</function|\[CALL:([A-Za-z0-9_]+)\]\s*(\{.*?\})", re.DOTALL
)


class GenerationEngine:
    def __init__(
        self,
        backend: Backend,
        tokenizer: Tokenizer,
        scheduler: Scheduler,
        kv_cache: PagedKVCache,
    ) -> None:
        self.backend = backend
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.kv_cache = kv_cache

    def generate(self, req: GenerationRequest) -> TokenOutput:
        """Synchronous single-request generation."""
        tokens: list[TokenOutput] = list(self.generate_stream(req))
        if not tokens:
            return TokenOutput(text="", finish_reason="error")
        return tokens[-1]

    def generate_stream(self, req: GenerationRequest) -> Generator[TokenOutput, None, None]:
        rr = self.scheduler.add(req)
        if rr is None:
            yield TokenOutput(text="", finish_reason="error")
            return
        self.kv_cache.ensure_sequence(req.request_id, 16)

        # prefill
        input_ids = np.array(self.tokenizer.encode(req.prompt), dtype=np.int32)
        if input_ids.size == 0:
            input_ids = np.array([0], dtype=np.int32)
        logits = self.backend.prefill(req.request_id, input_ids)
        if logits.ndim > 1:
            logits = logits[-1]  # sample from the last position's distribution
        rng = np.random.default_rng(hash(req.request_id) % (2**31))

        n_tokens = 0
        while True:
            tok_id = sample_token(logits, req, rr.input_ids, rng)
            text = self.tokenizer.decode([tok_id])
            rr.append(tok_id, text)
            n_tokens += 1
            finish = rr.should_stop()
            # tool-call extraction (best-effort, template-agnostic)
            if req.tools and n_tokens > 2:
                calls = _extract_tool_calls(rr.generated_text, req.tools)
                if calls:
                    rr.tool_calls = calls
                    finish = "tool_call"

            yield TokenOutput(
                text=text,
                token_id=tok_id,
                finish_reason=finish,
                tool_calls=rr.tool_calls,
                usage={"prompt_tokens": int(input_ids.size), "completion_tokens": n_tokens},
            )
            if finish:
                self.scheduler.finish(req.request_id)
                self.kv_cache.free(req.request_id)
                return

            logits = self.backend.decode([req.request_id], np.array([[tok_id]], dtype=np.int32))
            if logits.ndim > 1:
                logits = logits[-1]  # last position's distribution
            if rr.req.stream is False and n_tokens >= 4096:  # safety cap
                self.scheduler.finish(req.request_id)
                self.kv_cache.free(req.request_id)
                yield TokenOutput(text="", finish_reason="length")
                return


def _extract_tool_calls(text: str, tools: list[dict]) -> list[dict]:
    """Best-effort tool-call extraction for engines without native support."""
    out = []
    by_name = {t.get("name") or t.get("function", {}).get("name"): t for t in tools}
    for m in TOOL_CALL_RE.finditer(text):
        name = m.group(1) or m.group(3)
        raw_args = m.group(2) or m.group(4) or "{}"
        if name in by_name:
            try:
                import json

                args = json.loads(raw_args.strip())
            except Exception:
                args = {"raw": raw_args.strip()}
            out.append({"name": name, "arguments": args})
    return out
