"""Distributed generation engine (client-facing).

Sits in front of a running cluster and drives generation: it sends
prefill to rank-0, which flows activations through the pipeline; the
final rank returns logits, which the local sampler consumes. On a
single machine this degenerates to local inference with zero overhead.
"""

from __future__ import annotations

import logging
from collections.abc import Generator

import numpy as np

from nexinfer.distributed.coordinator import Coordinator
from nexinfer.distributed.worker import Worker
from nexinfer.engine.sampling import sample_token
from nexinfer.engine.tokenizer_helper import Tokenizer
from nexinfer.engine.types import GenerationRequest, TokenOutput

log = logging.getLogger("nexinfer.distributed.engine")


class DistributedEngine:
    def __init__(
        self,
        coordinator: Coordinator,
        tokenizer: Tokenizer,
        rank0_worker: Worker,
    ) -> None:
        self.coord = coordinator
        self.tokenizer = tokenizer
        self.rank0 = rank0_worker

    def generate(self, req: GenerationRequest) -> TokenOutput:
        tokens = list(self.generate_stream(req))
        return tokens[-1] if tokens else TokenOutput(text="", finish_reason="error")

    def generate_stream(self, req: GenerationRequest) -> Generator[TokenOutput, None, None]:
        input_ids = np.array(self.tokenizer.encode(req.prompt), dtype=np.int32)
        if input_ids.size == 0:
            input_ids = np.array([0], dtype=np.int32)
        rng = np.random.default_rng(hash(req.request_id) % (2**31))
        n = 0
        tok_id: int | None = None
        while True:
            # in-process pipeline step on the rank-0 worker (cluster mode
            # would send activations across the transport instead)
            x = input_ids if n == 0 else np.array([[tok_id]], dtype=np.int32)
            logits = np.asarray(self.rank0.backend.prefill(req.request_id, x), dtype=np.float32)
            if logits.ndim > 1 and logits.shape[0] > 1:
                logits = logits[-1]  # full-sequence prefill; sample the last position
            tok_id = sample_token(logits, req, rng=rng)
            text = self.tokenizer.decode([tok_id])
            n += 1
            finish = None
            if req.max_tokens > 0 and n >= req.max_tokens:
                finish = "length"
            for stop in req.stop_sequences:
                if stop and stop in text:
                    finish = "stop"
            yield TokenOutput(
                text=text,
                token_id=tok_id,
                finish_reason=finish,
                usage={"prompt_tokens": int(input_ids.size), "completion_tokens": n},
            )
            if finish:
                return
            input_ids = np.array([[tok_id]], dtype=np.int32)
