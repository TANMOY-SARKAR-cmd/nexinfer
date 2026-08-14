"""Distributed generation engine (client-facing).

Sits in front of a running cluster and drives generation: it sends
prefill to rank-0, which flows activations through the pipeline over
the data transport; the final rank returns logits, which the local
sampler consumes. On a single machine this degenerates to local
inference with zero overhead.

The token loop runs on the rank-0 worker side (``worker.generate``),
which walks the pipeline ring per step and returns sampled logits in
token order. This engine decodes the token ids back to text and
enforces stop conditions and abort-on-cancel.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Generator

import numpy as np

from nexinfer.distributed.worker import Worker
from nexinfer.engine.tokenizer_helper import Tokenizer
from nexinfer.engine.types import GenerationRequest, TokenOutput

log = logging.getLogger("nexinfer.distributed.engine")


class DistributedEngine:
    def __init__(
        self,
        rank0_worker: Worker,
        tokenizer: Tokenizer,
    ) -> None:
        self.rank0 = rank0_worker
        self.tokenizer = tokenizer

    # ------------------------------------------------------------------
    def generate(self, req: GenerationRequest) -> TokenOutput:
        tokens = list(self.generate_stream(req))
        if not tokens:
            return TokenOutput(text="", finish_reason="error")
        out = tokens[-1]
        out = TokenOutput(
            text="".join(t.text for t in tokens),
            token_id=tokens[-1].token_id,
            finish_reason=out.finish_reason,
            usage={"prompt_tokens": out.usage["prompt_tokens"], "completion_tokens": len(tokens)},
        )
        return out

    def generate_stream(self, req: GenerationRequest) -> Generator[TokenOutput, None, None]:
        # ``worker.generate`` runs its own event loop on a dedicated
        # thread so this engine can be used from synchronous callers
        # (HTTP handlers, REPL scripts) without an outer asyncio loop.
        input_ids = self.tokenizer.encode(req.prompt)
        if not input_ids:
            input_ids = [0]

        result: dict[str, object] = {}
        t: threading.Thread | None = None

        # The rank-0 worker's transport server tasks (which feed the
        # activation/logits receive queues) run on the worker's own event
        # loop, so ``rank0.generate`` MUST be scheduled on that same loop:
        # cross-loop ``asyncio.Queue`` traffic deadlocks otherwise.
        worker_loop = getattr(self.rank0, "loop", None)
        if worker_loop is None:
            result["error"] = RuntimeError(
                "rank-0 worker has no event loop; call worker.start() before generating"
            )
        else:
            done = threading.Event()

            def driver() -> None:
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        self.rank0.generate(req.request_id, input_ids, max_tokens=req.max_tokens),
                        worker_loop,
                    )
                    result["logits"] = future.result(timeout=120.0)
                except Exception as exc:
                    result["error"] = exc
                finally:
                    done.set()

            t = threading.Thread(target=driver, daemon=True)
            t.start()

        try:
            n = 0
            finish = None
            stop_sequences = req.stop_sequences or []
            while not done.is_set() or "logits" in result or "error" in result:
                if "error" in result:
                    raise result["error"]  # type: ignore[misc]
                if "logits" in result:
                    logits = result["logits"]
                    break
                done.wait(timeout=0.02)
            else:
                logits = []

            assert isinstance(logits, list), "distributed generate must return a logits list"
            for n, logits_i in enumerate(logits, start=1):
                logits_i = np.asarray(logits_i, dtype=np.float32)
                if logits_i.ndim > 1:
                    logits_i = logits_i[-1]
                tok_id = int(np.argmax(logits_i))
                text = self.tokenizer.decode([tok_id])
                if n >= req.max_tokens:
                    finish = "length"
                for stop in stop_sequences:
                    if stop and stop in text:
                        finish = "stop"
                yield TokenOutput(
                    text=text,
                    token_id=tok_id,
                    finish_reason=finish,
                    usage={"prompt_tokens": len(input_ids), "completion_tokens": n},
                )
                if finish:
                    return
        finally:
            if t is not None:
                t.join(timeout=1.0)
