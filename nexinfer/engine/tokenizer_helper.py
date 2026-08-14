"""Tokenizer wrapper over Hugging Face ``tokenizers`` / local vocab files.

Handles HF hub repos (``meta-llama/...``, ``Qwen/Qwen2-0.5B``), local
``tokenizer.json`` / ``tokenizer.model`` files, and a minimal BPE fallback
so unit tests run without downloading anything.
"""

from __future__ import annotations

import logging
import os
import re

log = logging.getLogger("nexinfer.tokenizer")

try:
    from tokenizers import Tokenizer as HFTokenizer

    _HAS_TOKENIZERS = True
except ImportError:  # pragma: no cover
    _HAS_TOKENIZERS = False


def _download_hub_file(repo_id: str, filename: str, cache_dir: str | None) -> str:
    """Download a tokenizer file from the HF hub (requires huggingface_hub)."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("huggingface_hub is required for hub tokenizer loading") from exc
    return hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)


class Tokenizer:
    """Thin wrapper giving ``encode`` / ``decode`` over HF tokenizers."""

    def __init__(self, hf: object) -> None:  # hf: HFTokenizer
        self._hf = hf

    @staticmethod
    def load(model_name_or_path: str, cache_dir: str | None = None) -> Tokenizer:
        """Load a tokenizer from a HF repo id or a local directory."""
        if not _HAS_TOKENIZERS:
            return Tokenizer(MinimalBPE(vocab_size=32000))

        if os.path.isdir(model_name_or_path):
            candidates = [
                os.path.join(model_name_or_path, "tokenizer.json"),
                os.path.join(model_name_or_path, "tokenizer.model"),
            ]
            path = next((c for c in candidates if os.path.exists(c)), None)
            if path is None:
                raise FileNotFoundError(f"no tokenizer file in {model_name_or_path}")
            if path.endswith(".json"):
                return Tokenizer(HFTokenizer.from_file(path))
            # sentencepiece model
            from tokenizers import SentencePieceBPETokenizer  # type: ignore

            return Tokenizer(SentencePieceBPETokenizer.from_file(path))

        # HF hub repo id
        for fname in ("tokenizer.json", "tokenizer.model"):
            try:
                path = _download_hub_file(model_name_or_path, fname, cache_dir)
                if fname.endswith(".json"):
                    return Tokenizer(HFTokenizer.from_file(path))
            except Exception:
                continue
        raise FileNotFoundError(f"could not load tokenizer for {model_name_or_path}")

    def encode(self, text: str) -> list[int]:
        raw = self._hf.encode(text)
        if hasattr(raw, "ids"):
            return list(raw.ids)  # HF tokenizers: Encoding object
        return list(raw)  # MinimalBPE fallback returns a plain list

    def decode(self, ids: list[int]) -> str:
        return self._hf.decode(ids)  # type: ignore[union-attr]

    def apply_chat_template(
        self, messages: list[dict[str, str]], template: str | None = None, add_generation_prompt: bool = True
    ) -> str:
        """Format a conversation into prompt text using a chat template.

        ``template`` may be a built-in template name (``smollm``,
        ``chatml``, ``alpaca``, ``minimal``) or raw template text. When the
        loaded tokenizer exposes ``chat_template`` metadata (HF repos), that
        template is used unless ``template`` is given explicitly.
        """
        from nexinfer.engine.chat_template import ChatTemplate

        if template is None:
            template = getattr(self._hf, "chat_template", None) or "minimal"
        return ChatTemplate(template).apply(messages, add_generation_prompt)

    @property
    def bos_token_id(self) -> int | None:
        try:
            return self._hf.token_to_id("<s>")  # type: ignore[union-attr]
        except Exception:
            return None

    @property
    def eos_token_id(self) -> int | None:
        try:
            return self._hf.token_to_id("</s>")  # type: ignore[union-attr]
        except Exception:
            return None

    @property
    def vocab_size(self) -> int:
        try:
            return self._hf.get_vocab_size()  # type: ignore[union-attr]
        except Exception:
            return 0


class MinimalBPE:
    """Minimal byte-level BPE tokenizer used only when `tokenizers` is
    unavailable (e.g. in CI). Not meant to be production-quality."""

    def __init__(self, vocab_size: int = 32000) -> None:
        self.vocab_size = vocab_size
        # regex split into words; map each unique word to an id
        self._word_re = re.compile(r"\w+|\s+|[^\w\s]")
        self._w2i: dict[str, int] = {}
        self._i2w: dict[int, str] = {}

    def _ensure(self, word: str) -> int:
        if word not in self._w2i:
            if len(self._w2i) >= self.vocab_size - 2:
                return 1  # fallback bucket
            idx = len(self._w2i) + 3
            self._w2i[word] = idx
            self._i2w[idx] = word
        return self._w2i[word]

    def encode(self, text: str) -> list[int]:
        return [self._ensure(w) for w in self._word_re.findall(text)]

    def decode(self, ids: list[int]) -> str:
        return "".join(self._i2w.get(i, "") for i in ids)
