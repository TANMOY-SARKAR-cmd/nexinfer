"""Chat templates for formatting conversation messages into prompt text.

NexusInfer ships a minimal template engine that covers the templates used
by popular small models (SmolLM, TinyLlama-style, ChatML and a plain
instruction format). When ``jinja2`` is available, full HuggingFace-style
``{% if %}`` / ``{% for %}`` templates are supported; otherwise the engine
falls back to a small built-in renderer for the bundled templates.
"""

from __future__ import annotations

import re
from typing import Any

log = __import__("logging").getLogger("nexinfer.engine.chat_template")

# ---------------------------------------------------------------------------
# Built-in templates (used when jinja2 is unavailable or a template name is
# given instead of raw template text).
# ---------------------------------------------------------------------------

BUILTIN_TEMPLATES: dict[str, str] = {
    # SmolLM / Llama-3-ish: system, user/assistant turns wrapped in role tags
    "smollm": (
        "{% for m in messages %}"
        "{% if m.role == 'system' %}<|system|>\n{{ m.content }}\n{% endif %}"
        "{% if m.role == 'user' %}<|user|>\n{{ m.content }}<|end|>\n{% endif %}"
        "{% if m.role == 'assistant' %}<|assistant|>\n{{ m.content }}<|end|>\n{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
    ),
    # ChatML used by many small GGUF models
    "chatml": (
        "{% for m in messages %}"
        "<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    ),
    # Plain instruction format (Alpaca-style)
    "alpaca": (
        "{% for m in messages %}"
        "{% if m.role == 'system' %}{{ m.content }}\n{% endif %}"
        "{% if m.role == 'user' %}### Instruction:\n{{ m.content }}\n{% endif %}"
        "{% if m.role == 'assistant' %}### Response:\n{{ m.content }}\n{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}### Response:\n{% endif %}"
    ),
    # Minimal single-turn instruction format used by the MinimalBPE tokenizer
    "minimal": (
        "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"
        "{% if add_generation_prompt %}assistant: {% endif %}"
    ),
}

# Regex used by the fallback renderer for the subset of jinja syntax the
# built-in templates rely on.
_FJ_FOR = re.compile(r"\{%[-\s]*for\s+(\w+)\s+in\s+(\w+)[\s-]*%\}")
_FJ_ENDFOR = re.compile(r"\{%[-\s]*endfor[\s-]*%\}")
_FJ_IF = re.compile(r"\{%[-\s]*if\s+(m\.role\s*==\s*'[\w]+')[\s-]*%\}")
_FJ_ENDIF = re.compile(r"\{%[-\s]*endif[\s-]*%\}")
_FJ_VAR = re.compile(r"\{\{\s*(m\.(\w+))\s*\}\}")
_JINJA_IMPORT = "jinja2"


def _render_fallback(
    template: str, messages: list[dict[str, str]], add_generation_prompt: bool = True
) -> str:
    """Render the small built-in template grammar without jinja2."""
    out: list[str] = []
    pos = 0
    for_iter = _FJ_FOR.match(template, pos)
    if not for_iter:
        # static template, just substitute variables of the last message
        body = template
        for name in ("content",):
            body = _FJ_VAR.sub(lambda mm, n=name: messages[-1].get(n, ""), body)
        return _strip_markers(body, add_generation_prompt)
    var = for_iter.group(1)  # noqa: F841  (kept for clarity)
    body = template[for_iter.end() :]
    chunks: list[tuple[str, str | None]] = []  # (text, condition or None)
    while body:
        ifm = _FJ_IF.match(body)
        if ifm:
            chunks.append(("", ifm.group(1)))
            body = body[ifm.end() :]
            continue
        eifm = _FJ_ENDIF.match(body)
        if eifm:
            chunks.append(("", None))
            body = body[eifm.end() :]
            continue
        endm = _FJ_ENDFOR.match(body)
        if endm:
            body = body[endm.end() :]
            break
        nxt = min(
            (m.start() for m in (_FJ_IF.match(body), _FJ_ENDIF.match(body), _FJ_ENDFOR.match(body)) if m),
            default=len(body),
        )
        chunks.append(
            (body[:nxt], None if chunks and chunks[-1][1] is not None else chunks[-1][1] if chunks else None)
        )
        body = body[nxt:]

    for m in messages:
        for text, cond in chunks:
            if cond is not None and not _eval_role_cond(cond, m):
                continue
            msg = m  # late binding: capture per-iteration value
            seg = _FJ_VAR.sub(lambda mm, _m=msg: _m.get(mm.group(2), ""), text)
            out.append(seg)
    tail = _FJ_VAR.sub(lambda mm: messages[-1].get(mm.group(2), ""), body)
    return _strip_markers("".join(out) + tail, add_generation_prompt)


def _eval_role_cond(cond: str, message: dict[str, str]) -> bool:
    match = re.match(r"m\.role\s*==\s*'([\w]+)'", cond)
    return bool(match) and message.get("role") == match.group(1)


def _strip_markers(text: str, add_generation_prompt: bool) -> str:
    """Remove the conditional ``{% if add_generation_prompt %}`` blocks.

    The fallback grammar inlines these blocks; when the flag is true the
    marker text that follows ``<|assistant|>\n`` or ``### Response:\n`` is
    kept (they are part of the prompt), otherwise such trailing markers
    are dropped.
    """
    if add_generation_prompt:
        return text
    for marker in ("<|assistant|>\n", "### Response:\n"):
        text = text.removesuffix(marker)
    return text


class ChatTemplate:
    """Format conversation messages into prompt text."""

    def __init__(self, template: str) -> None:
        if template in BUILTIN_TEMPLATES:
            self.raw = BUILTIN_TEMPLATES[template]
            self.name = template
        else:
            self.raw = template
            self.name = "custom"
        self._jinja_env: Any = None
        try:
            import jinja2  # noqa: F401

            self._jinja_env = __import__("jinja2").Environment(undefined=__import__("jinja2").StrictUndefined)
        except ImportError:
            log.info("jinja2 not installed; using fallback chat-template renderer")

    def apply(self, messages: list[dict[str, str]], add_generation_prompt: bool = True) -> str:
        if self._jinja_env is not None:
            tmpl = self._jinja_env.from_string(self.raw)
            try:
                return tmpl.render(messages=messages, add_generation_prompt=add_generation_prompt)
            except Exception as exc:
                log.warning("jinja render failed (%s), falling back", exc)
        return _render_fallback(self.raw, messages, add_generation_prompt)

    def __repr__(self) -> str:
        return f"ChatTemplate(name={self.name!r}, has_jinja={self._jinja_env is not None})"
