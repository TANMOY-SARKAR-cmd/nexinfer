"""HTTP security primitives for the NexusInfer API surface.

Three building blocks are provided, all usable with the stdlib HTTP server
(plus FastAPI/Flask if users host ``HttpServer`` elsewhere):

* ``ApiKeyAuth``   -- validates ``Authorization: Bearer <key>`` against a
  configured secret. When **no key is configured the gate is open**, so a
  dev-mode server keeps working out of the box. Configure via
  ``NEXINFER_API_KEY`` env var or ``--api-key`` CLI flag.
* ``RateLimiter``  -- token-bucket limiter keyed by client IP (or any
  string). ``--rate-limit`` is requests-per-minute; 0 / unset = unlimited.
* ``InputPolicy``  -- request validation: caps on prompt length,
  ``max_tokens``, stop-sequence counts, and a sanitiser for function-call
  tool schemas (rejects anything that is not a plain dict with safe
  primitives).

These deliberately have **no external dependencies** so the default
``nexinfer`` install stays tiny; everything here is pure stdlib.
"""

from __future__ import annotations

import hmac
import logging
import math
import re
import secrets
import threading
import time
from typing import Any

log = logging.getLogger("nexinfer.services.security")

ALLOWED_SCHEMA_TYPES = (str, int, float, bool, list, type(None))


class ApiKeyAuth:
    """Bearer-token gate.

    * ``api_key`` set      -> every request must carry a valid Bearer token
    * ``api_key`` unset    -> gate disabled (development mode)
    """

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or ""

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    @staticmethod
    def from_env_or_flag(api_key: str | None = None) -> ApiKeyAuth:
        """Prefer an explicit value, otherwise fall back to
        ``NEXINFER_API_KEY`` (stripped of surrounding whitespace)."""
        if not api_key:
            api_key = (__import__("os").getenv("NEXINFER_API_KEY") or "").strip()
        return ApiKeyAuth(api_key or None)

    def check(self, authorization_header: str | None) -> bool:
        """Return True if the request is allowed."""
        if not self.enabled:
            return True
        if not authorization_header:
            return False
        parts = authorization_header.split(None, 1)
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return False
        # constant-time comparison avoids leaking key-length timing info
        return hmac.compare_digest(parts[1].strip(), self.api_key)

    @staticmethod
    def generate_key() -> str:
        """Handy helper for generating a strong key (hex of 32 bytes)."""
        return secrets.token_hex(32)


class RateLimiter:
    """Fixed-window / token-bucket limiter keyed by arbitrary client id.

    ``limit`` is the maximum number of requests allowed per
    ``window_seconds`` (default window = 60 s, matching the documented
    "requests per minute" semantics). ``limit <= 0`` disables limiting.
    """

    def __init__(self, limit: int = 0, window_seconds: float = 60.0) -> None:
        self.limit = max(0, int(limit))
        self.window = window_seconds
        self._lock = threading.Lock()
        # key -> (window_start, count_in_window)
        self._buckets: dict[str, tuple[float, int]] = {}

    @property
    def enabled(self) -> bool:
        return self.limit > 0

    def allow(self, client_id: str, *, cost: int = 1) -> bool:
        """Return True when the request is allowed (and consume quota)."""
        if not self.enabled:
            return True
        now = time.time()
        with self._lock:
            start, count = self._buckets.get(client_id, (now, 0))
            if now - start >= self.window:
                start, count = now, 0
            if count + cost <= self.limit:
                self._buckets[client_id] = (start, count + cost)
                return True
            return False

    def clear(self, client_id: str | None = None) -> None:
        with self._lock:
            if client_id:
                self._buckets.pop(client_id, None)
            else:
                self._buckets.clear()

    def cleanup(self, *, max_age: float | None = None) -> int:
        """Drop stale buckets older than ``max_age`` (default 2 windows).
        Returns the number of removed entries."""
        max_age = self.window * 2 if max_age is None else max_age
        now = time.time()
        stale = []
        with self._lock:
            for key, (start, _count) in self._buckets.items():
                if now - start > max_age:
                    stale.append(key)
            for key in stale:
                self._buckets.pop(key, None)
        return len(stale)


class InputPolicy:
    """Server-side validation and sanitisation of generation requests."""

    DEFAULT_MAX_PROMPT_CHARS = 128 * 1024  # 128 KiB text ceiling
    DEFAULT_MAX_PROMPT_TOKENS = 8192
    DEFAULT_MAX_TOKENS_CAP = 8192
    DEFAULT_MAX_STOP_SEQUENCES = 4
    DEFAULT_MAX_STOP_CHARS = 64
    DEFAULT_MAX_MESSAGES = 256

    def __init__(
        self,
        max_prompt_chars: int | None = None,
        max_prompt_tokens: int | None = None,
        max_tokens_cap: int | None = None,
        max_stop_sequences: int | None = None,
        max_messages: int | None = None,
    ) -> None:
        self.max_prompt_chars = (
            max_prompt_chars if max_prompt_chars is not None else self.DEFAULT_MAX_PROMPT_CHARS
        )
        self.max_prompt_tokens = (
            max_prompt_tokens if max_prompt_tokens is not None else self.DEFAULT_MAX_PROMPT_TOKENS
        )
        self.max_tokens_cap = max_tokens_cap if max_tokens_cap is not None else self.DEFAULT_MAX_TOKENS_CAP
        self.max_stop_sequences = (
            max_stop_sequences if max_stop_sequences is not None else self.DEFAULT_MAX_STOP_SEQUENCES
        )
        self.max_messages = max_messages if max_messages is not None else self.DEFAULT_MAX_MESSAGES

    # -- validation ----------------------------------------------------------

    def validate_prompt(self, prompt: str) -> str | None:
        """Return an error string if the prompt is not acceptable, else None."""
        if not isinstance(prompt, str):
            return "prompt must be a string"
        if not prompt.strip():
            return "prompt must not be empty"
        if len(prompt) > self.max_prompt_chars:
            return f"prompt exceeds {self.max_prompt_chars} character limit"
        # strip embedded control chars that can break downstream pipelines
        if "\x00" in prompt:
            return "prompt contains null bytes"
        return None

    def validate_messages(self, messages: list) -> str | None:
        if not isinstance(messages, list) or not messages:
            return "messages must be a non-empty list"
        if len(messages) > self.max_messages:
            return f"messages exceeds {self.max_messages} limit"
        allowed_roles = {"system", "user", "assistant", "tool"}
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                return f"messages[{i}] must be an object"
            role = msg.get("role")
            if not isinstance(role, str) or role not in allowed_roles:
                return f"messages[{i}] has invalid role {role!r}"
            content = msg.get("content")
            if content is not None and not isinstance(content, (str, list)):
                return f"messages[{i}].content must be a string or list"
        return None

    def validate_max_tokens(self, max_tokens: int) -> str | None:
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
            return "max_tokens must be an integer"
        if max_tokens < 1:
            return "max_tokens must be >= 1"
        if max_tokens > self.max_tokens_cap:
            return f"max_tokens exceeds server cap of {self.max_tokens_cap}"
        return None

    def validate_sampling(self, temperature: float, top_p: float, top_k: int) -> str | None:
        try:
            if not (-2.0 <= float(temperature) <= 5.0):
                return "temperature out of safe range"
            if not (0.0 <= float(top_p) <= 1.0):
                return "top_p must be in [0, 1]"
            if not (0 <= int(top_k) <= 500):
                return "top_k must be in [0, 500]"
        except (TypeError, ValueError):
            return "sampling parameters must be numeric"
        return None

    def validate_stop_sequences(self, stop: list) -> str | None:
        if not isinstance(stop, list):
            return "stop must be a list"
        if len(stop) > self.max_stop_sequences:
            return f"stop exceeds {self.max_stop_sequences} limit"
        for i, s in enumerate(stop):
            if not isinstance(s, str) or not s or len(s) > self.DEFAULT_MAX_STOP_CHARS:
                return f"stop[{i}] must be a non-empty string <= {self.DEFAULT_MAX_STOP_CHARS} chars"
        return None

    # -- sanitisation --------------------------------------------------------

    @staticmethod
    def sanitize_tool_schemas(tools: list[Any]) -> tuple[list[dict[str, Any]], str | None]:
        """Reduce an incoming ``tools`` array to plain dicts of safe
        primitives. Returns (sanitised, error_string | None)."""
        if tools is None:
            return [], None
        if not isinstance(tools, list):
            return [], "tools must be a list"
        out: list[dict[str, Any]] = []
        for i, t in enumerate(tools):
            if not isinstance(t, dict):
                return [], f"tools[{i}] must be an object"
            safe, err = _sanitize_value(t)
            if err:
                return [], f"tools[{i}]: {err}"
            out.append(safe)
        return out, None


def _sanitize_value(value: Any) -> tuple[Any, str | None]:
    """Recursively restrict a JSON value to safe primitive types."""
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if not isinstance(k, str):
                return None, f"non-string dict key: {type(k).__name__}"
            safe, err = _sanitize_value(v)
            if err is not None:
                return None, err
            out[k] = safe
        return out, None
    if isinstance(value, list):
        out = []
        for v in value:
            safe, err = _sanitize_value(v)
            if err is not None:
                return None, err
            out.append(safe)
        return out, None
    # NaN floats are JSON-invalid; map to 0.0 before accepting numbers
    if isinstance(value, float) and math.isnan(value):
        return 0.0, None
    if isinstance(value, ALLOWED_SCHEMA_TYPES):
        return value, None
    return None, f"unsafe value type: {type(value).__name__}"


_IDENT = re.compile(r"^[A-Za-z0-9_.:-]+$")


def is_safe_trace_id(value: str) -> bool:
    """Guard against log-injection via trace ids (must be plain alphanum)."""
    return bool(value) and len(value) <= 64 and _IDENT.match(value) is not None
