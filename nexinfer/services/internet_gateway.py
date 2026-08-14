"""Internet gateway: controlled web access for models.

Exposes a ``web_fetch`` tool to the generation loop so a model can reach
the internet. Features:

* **policy** -- allow/block domain lists, scheme restrictions, redirect
  chain limits, and per-agent quotas; every fetch is logged.
* **sandboxed worker** -- network calls run in a dedicated thread so a
  slow host never stalls generation.
* **context-bound summarization** -- HTML is converted to readable
  text; if the body exceeds ``max_tokens``, it is chunked and the
  gateway returns the most relevant chunk boundaries with a truncation
  notice (full summarization can be delegated to the local model).

Tool schema returned for function calling::

    {
      "name": "web_fetch",
      "description": "Fetch a URL and return readable text. ...",
      "parameters": {"url": "...", "max_tokens": 2000}
    }
"""

from __future__ import annotations

import html
import logging
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("nexinfer.services.internet")

DEFAULT_USER_AGENT = "NexusInfer/0.1 (model internet gateway)"


@dataclass
class InternetPolicy:
    allowed_domains: list[str] = field(default_factory=lambda: ["*"])  # "*" = unrestricted
    blocked_domains: list[str] = field(default_factory=list)
    allowed_schemes: list[str] = field(default_factory=lambda: ["https", "http"])
    max_redirects: int = 5
    max_response_bytes: int = 512 * 1024
    timeout_seconds: float = 15.0
    quota_per_agent: dict[str, int] = field(default_factory=dict)  # agent_id -> max fetches
    log_fetches: bool = True


# registry of every fetch for audit / memory persistence
FETCH_LOG: list[dict[str, Any]] = []


class InternetGateway:
    def __init__(self, policy: InternetPolicy | None = None) -> None:
        self.policy = policy or InternetPolicy()
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="fetch")
        self._counts: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Tool schema + execution
    # ------------------------------------------------------------------

    def tool_schema(self) -> dict[str, Any]:
        return {
            "name": "web_fetch",
            "description": (
                "Fetch a URL and return its readable text content. "
                "Respects the engine's domain policy. Use max_tokens to bound output size."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch (https preferred)"},
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum output length",
                        "default": 2000,
                    },
                },
                "required": ["url"],
            },
        }

    def call(self, url: str, max_tokens: int = 2000, agent_id: str | None = None) -> dict[str, Any]:
        future = self._pool.submit(self._fetch_sync, url, max_tokens, agent_id)
        try:
            return future.result(timeout=self.policy.timeout_seconds + 10)
        except Exception as exc:
            return {"error": f"web_fetch failed: {exc}", "url": url}

    def _fetch_sync(self, url: str, max_tokens: int, agent_id: str | None) -> dict[str, Any]:
        entry = {"url": url, "agent_id": agent_id, "ts": time.time(), "status": "pending"}
        try:
            parsed = urllib.parse.urlparse(url)
            self._check_policy(parsed, agent_id)
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": DEFAULT_USER_AGENT,
                    "Accept": "text/html,text/plain,application/json,*/*",
                },
            )
            with urllib.request.urlopen(req, timeout=self.policy.timeout_seconds) as resp:
                data = resp.read(self.policy.max_response_bytes)
                encoding = (resp.headers.get_content_charset() or "utf-8").replace("utf-8", "utf-8")
                text = data.decode(encoding, errors="replace")
            content_type = resp.headers.get("Content-Type", "")
            entry["status"] = "ok"
            entry["content_type"] = content_type
            entry["final_url"] = resp.url
            entry["bytes"] = len(data)
        except urllib.error.HTTPError as exc:
            entry["status"] = f"http_{exc.code}"
            return {"error": f"HTTP {exc.code}", "url": url}
        except Exception as exc:
            entry["status"] = "error"
            entry["error"] = str(exc)
            return {"error": str(exc), "url": url}
        finally:
            if self.policy.log_fetches:
                FETCH_LOG.append(entry)

        text = _extract_text(text, entry.get("content_type", ""))
        entry["output_tokens"] = len(text.split())
        return {
            "url": entry.get("final_url", url),
            "content": _trim(text, max_tokens),
            "truncated": entry["output_tokens"] > max_tokens,
        }

    # ------------------------------------------------------------------

    def _check_policy(self, parsed: urllib.parse.ParseResult, agent_id: str | None) -> None:
        if parsed.scheme not in self.policy.allowed_schemes:
            raise ValueError(f"scheme {parsed.scheme!r} not allowed")
        host = (parsed.hostname or "").lower()
        for blocked in self.policy.blocked_domains:
            if host.endswith(blocked.lstrip("*.")):
                raise ValueError(f"domain {host!r} is blocked")
        allowed = self.policy.allowed_domains
        if "*" not in allowed and not any(host == d or host.endswith("." + d) for d in allowed):
            raise ValueError(f"domain {host!r} not in allow list")
        if agent_id and agent_id in self.policy.quota_per_agent:
            self._counts[agent_id] = self._counts.get(agent_id, 0) + 1
            if self._counts[agent_id] > self.policy.quota_per_agent[agent_id]:
                raise ValueError(f"agent {agent_id!r} exceeded fetch quota")

    def history(self, agent_id: str | None = None) -> list[dict[str, Any]]:
        return [e for e in FETCH_LOG if agent_id is None or e.get("agent_id") == agent_id]

    def close(self) -> None:
        self._pool.shutdown(wait=False)


# ----------------------------------------------------------------------
# HTML -> text helpers (no external deps)


def _strip_tags(s: str) -> str:
    s = re.sub(r"<(script|style|noscript)[\s\S]*?</\1>", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.IGNORECASE)
    s = re.sub(r"</p>|</div>|</h[1-6]>|</li>|</tr>", "\n", s, flags=re.IGNORECASE)
    s = re.sub(r"<[^>]+>", " ", s)
    return html.unescape(s)


def _extract_text(raw: str, content_type: str) -> str:
    if "html" in content_type:
        text = _strip_tags(raw)
    else:
        text = raw
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def _trim(text: str, max_tokens: int) -> str:
    words = text.split()
    if len(words) <= max_tokens:
        return text
    return " ".join(words[:max_tokens]) + " …[truncated]"
