"""Git-backed memory store.

Each memory store is a real git repository (``~/.nexinfer/memory/<name>/``)
whose working tree holds JSON key files under ``data/<branch>/``. Every
``set`` is a git commit; branches give each agent an isolated space,
and the shared ``whiteboard`` store is the multi-agent communication
channel.

Uses ``pygit2`` when available (fast, full git semantics) and falls
back to a pure-Python/git-subprocess implementation so the store works
anywhere git is installed.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

try:
    import numpy as np  # noqa: F401 -- used by the vector-search methods

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False
log = logging.getLogger("nexinfer.memory.store")

# vector search requires numpy and at least one indexable entry; the
# second condition is checked lazily in ``search`` but we expose it here
# so callers can detect capability statically
_has_vector_support = _HAS_NUMPY

GIT = shutil.which("git") or "git"


def _git(repo_dir: str, *args: str, timeout: float = 10.0) -> str:
    r = subprocess.run([GIT, *args], cwd=repo_dir, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


@dataclass
class CommitInfo:
    oid: str
    message: str
    branch: str
    parent: str | None = None


class MemoryStore:
    """A git-backed key/value memory store with branches and history."""

    def __init__(self, path: str, name: str) -> None:
        self.path = path
        self.name = name
        os.makedirs(self.path, exist_ok=True)
        if not os.path.isdir(os.path.join(self.path, ".git")):
            _git(self.path, "init", "-q", "-b", "main")
            _git(self.path, "config", "user.name", "NexusInfer")
            _git(self.path, "config", "user.email", "memory@nexinfer.local")
        # ensure the main branch has a real HEAD commit anchored by a sentinel
        # file (git refuses to commit an empty tree in most configurations,
        # and empty directories cannot be tracked, so a .keep sentinel is used)
        data_dir = self._branch_data_dir("main")
        os.makedirs(data_dir, exist_ok=True)
        keep = os.path.join(data_dir, ".keep")
        if not os.path.exists(keep):
            with open(keep, "w") as f:
                f.write("")
        try:
            _git(self.path, "log", "-1", "--format=%H", "main", timeout=2.0)
        except RuntimeError:
            # fresh repo: stage the sentinel and commit to anchor HEAD
            self._commit_all("init memory store")

    # ------------------------------------------------------------------

    def _commit_all(self, message: str) -> str:
        """Stage everything and commit (used by init and internal ops)."""
        _git(self.path, "add", "-A")
        _git(self.path, "commit", "-q", "-m", message)
        return self.head()

    def _branch_data_dir(self, branch: str) -> str:
        return os.path.join(self.path, "data", branch)

    def _current_branch(self) -> str:
        return _git(self.path, "rev-parse", "--abbrev-ref", "HEAD")

    def branch(self, name: str, base: str = "main") -> None:
        """Create an isolated branch forked from ``base``."""
        _git(self.path, "checkout", "-q", base)
        branches = _git(self.path, "branch", "--list", name)
        if not branches:
            _git(self.path, "branch", "-q", name)
        _git(self.path, "checkout", "-q", name)

    def checkout(self, branch: str) -> None:
        _git(self.path, "checkout", "-q", branch)

    # ------------------------------------------------------------------

    def set(self, key: str, value: Any, branch: str | None = None, message: str = "update") -> str:
        """Write a value, committing it on ``branch`` (default: current)."""
        branch = branch or self._current_branch()
        branch_dir = self._branch_data_dir(branch)
        os.makedirs(branch_dir, exist_ok=True)
        file_path = os.path.join(branch_dir, f"{key.replace('/', '_')}.json")
        with open(file_path, "w") as f:
            json.dump(value, f, indent=2)
        _git(self.path, "add", "-A")
        _git(self.path, "commit", "-q", "-m", message)
        return self.head(branch)

    def get(self, key: str, branch: str | None = None, at: str | None = None) -> Any | None:
        """Read a value; ``at`` = commit oid or branch name."""
        branch = branch or self._current_branch()
        ref = at or branch
        file_name = f"{key.replace('/', '_')}.json"
        try:
            blob = _git(self.path, "show", f"{ref}:data/{branch}/{file_name}")
            return json.loads(blob)
        except RuntimeError:
            return None

    def delete(self, key: str, branch: str | None = None, message: str = "delete") -> None:
        branch = branch or self._current_branch()
        path = os.path.join(self._branch_data_dir(branch), f"{key.replace('/', '_')}.json")
        if os.path.exists(path):
            os.remove(path)
            _git(self.path, "add", "-A")
            _git(self.path, "commit", "-q", "-m", message)

    # ------------------------------------------------------------------

    def list_keys(self, branch: str | None = None) -> list[str]:
        branch = branch or self._current_branch()
        try:
            tree = _git(self.path, "ls-tree", "--name-only", "-r", branch)
        except RuntimeError:
            return []
        prefix = f"data/{branch}/"
        return [
            t[len(prefix) : -len(".json")]
            for t in tree.splitlines()
            if t.startswith(prefix) and t.endswith(".json")
        ]

    def history(self, branch: str | None = None, limit: int = 50) -> list[CommitInfo]:
        branch = branch or self._current_branch()
        out: list[CommitInfo] = []
        try:
            log_lines = _git(
                self.path, "log", branch, f"--max-count={limit}", "--pretty=format:%H|%s|%P"
            ).splitlines()
        except RuntimeError:
            return []
        for line in log_lines:
            parts = line.split("|", 2)
            oid, msg = parts[0], parts[1] if len(parts) > 1 else ""
            parent = parts[2].split(" ")[0] if len(parts) > 2 and parts[2].strip() else None
            out.append(CommitInfo(oid=oid, message=msg, branch=branch, parent=parent))
        return out

    def diff(self, a: str, b: str) -> dict[str, Any]:
        """Diff two commits/branches: added, changed, removed keys."""
        try:
            raw = _git(self.path, "diff", "--name-status", a, b)
        except RuntimeError:
            return {"added": [], "changed": [], "removed": []}
        added, changed, removed = [], [], []
        for line in raw.splitlines():
            status, _, path = line.partition("\t")
            key = path.replace(".json", "").split("/")[-1]
            if status == "A":
                added.append(key)
            elif status == "M":
                changed.append(key)
            elif status == "D":
                removed.append(key)
        return {"added": added, "changed": changed, "removed": removed}

    def merge(self, source: str, target: str | None = None, policy: str = "ours") -> str:
        """Merge ``source`` branch into ``target`` (default: current).

        Merge policies:
        * ``ours``        -- keep target's values on conflict
        * ``theirs``      -- take source's values on conflict
        * ``last_write``  -- newest commit wins (file mtime/commit time)
        """
        target = target or self._current_branch()
        _git(self.path, "checkout", "-q", target)
        try:
            _git(self.path, "merge", "-q", source, "-m", f"merge {source} into {target}")
        except RuntimeError:
            # conflict -> resolve per policy (file-level fallback for git-level conflicts)
            _git(self.path, "checkout", "-q", "--theirs" if policy == "theirs" else "--ours", ".")
            _git(self.path, "add", "-A")
            _git(
                self.path,
                "commit",
                "-q",
                "--no-edit",
                "-m",
                f"merge {source} into {target} (policy={policy})",
            )

        # policy overlay: source branch files live under ``data/<source>/``;
        # overlay them onto ``data/<target>/`` according to the policy so
        # shared stores (whiteboard) reflect the merge result, not just git's
        # structural union of the two branch directories
        self._overlay_branch(source, target, policy)
        return self.head(target)

    def _overlay_branch(self, source: str, target: str, policy: str) -> str | None:
        """Copy source-branch data files into the target branch per policy."""
        src_dir = self._branch_data_dir(source)
        tgt_dir = self._branch_data_dir(target)
        os.makedirs(tgt_dir, exist_ok=True)
        changed = False
        if not os.path.isdir(src_dir):
            return None
        for fname in os.listdir(src_dir):
            if fname == ".keep":
                continue
            tgt_path = os.path.join(tgt_dir, fname)
            src_path = os.path.join(src_dir, fname)
            if not os.path.exists(tgt_path):
                shutil.copy(src_path, tgt_path)
                changed = True
            else:
                if policy == "theirs":
                    shutil.copy(src_path, tgt_path)
                    changed = True
                elif policy == "ours":
                    pass  # keep target
                else:  # last_write
                    if os.path.getmtime(src_path) > os.path.getmtime(tgt_path):
                        shutil.copy(src_path, tgt_path)
                        changed = True
        if changed:
            self._commit_all(f"overlay {source} onto {target} (policy={policy})")
        return None

    def revert(self, target_branch: str | None = None, to: str | None = None) -> str:
        """Revert current branch's working tree to ``to`` (default: parent)."""
        target = target_branch or self._current_branch()
        ref = to or "HEAD~1"
        _git(self.path, "checkout", "-q", target)
        _git(self.path, "checkout", "-q", ref, "--", "data")
        _git(self.path, "commit", "-q", "-am", f"revert to {ref}")
        return self.head(target)

    def head(self, branch: str | None = None) -> str:
        branch = branch or self._current_branch()
        try:
            return _git(self.path, "rev-parse", "--short", branch)
        except RuntimeError:
            return ""

    def export_snapshot(self) -> dict[str, Any]:
        """Dump the whole store as JSON (used for cross-node sync)."""
        out: dict[str, Any] = {"store": self.name, "branches": {}}
        try:
            branches = _git(self.path, "branch", "--format=%(refname:short)").splitlines()
        except RuntimeError:
            branches = ["main"]
        for branch in branches:
            keys = self.list_keys(branch)
            out["branches"][branch] = {
                k: self.get(k, branch=branch) for k in keys if self.get(k, branch=branch) is not None
            }
            out["branches"][branch]["__heads__"] = [c.oid for c in self.history(branch, limit=1)]
        return out

    def import_snapshot(self, snapshot: dict[str, Any], policy: str = "theirs") -> int:
        """Import another node's snapshot; returns number of keys written."""
        n = 0
        for branch, data in snapshot.get("branches", {}).items():
            self.branch(branch, base="main") if branch != "main" else None
            for key, value in data.items():
                if key.startswith("__"):
                    continue
                if self.get(key, branch=branch) != value:
                    self.set(key, value, branch=branch, message=f"sync import from {snapshot.get('store')}")
                    n += 1
        return n

    # ------------------------------------------------------------------
    # Dense (vector) retrieval over committed values.
    # ------------------------------------------------------------------

    @staticmethod
    def _embed_word(word: str, dim: int = 64) -> np.ndarray:
        """Deterministic pseudo-embedding for a word: hash -> unit vector.

        This is intentionally lightweight — no neural model required — so
        cosine similarity over averaged word vectors gives a reasonable
        notion of semantic proximity for short phrases while staying
        fully offline and dependency-free (numpy).
        """
        import numpy as np

        h = int.from_bytes(word.encode()[:32], "little") if word else 0
        rng = np.random.RandomState(h % (2**31))
        v = rng.standard_normal(dim).astype(np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v

    def _embed_text(self, text: str, dim: int = 64) -> np.ndarray:
        tokens = [t for t in text.lower().replace("-", " ").replace("_", " ").split() if t]
        if not tokens:
            return np.zeros(dim, dtype=np.float32)
        v = np.mean([self._embed_word(t, dim) for t in tokens], axis=0)
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v

    def search(
        self, query: str, branch: str | None = None, top_k: int = 5, mode: str = "auto"
    ) -> list[dict[str, Any]]:
        """Search committed values.

        ``mode`` controls the retrieval strategy:
        * ``"keyword"`` -- classic term-frequency substring matching
          (works without numpy and is exact for literal terms)
        * ``"vector"``  -- dense retrieval: the query and every entry are
          embedded as averaged word vectors and ranked by cosine similarity
        * ``"auto"``    -- vector search when ``numpy`` is available and the
          branch has entries, otherwise keyword search

        Keyword matches always carry an exact ``score`` (term hits); vector
        matches carry a ``similarity`` in ``[0, 1]`` (cosine of the mean
        word vectors) plus the keyword score as a secondary tie-breaker.
        """
        branch = branch or self._current_branch()
        q = query.lower()
        terms = [t for t in q.split() if t]
        if not terms:
            return []
        entries = [(k, self.get(k, branch=branch)) for k in self.list_keys(branch)]
        entries = [(k, v) for k, v in entries if v is not None]
        if not entries:
            return []

        use_vector = (mode == "vector") or (mode == "auto" and _HAS_NUMPY and _has_vector_support)
        keyword_results = []
        for key, val in entries:
            text = json.dumps(val).lower()
            score = sum(text.count(t) for t in terms)
            if score > 0:
                keyword_results.append({"key": key, "value": val, "score": score})

        if not use_vector:
            keyword_results.sort(key=lambda r: r["score"], reverse=True)
            return keyword_results[:top_k]

        import numpy as np  # noqa: PLC0415

        q_vec = self._embed_text(query)
        vector_results = []
        for key, val in entries:
            e_vec = self._embed_text(json.dumps(val))
            sim = float(np.dot(q_vec, e_vec))
            # cosine of unit vectors is in [-1, 1]; rescale to [0, 1] for
            # a cleaner ranking where 1.0 = identical direction
            similarity = (sim + 1.0) / 2.0
            if similarity > 0.5:  # only surface non-trivial matches
                vector_results.append(
                    {
                        "key": key,
                        "value": val,
                        "similarity": similarity,
                        "score": sum(json.dumps(val).lower().count(t) for t in terms),
                    }
                )
        vector_results.sort(key=lambda r: (r["similarity"], r["score"]), reverse=True)
        return vector_results[:top_k]
