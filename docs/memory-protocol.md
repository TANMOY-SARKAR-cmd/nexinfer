# Memory Protocol

NexusInfer gives every model or agent a git-backed memory fabric: versioned,
branch-isolated, mergeable, and syncable across machines. This document
describes the model, the commands, and how multi-agent communication works.

## 1. Storage model

Each memory **store** is a git repository at `~/.nexinfer/memory/<name>/`.
Values are JSON files under `data/<branch>/<key>.json`; every write is a
commit. The git history is the memory's full audit trail.

| Concept | Git equivalent | Agent meaning |
|---|---|---|
| store | repository | a namespace (e.g. `notes`, `whiteboard`, `agent-alice`) |
| branch | branch | one agent's or task's isolated workspace |
| commit | commit | one memory write, with message |
| merge | merge | combining agents' work with a conflict policy |
| snapshot | export/import JSON | syncing a store between machines |

## 2. Isolated vs shared space

- **private store** (`kind=private`): only listed owners can write. Perfect
  for an agent's scratch space.
- **shared store** (`kind=shared`): anyone may read and write — the
  **whiteboard** is the canonical shared store, created automatically by
  `fabric.whiteboard()`.
- **grant store** (`kind=grant`): write access only via explicit grants.

```bash
nexinfer memory create agent-alice --kind private --owners alice
nexinfer memory grant agent-alice bob        # explicit hand-off
```

Two agents never see each other's private branches; communication happens
through shared stores, exactly the blackboard pattern used in multi-agent
systems.

## 3. Merge policies

When branches are merged, conflicts are resolved per store:

| Policy | Behavior |
|---|---|
| `ours` | target branch wins on conflict |
| `theirs` | source branch wins on conflict |
| `last_write` | newest commit wins (time-based) |

The whiteboard defaults to `last_write` — agents post, latest post visible.
Research stores usually want `theirs` (incoming results take precedence) or
a deliberate `ours`.

```bash
nexinfer memory merge agent-alice source main --policy theirs
```

## 4. The whiteboard (multi-model / agent communication)

The shared whiteboard store is how different models and agents coordinate
within one NexusInfer deployment — or, via snapshot sync, across machines:

```bash
nexinfer memory write whiteboard "alice:1" "hypothesis found"
nexinfer memory read   whiteboard --branch main
```

Through the MCP server these become `whiteboard_read` / `whiteboard_write`
tools any external agent can call; through the engine they are
`memory_write` / `memory_search` tools available to skills.

## 5. Sync between machines

Nodes of a cluster (or independent deployments) exchange a store's full
state as a JSON snapshot:

```bash
# machine A
nexinfer memory sync push notes /tmp/snap        # writes notes.snapshot.json

# machine B (same model / same deployment)
nexinfer memory sync pull notes /tmp/snap/notes.snapshot.json
```

Programmatic: `fabric.push_snapshot(name, dir)` / `fabric.pull_snapshot(name,
path)`, which call `MemoryStore.import_snapshot()` under the hood (default
import policy `theirs` — remote values win).

## 6. History, diff, undo

```bash
nexinfer memory history agent-alice --branch main
# commit a1b2c3: hypothesis draft
# commit d4e5f6: refined after web research

nexinfer memory list agent-alice --branch main   # keys
nexinfer memory read  agent-alice hypothesis     # value
```

Programmatic: `store.history()`, `store.diff(a, b)` (added/changed/removed
keys), `store.revert(branch, to=oid)`.

## 7. Searching memory

`store.search(query, top_k=5)` is currently a keyword/substring scorer over
committed JSON values — sufficient and dependency-free. For semantic
retrieval, plug a small embedding model into the backend registry and replace
the scorer with cosine similarity; the hook point is already there.

## 8. From the model's point of view

During generation, an agent with the `research` or `memory` skill sees three
tools:

- `memory_write(store, branch, key, value, message)` — commit
- `memory_read(store, branch, key)` — read
- `memory_search(store, query, branch, top_k)` — find

A typical research loop: write findings to its own branch as it goes, post a
summary to the whiteboard, and a second agent later reads the whiteboard and
merges the branch that looks most promising. Every step is a git commit, so
the whole reasoning trail is replayable.
