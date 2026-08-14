# Cluster Guide

Run the same model across several machines — even different ones, as long as
the architecture and quantization match — and let NexusInfer split the work
between them.

## 1. Concepts

| Term | Meaning |
|---|---|
| coordinator | rank-0 node: registers peers, pushes the parallel plan, relays logits |
| worker | a node that owns a layer range (PP) or head slice (TP) |
| pipeline parallelism (PP) | contiguous layers on different nodes; activations cross the network |
| tensor parallelism (TP) | attention heads / FFN columns split across ranks; needs homogeneous hardware and divides `num_attention_heads` |
| same-model constraint | every node advertises a `model_hash`; mismatching hashes are warned because output would be silently wrong |
| transport | the data path between ranks: `tcp`, `grpc`, `webrtc`, `rdma` |

The planner chooses automatically: homogeneous clusters with divisible head
counts get **TP**, everything else gets **PP**. Override with
`--parallel-mode` (a flag wired through the coordinator).

## 2. LAN cluster (simplest)

Two machines on the same network, both running the same model build:

```bash
# node 1 — coordinator
nexinfer cluster coordinator --node-id node-0 --port 9000

# node 2 — worker
nexinfer cluster worker --node-id node-1 --host 0.0.0.0 \
                        --port 9000 --peers 10.0.0.1:9000
```

LAN nodes also announce themselves via mDNS, so `--peers` is technically
optional on the same subnet; manual peers are the robust choice.

## 3. Cross-network cluster (different machines, different locations)

Nodes behind NAT cannot use raw TCP. Use the **WebRTC** transport with the
bundled SDP signaling server (put the signaling server anywhere reachable by
both nodes):

```bash
# anywhere public (or on one of the nodes, port-forwarded)
python -m nexinfer.distributed.signaling --port 8900

# both nodes
nexinfer cluster coordinator --node-id node-0 --transport webrtc \
    --signaling 1.2.3.4:8900
nexinfer cluster worker --node-id node-1 --transport webrtc \
    --signaling 1.2.3.4:8900
```

The signaling server only exchanges SDP offers/answers; all tensor traffic
then flows peer-to-peer, optionally relayed through the default public STUN
servers. For fully offline environments, paste the SDP strings manually with
`--offer`/`--answer`.

## 4. Transport comparison

| Transport | Setup effort | NAT traversal | Throughput | Best for |
|---|---|---|---|---|
| tcp | none | no | good | LAN clusters |
| grpc | none | no | best (multiplexed) | busy LAN clusters |
| webrtc | signaling server or SDP paste | yes | good (DTLS overhead) | cross-network, p2p |
| rdma | `rdma link` shows RoCE/IB | n/a (fabric only) | excellent (zero-copy) | datacenter fabrics; falls back to TCP elsewhere |

## 5. Scaling advice

- **Heterogeneous hardware** (laptop + desktop GPU): pipeline parallelism is
  the natural fit — give the GPU the larger layer range; the planner does
  this automatically via compute scores.
- **Identical GPUs**: tensor parallelism halves per-node activation traffic
  once head counts divide evenly.
- **Slow network** is the killer for PP across the internet; prefer serving
  smaller models there, or keep distribution inside one LAN/region and use
  the MCP gateway (below) to coordinate remote agents instead of splitting
  one model.
- **Membership changes**: the coordinator recomputes and re-pushes the plan
  on every join/leave; workers re-map their layer ranges on `plan` messages.

## 6. Observing the cluster

```bash
nexinfer cluster coordinator --node-id node-0 -v   # verbose plan logs
```

The coordinator logs the elected mode, per-node layer/head assignments, and
heartbeat timeouts (a node silent for 30 s is removed and the plan is
recomputed).

## 7. Beyond splitting one model

When nodes are on different networks with very different hardware, the more
practical pattern is often many independent NexusInfer nodes exposing MCP
servers and coordinating through the shared memory whiteboard (see
[memory protocol](memory-protocol.md)). The cluster machinery is for
splitting a single model; the MCP + memory fabric is for coordinating many
models/agents of any kind.
