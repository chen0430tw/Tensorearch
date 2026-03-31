# Oscillator Trace Checklist

Use this checklist when constructing or validating a Tensorearch trace for an Oscillator-style architecture.

Oscillator architectures differ from standard Transformers in that they use **phase encoding**, **topology propagation**, and **mixed-local routing** rather than standard QKV attention and dense FFN.

---

## System Fields

- [ ] `name` — descriptive string (e.g. `"oscillator-v1-trace"`)
- [ ] `model_arch` — set to `"oscillator"`
- [ ] `num_layers` — number of oscillator blocks
- [ ] `hidden_size` — embedding / phase-state dimension
- [ ] `intermediate_size` — FFN expansion dimension (if FFN slices are present)
- [ ] `num_heads` — number of oscillation heads / channels
- [ ] `kv_heads` — if asymmetric head count is used (often equal to `num_heads`)
- [ ] `batch_size` — sequences in batch
- [ ] `seq_len` — sequence / context length
- [ ] `dtype` — `"bf16"`, `"fp16"`, etc.
- [ ] `device` — accelerator type
- [ ] `tp_degree` — tensor-parallel degree
- [ ] `pp_degree` — pipeline-parallel degree
- [ ] `measured_latency_ms` — end-to-end wall clock (optional)
- [ ] `measured_tokens_per_sec` — throughput (optional)

---

## Slice Coverage

### Required per block

- [ ] `blkN.phase` — phase encoding slice
  - `kind: "phase"`, `op_type: "attn_qkv"`
  - encodes token-level oscillation state
  - `kv_bytes` required (oscillator maintains phase state similar to KV cache)
  - `sync_cost` elevated: phase synchronization across heads is expensive
  - `write_magnitude` > 1.0 typical: phase writes dominate downstream propagation

- [ ] `blkN.prop` — topology propagation slice
  - `kind: "propagation"`, `op_type: "attn_score"`
  - propagates phase state across the adjacency / coupling topology
  - highest `activation_bytes` in the block (propagation is memory-intensive)
  - `stall_ms` typically highest in the block (topology gather creates stalls)
  - TP communication self-loop edge is expected here

### Optional per block

- [ ] `blkN.ffn` — point-wise feed-forward (same as Transformer)
  - `kind: "ffn"`, `op_type: "ffn"`
  - present when Oscillator includes a local mixing step

- [ ] `blkN.local` — local routing gate
  - `kind: "router"`, `op_type: "router"`
  - governs mixed-local / learned routing decisions
  - `read_sensitivity` > 1.0 (gating quality is sensitive to incoming phase state)
  - should fan out to at least two downstream slices

- [ ] `blkN.mixed` — mixed-signal recombination
  - `kind: "propagation"`, `op_type: "attn_out"`
  - merges local-route and full-propagation signals
  - receives edges from both `blkN.local` and `blkN.ffn`

### Model boundaries

- [ ] `embed` — phase-state initialization (optional, if trace starts from tokens)
  - `kind: "embedding"`, `op_type: "embed"`
  - low flops, no kv_bytes

- [ ] `lm_head` — readout / phase-state projection to vocabulary
  - `kind: "readout"`, `op_type: "lm_head"`
  - `read_sensitivity` > 1.0

---

## Edge Coverage

### Standard block flow (no local routing)

- [ ] `embed → blk0.phase`
- [ ] `blkN.phase → blkN.prop`
- [ ] `blkN.prop → blkN.ffn`
- [ ] `blkN.ffn → blk(N+1).phase` (cross-block)
- [ ] `blk(L-1).phase → lm_head` or `blk(L-1).prop → lm_head`

### With local routing (mixed-local variant)

- [ ] `blkN.prop → blkN.local` (router fan-out, weight ≈ 0.5)
- [ ] `blkN.prop → blkN.ffn` (direct fan-out, weight ≈ 0.45)
- [ ] `blkN.local → blkN.mixed`
- [ ] `blkN.ffn → blkN.mixed`
- [ ] `blkN.mixed → blk(N+1).phase`

### Communication edges

- [ ] `blkN.prop → blkN.prop` (self-loop) — TP allreduce for propagation state
  - `collective: "allreduce"`, `same_device: false`, `same_stage: true`
  - `edge_bytes` larger than Transformer TP edges (propagation gathers more data)
  - weight typically 0.4–0.6

### Edge field checklist

- [ ] Router fan-out weights sum to approximately 1.0 across outgoing edges
- [ ] `edge_type: "attention"` for phase→prop edges
- [ ] `edge_type: "router"` for prop→local and local→mixed edges
- [ ] `edge_type: "residual"` for standard sequential flow
- [ ] `transport_scale` ≥ 1.2 for TP allreduce edges (communication overhead)

---

## Metric Plausibility Checks

After loading and inspecting the trace:

- [ ] `predicted_bottleneck` is one of the `prop` slices (propagation dominates in typical Oscillator)
- [ ] `prop` slices have higher `propagated_cost` than `phase` slices
- [ ] `local` (router) slice has `route_entropy` > 0 (it fans out)
- [ ] `global_coupling_efficiency` lower than equivalent Transformer (Oscillator graphs are sparser)
- [ ] `freedom_index` for `prop` slices > Transformer attention slices (topology propagation is less constrained)
- [ ] TP allreduce self-loops on `prop` appear in `top_edge_attributions`
- [ ] `compliance_index` for `local` router slices < `prop` slices (router has more structural freedom)

---

## Oscillator vs Transformer Comparison Expectations

When comparing an Oscillator trace to a Transformer trace of similar size:

| Metric | Expected Direction |
|---|---|
| `intelligence_delta` | positive (Oscillator typically higher II) |
| `coupling_delta` | negative (Oscillator graphs are sparser) |
| `obedience_delta` | variable (depends on routing configuration) |
| `left_predicted_bottleneck` | Transformer: attention; Oscillator: propagation |

---

## Common Mistakes

| Mistake | Fix |
|---|---|
| Missing `kv_bytes` on `phase` slices | Phase slices maintain oscillation state analogous to KV cache |
| `prop` slice has low `stall_ms` | Propagation is topology-gather-bound; stall_ms should be elevated |
| Router fan-out weights > 1.0 total | Normalize to sum ≈ 1.0 |
| No self-loop TP edge on `prop` | Oscillator topology gather requires cross-device reduce |
| `blkN.mixed` has a single incoming edge | Mixed slice should receive from both `local` and `ffn` paths |
| Same `local_vector_space` for `phase` and `prop` | These slices occupy different structural positions; vectors should differ |

---

## Reference: Oscillator-Specific Slice Kind Vocabulary

| kind | op_type | Description |
|---|---|---|
| `phase` | `attn_qkv` | phase-state encoding (QKV analog) |
| `propagation` | `attn_score` | topology propagation (attention-score analog) |
| `propagation` | `attn_out` | mixed-signal recombination (output-projection analog) |
| `router` | `router` | mixed-local routing gate |
| `ffn` | `ffn` | point-wise feed-forward |
| `embedding` | `embed` | token-to-phase initialization |
| `readout` | `lm_head` | phase-state to vocabulary projection |
