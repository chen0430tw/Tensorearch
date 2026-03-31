# Transformer Trace Checklist

Use this checklist when constructing or validating a Tensorearch trace for a standard Transformer architecture.

---

## System Fields

- [ ] `name` — descriptive string (e.g. `"llama3-8b-trace"`)
- [ ] `model_arch` — set to `"transformer"`
- [ ] `num_layers` — total decoder layers
- [ ] `hidden_size` — model dimension (e.g. 4096)
- [ ] `intermediate_size` — FFN expansion dimension (e.g. 11008 for LLaMA-style 2.7× ratio)
- [ ] `num_heads` — total attention heads
- [ ] `kv_heads` — KV heads (= `num_heads` for MHA, < `num_heads` for GQA/MQA)
- [ ] `batch_size` — number of sequences in batch
- [ ] `seq_len` — sequence length
- [ ] `dtype` — `"bf16"`, `"fp16"`, `"fp8"`, etc.
- [ ] `device` — `"H100"`, `"A100"`, `"V100"`, etc.
- [ ] `tp_degree` — tensor-parallel degree (1 = no TP)
- [ ] `pp_degree` — pipeline-parallel degree (1 = no PP)
- [ ] `measured_latency_ms` — wall-clock end-to-end latency (optional but valuable)
- [ ] `measured_tokens_per_sec` — throughput measurement (optional)

---

## Slice Coverage

A complete Transformer trace should include one or more of these slice kinds per layer:

### Required per layer

- [ ] `blkN.attn` — self-attention block
  - `kind: "attention"`, `op_type: "attn"`
  - fields: `hidden_size`, `num_heads`, `kv_heads`, `kv_bytes`, `tokens_in`, `tokens_out`
  - realistic flops: ~6–8 per layer (relative units), rising slightly with depth

- [ ] `blkN.ffn` — feed-forward block
  - `kind: "ffn"`, `op_type: "ffn"`
  - fields: `hidden_size`, `intermediate_size`, `tokens_in`, `tokens_out`
  - realistic flops: ~5–7, with `weight_bytes` > `activation_bytes` for large FFN

### Optional per layer (MoE and routing)

- [ ] `blkN.router` — routing gate for MoE
  - `kind: "router"`, `op_type: "router"`
  - `read_sensitivity` > 1.0 (routers are sensitive to incoming residual quality)
  - should have **two outgoing edges** with weights summing to ~1.0

- [ ] `blkN.expert` — sparse expert block
  - `kind: "expert"`, `op_type: "ffn_up"`
  - `tokens_in` / `tokens_out` ≈ half of full batch (top-1 routing)
  - `weight_bytes` larger than standard FFN (larger intermediate dimension)

### Model boundaries

- [ ] `embed` — embedding slice (layer_index: -1)
  - `kind: "embedding"`, `op_type: "embed"`
  - low flops, no kv_bytes, low sync_cost

- [ ] `lm_head` — readout/unembedding
  - `kind: "readout"`, `op_type: "lm_head"`
  - `read_sensitivity` > 1.0 (high sensitivity to final hidden state quality)

---

## Edge Coverage

- [ ] `embed → blk0.attn` — residual dataflow entry
- [ ] `blkN.attn → blkN.ffn` — within-block residual
- [ ] `blkN.ffn → blk(N+1).attn` — cross-block residual
- [ ] `blkN.router → blkN.ffn` and `blkN.router → blkN.expert` — routing fan-out (MoE)
- [ ] `blk(L-1).attn → lm_head` — final readout edge
- [ ] `blkN.attn → blkN.attn` (self-loop) — TP allreduce communication edge

### Edge field checklist

For each edge:
- [ ] `src` and `dst` match valid `slice_id` values
- [ ] `weight` is in [0, 1] range for dataflow edges
- [ ] `edge_type` is one of: `residual`, `attention`, `router`, `readout`, `tp_comm`, `pp_comm`
- [ ] `edge_bytes` is a positive float (communication volume)
- [ ] `transport_scale` ≥ 1.0 for cross-device edges
- [ ] `collective` is `"none"` for intra-device, `"allreduce"` / `"allgather"` / `"reduce_scatter"` for TP
- [ ] `same_device: false` for TP communication edges
- [ ] `same_stage: true` unless crossing a PP boundary

---

## Metric Plausibility Checks

After loading and inspecting the trace, check:

- [ ] `predicted_bottleneck` is one of the attention or FFN slices (not embed or lm_head in typical cases)
- [ ] `global_coupling_efficiency` in [0.05, 0.5] — very high (>0.5) suggests over-coupling; very low (<0.05) suggests a thin graph
- [ ] `freedom_index` for attention slices > FFN slices (attention has more routing freedom)
- [ ] `compliance_index` for communication slices ≈ near-max (protocol-bound)
- [ ] `route_entropy` > 0 for router slices (they should fan out)
- [ ] `route_entropy` ≈ 0 for embed slice (single outgoing edge)
- [ ] TP allreduce edges appear in `top_edge_attributions` when `tp_degree > 1`

---

## Common Mistakes

| Mistake | Fix |
|---|---|
| `tokens_in = seq_len` instead of `batch_size × seq_len` | multiply both dimensions |
| `kv_bytes = 0` for attention slices | estimate from seq_len × kv_heads × head_dim × 2 (K+V) × dtype_bytes |
| Router fan-out weights summing > 1.0 | normalize to ≈ 1.0 total outgoing weight |
| `same_device: true` on a TP allreduce edge | set `same_device: false` |
| Missing `lm_head` slice | always include the final readout slice |
| All slices have identical `local_vector_space` | derive from actual profiling or use type-differentiated defaults |

---

## Reference: Slice Kind Vocabulary

| kind | op_type | Description |
|---|---|---|
| `embedding` | `embed` | token embedding lookup |
| `attention` | `attn` | full self-attention |
| `attention` | `attn_qkv` | QKV projection only |
| `attention` | `attn_score` | attention score computation |
| `attention` | `attn_out` | attention output projection |
| `ffn` | `ffn` | full FFN block |
| `ffn` | `ffn_up` | FFN up-projection |
| `ffn` | `ffn_down` | FFN down-projection |
| `router` | `router` | MoE routing gate |
| `expert` | `ffn_up` | sparse expert block |
| `readout` | `lm_head` | unembedding / logit projection |
| `comm` | `allreduce` | TP all-reduce |
| `comm` | `allgather` | TP all-gather |
| `comm` | `reduce_scatter` | TP reduce-scatter |
