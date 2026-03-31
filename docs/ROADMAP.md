# Tensorearch Roadmap

## Goal

Tensorearch is an architecture-inspection system for large models.

It should answer four classes of questions:

1. throughput
2. bottlenecks
3. compliance vs covert deviation
4. architecture intelligence / adaptive structure

The project is not just a math note and not just a profiler.
It should become a trace-driven inspection toolkit with explicit structural metrics.

---

## Phase Map

| Phase | Status | Purpose |
|---|---|---|
| P0. Mathematical base | in progress | define objects, metrics, propagation, entropy |
| P1. Trace schema + ingestion | in progress | accept real model slice traces |
| P2. Feature extraction | in progress | map Transformer/Oscillator structure into local vectors and transport terms |
| P3. Core metrics engine | in progress | compute bottleneck/compliance/intelligence metrics |
| P4. Intervention engine | planned | support ablation, rerouting, edge masking, topology swaps |
| P5. Real model adapters | planned | ingest Transformer / Oscillator traces |
| P6. Comparison reports | planned | architecture-vs-architecture report generation |
| P7. Validation loop | planned | correlate predictions with measured throughput and quality |

---

## Functional Plan

## 1. Trace Schema

### 1.1 System-Level Fields

- `name`
- `model_arch`
- `num_layers`
- `hidden_size`
- `intermediate_size`
- `num_heads`
- `kv_heads`
- `batch_size`
- `seq_len`
- `dtype`
- `device`
- `tp_degree`
- `pp_degree`
- `measured_latency_ms`
- `measured_tokens_per_sec`
- `metadata`

### 1.2 Slice-Level Fields

- `slice_id`
- `layer_index`
- `kind`
- `op_type`
- `hidden_size`
- `intermediate_size`
- `num_heads`
- `kv_heads`
- `tokens_in`
- `tokens_out`
- `flops`
- `activation_bytes`
- `weight_bytes`
- `kv_bytes`
- `memory_bytes`
- `comm_bytes`
- `sync_cost`
- `kernel_time_ms`
- `stall_ms`
- `measured_latency_ms`
- `write_magnitude`
- `read_sensitivity`
- `doi_alignment`
- `obedience_target`
- `local_vector_space`
- `metadata`

### 1.3 Edge-Level Fields

- `src`
- `dst`
- `kind`
- `edge_type`
- `weight`
- `edge_bytes`
- `transport_scale`
- `collective`
- `same_device`
- `same_stage`
- `metadata`

### 1.4 Intervention-Level Fields

- `kind`
- `target`
- `value`
- `metadata`

---

## 2. Feature Extraction

### 2.1 Slice Local Vector Spaces

Need explicit mapping rules for:

- `embed`
- `attn`
- `attn_qkv`
- `attn_score`
- `attn_out`
- `ffn`
- `ffn_up`
- `ffn_down`
- `router`
- `expert`
- `kv_write`
- `kv_read`
- `allreduce`
- `allgather`
- `reduce_scatter`
- `tp_comm`
- `pp_comm`
- `lm_head`

### 2.2 Attention Slice Features

- qkv flops density
- kv-cache pressure
- head / kv-head asymmetry
- synchronization pressure
- communication-to-memory ratio
- write-to-read ratio
- context sensitivity
- attention sparsity when available

### 2.3 FFN Slice Features

- expansion ratio
- token-normalized flops
- weight / activation ratio
- memory / latency ratio
- stall / latency ratio
- write-to-read ratio
- gating sparsity when available

### 2.4 Communication Slice Features

- edge-bytes log scale
- bytes per token
- sync / latency ratio
- stall / latency ratio
- collective burden
- cross-device penalty
- cross-stage penalty

### 2.5 Edge Transport Estimation

Need a stable formula using:

- `edge_bytes`
- `collective`
- `same_device`
- `same_stage`
- `edge_type`
- optional measured transfer time

---

## 3. Core Metrics

### 3.1 Bottleneck / Throughput

- `base_cost`
- `topological_congestion`
- `propagated_cost`
- `slice_bottleneck_index`
- `global_coupling_efficiency`
- `predicted_bottleneck`

### 3.2 TDB-Inspired Metrics

- `direction_of_interest`
- `direct_effect`
- `estimated_total_effect`
- `edge_attribution`

### 3.3 Compliance Metrics

- `freedom_index`
- `compliance_index`
- `global_obedience_score`

### 3.4 Dynamical Entropy Metrics

- `route_entropy`
- `effect_entropy`
- `compliance_entropy`
- `global_entropy_summary`

### 3.5 Intelligence Metrics

- `slice_intelligence_index`
- `global_intelligence_score`
- `adaptive_vs_deceptive_regime`

---

## 4. Algorithm Upgrades Still Needed

### 4.1 Better Weight-Chain Model

Current version is acceptable as prototype but still simple.
Need future upgrades:

- residual-path accumulation
- skip-connection reweighting
- multi-edge competing path normalization
- expert routing sparsity support
- learned-vs-fixed routing comparison support

### 4.2 Better Propagation

Current OTAL-like propagation is single-state and linear-ish.
Need:

- multi-channel propagation
- local/global propagation split
- residual-stabilized propagation
- path attenuation
- congestion backpressure term

### 4.3 Better Smartness Model

Current intelligence index is a prototype.
Need:

- separate "adaptive freedom" from "deceptive freedom"
- path-consistency score
- policy-divergence score
- hidden reroute detection

---

## 5. Intervention Engine

Must support:

- remove a slice
- zero an edge
- reduce edge bandwidth
- swap learned routing to fixed local routing
- freeze a subgraph
- reduce communication scale
- replace topology with identity / local / mixed

Outputs:

- delta latency
- delta throughput
- delta bottleneck
- delta compliance
- delta intelligence

---

## 6. Real Model Adapters

### 6.1 Transformer Adapter

Need trace extraction for:

- embedding
- attention
- FFN
- residual stream transitions
- tensor-parallel comm
- readout

### 6.2 Oscillator Adapter

Need trace extraction for:

- phase encoding
- adjacency builder
- topology propagation
- phase readout
- FFN
- mixed-local routing
- communication edges

### 6.3 Shared Adapter Features

- wall-clock timing
- CUDA kernel timing when available
- memory allocation / reserved / peak
- per-slice bytes estimates
- route metadata

---

## 7. Report Generation

### 7.1 Single-Run Report

Should output:

- system summary
- bottleneck ranking
- top edge attribution
- freedom / compliance summary
- intelligence summary

### 7.2 Comparative Report

Should output:

- model A vs model B
- bottleneck differences
- coupling pattern differences
- compliance / intelligence differences
- throughput prediction differences

### 7.3 Paper-Style Export

Need exportable tables/figures for:

- bottleneck trajectory
- entropy trajectory
- coupling efficiency
- intelligence vs throughput plots

---

## 8. Validation Plan

### 8.1 Internal Validation

- unit tests
- schema tests
- feature extraction tests
- deterministic intervention tests

### 8.2 Empirical Validation

- correlate predicted bottleneck with real latency
- correlate edge attribution with intervention impact
- compare Transformer vs Oscillator
- compare local vs mixed vs learned topology

### 8.3 Failure Modes to Track

- metric collapse to one dominant slice
- entropy trivially zero because graph is too thin
- compliance inflated by poor target inference
- intelligence score rewarding chaos

---

## 9. CLI / UX Plan

Need commands like:

- `tensorearch inspect trace.json`
- `tensorearch compare a.json b.json`
- `tensorearch ablate trace.json --slice blk0.attn`
- `tensorearch reroute trace.json --mode local_window`
- `tensorearch export trace.json --format markdown`

---

## 10. Immediate Build Order

This is the concrete order to stop incremental drift:

1. finalize schema and feature mapping
2. add richer sample traces for branching graphs
3. build intervention engine
4. build comparison engine
5. ingest real Transformer trace
6. ingest real Oscillator trace
7. run first real architecture comparison
8. generate first paper-style report

---

## 11. Definition of Done for v0.1

Tensorearch v0.1 is done only if it can:

- load real Transformer trace JSON
- load real Oscillator trace JSON
- compute bottleneck / compliance / intelligence metrics
- perform at least one intervention
- produce a comparison report
- explain why one architecture is faster or slower in explicit structural terms
