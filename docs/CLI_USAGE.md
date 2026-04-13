# Tensorearch CLI Usage

## Overview

Tensorearch is invoked as a Python module:

```
python -m tensorearch [--verbose] <command> [options]
```

All commands support `--json` for machine-readable output.

---

## Commands

### inspect

Inspect a single trace and print a report.

```bash
python -m tensorearch inspect <trace.json>
python -m tensorearch inspect <trace.json> --json
python -m tensorearch inspect <trace.json> --json --output report.json
```

**Arguments:**

| Argument | Required | Description |
|---|---|---|
| `trace` | yes | path to trace JSON file |
| `--json` | no | emit machine-readable JSON instead of human-readable text |
| `--output` | no | write report to this file path in addition to stdout |

**Human-readable output includes:**
- system summary (name, arch, latency, tokens/sec)
- predicted bottleneck slice
- global coupling efficiency
- global obedience score
- global intelligence score
- per-slice: base cost, propagated cost, bottleneck index, direct effect, estimated total effect, freedom index, compliance index, route entropy, effect entropy, intelligence index
- top edge attributions

**JSON output keys:**

```json
{
  "system": { "name": "...", "latency_ms": ..., "tokens_per_sec": ... },
  "summary": {
    "predicted_bottleneck": "blk1.attn",
    "global_coupling_efficiency": 0.24,
    "global_obedience_score": 0.41,
    "global_intelligence_score": 0.08
  },
  "slices": [
    {
      "slice_id": "blk0.attn",
      "kind": "attention",
      "op_type": "attn",
      "cost": 12.0,
      "propagated_cost": 17.2,
      "slice_bottleneck_index": 0.21,
      "topological_congestion": 5.1,
      "direct_effect": 15.3,
      "estimated_total_effect": 22.4,
      "freedom_index": 1.8,
      "compliance_index": 0.4,
      "route_entropy": 0.69,
      "effect_entropy": 0.65,
      "compliance_entropy": 0.99,
      "intelligence_index": 0.09,
      "propagated_state": 0.58
    }
  ],
  "top_edge_attributions": [
    { "src": "blk1.attn", "dst": "lm_head", "edge_attribution": 21.4 }
  ]
}
```

---

### compare

Compare two traces and print a delta report.

```bash
python -m tensorearch compare <left.json> <right.json>
python -m tensorearch compare <left.json> <right.json> --json
python -m tensorearch compare <left.json> <right.json> --json --output delta.json
```

**Arguments:**

| Argument | Required | Description |
|---|---|---|
| `left` | yes | first trace (baseline) |
| `right` | yes | second trace (variant) |
| `--json` | no | machine-readable JSON |
| `--output` | no | write report to file |

**JSON output keys:**

```json
{
  "left_name": "...",
  "right_name": "...",
  "left_predicted_bottleneck": "blk1.attn",
  "right_predicted_bottleneck": "blk0.prop",
  "left_obedience": 0.41,
  "right_obedience": 0.46,
  "obedience_delta": 0.05,
  "left_intelligence": 0.08,
  "right_intelligence": 0.14,
  "intelligence_delta": 0.06,
  "left_coupling": 0.24,
  "right_coupling": 0.09,
  "coupling_delta": -0.15
}
```

---

### ablate

Apply one intervention and compare before and after.

```bash
python -m tensorearch ablate <trace.json> --kind <kind> --target <target>
python -m tensorearch ablate <trace.json> --kind <kind> --target <target> --value <float> --json
```

**Arguments:**

| Argument | Required | Description |
|---|---|---|
| `trace` | yes | trace JSON path |
| `--kind` | yes | intervention type (see table below) |
| `--target` | yes | slice ID or `src->dst` edge ID |
| `--value` | no | numeric parameter (default 0.0) |
| `--json` | no | machine-readable JSON |

**Intervention kinds:**

| kind | target format | value | effect |
|---|---|---|---|
| `remove_slice` | slice ID | — | remove slice and all its edges |
| `mask_edge` | `src->dst` | — | set edge weight to 0.0 |
| `set_write_magnitude` | slice ID | float | override write_magnitude |
| `set_read_sensitivity` | slice ID | float | override read_sensitivity |
| `scale_comm_bytes` | slice ID | float | multiply comm_bytes |
| `set_edge_weight` | `src->dst` | float | override edge weight |

Output is the same format as `compare --json`.

**Examples:**

```bash
# remove attention slice and see bottleneck shift
python -m tensorearch ablate trace.json --kind remove_slice --target blk0.attn --json

# zero a cross-device TP communication edge
python -m tensorearch ablate trace.json --kind mask_edge --target blk0.attn->blk1.attn --json

# reduce write magnitude on a router slice
python -m tensorearch ablate trace.json --kind set_write_magnitude --target blk0.router --value 0.5 --json
```

---

### export

Export inspect or compare results to a file.

```bash
python -m tensorearch export --mode inspect --left <trace.json> --output out.json --json
python -m tensorearch export --mode compare --left <left.json> --right <right.json> --output out.json --json
```

**Arguments:**

| Argument | Required | Description |
|---|---|---|
| `--mode` | yes | `inspect` or `compare` |
| `--left` | yes | trace for inspect, or left trace for compare |
| `--right` | compare only | right trace path |
| `--output` | yes | output file path |
| `--json` | no | write JSON format instead of text |

---

### adapt

Convert a high-level architecture description into a full trace JSON.

```bash
python -m tensorearch adapt --adapter transformer --input input.json --output trace.json
python -m tensorearch adapt --adapter oscillator --input input.json --output trace.json
```

**Arguments:**

| Argument | Required | Description |
|---|---|---|
| `--adapter` | yes | `transformer` or `oscillator` |
| `--input` | yes | adapter input JSON (see examples/) |
| `--output` | yes | output path for the generated trace JSON |

The adapter synthesizes slice states and edges from a compact architecture description, enriches them with feature vectors, and emits the same JSON format that `inspect` can read directly.

**Adapter input format:**

```json
{
  "name": "my-model",
  "num_layers": 4,
  "hidden_size": 4096,
  "intermediate_size": 11008,
  "num_heads": 32,
  "kv_heads": 8,
  "batch_size": 4,
  "seq_len": 2048,
  "dtype": "bf16",
  "device": "H100",
  "tp_degree": 2,
  "pp_degree": 1,
  "measured_latency_ms": 24.1,
  "measured_tokens_per_sec": 181200.0
}
```

See `examples/transformer_adapter_input.json` and `examples/oscillator_adapter_input.json`.

---

### diagnose

Diagnose source-level logic smells in Python or shell scripts.

```bash
python -m tensorearch diagnose --source-file path/to/script.py
python -m tensorearch diagnose --source-file path/to/script.py --json
```

The diagnostic report is intended for:

- multi-stage scoring logic
- conflicting weighted signals
- repeated score mutation
- fallback reconstruction from downstream outputs
- shell script overwrite / branch / pipeline visibility
- entropy-based grouping of logic scopes into low / medium / high entropy clusters
- modular-flow analysis of whether logic events are evenly distributed or collapse into hotspots

**Arguments:**

| Argument | Required | Description |
|---|---|---|
| `--source-file` | yes | path to a Python or shell script |
| `--json` | no | emit machine-readable JSON output |
| `--output` | no | write the report to this file path in addition to stdout |

**JSON output keys:**

```json
{
  "language": "python",
  "source_file": "path/to/script.py",
  "summary": {
    "n_findings": 3,
    "n_strengths": 2,
    "n_entropy_clusters": 2,
    "overall_assessment": "needs_review"
  },
  "entropy_clusters": [
    {
      "scope": "function",
      "name": "score",
      "cluster": "medium_entropy",
      "entropy": 1.58,
      "dominant_signal": "call",
      "logic_labels": ["scoring_logic", "gating_logic"],
      "modular_flow": {
        "modulus": 5,
        "event_count": 14,
        "modular_uniformity": 0.88,
        "topological_uniformity": 0.91,
        "modular_shrinking_number": 0.10,
        "assessment": "uniform_flow",
        "hotspots": ["topo:2"]
      },
      "counts": {
        "assign": 1,
        "aug_assign": 2,
        "call": 2
      }
    }
  ],
  "findings": [
    {
      "severity": "warning",
      "kind": "conflicting_signal",
      "message": "Feature 'x' drives both align and oppose rules.",
      "line": 12,
      "symbol": "x"
    }
  ],
  "strengths": [
    {
      "severity": "info",
      "kind": "score_normalization",
      "message": "Function 'score' combines normalization with explicit score bounds.",
      "line": 40,
      "symbol": "score"
    }
  ]
}
```

**Modular flow fields:**

| Field | Meaning |
|---|---|
| `modulus` | modulo bucket count used for the scope |
| `event_count` | number of logic events observed in the scope |
| `modular_uniformity` | normalized entropy of logic events across modulo buckets |
| `topological_uniformity` | normalized entropy of logic events across source-span bins |
| `modular_shrinking_number` | concentration score derived from both uniformities; higher means stronger collapse into fewer regions |
| `assessment` | one of `uniform_flow`, `mixed_flow`, `concentrated_flow`, or `insufficient_signal` |
| `hotspots` | unusually dense modulo residues or topological bins |

**Interpretation guide:**

- `uniform_flow`
  - logic events are spread relatively evenly across modulo and topological partitions
- `mixed_flow`
  - one partitioning view is even while another has visible concentration
- `concentrated_flow`
  - logic events collapse into a few residues / bins, suggesting hotspot-like logic packing
- `insufficient_signal`
  - scope is too small to support a meaningful modular-flow judgment

**When to use it:**

- when training metrics plateau and repeated tuning is not clarifying the cause
- when pairwise ranking looks good but top-1 keeps missing
- when the code "works" but decisions still feel black-box
- when you need to tell whether the bottleneck is in optimization logic or in fine-grained feature integration

Recommended workflow:

1. run `diagnose` on both the runtime scorer and the training script
2. inspect the `modular_flow` summary section
3. if the trainer is mostly `uniform_flow` but the scorer has `mixed_flow` or `concentrated_flow`, fix representation / integration first
4. only keep tuning weights after the concentrated logic hotspots are understood

---

## Global Flags

| Flag | Description |
|---|---|
| `-v` / `--verbose` | print extra details (trace paths, output file confirmations) |

---

## Common Patterns

### Inspect and save JSON for downstream processing

```bash
python -m tensorearch inspect examples/sample_trace.json --json --output inspect.json
```

### Compare before and after an ablation

```bash
python -m tensorearch ablate examples/sample_trace.json \
  --kind set_write_magnitude --target blk0.attn --value 0.5 --json
```

### Generate a synthetic trace from architecture params and inspect it

```bash
python -m tensorearch adapt --adapter transformer \
  --input examples/transformer_adapter_input.json \
  --output synth.json

python -m tensorearch inspect synth.json --json
```

### Cross-architecture comparison

```bash
python -m tensorearch compare \
  examples/branching_transformer_trace.json \
  examples/branching_oscillator_trace.json \
  --json
```
