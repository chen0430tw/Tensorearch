# Agent JSON Usage Guide

This document shows how to consume Tensorearch `--json` output programmatically — for automated pipelines, agents, or downstream tooling.

---

## inspect --json

### Command

```bash
python -m tensorearch inspect trace.json --json
```

### Parsing the output

```python
import json, subprocess, sys

result = subprocess.run(
    [sys.executable, "-m", "tensorearch", "inspect", "trace.json", "--json"],
    capture_output=True, text=True, check=True,
)
data = json.loads(result.stdout)

# Top-level keys
summary   = data["summary"]        # global metrics
slices    = data["slices"]         # per-slice metrics list
edges     = data["top_edge_attributions"]
system    = data["system"]
```

### Extracting bottleneck and scores

```python
bottleneck      = summary["predicted_bottleneck"]       # str: slice_id
coupling        = summary["global_coupling_efficiency"]  # float
obedience       = summary["global_obedience_score"]      # float
intelligence    = summary["global_intelligence_score"]   # float

print(f"bottleneck={bottleneck}  coupling={coupling:.3f}  ii={intelligence:.4f}")
```

### Ranking slices by bottleneck index

```python
ranked = sorted(slices, key=lambda s: s["slice_bottleneck_index"], reverse=True)
for s in ranked[:5]:
    print(f"{s['slice_id']:20s}  sbi={s['slice_bottleneck_index']:.4f}  ii={s['intelligence_index']:.4f}")
```

### Finding high-freedom slices

```python
high_freedom = [s for s in slices if s["freedom_index"] > 1.5]
for s in sorted(high_freedom, key=lambda x: x["freedom_index"], reverse=True):
    print(f"{s['slice_id']:20s}  freedom={s['freedom_index']:.3f}  compliance={s['compliance_index']:.3f}")
```

### Top edge attributions

```python
for e in edges[:3]:
    print(f"{e['src']} -> {e['dst']}  attribution={e['edge_attribution']:.2f}")
```

---

## compare --json

### Command

```bash
python -m tensorearch compare left.json right.json --json
```

### Parsing

```python
result = subprocess.run(
    [sys.executable, "-m", "tensorearch", "compare", "left.json", "right.json", "--json"],
    capture_output=True, text=True, check=True,
)
data = json.loads(result.stdout)

left_bottleneck  = data["left_predicted_bottleneck"]
right_bottleneck = data["right_predicted_bottleneck"]
ii_delta         = data["intelligence_delta"]
obedience_delta  = data["obedience_delta"]
coupling_delta   = data["coupling_delta"]

print(f"bottleneck: {left_bottleneck} -> {right_bottleneck}")
print(f"intelligence delta: {ii_delta:+.4f}")
print(f"coupling delta: {coupling_delta:+.4f}")
```

### Deciding which architecture is better

```python
# Higher intelligence + lower coupling = better structural quality
if ii_delta > 0 and coupling_delta < 0:
    print("right architecture has better intelligence with less coupling overhead")
elif ii_delta < 0:
    print("right architecture lost intelligence — investigate bottleneck shift")
```

---

## ablate --json

### Command

```bash
python -m tensorearch ablate trace.json --kind remove_slice --target blk0.attn --json
```

Output is the same format as `compare --json`.

### Measuring bottleneck shift after ablation

```python
result = subprocess.run(
    [sys.executable, "-m", "tensorearch", "ablate", "trace.json",
     "--kind", "set_write_magnitude", "--target", "blk0.router", "--value", "0.5", "--json"],
    capture_output=True, text=True, check=True,
)
data = json.loads(result.stdout)
print(f"bottleneck before: {data['left_predicted_bottleneck']}")
print(f"bottleneck after:  {data['right_predicted_bottleneck']}")
print(f"ii delta:          {data['intelligence_delta']:+.4f}")
```

---

## export --json

### Command

```bash
python -m tensorearch export --mode inspect --left trace.json --output report.json --json
```

This writes the same JSON as `inspect --json` to a file instead of stdout.
Reading it back:

```python
import json
from pathlib import Path

data = json.loads(Path("report.json").read_text(encoding="utf-8"))
bottleneck = data["summary"]["predicted_bottleneck"]
```

### Compare export

```bash
python -m tensorearch export \
  --mode compare \
  --left left.json \
  --right right.json \
  --output compare_report.json \
  --json
```

---

## adapt → inspect pipeline

The `adapt` command outputs the same JSON schema that `inspect` reads.
Full pipeline in Python:

```python
import json, subprocess, sys, tempfile
from pathlib import Path

adapter_input = {
    "name": "my-4layer-transformer",
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
    "measured_tokens_per_sec": 181200.0,
}

with tempfile.TemporaryDirectory() as tmp:
    inp = Path(tmp) / "input.json"
    out = Path(tmp) / "trace.json"
    inp.write_text(json.dumps(adapter_input), encoding="utf-8")

    subprocess.run(
        [sys.executable, "-m", "tensorearch", "adapt",
         "--adapter", "transformer", "--input", str(inp), "--output", str(out)],
        check=True,
    )

    result = subprocess.run(
        [sys.executable, "-m", "tensorearch", "inspect", str(out), "--json"],
        capture_output=True, text=True, check=True,
    )

data = json.loads(result.stdout)
print("bottleneck:", data["summary"]["predicted_bottleneck"])
print("intelligence:", data["summary"]["global_intelligence_score"])
```

---

## JSON schema reference

### inspect output

```
{
  "system": {
    "name": str,
    "latency_ms": float,
    "tokens_per_sec": float
  },
  "summary": {
    "predicted_bottleneck": str,
    "global_coupling_efficiency": float,
    "global_obedience_score": float,
    "global_intelligence_score": float
  },
  "slices": [
    {
      "slice_id": str,
      "kind": str,
      "op_type": str,
      "cost": float,
      "propagated_cost": float,
      "slice_bottleneck_index": float,
      "topological_congestion": float,
      "direct_effect": float,
      "estimated_total_effect": float,
      "freedom_index": float,
      "compliance_index": float,
      "route_entropy": float,
      "effect_entropy": float,
      "compliance_entropy": float,
      "intelligence_index": float,
      "propagated_state": float
    }
  ],
  "top_edge_attributions": [
    { "src": str, "dst": str, "edge_attribution": float }
  ]
}
```

### compare / ablate output

```
{
  "left_name": str,
  "right_name": str,
  "left_predicted_bottleneck": str,
  "right_predicted_bottleneck": str,
  "left_obedience": float,
  "right_obedience": float,
  "obedience_delta": float,
  "left_intelligence": float,
  "right_intelligence": float,
  "intelligence_delta": float,
  "left_coupling": float,
  "right_coupling": float,
  "coupling_delta": float
}
```
