# Tensorearch

Tensorearch is a topology-aware model-architecture inspection tool for analyzing how slice structure, coupling topology, and execution geometry shape throughput in large language models.

The name should be read as:

- `Tensor`
- `Research`
- `Search`

It is not a training framework. It is an architecture debugger.

## What It Combines

Tensorearch combines four ideas:

- OpenAI Transformer Debugger style inspection workflow
- OTAL-inspired propagation topology
- weighted-chain coupling analysis
- local-vector-space structural modeling

## Core Question

Given a model execution graph, Tensorearch asks:

- which slices dominate latency?
- which couplings amplify congestion?
- which structural regimes create poor throughput despite similar parameter count?
- where is the bottleneck actually localized?
- is the model structurally obedient, overly free, or strategically adaptive?

## Conceptual Position

Transformer Debugger focuses on:

- neurons
- heads
- SAE latents
- behavioral circuits

Tensorearch shifts the inspection unit upward to:

- layers
- sub-layers
- routing blocks
- communication edges
- memory-transfer boundaries
- execution slices

The intended result is an architecture debugger rather than a neuron debugger.

## Relation To Similar Tools

Tensorearch is not trying to replace existing tooling. It sits at a different layer.

### Compared with Transformer Debugger

Transformer Debugger is strongest when the question is:

- why did the model produce this token?
- which head / neuron / latent drove this behavior?
- what circuit implements this behavior?

Tensorearch is strongest when the question is:

- why is this architecture slower than another one?
- which slice is the true bottleneck?
- where does coupling topology fail to become useful structure?
- why does a new architecture underperform a simpler baseline?

Short version:

- Transformer Debugger = behavior / circuit microscope
- Tensorearch = architecture / topology microscope

### Compared with Profilers

Classic profilers answer:

- where time is spent
- where memory is spent
- which kernels are hot

Tensorearch uses profiler-like inputs, but pushes one level higher:

- how slices couple
- how bottlenecks propagate
- which structural regimes create congestion
- whether a path is merely expensive or strategically meaningful

So a profiler is an input source; Tensorearch is an interpretation layer over model structure.

### Compared with Benchmark Dashboards

Benchmark dashboards tell you:

- final loss
- final quality
- throughput
- memory footprint

Tensorearch tries to explain *why* those numbers happen.

It is intended for cases where benchmark outcomes alone are too coarse, especially for:

- new architectures
- routing-heavy models
- non-standard attention replacements
- models that may be paying extra compute cost without getting matching capability gains

### Compared with Generic Interpretability Tools

Many interpretability tools stay close to:

- token attribution
- feature activation
- neuron or head saliency

Tensorearch instead reasons over:

- execution slices
- transport edges
- local vector spaces
- weighted chains
- structural obedience / freedom / intelligence

That makes it more suitable for architecture diagnosis than for token-level behavioral explanation.

### Practical Positioning

In practice, Tensorearch is best used together with other tools:

- profiler: collect timing / memory / kernel data
- benchmark suite: establish empirical performance
- interpretability tooling: inspect token-level behavior
- Tensorearch: connect those observations back to architecture-level failure modes

Its purpose is not to say only that a model is worse than a baseline.
Its purpose is to localize *why* it is worse, and whether the failure comes from:

- routing
- coupling
- propagation
- communication
- structural inefficiency
- or a mismatch between extra structure and actual downstream gain

## Implemented Scope

Tensorearch already includes:

- JSON trace schema for model/system/slice/edge data
- feature enrichment for:
  - local vector spaces
  - transport-scale estimation
  - obedience-target inference
- weighted-chain propagation
- bottleneck and congestion metrics
- freedom / compliance / intelligence metrics
- intervention engine
- cross-architecture comparison engine
- agent-friendly JSON CLI
- Windows `exe` packaging

## Current Metrics

Current reports can emit:

- predicted bottleneck
- global coupling efficiency
- global obedience score
- global intelligence score
- per-slice:
  - propagated cost
  - bottleneck index
  - congestion
  - freedom index
  - compliance index
  - route entropy
  - effect entropy
  - intelligence index
- edge attribution rankings

## CLI

Source mode:

```powershell
python -m tensorearch --help
python -m tensorearch inspect <trace.json> --json
python -m tensorearch compare <left.json> <right.json> --json
python -m tensorearch ablate <trace.json> --kind <kind> --target <target> --json
python -m tensorearch export --mode inspect --left <trace.json> --output out.json --json
python -m tensorearch adapt --adapter transformer --input adapter_input.json --output out.json
python -m tensorearch space --source-file path/to/model.py --json
python -m tensorearch diagnose --source-file path/to/script.py --json
```

Packaged Windows CLI:

```powershell
dist\tensorearch.exe --help
dist\tensorearch.exe inspect examples\sample_trace.json --json
dist\tensorearch.exe space --source-file examples\sample_model.py --json
dist\tensorearch.exe diagnose --source-file path\to\script.py --json
```

Global CLI behavior:

- `--help`
- `-v / --verbose`
- `--json`

## Real Trace Workflow

Tensorearch is designed to consume either:

- hand-authored trace JSON
- adapted JSON from another source
- real profiling traces exported from training scripts

Current integration work already connects to `APT-Transformer` quickcook profiling so short profiling runs can emit:

- `tensorearch_trace.json`

This enables direct comparison of real short-profile traces from:

- `Transformer`
- `Oscillator`

## Quadrupole Space Projection

Tensorearch now also supports a source-driven quadrupole space projection.

The current quadrupole axes are:

- `X`: residual
- `Y`: latent-attention
- `Z`: kv-transport
- `W`: propagation

The CLI can infer a density profile from source code and project it into this shared coordinate system:

```powershell
python -m tensorearch space --source-file path/to/model.py --json
```

This is meant to answer:

- is this architecture still residual-dominant?
- is it primarily shifting into latent-attention space?
- is it mostly a KV backend variant?
- is it propagation-driven?

It also emits extension terms for:

- `expert_extension`
- `ffn_extension`

## Current Architectural Findings

On the current `QuickCook + HLBD` line:

- Transformer remains the stronger practical baseline in long-run quickcook quality / throughput.
- Oscillator currently pays extra cost for its phase/propagation path.
- After finer-grained profiling, the main local hotspots are:
  - Transformer: `score`
  - Oscillator: `adj`

This is exactly the kind of localization Tensorearch is meant to provide.

## Repository Layout

- `docs/MATH_MODEL.md`
  - core mathematical model
- `docs/TDB_MAPPING.md`
  - Transformer Debugger to Tensorearch mapping
- `docs/CLI_USAGE.md`
  - full CLI usage and JSON behavior
- `docs/TRANSFORMER_TRACE_CHECKLIST.md`
  - transformer trace checklist
- `docs/OSCILLATOR_TRACE_CHECKLIST.md`
  - oscillator trace checklist
- `docs/AGENT_JSON_USAGE.md`
  - agent / pipeline JSON consumption guide
- `docs/CLAUDE_DEVELOPMENT_GUIDE.md`
  - collaboration guide for Claude
- `src/tensorearch/`
  - package source
- `tests/`
  - schema, fixtures, CLI, and comparison tests
- `examples/`
  - sample traces and adapter inputs
- `CLAUDE_READY_INDEX.md`
  - quick handoff index for agents

## Testing Status

The current repository test suite passes locally with:

- `37 passed`

This includes:

- fixture traces
- CLI subcommands
- comparison flows
- intervention flows
- export / adapt behavior

## Intended Outputs

Tensorearch emits:

- slice bottleneck rankings
- coupling-congestion maps
- architecture-level compliance / intelligence summaries
- intervention deltas
- cross-architecture comparison reports
- agent-friendly JSON summaries for downstream tooling
