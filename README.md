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
- source-level modular flow analysis for logic distribution uniformity
- multi-language diagnose engine (20 languages, regex + AST hybrid)
- comment/string pre-filter layer for regex precision
- PPM pseudocode bridge for binary diagnosis
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
- source-diagnose flow metrics:
  - modular shrinking number
  - modular uniformity
  - topological uniformity
  - modular flow hotspots

## CLI

8 commands:

| Command | Description |
|---------|-------------|
| `inspect` | Analyze a trace, report bottlenecks and compliance metrics |
| `compare` | Compare two architectures side by side |
| `ablate` | Simulate an intervention (remove slice, scale edge, etc.) and show delta |
| `adapt` | Convert source code or architecture description into a trace (16 families) |
| `space` | Classify source code into one of 15 model families (18-dim projection) |
| `diagnose` | Audit source-level logic: entropy clusters, modular flow, mutation tracking (20 languages) |
| `export` | Write inspect or compare results to a file |
| `help` | Show complete usage guide with all commands, families, and examples |

```powershell
# Quick reference
python -m tensorearch help
python -m tensorearch inspect <trace.json> --json
python -m tensorearch compare <left.json> <right.json> --json
python -m tensorearch ablate <trace.json> --kind <kind> --target <target> --json
python -m tensorearch adapt --adapter source --input model.py --output trace.json
python -m tensorearch adapt --adapter transformer --input desc.json --output trace.json
python -m tensorearch adapt --adapter family --family diffusion_unet --input desc.json --output trace.json
python -m tensorearch space --source-file path/to/model.py --json
python -m tensorearch diagnose --source-file path/to/script.py --json
python -m tensorearch export --mode inspect --left <trace.json> --output out.json --json
```

Packaged Windows CLI:

```powershell
dist\tensorearch.exe help
dist\tensorearch.exe inspect examples\sample_trace.json --json
dist\tensorearch.exe space --source-file model.py --json
dist\tensorearch.exe diagnose --source-file script.py --json
dist\tensorearch.exe adapt --adapter source --input model.py --output trace.json
```

### Supported Languages (20)

| # | Language | Extensions | Method | Key Detections |
|---|----------|-----------|--------|----------------|
| 1 | Python | .py | AST | multi-stage mutation, conflicting signals, score normalization |
| 2 | Shell | .sh/.bash/.ps1/.cmd/.bat | regex | destructive commands, repeated overwrites, pipelines |
| 3 | Go | .go | regex | panic, error handling (if err != nil), defer, goroutines |
| 4 | C-pseudo | .c/.h/.pseudo/.ppc | regex | dangerous APIs, goto, bitmask ops (PPM bridge) |
| 5 | Rust | .rs | regex | unsafe blocks, unwrap(), panic!(), Result/Option |
| 6 | JavaScript | .js/.jsx/.mjs | regex | eval(), var usage, console.log, callback hell |
| 7 | TypeScript | .ts/.tsx | regex | any type, @ts-ignore, generics, interfaces |
| 8 | Java | .java | regex | empty catch, System.exit(), @Override, try-with-resources |
| 9 | Zig | .zig | regex | @panic, defer/errdefer, comptime, error unions |
| 10 | C++ | .cpp/.cc/.cxx/.hpp | regex | raw new/delete, goto, reinterpret_cast, RAII, smart ptrs |
| 11 | YAML | .yaml/.yml | regex | deep nesting (>6), anchors, long lines |
| 12 | SQL | .sql | regex | SELECT *, DELETE without WHERE, DROP TABLE, subqueries |
| 13 | Dockerfile | Dockerfile | regex | FROM latest, no HEALTHCHECK, curl\|bash, multi-stage |
| 14 | Ruby | .rb | regex | eval(), system(), rescue Exception, blocks/yield |
| 15 | Lua | .lua | regex | loadstring, global vars, coroutines, metatables |
| 16 | C# | .cs | regex | empty catch, GC.Collect(), async/await, LINQ, nullable |
| 17 | PHP | .php | regex | eval(), exec(), extract(), mysql_*, PDO, namespaces |
| 18 | Basic/VB | .bas/.vb/.vbs/.frm | regex | On Error Resume Next, GoTo, Option Explicit |
| 19 | EPL | .e/.ec | regex | shell execution, DLL without error handling (Chinese keywords) |
| 20 | Kotlin | .kt/.kts | regex | !! force-unwrap, lateinit, coroutines, null-safe ops, when |

All regex-based analyzers include a **comment/string pre-filter layer** (`_strip_line`) that removes string literals and line comments before pattern matching, achieving near-AST precision without external parser dependencies.

For binary files (.sys/.exe/.dll), use PPM (ParadexMonitor) to generate C-like pseudocode, then feed into Tensorearch's C-pseudo analyzer.

For source-level `diagnose`, each module/function/script scope now includes a `modular_flow` profile:

- `assessment`
  - `uniform_flow`
  - `mixed_flow`
  - `concentrated_flow`
  - `insufficient_signal`
- `modular_shrinking_number`
  - higher means logic events collapse into fewer modular / topological regions
- `modular_uniformity`
  - how evenly logic events spread under modulo bucketing
- `topological_uniformity`
  - how evenly logic events spread across source span bins
- `hotspots`
  - modular residues or topological bins with unusually dense activity

This is especially useful when a system is "stuck but not obviously broken".
In that situation, Tensorearch should be used as an instrumentation layer first:

- do not keep tuning weights blindly
- diagnose which fine-grained scopes have concentrated or mixed flow
- separate training-loop bottlenecks from representation-layer bottlenecks
- only then decide whether to change optimization, features, or core logic

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

## Space Projection: Quadrupole + Multi-Family

Tensorearch projects source code into a structural coordinate system for architecture classification.

### Legacy Quadrupole Axes

The original 4-axis projection for LLM-centric architectures:

- `X`: residual
- `Y`: latent-attention
- `Z`: kv-transport
- `W`: propagation

### Extended Family Axes (14 dimensions)

For architectures beyond LLMs, Tensorearch now supports 14 additional family axes:

| Axis | Family | Example Models |
|------|--------|----------------|
| `D` | diffusion-denoising | Stable Diffusion, DDPM |
| `T` | timestep-conditioning | diffusion schedulers |
| `U` | multiscale-unet | U-Net encoder-decoder |
| `A` | adapterization | LoRA, hypernetworks, PEFT |
| `R` | runtime-wrapper | Triton kernels, quantization bridges |
| `V` | temporal-video | AnimateDiff, UNet3D |
| `O` | audio-spectral | mel spectrograms, vocoders |
| `G` | 3d-generative | NeRF, Gaussian splatting, point clouds |
| `S` | speech-language | Whisper, ASR/TTS |
| `M` | world-model | MDRNN, environment simulators |
| `L` | multimodal-alignment | BLIP-2, Q-Former, VLMs |
| `H` | graph-message-passing | GCN, GAT, PyG |
| `I` | vision-detection | Detectron2, Mask R-CNN, FPN |
| `B` | bio-sequence | ESM, Evoformer, protein folding |

### Family Classification

The `space` command classifies source files into one of 15 families:

```powershell
python -m tensorearch space --source-file path/to/model.py --json
```

Output includes:
- `quadrupole_projection`: legacy 4-axis (X/Y/Z/W)
- `space_family_projection`: 15-family scores + classification
- `classification`: the dominant family label

### Verified Results

| Source | Family |
|--------|--------|
| Stable Diffusion U-Net | diffusion-unet dominant |
| SD attention.py | latent-attention dominant |
| Microsoft LoRA layers.py | adapterization dominant |
| APT-Transformer wrapper | runtime-wrapper dominant |
| diffusers UNet3D | video-temporal dominant |
| audio-diffusion mel.py | audio-spectral dominant |
| torch-splatting train.py | 3d-generative dominant |
| GPT-SoVITS models.py | audio-spectral dominant |
| Coqui TTS VITS | audio-spectral dominant |
| OpenAI Whisper model.py | speech-language dominant |
| world-models MDRNN | world-model dominant |
| GameGAN trainer | world-model dominant |
| BLIP-2 modeling | multimodal-alignment dominant |
| PyG GCNConv | graph-message-passing dominant |
| Detectron2 RCNN | vision-detection dominant |
| ESM-2 | bio-sequence dominant |

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

- `22 passed` (as of 2026-04-14)

This includes:

- fixture traces
- CLI subcommands (inspect, compare, ablate, export, adapt, space, diagnose)
- comparison flows
- intervention flows
- export / adapt behavior
- space family classification
- diagnose modular flow profiles
- short boolean helper entropy correction

## Intended Outputs

Tensorearch emits:

- slice bottleneck rankings
- coupling-congestion maps
- architecture-level compliance / intelligence summaries
- intervention deltas
- cross-architecture comparison reports
- agent-friendly JSON summaries for downstream tooling
