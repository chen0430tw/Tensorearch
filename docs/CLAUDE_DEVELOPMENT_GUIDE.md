# Claude Development Guide

## Purpose

This file tells Claude what it should and should not work on inside `Tensorearch`.

The goal is to keep collaboration parallel without breaking the mathematical core.

## Project Name Meaning

`Tensorearch` should be understood as:

- `Tensor`
- `Research`
- `Search`

The intended sense is:

- a research tool for tensor-structured model architectures
- a search / inspection tool over slices, routes, and coupling structure

Do not casually rename or reinterpret the project as a generic training framework.
Its identity is:

- architecture inspection
- topology-aware analysis
- bottleneck / compliance / intelligence study

## Ownership Split

### Codex Owns

- schema evolution
- core metrics
- propagation logic
- weighted-chain logic
- freedom / compliance / intelligence logic
- intervention engine
- comparison engine
- performance / parallel execution decisions
- final experiment interpretation

### Claude May Own

- docs
- examples
- tests
- report templates
- sample traces
- adapter scaffolding
- utility scripts
- non-core CLI polish

Claude may also extend:

- `export` usage docs
- `adapt` input examples
- agent-facing JSON fixtures

## Do Not Edit Without Coordination

Claude should not independently change:

- `src/tensorearch/metrics.py`
- `src/tensorearch/propagation.py`
- `src/tensorearch/features.py`
- metric formulas in `docs/MATH_MODEL.md`
- CLI output schema keys without coordination

If Claude proposes changes there, it should write them as suggestions in docs or in a side note, not silently rewrite the math.

## Preferred Work Items For Claude

Good parallel tasks:

- add richer branching sample traces
- add Transformer trace checklist
- add Oscillator trace checklist
- write report templates
- write CLI usage docs
- add fixture-heavy tests
- build JSON conversion helpers
- add `adapt` input examples
- add agent-consumption examples for `--json`

## Testing Expectations

Before handing work back, Claude should:

1. keep changes targeted
2. avoid broad rewrites
3. run the smallest relevant test
4. report exactly which files changed

## Trace Design Guidance

When Claude creates traces:

- prefer explicit numbers over placeholders
- include branching edges when testing entropy metrics
- use realistic `edge_type`, `collective`, and `same_device` fields
- avoid inventing impossible combinations

## Reporting Format

Claude should report:

- what changed
- why
- what was tested
- any assumptions left unresolved

## CLI Notes

Current agent-friendly commands:

- `python -m tensorearch inspect <trace.json> --json`
- `python -m tensorearch compare <left.json> <right.json> --json`
- `python -m tensorearch ablate <trace.json> --kind ... --target ... --json`
- `python -m tensorearch export --mode inspect --left <trace.json> --output out.json --json`
- `python -m tensorearch adapt --adapter transformer --input adapter_input.json --output out.json`

Claude should preserve these stable JSON-oriented workflows unless explicitly asked to redesign them.
