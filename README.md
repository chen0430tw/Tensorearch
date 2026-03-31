# Tensorearch

Tensorearch is a model-architecture inspection tool for analyzing how slice structure, coupling topology, and execution geometry shape throughput in large language models.

It combines four ideas:

- OpenAI Transformer Debugger style inspection workflow
- OTAL-inspired propagation topology
- weighted chain analysis
- simulated-topology / multi-scale structural modeling

## Core Question

Given a model execution graph, Tensorearch asks:

- which slices dominate latency?
- which couplings amplify congestion?
- which structural regimes create poor throughput despite similar parameter count?
- can architecture quality be inspected mathematically rather than only benchmarked empirically?

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

## Current Scope

Phase 1 is mathematical and structural:

1. define slice-state encoding
2. define weighted-chain coupling
3. define simulated-topology observables
4. define OTAL-like propagation over architecture graphs
5. derive bottleneck and throughput metrics

## Repository Layout

- `docs/MATH_MODEL.md`
  - core mathematical model
- `docs/TDB_MAPPING.md`
  - Transformer Debugger to Tensorearch mapping
- `src/tensorearch/`
  - package skeleton
- `tests/`
  - future metric and schema tests
- `CLAUDE_READY_INDEX.md`
  - quick handoff index for agents

## Planned Outputs

Tensorearch should eventually emit:

- slice bottleneck rankings
- coupling-congestion maps
- throughput predictors
- architecture-level circuit summaries
- cross-architecture comparison reports

## Near-Term Plan

1. finalize the mathematical schema
2. define trace ingestion format
3. build graph + metric primitives
4. validate on Transformer vs Oscillator style architectures
