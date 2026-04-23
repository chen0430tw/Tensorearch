# Training-Analysis Layers

Tensorearch's training-analysis pipeline is split into **three layers** that
answer three different questions. Each layer has its own module, its own
contract, and its own literature line. Mixing them is a known way to ship
overconfident conclusions.

| Layer | Module | Question | Status |
|-------|--------|----------|--------|
| 1. Zombie gate         | `zombie.py`   | Is this run still alive? | implemented |
| 2. Single-run forecast | `forecast.py` | What will it converge to? | implemented (heuristic); LC-PFN stub |
| 3. Multi-run scheduler | _(future)_    | Given many live candidates, which should we keep budget on? | not yet |

The layers are intended to be composed as a pipeline:

```
trace ──► zombie gate ──► (alive)  ──► single-run predictor ──► forecast result
                      └─► (zombie) ──► terminate, do NOT predict
```

A scheduler, when added, would wrap many such pipelines and decide which runs
to allocate further compute to. It does **not** replace the predictor.

---

## Layer 1: Zombie gate (`zombie.py`)

**Question:** is the run still producing meaningful signal?

A "zombie" is a run that has died but keeps emitting numbers — NaN/Inf
contamination, frozen-but-clamped loss, fake plateau with collapsed direction
consistency, or loss/metric drift. Naive forecasters can mistake zombies for
healthy progress (a NaN run can score ~1.0 under sloppy clamping).

This is a **sanity gate**, not a predictor. It is intentionally separate from
`forecast.py` so a forecaster cannot accidentally reward dead runs.

This module appears to be original to Tensorearch — there is no published
"zombie detector for ML training" we are aware of. It is engineered, not
calibrated against a benchmark.

---

## Layer 2: Single-run forecast (`forecast.py`)

**Question:** given a short prefix from one run, what is the final metric
likely to be, and is the answer already settled enough to stop early?

### Predictor abstraction

`forecast.py` exposes a `ForecastPredictor` Protocol. `forecast_trace()`
dispatches to a concrete predictor and defaults to the heuristic implementation,
so existing callers and the CLI are unchanged.

```python
forecast_trace(trace)                              # -> heuristic (default)
forecast_trace(trace, predictor=HeuristicForecastPredictor())
forecast_trace(trace, predictor=LCPFNForecastPredictor())   # raises today
```

### `HeuristicForecastPredictor` (default)

- A 5-dim "growth state" updated by EMA over local + prefix signals, projected
  to a scalar fitness, mixed with the most recent observed metric and a
  discounted recent-gain term.
- Stops early when prediction-shift, uncertainty, and stability all cross
  hand-tuned thresholds.
- All scalar weights are **bootstrap defaults** pending real-data calibration.
- The "gene growth" framing is narrative — there is **no peer-reviewed analog**
  of this scheme as a learning-curve forecaster. NEAT/HyperNEAT/HyperNCA evolve
  network topology or weights, not training-curve forecasts.

### `LCPFNForecastPredictor` (experimental stub)

LC-PFN (Adriaensen et al., NeurIPS 2023, https://arxiv.org/abs/2310.20447) is
the most promising direction for replacing the heuristic. **It is not a
drop-in.** Filling in this stub requires solving:

1. **Input adaptation.** LC-PFN consumes a validation-metric curve under a
   curve-shape prior. Our `TrainingStep` carries `grad_norm`, `curvature`,
   `direction_consistency` as additional signals. Either project to LC-PFN's
   format (information loss) or design a hybrid head that consumes both.
2. **Runtime.** `automl/lc-pfns` is a research repo with model artifacts, not
   a maintained pip package. Vendoring or wrapping a snapshot is a real cost.
3. **Composition.** Layer-1 (zombie) and a future layer-3 (scheduler) sit
   outside LC-PFN's scope. The pipeline contract — when does the scheduler
   ask the predictor, what does the gate do on a zombie — must be agreed
   first, before any model integration.

Until those are decided, calling `.predict()` raises `NotImplementedError`.

### Calibration / benchmarks

The right literature line for layer 2 is **learning-curve extrapolation**:

| Year | Method | Notes |
|------|--------|-------|
| 2015 | Domhan et al. (IJCAI) | Parametric curve ensemble + MCMC. |
| 2017 | LCNet (Klein et al., ICLR) | Bayesian NN over hyperparameters + prefix. |
| 2023 | LC-PFN (Adriaensen et al., NeurIPS) | Prior-data fitted transformer; SOTA on its benchmark. |

**Benchmark warning.** LCDB 1.1 (Mohr et al., 2025, https://arxiv.org/abs/2505.15657)
is often cited as the headline curve benchmark, but it studies **sample-wise**
learning curves (performance vs training set size), not the
**step/epoch-prefix** problem this module solves. Its lesson — real curves are
non-monotone and zigzag — is useful as a sanity check, but it is not the most
direct benchmark for our task. A step/epoch-prefix curve set such as
**NAS-Bench-201**'s training trajectories is a better apples-to-apples target.
This is a future calibration item, not a one-step swap.

---

## Layer 3: Multi-run scheduler (not yet)

**Question:** when there are many live candidate runs, how should we allocate
the next slice of compute?

This is the **rank-and-cull** family — it is a different problem from
single-run forecasting and is not a substitute for layer 2:

| Method | Citation | Role |
|--------|----------|------|
| Hyperband | Li et al., JMLR 2017 | Bandit-style multi-fidelity scheduler. |
| ASHA | Li et al., MLSys 2020 | Async parallel successive halving. Still has parameters (rung sizes, reduction factor). |
| BOHB | Falkner et al., ICML 2018 | Bayesian + Hyperband. |
| PBT | Jaderberg et al., 2017 | Population-Based Training; closest analog to "evolutionary early selection on partial training prefixes". |

**Note on terminology.** Hyperband / ASHA / BOHB are bandit / multi-fidelity
schedulers, not evolutionary algorithms. PBT is the EA-flavored one. Earlier
notes that called the whole family "evolutionary" were imprecise.

---

## Reading guide for prior research notes

Earlier session notes about LC-PFN being a "drop-in head", or about a
candidate-count threshold (e.g. "≥4 runs → ASHA") are **withdrawn**. The
correct, narrower claims are:

- LC-PFN is the most promising **experimental** direction for replacing the
  heuristic, behind an adapter that handles input format and the auxiliary
  signals. Not drop-in.
- ASHA / Hyperband / BOHB / PBT belong to layer 3 and only become relevant
  when there are multiple live candidate runs. They do not replace layer 2.
- LCDB 1.1 is one reference for "curves can be ill-behaved", not the only or
  even the most direct benchmark for step/epoch-prefix forecasting.

---

## Delivery order

1. **Done:** layer-2 predictor abstraction + `HeuristicForecastPredictor`
   (default, behavior unchanged) + `LCPFNForecastPredictor` stub.
2. **Done:** this docs file documenting the three-layer split.
3. **Next:** decide whether the LC-PFN integration cost (vendoring, input
   adaptation, calibration set) is worth paying now, or whether the heuristic
   plus zombie gate is enough for current usage.
4. **Later:** introduce a layer-3 scheduler interface only when there is a
   real multi-run workload to serve.
