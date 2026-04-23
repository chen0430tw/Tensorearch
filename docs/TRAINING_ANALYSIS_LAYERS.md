# Training-Analysis Layers

Tensorearch's training-analysis pipeline is split into **three layers** that
answer three different questions. Each layer has its own module, its own
contract, and its own literature line. Mixing them is a known way to ship
overconfident conclusions.

| Layer | Module | Question | Status |
|-------|--------|----------|--------|
| 0. Trace contract validator | `training_contract.py` | Is the trace structurally trustworthy? | implemented |
| 1. Zombie gate         | `zombie.py`   | Is this run still alive? | implemented (combined-rule explosive) |
| 2. Single-run forecast | `forecast.py` | What will it converge to? Where to stop? | implemented (heuristic + stop window); LC-PFN stub |
| 3. Multi-run scheduler | _(future)_    | Given many live candidates, which should we keep budget on? | not yet |

The layers are intended to be composed as a pipeline:

```
trace ──► zombie gate ──► (alive)  ──► single-run predictor ──► forecast result
                      └─► (zombie) ──► terminate, do NOT predict
```

A scheduler, when added, would wrap many such pipelines and decide which runs
to allocate further compute to. It does **not** replace the predictor.

---

## Layer 0: Trace contract validator (`training_contract.py`)

**Question:** does this trace look trustworthy? Or is the upstream training
script writing display-smoothed values that will fool zombie/forecast?

This layer runs first, on every `forecast_trace()` and `assess_zombie()` call,
and attaches its output as `metadata.contract` (forecast) or `report.contract`
(zombie). It does **not** modify the trace; it only flags suspicious patterns:

| Code | Severity | Triggered when |
|------|----------|----------------|
| `EMPTY_TRACE`              | CRITICAL | trace.steps is empty |
| `DISPLAY_SMOOTHED_LOSS`    | CRITICAL | any step has `train_loss_kind="display_smoothed"` |
| `LOSS_SAWTOOTH_PATTERN`    | CRITICAL | sign-flip period of `suspected_log_interval` (= display reset) |
| `UNKNOWN_LOSS_KIND`        | WARN     | every step has `train_loss_kind="unknown"` |
| `VAL_METRIC_NEVER_OBSERVED`| WARN     | no step has `val_metric_observed=True` |
| `UNKNOWN_GRAD_NORM_KIND`   | WARN     | every step has `grad_norm_kind="unknown"` |
| `PRE_CLIP_WITHOUT_THRESHOLD`| WARN    | grad_norm is pre-clip but `gradient_clip` is unset |
| `LOSS_NONFINITE`           | INFO     | any step has NaN/Inf train_loss |

**Effect on layer 1 (zombie):**
- `valid_for_zombie=False` (e.g. when grad_norm_kind is unknown) means
  zombie only safely catches NaN/Inf; combined-rule explosive falls back to
  legacy single-point check.

**Effect on layer 2 (forecast):**
- `valid_for_forecast=False` (e.g. display-smoothed or sawtooth) floors
  `confidence` to 0.25 and prepends "trace contract violation" to `reason`.

The validator is cheap (a few list comprehensions over the trace) so it runs
unconditionally.

### Trace field contract

Trace JSON `steps[i]` schema (with the v4 contract additions):

```json
{
  "step": 100,
  "train_loss": 1.85,
  "train_loss_kind": "raw_step_mean",   // or "display_smoothed" | "unknown"
  "val_metric": 0.42,
  "val_metric_observed": true,           // false if val_metric is placeholder
  "grad_norm": 178.31,
  "grad_norm_kind": "pre_clip",          // or "post_clip" | "unknown"
  "post_clip_grad_norm": 1.0,            // 0 if not captured
  "gradient_clip": 1.0,                  // 0 if unknown
  "curvature": 0.018,
  "direction_consistency": 1.0
}
```

**Rule of thumb for trace writers:**
- `train_loss` MUST be the raw per-optimizer-step loss (mean of grad_accum
  micro-batches), NOT a rolling display average. If you only have a
  display-smoothed value, set `train_loss_kind="display_smoothed"` so
  forecast knows not to trust its curvature/direction proxies.
- `val_metric_observed=true` **only** when this step actually ran a fresh
  validation forward. A stale-but-recent value is still observed=true; a
  placeholder (0 or copy of train_loss) is observed=false.
- `grad_norm_kind="pre_clip"` is the default since `clip_grad_norm_` returns
  the pre-clip norm. If you have access to post-clip too, fill it in;
  zombie's combined-rule prefers it.

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

### Combined-rule explosive detection (post-Codex review)

The original v1 detector used a single threshold `inf_grad_threshold=50.0` on
pre-clip grad_norm. **This was too aggressive** — transformer training with
default warmup routinely sees pre-clip norms in the 100-1000 range that the
gradient clipper neutralizes. v2 fixes this with two paths:

**Optimal path** — when the trace records `post_clip_grad_norm` and
`gradient_clip`:
- `post_clip > gradient_clip * explosive_postclip_factor` (default 1.2)
- sustained for `explosive_persist_steps` (default 3)
- AND loss not improving in the recent window
- → INFECTED

**Fallback path** — when only pre-clip is available:
- `grad_norm > explosive_preclip_abs_threshold` (default 1000.0)
- AND `> explosive_preclip_spike_ratio × recent_median` (default 4.0)
- sustained for `explosive_persist_steps`
- AND loss not improving
- → INFECTED

**Single high pre-clip without persistence/spike → SUSPECT, not INFECTED.**
This avoids the false-positive flood from APT-Transformer-style early-step
gradient bursts that the clipper handles fine.

---

## Layer 2: Single-run forecast (`forecast.py`)

**Question:** given a short prefix from one run, what is the final metric
likely to be, and is the answer already settled enough to stop early?

### Stop-window output (post-Codex review)

`ForecastResult` reports three additional fields beyond a single point estimate:

| Field | Meaning | Value when not-yet-recommended |
|-------|---------|---------------------------------|
| `decision_window_start` | earliest step where stop is safe | 0 |
| `decision_window_end`   | latest step beyond which further training is unlikely to help | 0 |
| `recommended_stop_step` | the forecaster's pick inside the window | 0 |

When `continue_training_recommended=True` (criteria not met), all three are 0
and the caller should keep training. When the early-decision criteria first
trigger, `recommended_stop_step = decision_window_start = current_step` and
`decision_window_end = current_step + delta_steps` (a small confirmation
buffer). Future versions may project a window further forward when stability
is partial; today the window is conservative.

### Training-time broadcast

When `--tensorearch-trace` is enabled in `pretrain_quickcook.py` (or any
trainer that adopts the same pattern), every `--tensorearch-val-interval`
global steps the trainer:

1. reads its own accumulated `training_trace.jsonl`
2. runs `validate_trace()` + `assess_zombie()` + `forecast_trace()`
3. writes `training_forecast.json` with the combined report
4. logs one line: `[Tensorearch] status=hold predicted=0.81 conf=0.62 stop_window=[1200,1600]`

`status` is one of:
- `zombie` — zombie gate said INFECTED or ZOMBIE (training is dying)
- `stop` — forecast says stop now (`recommended_stop_step > 0` and
   `continue_training_recommended=False`)
- `hold` — keep training; either no recommendation yet or window not reached

`training_forecast.json` schema:

```json
{
  "status": "hold" | "stop" | "zombie",
  "predicted_final_score": 0.81,
  "uncertainty": 0.07,
  "confidence": 0.62,
  "earliest_decision_step": 1200,
  "decision_window_start": 1200,
  "decision_window_end": 1600,
  "recommended_stop_step": 1400,
  "zombie": {...},               // full ZombieReport
  "trace_contract": {...}        // ContractReport.to_dict()
}
```

Failure mode: if `tensorearch` is not importable from the training script's
PYTHONPATH, broadcast logs a one-time warning and silently no-ops. Training
itself is never blocked by broadcast failure.

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
