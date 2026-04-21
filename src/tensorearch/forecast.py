from __future__ import annotations

"""
Forecast training outcomes from short training prefixes.

This module is intentionally a heuristic bootstrap model, not a paper-faithful
implementation of any single forecasting method. Its current defaults are based
on two design ideas:

1. Gene-growth style propagation
   Early local signals do not determine the outcome alone. We keep a rolling
   "growth state" and let each new step inherit part of the previous state,
   similar to how a growing organism preserves and propagates structure while
   still reacting to local conditions.

2. Evolutionary early selection
   The system is meant to decide whether a run already looks promising enough
   to keep, or stable enough to stop early, before full training completes.
   The thresholds therefore act like practical early-selection cutoffs rather
   than theoretically derived constants.

All scalar weights below are heuristic bootstrap defaults. They are not
paper-derived constants and should be treated as calibration targets once real
training-trace data is available.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .schema import ForecastResult, TrainingStep, TrainingTrace

_ACTIVE_RUN_ID: str | None = None
_ACTIVE_FINGERPRINT: str | None = None
_ACTIVE_PREFIX_LEN: int = 0
_GROWTH_CACHE: dict[str, list[float]] = {}
_PREDICTION_CACHE: dict[str, list[float]] = {}
_UNCERTAINTY_CACHE: dict[str, list[float]] = {}

@dataclass(frozen=True)
class ForecastConfig:
    # Heuristic bootstrap defaults for projecting the 5-dim growth state into
    # a single fitness scalar. Inspired by "gene growth" intuition: some axes
    # should dominate inherited viability more than others in the first pass.
    fitness_weights: list[float] = field(default_factory=lambda: [0.32, 0.26, 0.18, 0.14, 0.10])

    # State propagation controls.
    alpha: float = 0.72
    beta: float = 0.22
    lambda_stability: float = 5.0
    gamma_discount: float = 0.88

    # Early-selection cutoffs.
    epsilon_pred: float = 0.015
    epsilon_uncertainty: float = 0.08
    epsilon_stability: float = 0.82
    min_prefix_steps: int = 6
    delta_steps: int = 3

    # Prediction mixer.
    current_metric_weight: float = 0.58
    fitness_weight: float = 0.24
    stability_weight: float = 0.12
    discounted_gain_weight: float = 0.06

    # Uncertainty estimator.
    prefix_penalty_weight: float = 0.18
    metric_volatility_weight: float = 0.45
    loss_volatility_weight: float = 0.35
    maturity_weight: float = 0.20

    # Confidence collapse scale.
    tau_confidence: float = 0.22


DEFAULT_FORECAST_CONFIG = ForecastConfig()


def _zero_state() -> list[float]:
    return [0.0] * 5


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _safe_ratio(num: float, den: float) -> float:
    return 0.0 if abs(den) < 1e-12 else num / den


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _variance(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = _mean(values)
    return sum((value - mu) ** 2 for value in values) / len(values)


def _slope(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return (values[-1] - values[0]) / max(len(values) - 1, 1)


def _monotonicity(values: list[float], direction: str) -> float:
    if len(values) <= 1:
        return 1.0
    good = 0
    total = 0
    for prev, cur in zip(values[:-1], values[1:]):
        total += 1
        if direction == "decreasing" and cur <= prev:
            good += 1
        if direction == "increasing" and cur >= prev:
            good += 1
    return _safe_ratio(good, total)


def _saturation_index(values: list[float]) -> float:
    if len(values) <= 2:
        return 0.0
    early = abs(values[min(1, len(values) - 1)] - values[0])
    late = abs(values[-1] - values[-2])
    if early <= 1e-8:
        return 0.0
    return _clamp(1.0 - _safe_ratio(late, early))


def _existing_checkpoint_signature(path: str) -> tuple[float, int]:
    if not path:
        return (0.0, 0)
    source = Path(path)
    if not source.exists():
        return (0.0, 0)
    stat = source.stat()
    return (stat.st_mtime, stat.st_size)


def _fingerprint_trace(trace: TrainingTrace) -> str:
    mtime, size = _existing_checkpoint_signature(trace.checkpoint_path)
    raw = "|".join(
        [
            trace.run_id,
            trace.checkpoint_path,
            f"{mtime:.6f}",
            str(size),
            trace.target_metric,
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def reset_run_state(run_id: str) -> None:
    _GROWTH_CACHE.pop(run_id, None)
    _PREDICTION_CACHE.pop(run_id, None)
    _UNCERTAINTY_CACHE.pop(run_id, None)


def _ensure_clean_context(trace: TrainingTrace, prefix_len: int) -> None:
    global _ACTIVE_RUN_ID, _ACTIVE_FINGERPRINT, _ACTIVE_PREFIX_LEN

    fingerprint = _fingerprint_trace(trace)
    if _ACTIVE_RUN_ID != trace.run_id:
        _ACTIVE_RUN_ID = trace.run_id
        _ACTIVE_FINGERPRINT = fingerprint
        _ACTIVE_PREFIX_LEN = prefix_len
        _GROWTH_CACHE[trace.run_id] = _zero_state()
        _PREDICTION_CACHE[trace.run_id] = []
        _UNCERTAINTY_CACHE[trace.run_id] = []
        return

    if _ACTIVE_FINGERPRINT != fingerprint or prefix_len < _ACTIVE_PREFIX_LEN:
        reset_run_state(trace.run_id)
        _GROWTH_CACHE[trace.run_id] = _zero_state()
        _PREDICTION_CACHE[trace.run_id] = []
        _UNCERTAINTY_CACHE[trace.run_id] = []
        _ACTIVE_FINGERPRINT = fingerprint
    _ACTIVE_PREFIX_LEN = prefix_len


def _end_run(trace: TrainingTrace) -> None:
    global _ACTIVE_RUN_ID, _ACTIVE_FINGERPRINT, _ACTIVE_PREFIX_LEN
    reset_run_state(trace.run_id)
    if _ACTIVE_RUN_ID == trace.run_id:
        _ACTIVE_RUN_ID = None
        _ACTIVE_FINGERPRINT = None
        _ACTIVE_PREFIX_LEN = 0


def _delta(current: float, previous: float) -> float:
    return current - previous


def _encode_local_signal(current: TrainingStep, previous: TrainingStep | None) -> list[float]:
    prev_loss = previous.train_loss if previous else current.train_loss
    prev_metric = previous.val_metric if previous else current.val_metric
    return [
        _clamp(1.0 - current.train_loss),
        _clamp(current.val_metric),
        _clamp(1.0 - min(current.grad_norm, 10.0) / 10.0),
        _clamp(1.0 - min(current.curvature, 10.0) / 10.0),
        _clamp(0.5 + 0.25 * (_delta(prev_loss, current.train_loss) + _delta(current.val_metric, prev_metric)) + 0.25 * current.direction_consistency),
    ]


def _encode_prefix_shape(prefix: list[TrainingStep]) -> list[float]:
    losses = [step.train_loss for step in prefix]
    metrics = [step.val_metric for step in prefix]
    grad_norms = [step.grad_norm for step in prefix]
    curvatures = [step.curvature for step in prefix]
    direction = [step.direction_consistency for step in prefix]
    return [
        _clamp(0.5 - _slope(losses)),
        _clamp(0.5 + _slope(metrics)),
        _clamp(1.0 - _variance(losses)),
        _clamp(1.0 - _variance(metrics)),
        _clamp(0.4 * _saturation_index(metrics) + 0.3 * _monotonicity(losses, "decreasing") + 0.3 * _monotonicity(metrics, "increasing")),
        _clamp(1.0 - _mean(grad_norms) / 10.0),
        _clamp(1.0 - _mean(curvatures) / 10.0 + 0.2 * _mean(direction)),
    ]


def _update_growth_state(
    z_prev: list[float],
    local_signal: list[float],
    prefix_signal: list[float],
    config: ForecastConfig,
) -> list[float]:
    merged_prefix = [
        prefix_signal[0],
        prefix_signal[1],
        prefix_signal[2],
        prefix_signal[4],
        0.5 * prefix_signal[5] + 0.5 * prefix_signal[6],
    ]
    return [
        _clamp(config.alpha * prev + (1.0 - config.alpha) * local + config.beta * prefix)
        for prev, local, prefix in zip(z_prev, local_signal, merged_prefix)
    ]


def _compute_growth_fitness(z_t: list[float], config: ForecastConfig) -> float:
    return _clamp(sum(weight * value for weight, value in zip(config.fitness_weights, z_t)))


def _compute_growth_gain(current: float, previous: float) -> float:
    return current - previous


def _compute_stability(history: list[float], config: ForecastConfig) -> float:
    recent = history[-5:]
    return math.exp(-config.lambda_stability * _variance(recent))


def _predict_final_score(
    prefix: list[TrainingStep],
    fitness: float,
    stability: float,
    gain_history: list[float],
    config: ForecastConfig,
) -> float:
    current_metric = prefix[-1].val_metric if prefix else 0.0
    discounted_gain = 0.0
    for idx, gain in enumerate(reversed(gain_history[-8:])):
        discounted_gain += (config.gamma_discount**idx) * gain
    # Heuristic mixture:
    # - 0.58 current observed metric: strongest anchor to actual measured progress
    # - 0.24 growth fitness: inherited trajectory quality
    # - 0.12 stability: whether the pattern has settled
    # - 0.06 discounted gain: recent momentum bonus
    return _clamp(
        config.current_metric_weight * current_metric
        + config.fitness_weight * fitness
        + config.stability_weight * stability
        + config.discounted_gain_weight * discounted_gain
    )


def _estimate_uncertainty(prefix: list[TrainingStep], prediction: float, config: ForecastConfig) -> float:
    metrics = [step.val_metric for step in prefix]
    losses = [step.train_loss for step in prefix]
    # Heuristic uncertainty composition:
    # - prefix_penalty: short traces should stay uncertain
    # - volatility: noisy validation/loss curves reduce confidence
    # - maturity: if the prediction is far from the latest observed metric,
    #   treat the run as not fully "grown into" its predicted phenotype yet
    prefix_penalty = config.prefix_penalty_weight / math.sqrt(max(len(prefix), 1))
    volatility = (
        config.metric_volatility_weight * math.sqrt(_variance(metrics))
        + config.loss_volatility_weight * math.sqrt(_variance(losses))
    )
    maturity = abs(prediction - (metrics[-1] if metrics else 0.0)) * config.maturity_weight
    return _clamp(prefix_penalty + volatility + maturity, 0.0, 1.0)


def _confidence_from_uncertainty(sigma: float, config: ForecastConfig) -> float:
    # `tau` sets the scale of the uncertainty-to-confidence collapse.
    # Smaller tau makes confidence decay faster as uncertainty rises.
    tau = config.tau_confidence
    return _clamp(math.exp(-((sigma * sigma) / (tau * tau))))


def forecast_trace(trace: TrainingTrace, config: ForecastConfig = DEFAULT_FORECAST_CONFIG) -> ForecastResult:
    if not trace.steps:
        return ForecastResult(
            run_id=trace.run_id,
            predicted_final_score=0.0,
            uncertainty=1.0,
            confidence=0.0,
            earliest_decision_step=0,
            continue_training_recommended=True,
            stability=0.0,
            growth_fitness=0.0,
            growth_gain=0.0,
            reason="empty training trace",
            metadata={"target_metric": trace.target_metric},
        )

    reset_run_state(trace.run_id)
    z_prev = _zero_state()
    prev_fitness = 0.0
    fitness_history: list[float] = []
    gain_history: list[float] = []

    last_prediction = 0.0
    last_uncertainty = 1.0
    last_confidence = 0.0
    last_stability = 0.0
    last_fitness = 0.0
    last_gain = 0.0

    try:
        for idx, step in enumerate(trace.steps, start=1):
            _ensure_clean_context(trace, idx)
            prefix = trace.steps[:idx]
            previous = trace.steps[idx - 2] if idx > 1 else None
            local_signal = _encode_local_signal(step, previous)
            prefix_signal = _encode_prefix_shape(prefix)
            z_t = _update_growth_state(z_prev, local_signal, prefix_signal, config)
            fitness = _compute_growth_fitness(z_t, config)
            gain = _compute_growth_gain(fitness, prev_fitness)
            fitness_history.append(fitness)
            gain_history.append(gain)
            stability = _compute_stability(fitness_history, config)
            prediction = _predict_final_score(prefix, fitness, stability, gain_history, config)
            uncertainty = _estimate_uncertainty(prefix, prediction, config)
            confidence = _confidence_from_uncertainty(uncertainty, config)

            _GROWTH_CACHE[trace.run_id] = z_t
            _PREDICTION_CACHE[trace.run_id].append(prediction)
            _UNCERTAINTY_CACHE[trace.run_id].append(uncertainty)

            last_prediction = prediction
            last_uncertainty = uncertainty
            last_confidence = confidence
            last_stability = stability
            last_fitness = fitness
            last_gain = gain

            if idx >= max(config.min_prefix_steps, config.delta_steps + 1):
                previous_prediction = _PREDICTION_CACHE[trace.run_id][-1 - config.delta_steps]
                prediction_shift = abs(prediction - previous_prediction)
                if (
                    prediction_shift < config.epsilon_pred
                    and uncertainty < config.epsilon_uncertainty
                    and stability > config.epsilon_stability
                ):
                    return ForecastResult(
                        run_id=trace.run_id,
                        predicted_final_score=prediction,
                        uncertainty=uncertainty,
                        confidence=confidence,
                        earliest_decision_step=step.step,
                        continue_training_recommended=False,
                        stability=stability,
                        growth_fitness=fitness,
                        growth_gain=gain,
                        reason="prediction stabilized and uncertainty is low",
                        metadata={
                            "target_metric": trace.target_metric,
                            "decision_index": idx,
                            "prediction_shift": round(prediction_shift, 6),
                        },
                    )

            z_prev = z_t
            prev_fitness = fitness

        return ForecastResult(
            run_id=trace.run_id,
            predicted_final_score=last_prediction,
            uncertainty=last_uncertainty,
            confidence=last_confidence,
            earliest_decision_step=trace.steps[-1].step,
            continue_training_recommended=True,
            stability=last_stability,
            growth_fitness=last_fitness,
            growth_gain=last_gain,
            reason="signal not yet stable enough for early decision",
            metadata={
                "target_metric": trace.target_metric,
                "decision_index": len(trace.steps),
            },
        )
    finally:
        _end_run(trace)


def forecast_payload(trace: TrainingTrace, config: ForecastConfig = DEFAULT_FORECAST_CONFIG) -> dict[str, object]:
    return asdict(forecast_trace(trace, config=config))


def forecast_report(trace: TrainingTrace, config: ForecastConfig = DEFAULT_FORECAST_CONFIG) -> str:
    result = forecast_trace(trace, config=config)
    lines = [
        "Tensorearch forecast report",
        f"run_id={result.run_id}",
        f"target_metric={trace.target_metric}",
        f"predicted_final_score={result.predicted_final_score:.4f}",
        f"uncertainty={result.uncertainty:.4f}",
        f"confidence={result.confidence:.4f}",
        f"earliest_decision_step={result.earliest_decision_step}",
        f"continue_training_recommended={'true' if result.continue_training_recommended else 'false'}",
        f"stability={result.stability:.4f}",
        f"growth_fitness={result.growth_fitness:.4f}",
        f"growth_gain={result.growth_gain:.4f}",
        f"reason={result.reason}",
    ]
    return "\n".join(lines)


def forecast_report_json(trace: TrainingTrace, config: ForecastConfig = DEFAULT_FORECAST_CONFIG) -> str:
    return json.dumps(forecast_payload(trace, config=config), ensure_ascii=False, indent=2)
