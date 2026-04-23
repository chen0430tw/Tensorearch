from __future__ import annotations

"""
Training-trace contract validator.

The forecast / zombie modules consume TrainingTrace JSONs that are written by
external training scripts. Without a contract check, badly-written traces can
silently mislead the analysis (e.g. a display-smoothed loss looks like a
sawtooth-noisy raw loss to the forecaster, and the forecaster's curvature /
direction-consistency proxies become garbage).

This module performs cheap structural checks on a TrainingTrace and returns
warnings that downstream layers can either surface in their reports or use to
gate their own confidence. It does NOT modify the trace.

Output schema (`ContractReport`):
    valid_for_forecast: bool   - safe to feed to the forecaster
    valid_for_zombie:   bool   - safe to feed to the zombie gate
    warnings:           list[ContractWarning]

A `valid_for_*` flag is False when one or more critical warnings would make
that layer's output unreliable. SUSPECT-level warnings do NOT flip the flag —
they only enrich the report.
"""

import math
from dataclasses import asdict, dataclass, field
from typing import Literal

from .schema import TrainingStep, TrainingTrace


WarningSeverity = Literal["INFO", "WARN", "CRITICAL"]


@dataclass
class ContractWarning:
    code: str
    severity: WarningSeverity
    message: str
    evidence: dict = field(default_factory=dict)


@dataclass
class ContractReport:
    valid_for_forecast: bool
    valid_for_zombie: bool
    warnings: list[ContractWarning] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "valid_for_forecast": self.valid_for_forecast,
            "valid_for_zombie": self.valid_for_zombie,
            "warnings": [asdict(w) for w in self.warnings],
        }


def _is_finite(x: float) -> bool:
    return math.isfinite(x) if isinstance(x, float) else True


def _detect_sawtooth(losses: list[float], period: int) -> bool:
    """Detect display-smoothing reset pattern: every `period` steps the loss
    drops sharply (closer to single-batch raw value), then climbs back to a
    rolling-average plateau before the next reset.

    Heuristic: look at first-differences. If they alternate in sign with a
    period close to log_interval, that's the display-smoothing artifact.
    Returns True on strong evidence of display smoothing.
    """
    if len(losses) < period * 2:
        return False
    diffs = [losses[i + 1] - losses[i] for i in range(len(losses) - 1)]
    # count sign-flips at every 'period' boundary
    flip_at_boundary = 0
    other_flips = 0
    for i, d in enumerate(diffs):
        sign_changes = (i > 0) and (diffs[i - 1] * d < 0)
        if not sign_changes:
            continue
        if (i + 1) % period == 0:
            flip_at_boundary += 1
        else:
            other_flips += 1
    n_periods = max(1, len(diffs) // period)
    boundary_rate = flip_at_boundary / n_periods
    return boundary_rate > 0.6


def validate_trace(
    trace: TrainingTrace,
    *,
    suspected_log_interval: int | None = None,
) -> ContractReport:
    """Run all contract checks. `suspected_log_interval`, if provided, is used
    to look for display-smoothing sawtooth at that period. Pass None to skip
    sawtooth analysis (no false positives on traces that just look noisy)."""
    warnings: list[ContractWarning] = []
    valid_for_forecast = True
    valid_for_zombie = True

    if not trace.steps:
        warnings.append(ContractWarning(
            code="EMPTY_TRACE",
            severity="CRITICAL",
            message="Trace has no steps; nothing to analyze.",
        ))
        return ContractReport(False, False, warnings)

    # 1. train_loss_kind — display_smoothed / unknown -> warn (forecast-critical)
    kinds = {s.train_loss_kind for s in trace.steps}
    if "display_smoothed" in kinds:
        warnings.append(ContractWarning(
            code="DISPLAY_SMOOTHED_LOSS",
            severity="CRITICAL",
            message=(
                "train_loss_kind='display_smoothed' detected — the writer is "
                "passing display-smoothed values, not raw per-step loss. "
                "Curvature and direction_consistency derived from this are "
                "unreliable; forecast confidence will be artificially low."
            ),
            evidence={"kinds_seen": sorted(kinds)},
        ))
        valid_for_forecast = False
    elif kinds == {"unknown"}:
        warnings.append(ContractWarning(
            code="UNKNOWN_LOSS_KIND",
            severity="WARN",
            message=(
                "train_loss_kind='unknown' for all steps — the writer did not "
                "declare what kind of loss it recorded. Assuming raw step loss "
                "but cannot verify."
            ),
        ))
        # Don't flip valid_for_forecast — old traces look like this.

    # 2. val_metric_observed — never observed -> forecast trusts val_metric=0
    if not any(s.val_metric_observed for s in trace.steps):
        observed_pct = 100.0 * sum(1 for s in trace.steps if s.val_metric > 0) / len(trace.steps)
        warnings.append(ContractWarning(
            code="VAL_METRIC_NEVER_OBSERVED",
            severity="WARN",
            message=(
                "val_metric_observed=False for every step. val_metric values "
                "in the trace are placeholders (likely 0 or copied from "
                "train_loss). Forecast's predicted_final_score should not be "
                "interpreted as a calibrated final accuracy."
            ),
            evidence={"steps_with_nonzero_val_metric_pct": round(observed_pct, 1)},
        ))
        # Forecast still works (signal-poverty already handles this) — don't flip.

    # 3. grad_norm_kind / gradient_clip pairing
    grad_kinds = {s.grad_norm_kind for s in trace.steps}
    if grad_kinds == {"unknown"}:
        warnings.append(ContractWarning(
            code="UNKNOWN_GRAD_NORM_KIND",
            severity="WARN",
            message=(
                "grad_norm_kind='unknown' for all steps — zombie cannot tell if "
                "grad_norm is pre- or post-clip. Falling back to legacy single-"
                "point heuristic."
            ),
        ))
        valid_for_zombie = False  # zombie only safe for NaN/Inf detection
    elif "pre_clip" in grad_kinds and not any(s.gradient_clip > 0 for s in trace.steps):
        warnings.append(ContractWarning(
            code="PRE_CLIP_WITHOUT_THRESHOLD",
            severity="WARN",
            message=(
                "grad_norm is pre-clip but gradient_clip is unset (0 for all "
                "steps). Zombie can apply abs/spike rule but cannot compare "
                "to clip threshold. Trace writer should record the active "
                "gradient_clip value each step."
            ),
        ))

    # 4. Sawtooth detection (if log interval suspected)
    if suspected_log_interval and suspected_log_interval >= 2:
        finite_losses = [s.train_loss for s in trace.steps if _is_finite(s.train_loss)]
        if _detect_sawtooth(finite_losses, suspected_log_interval):
            warnings.append(ContractWarning(
                code="LOSS_SAWTOOTH_PATTERN",
                severity="CRITICAL",
                message=(
                    f"Loss series shows a sign-flip period of {suspected_log_interval} "
                    "consistent with display-smoothing reset (running_loss / N reset "
                    "every N steps). Forecast / zombie trusting this are likely "
                    "consuming display values, not raw per-step loss."
                ),
                evidence={"suspected_period": suspected_log_interval},
            ))
            valid_for_forecast = False

    # 5. NaN / Inf in raw loss series — forecast cannot extrapolate cleanly
    nan_count = sum(1 for s in trace.steps if not _is_finite(s.train_loss))
    if nan_count > 0:
        warnings.append(ContractWarning(
            code="LOSS_NONFINITE",
            severity="INFO",
            message=f"{nan_count} step(s) have non-finite train_loss. Zombie "
                    "will detect (NAN_HOST / INF_HOST); forecast skips them.",
        ))

    return ContractReport(
        valid_for_forecast=valid_for_forecast,
        valid_for_zombie=valid_for_zombie,
        warnings=warnings,
    )
