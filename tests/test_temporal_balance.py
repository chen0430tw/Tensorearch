"""Regression tests for the temporal-balance probe.

Three synthetic cases cover the verdict taxonomy:

  1. potential_consistent — response perfectly geostrophic to potential
  2. static_forcing_overrides — adding a static forcing flips the
     effective balance direction (the 2026-04-22 TD bug shape)
  3. magnitude mismatch — direction correct but speed 4× too large
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from tensorearch.temporal_balance import (
    BalanceOperator, BalanceSpec,
    analyze_temporal_balance, analyze_temporal_balance_file,
    load_spec_from_dict,
    temporal_balance_report, temporal_balance_report_json,
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _linear_ramp(T: int, H: int, W: int, direction: str) -> np.ndarray:
    """Scalar field with a uniform linear gradient in the given direction.

    direction ∈ {'x', 'y', '-x', '-y'}. Amplitude 1.0 per cell.
    """
    x = np.arange(W, dtype=np.float64)
    y = np.arange(H, dtype=np.float64)
    YY, XX = np.meshgrid(y, x, indexing="ij")
    if direction == "x":
        f = XX
    elif direction == "-x":
        f = -XX
    elif direction == "y":
        f = YY
    elif direction == "-y":
        f = -YY
    else:
        raise ValueError(direction)
    return np.broadcast_to(f[None, :, :], (T, H, W)).copy()


def _rotated_grad(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Helper — return (-∂/∂y, +∂/∂x) of field with dx=dy=1."""
    gy, gx = np.gradient(field, axis=(1, 2))
    return -gy, gx


# ---------------------------------------------------------------------------
# Case 1: potential_consistent
# ---------------------------------------------------------------------------

def _case_potential_consistent():
    T, H, W = 10, 8, 12
    # Potential: +x ramp ⇒ rotated_grad is purely in +y.
    pot = _linear_ramp(T, H, W, "x")
    bx, by = _rotated_grad(pot)
    arrays = {"h": pot, "u": bx, "v": by}
    spec = BalanceSpec(
        potential_key="h", response_u_key="u", response_v_key="v",
        operator=BalanceOperator(kind="rotated_gradient", scale=1.0),
    )
    spec.analysis.time_bins = 4
    spec.analysis.y_bins = 4
    spec.analysis.x_bins = 4
    return arrays, spec


def test_potential_consistent_synthetic():
    arrays, spec = _case_potential_consistent()
    report = analyze_temporal_balance(arrays, spec)
    g = report.global_
    assert g["alignment_potential_only_mean"] > 0.95
    assert g["alignment_combined_mean"]       > 0.95
    assert g["best_mode"] == "potential_only" or g["best_mode"] == "combined"
    assert g["verdict"] == "potential_consistent"


# ---------------------------------------------------------------------------
# Case 2: static_forcing_overrides_potential_balance
# ---------------------------------------------------------------------------

def _case_static_overrides():
    """The TD-style bug shape:
        potential = +x ramp   → rotated_grad = (0, +1)    (response picks this)
        static    = -x ramp   → rotated_grad = (0, -1)
        combined  = (1-w)·x   → rotated_grad = (0, 1-w)
        With weight = 10, combined rotated = (0, -9), anti-parallel to response.
        → alignment_potential ≈ +1 BUT alignment_combined ≈ -1
        → exact ``static_forcing_overrides_potential_balance`` shape.
    """
    from tensorearch.temporal_balance import StaticForcingSpec
    T, H, W = 10, 8, 12
    pot = _linear_ramp(T, H, W, "x")
    r_x, r_y = _rotated_grad(pot)          # response tracks potential_only
    static2d = _linear_ramp(T, H, W, "-x")[0]   # same axis, opposite sign
    arrays = {"h": pot, "u": r_x, "v": r_y, "topo": static2d}
    spec = BalanceSpec(
        potential_key="h", response_u_key="u", response_v_key="v",
        operator=BalanceOperator(kind="rotated_gradient", scale=1.0),
    )
    spec.static_forcings.append(StaticForcingSpec(key="topo", weight=10.0))
    spec.analysis.time_bins = 4
    spec.analysis.y_bins = 4
    spec.analysis.x_bins = 4
    return arrays, spec


def test_static_forcing_overrides_synthetic():
    arrays, spec = _case_static_overrides()
    report = analyze_temporal_balance(arrays, spec)
    g = report.global_
    # potential_only alignment is still strong
    assert g["alignment_potential_only_mean"] > 0.8
    # combined alignment has been dragged negative by the huge static
    assert g["alignment_combined_mean"] < 0.0
    # verdict must flag the override
    assert g["verdict"] == "static_forcing_overrides_potential_balance"


# ---------------------------------------------------------------------------
# Case 3: magnitude mismatch
# ---------------------------------------------------------------------------

def _case_magnitude_overpowered():
    T, H, W = 10, 8, 12
    pot = _linear_ramp(T, H, W, "x")
    bx, by = _rotated_grad(pot)
    scale = 8.0  # response is 8× stronger than theoretical
    arrays = {"h": pot, "u": scale * bx, "v": scale * by}
    spec = BalanceSpec(
        potential_key="h", response_u_key="u", response_v_key="v",
        operator=BalanceOperator(kind="rotated_gradient", scale=1.0),
    )
    spec.analysis.time_bins = 4
    spec.analysis.y_bins = 4
    spec.analysis.x_bins = 4
    return arrays, spec


def _case_magnitude_underpowered():
    T, H, W = 10, 8, 12
    pot = _linear_ramp(T, H, W, "x")
    bx, by = _rotated_grad(pot)
    scale = 0.1  # response is 10× weaker
    arrays = {"h": pot, "u": scale * bx, "v": scale * by}
    spec = BalanceSpec(
        potential_key="h", response_u_key="u", response_v_key="v",
        operator=BalanceOperator(kind="rotated_gradient", scale=1.0),
    )
    spec.analysis.time_bins = 4
    spec.analysis.y_bins = 4
    spec.analysis.x_bins = 4
    return arrays, spec


def test_magnitude_overpowered_synthetic():
    arrays, spec = _case_magnitude_overpowered()
    report = analyze_temporal_balance(arrays, spec)
    g = report.global_
    # Direction is still near-perfect
    assert g["alignment_potential_only_mean"] > 0.95
    # But magnitude is much larger than potential_only balance implies
    assert g["verdict"] == "response_magnitude_overpowered"


def test_magnitude_underpowered_synthetic():
    arrays, spec = _case_magnitude_underpowered()
    report = analyze_temporal_balance(arrays, spec)
    g = report.global_
    assert g["alignment_potential_only_mean"] > 0.95
    assert g["verdict"] == "response_magnitude_underpowered"


# ---------------------------------------------------------------------------
# Schema / CLI format tests
# ---------------------------------------------------------------------------

def test_report_schema_has_required_fields():
    arrays, spec = _case_potential_consistent()
    report = analyze_temporal_balance(arrays, spec)
    payload = json.loads(temporal_balance_report_json(report))
    assert payload["version"] == "trbalance_report.v1"
    for key in ("alignment_potential_only_mean", "alignment_static_only_mean",
                "alignment_combined_mean", "speed_potential_only_mean",
                "speed_static_only_mean", "speed_combined_mean",
                "consistency_gap_mean", "best_mode", "verdict"):
        assert key in payload["global"], f"global.{key} missing"
    assert isinstance(payload["time_series"], list)
    assert isinstance(payload["windows"], list)
    # Top windows carry per-window verdicts
    for row in payload["windows"]:
        for key in ("time_bin", "y_bin", "x_bin",
                    "alignment_potential", "alignment_static", "alignment_combined",
                    "consistency_gap_combined", "verdict", "confidence"):
            assert key in row, f"window.{key} missing"


def test_load_spec_from_dict_roundtrip():
    spec_dict = {
        "version": "trbalance.v1",
        "meta": {"case_id": "synthetic", "dt": 1.0, "dx": 1.0, "dy": 1.0},
        "fields": {
            "potential": "h",
            "response_u": "u",
            "response_v": "v",
            "static": {"topo": {"weight": 0.5}},
        },
        "operator": {"kind": "rotated_gradient", "scale": 1.0},
        "analysis": {"time_bins": 4, "y_bins": 4, "x_bins": 4},
    }
    spec = load_spec_from_dict(spec_dict)
    assert spec.potential_key == "h"
    assert spec.operator.kind == "rotated_gradient"
    assert len(spec.static_forcings) == 1
    assert spec.static_forcings[0].key == "topo"
    assert spec.static_forcings[0].weight == 0.5
    assert spec.analysis.time_bins == 4


def test_file_roundtrip(tmp_path):
    arrays, spec = _case_potential_consistent()
    probe = tmp_path / "probe.npz"
    np.savez(probe, **arrays)
    report = analyze_temporal_balance_file(probe, spec)
    text = temporal_balance_report(report)
    assert "Temporal Balance" in text
    assert "verdict" in text


def test_missing_required_field_errors():
    spec = BalanceSpec()
    with pytest.raises(KeyError):
        analyze_temporal_balance({}, spec)
    with pytest.raises(KeyError):
        analyze_temporal_balance({"h": np.zeros((3, 4, 4))}, spec)
