"""Tests for the temporal-topology probe (对称循环数)."""
import math

import numpy as np
import pytest

from tensorearch.temporal import (
    analyze_temporal_topology,
    temporal_report,
    temporal_report_json,
)


def _stable_wave(T=200, N=64, dt=0.1, omega=0.3):
    """Benign traveling wave — low ωΔt, unit amplitude, no growth."""
    t = np.arange(T) * dt
    x = np.arange(N)
    u = np.cos(omega * t[:, None] - 0.1 * x[None, :])
    return u


def _checkerboard_blowup(T=200, N=64, growth=1.01):
    """2Δt checkerboard mode with mild amplitude growth — CFL-violation proxy."""
    u = np.zeros((T, N))
    rng = np.random.default_rng(0)
    u[0] = rng.normal(0, 1, size=N)
    for k in range(1, T):
        u[k] = -growth * u[k - 1]  # flip sign + grow
    return u


def _stable_quarter_period(T=200, N=64):
    """ωΔt = π/2 resonance, unit amplitude, no growth — benign."""
    k = np.arange(T)[:, None]
    phase = 0.5 * math.pi * k + np.arange(N)[None, :] * 0.2
    return np.cos(phase)


def test_stable_wave_is_flagged_stable():
    u = _stable_wave()
    r = analyze_temporal_topology(u)
    assert r.verdict == "stable", f"expected stable, got {r.verdict} ({r.verdict_reason})"
    assert r.cfl_checkerboard_fraction < 0.01
    assert r.rho_median < 1.05


def test_checkerboard_flags_cfl_violation():
    u = _checkerboard_blowup()
    r = analyze_temporal_topology(u)
    assert r.verdict == "CFL_violation", (
        f"expected CFL_violation, got {r.verdict} ({r.verdict_reason})"
    )
    assert r.cfl_checkerboard_fraction > 0.5, (
        f"checkerboard fraction {r.cfl_checkerboard_fraction} — probe too lax"
    )


def test_quarter_period_is_stable_not_cfl():
    u = _stable_quarter_period()
    r = analyze_temporal_topology(u)
    assert r.verdict in {"stable", "mild_growth"}, f"got {r.verdict}"
    assert r.cfl_checkerboard_fraction < 0.01
    # Quarter-period signature should register
    assert r.quarter_resonance_fraction > 0.5


def test_runaway_growth_flagged_diverging():
    T, N = 30, 16
    u = np.ones((T, N))
    for k in range(1, T):
        u[k] = 3.0 * u[k - 1]  # pure growth, no sign flip
    r = analyze_temporal_topology(u)
    assert r.verdict in {"diverging", "CFL_violation"}
    assert r.rho_max > 2.0


def test_shape_too_short_raises():
    with pytest.raises(ValueError):
        analyze_temporal_topology(np.zeros((2, 4)))


def test_report_text_and_json_roundtrip():
    u = _checkerboard_blowup()
    r = analyze_temporal_topology(u)
    txt = temporal_report(r)
    assert "CFL_VIOLATION" in txt.upper()
    assert "对称循环数" in txt
    js = temporal_report_json(r)
    import json as _json
    d = _json.loads(js)
    assert d["verdict"] == "CFL_violation"
    assert d["shape"]["T"] == u.shape[0]


def test_hotspots_localize_instability():
    """Only cells 10-15 blow up; others stable. Hotspots should report those."""
    T, N = 100, 64
    rng = np.random.default_rng(42)
    u = np.cos(0.2 * np.arange(T)[:, None]) + 0.01 * rng.normal(size=(T, N))
    for k in range(1, T):
        u[k, 10:16] = -1.02 * u[k - 1, 10:16]
    r = analyze_temporal_topology(u, top_k_hotspots=6)
    flagged_coords = {h["coord"][0] for h in r.hotspots}
    assert flagged_coords.issubset(set(range(10, 16))), (
        f"hotspots {flagged_coords} leaked outside 10-15"
    )


def test_2d_spatial_works():
    """Sanity: (T, H, W) array accepted, hotspots carry 2D coords."""
    T, H, W = 60, 8, 8
    rng = np.random.default_rng(1)
    u = 0.1 * rng.normal(size=(T, H, W))
    for k in range(1, T):
        u[k, 3, 4] = -1.05 * u[k - 1, 3, 4]
    r = analyze_temporal_topology(u, top_k_hotspots=3)
    assert r.shape_spatial == [H, W]
    top_coord = r.hotspots[0]["coord"]
    assert top_coord == [3, 4], f"expected hotspot at [3,4], got {top_coord}"
