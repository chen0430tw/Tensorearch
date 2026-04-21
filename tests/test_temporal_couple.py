import json

import numpy as np
import pytest

from tensorearch.temporal_couple import (
    analyze_temporal_couple,
    analyze_temporal_couple_file,
    temporal_couple_report,
    temporal_couple_report_json,
)


def _make_anti_geostrophic_probe(T=16, H=8, W=8):
    """Synthetic case: h has a localised bump in the top-left quadrant
    that grows with time, and uv flows AGAINST the geostrophic direction
    (i.e. parallel to +∇h instead of perpendicular). Verdict must be
    ``anti_geostrophic`` and the lock must land in the bump region.
    """
    rng = np.random.default_rng(0)
    h = np.zeros((T, H, W), dtype=np.float64)
    u = np.zeros_like(h)
    v = np.zeros_like(h)

    # Bump centred at (1, 1), growing quadratically in time.
    yy, xx = np.mgrid[0:H, 0:W]
    bump = np.exp(-((yy - 1.0) ** 2 + (xx - 1.0) ** 2) / 2.5)
    for t in range(T):
        h[t] = 0.05 * t * bump + 0.01 * rng.standard_normal(h[t].shape)

        # Inside the bump region (top-left), force uv ANTI-parallel to
        # the geostrophic direction. rotated ∇h = (-∂h/∂y, +∂h/∂x), so
        # for coherence = -1 we set uv = (+∂h/∂y, -∂h/∂x).
        gy_t, gx_t = np.gradient(h[t])  # np.gradient returns (axis0=y, axis1=x)
        mag = np.sqrt(gx_t ** 2 + gy_t ** 2) + 1e-6
        mask = (yy < 3) & (xx < 3)
        u[t] = np.where(mask,  gy_t / mag, 0.01 * rng.standard_normal(h[t].shape))
        v[t] = np.where(mask, -gx_t / mag, 0.01 * rng.standard_normal(h[t].shape))

    return {"h": h, "u": u, "v": v}


def _make_geostrophic_probe(T=12, H=8, W=8):
    """h has a linear ridge in x; uv flows perpendicular (+y) to ∇h,
    which is the geostrophic direction. Verdict should be
    ``coupled_geostrophic``."""
    h = np.zeros((T, H, W), dtype=np.float64)
    u = np.zeros_like(h)
    v = np.zeros_like(h)
    for t in range(T):
        h[t] = np.linspace(0.0, 1.0, W)[None, :].repeat(H, axis=0) * (1.0 + 0.02 * t)
        # ∇h points in +x. rotated ∇h (ẑ × ∇h) points in +y. Geostrophic
        # wind should be in +y direction, so v > 0 and u = 0.
        u[t, :, :] = 0.0
        v[t, :, :] = 1.0
    return {"h": h, "u": u, "v": v}


def test_couple_locks_anti_geostrophic_bin():
    arrays = _make_anti_geostrophic_probe()
    report = analyze_temporal_couple(
        arrays, dt=1.0, case_id="unit-anti-geo", time_bins=4, y_bins=2, x_bins=2
    )
    # Bump region is top-left → y_bin == 0 AND x_bin == 0.
    assert report.lock.y_bin == 0
    assert report.lock.x_bin == 0
    assert report.lock.verdict["kind"] == "anti_geostrophic"
    # Anti-geostrophic majority → fraction > 0.5 in the lock bin.
    assert report.lock.coupling_metrics["h_uv_anti_geo_fraction"] > 0.55


def test_couple_recognises_healthy_geostrophy():
    arrays = _make_geostrophic_probe()
    report = analyze_temporal_couple(
        arrays, dt=1.0, case_id="unit-geo", time_bins=4, y_bins=2, x_bins=2
    )
    # With uv perfectly along rotated ∇h everywhere, coupling magnitude is
    # essentially 1.0 and anti-geostrophic fraction is ~0.
    assert report.coupling_scores["h_uv_coupling_mean"] > 0.95
    assert report.coupling_scores["h_uv_anti_geo_fraction"] < 0.05
    assert report.lock.verdict["kind"] == "coupled_geostrophic"


def test_couple_schema_has_required_fields():
    arrays = _make_anti_geostrophic_probe()
    report = analyze_temporal_couple(arrays, dt=1.0, time_bins=4, y_bins=2, x_bins=2)
    payload = json.loads(temporal_couple_report_json(report))

    assert payload["version"] == "trcouple.v1"
    assert payload["meta"]["pair"] == ["h", "uv"]
    assert payload["gate_indexing"]["pair"] == "h-uv"

    cs = payload["coupling_scores"]
    for required in (
        "h_uv_coupling_mean",
        "h_uv_anti_geo_fraction",
        "grad_growth_mean",
        "geostrophic_coherence_mean",
    ):
        assert required in cs, f"missing coupling_scores.{required}"
        assert cs[required] is not None

    lk = payload["lock"]
    for required in ("time_window", "spatial_window", "direction_metrics",
                     "coupling_metrics", "crt_locator", "verdict"):
        assert required in lk, f"missing lock.{required}"

    assert lk["crt_locator"]["moduli"] == [31, 37, 41]
    assert len(lk["crt_locator"]["residues"]) == 3


def test_couple_without_references_keeps_core_metrics():
    # No bg_*/obs_* — only h/u/v. Coupling metrics that need refs should
    # come back as None, but h→uv coupling itself must still compute.
    arrays = _make_geostrophic_probe()
    report = analyze_temporal_couple(arrays, dt=1.0, time_bins=4, y_bins=2, x_bins=2)
    assert report.references_available == {
        "h": True, "background_uv": False, "obs_uv": False,
    }
    assert report.coupling_scores["h_uv_coupling_mean"] is not None
    assert report.coupling_scores["background_lock_mean"] is None
    assert report.coupling_scores["obs_response_mean"] is None


def test_couple_file_and_json_roundtrip(tmp_path):
    arrays = _make_anti_geostrophic_probe()
    probe = tmp_path / "probe.npz"
    np.savez(probe, dt=np.asarray(1.0), **arrays)

    report = analyze_temporal_couple_file(probe, time_bins=4, y_bins=2, x_bins=2)
    payload = json.loads(temporal_couple_report_json(report))
    assert payload["version"] == "trcouple.v1"

    text = temporal_couple_report(report)
    assert "Temporal Couple" in text
    assert "coupling scores" in text


def test_couple_requires_h_and_uv():
    # Missing h → explicit error.
    with pytest.raises(KeyError, match="h"):
        analyze_temporal_couple({"u": np.zeros((4, 3, 3)), "v": np.zeros((4, 3, 3))}, dt=1.0)
    # Missing u → explicit error.
    with pytest.raises(KeyError, match="u"):
        analyze_temporal_couple({"h": np.zeros((4, 3, 3)), "v": np.zeros((4, 3, 3))}, dt=1.0)


def test_couple_shape_validation():
    with pytest.raises(ValueError, match="shape"):
        analyze_temporal_couple(
            {"h": np.zeros((4, 3, 3)), "u": np.zeros((4, 3, 4)), "v": np.zeros((4, 3, 3))},
            dt=1.0,
        )
