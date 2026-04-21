import json

import numpy as np

from tensorearch.temporal_radio import (
    analyze_temporal_radio,
    analyze_temporal_radio_file,
    temporal_radio_report,
    temporal_radio_report_json,
)


def _make_radio_probe(T=12, H=6, W=6):
    u = np.ones((T, H, W), dtype=np.float64)
    v = np.zeros((T, H, W), dtype=np.float64)
    h = np.zeros((T, H, W), dtype=np.float64)
    bg_u = np.ones((H, W), dtype=np.float64)
    bg_v = np.zeros((H, W), dtype=np.float64)
    obs_u = np.zeros((H, W), dtype=np.float64)
    obs_v = np.ones((H, W), dtype=np.float64)

    for t in range(1, T):
        u[t] = u[t - 1]
        v[t] = v[t - 1]
        if t >= 4:
            u[t, :3, :3] = 1.25 * u[t - 1, :3, :3]
            v[t, :3, :3] = 0.0
        h[t] = h[t - 1]
        h[t, :, :] += np.linspace(0.0, 1.0, W)[None, :]

    return {
        "u": u,
        "v": v,
        "h": h,
        "bg_u": bg_u,
        "bg_v": bg_v,
        "obs_u": obs_u,
        "obs_v": obs_v,
    }


def test_temporal_radio_locks_expected_hotspot():
    arrays = _make_radio_probe()
    report = analyze_temporal_radio(arrays, dt=1.0, case_id="unit-probe", time_bins=4, y_bins=2, x_bins=2)
    assert report.lock.field == "uv"
    assert report.lock.y_bin == 0
    assert report.lock.x_bin == 0
    assert report.lock.verdict["kind"] == "background_locked_wind_divergence"
    assert report.vector_scores["background_lock_mean"] is not None
    assert report.vector_scores["geostrophic_coherence_mean"] is not None
    assert report.vector_scores["obs_alignment_mean"] is not None
    assert report.coupled_scores["h_uv_coupling_mean"] is not None


def test_temporal_radio_file_and_json_roundtrip(tmp_path):
    arrays = _make_radio_probe()
    probe = tmp_path / "probe.npz"
    np.savez(probe, dt=np.asarray(1.0), **arrays)
    report = analyze_temporal_radio_file(probe, time_bins=4, y_bins=2, x_bins=2)
    payload = json.loads(temporal_radio_report_json(report))
    assert payload["version"] == "tradio.v1"
    assert payload["lock"]["field"] == "uv"
    assert "coupled_scores" in payload
    text = temporal_radio_report(report)
    assert "Temporal Radio" in text
