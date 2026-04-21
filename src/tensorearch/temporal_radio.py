"""Temporal radio probe for vector-field diagnostics.

This module turns time-domain diagnosis into a scan -> lock workflow:
1. scan coarse time/spatial bins for anomalous vector-field behaviour
2. lock onto the strongest channel
3. emit reversible gate coordinates and CRT residues for precise follow-up

MVP scope:
  - required fields: u(T,H,W), v(T,H,W)
  - optional references: h(T,H,W), bg_u/bg_v(H,W), obs_u/obs_v(H,W)
  - outputs: scan + lock JSON in tradio.v1 format
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .temporal import analyze_temporal_topology


_TRADIO_VERSION = "tradio.v1"
_CRT_MODULI = (31, 37, 41)


@dataclass
class TemporalRadioChannel:
    bank_id: int
    omega: float
    activation: float
    energy: float
    dominant_fields: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "bank_id": self.bank_id,
            "omega": self.omega,
            "activation": self.activation,
            "energy": self.energy,
            "dominant_fields": self.dominant_fields,
        }


@dataclass
class TemporalRadioLock:
    field: str
    time_bin: int
    y_bin: int
    x_bin: int
    score: float
    gate_id: int
    time_window: list[int]
    spatial_window: dict[str, int]
    direction_metrics: dict[str, float | None]
    crt_locator: dict[str, Any]
    hotspots: list[dict[str, Any]]
    verdict: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "time_window": self.time_window,
            "spatial_window": self.spatial_window,
            "score": self.score,
            "gate_id": self.gate_id,
            "direction_metrics": self.direction_metrics,
            "crt_locator": self.crt_locator,
            "hotspots": self.hotspots,
            "verdict": self.verdict,
        }


@dataclass
class TemporalRadioReport:
    case_id: str
    dt: float
    grid_shape: list[int]
    time_steps: int
    time_bins: int
    y_bins: int
    x_bins: int
    field_scores: dict[str, dict[str, float]]
    vector_scores: dict[str, float | None]
    coupled_scores: dict[str, float | None]
    frequency_banks: list[TemporalRadioChannel]
    top_channels: list[dict[str, Any]]
    lock: TemporalRadioLock
    references_available: dict[str, bool]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": _TRADIO_VERSION,
            "meta": {
                "case_id": self.case_id,
                "dt": self.dt,
                "grid_shape": self.grid_shape,
                "time_steps": self.time_steps,
            },
            "gate_indexing": {
                "fields": ["uv"],
                "time_bins": self.time_bins,
                "y_bins": self.y_bins,
                "x_bins": self.x_bins,
                "flatten_order": ["field", "time_bin", "y_bin", "x_bin"],
            },
            "references_available": self.references_available,
            "field_scores": self.field_scores,
            "vector_scores": self.vector_scores,
            "coupled_scores": self.coupled_scores,
            "scan": {
                "frequency_banks": [b.to_dict() for b in self.frequency_banks],
                "top_channels": self.top_channels,
            },
            "lock": self.lock.to_dict(),
            "metadata": self.metadata,
        }


def _load_radio_input(path: str | Path) -> tuple[dict[str, np.ndarray], float]:
    p = Path(path)
    if p.suffix.lower() == ".npz":
        with np.load(p) as npz:
            arrays = {k: np.asarray(npz[k]) for k in npz.files if k != "dt"}
            dt = float(np.asarray(npz["dt"]).item()) if "dt" in npz.files else 1.0
        return arrays, dt
    if p.suffix.lower() == ".json":
        payload = json.loads(p.read_text(encoding="utf-8"))
        arrays = {k: np.asarray(v) for k, v in payload.items() if k != "dt"}
        dt = float(payload.get("dt", 1.0))
        return arrays, dt
    raise ValueError("temporal-radio expects .npz or .json input")


def _bin_edges(length: int, bins: int) -> list[tuple[int, int]]:
    edges = np.linspace(0, length, bins + 1, dtype=int)
    out: list[tuple[int, int]] = []
    for i in range(bins):
        a = int(edges[i])
        b = int(edges[i + 1])
        if b <= a:
            b = min(length, a + 1)
        out.append((a, b))
    return out


def _grad_xy(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gy, gx = np.gradient(h)
    return gx, gy


def _norm2(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    return np.sqrt(a * a + b * b + eps)


def _coarsen_mean(arr: np.ndarray, y_bins: int, x_bins: int) -> np.ndarray:
    H, W = arr.shape
    y_edges = _bin_edges(H, y_bins)
    x_edges = _bin_edges(W, x_bins)
    out = np.zeros((y_bins, x_bins), dtype=np.float64)
    for iy, (ya, yb) in enumerate(y_edges):
        for ix, (xa, xb) in enumerate(x_edges):
            out[iy, ix] = float(arr[ya:yb, xa:xb].mean())
    return out


def _mean_or_none(arr: np.ndarray | None) -> float | None:
    if arr is None:
        return None
    return float(arr.mean())


def _alignment(u: np.ndarray, v: np.ndarray, ref_u: np.ndarray, ref_v: np.ndarray) -> np.ndarray:
    return (u * ref_u[None] + v * ref_v[None]) / (_norm2(u, v) * _norm2(ref_u[None], ref_v[None]) + 1e-9)


def _merge_hotspots(*hotspot_lists: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
    merged: dict[tuple[int, ...], dict[str, Any]] = {}
    for hotspots in hotspot_lists:
        for row in hotspots:
            coord = tuple(int(c) for c in row["coord"])
            slot = merged.setdefault(
                coord,
                {"coord": list(coord), "cumulative_score": 0.0, "flat_index": int(row["flat_index"])},
            )
            slot["cumulative_score"] += float(row["cumulative_score"])
    ranked = sorted(merged.values(), key=lambda x: (-x["cumulative_score"], x["coord"]))
    return ranked[:top_k]


def _flatten_gate_id(time_bin: int, y_bin: int, x_bin: int, *, y_bins: int, x_bins: int) -> int:
    return x_bin + x_bins * (y_bin + y_bins * time_bin)


def _crt_residues(gate_id: int, moduli: tuple[int, ...] = _CRT_MODULI) -> list[int]:
    return [int(gate_id % m) for m in moduli]


def _frequency_banks(scores: np.ndarray, beta: float = 2.0, n_banks: int = 4) -> list[TemporalRadioChannel]:
    gate_ids = np.arange(scores.size, dtype=np.float64)
    total_energy = float(scores.sum()) + 1e-12
    banks: list[TemporalRadioChannel] = []
    for bank_id in range(n_banks):
        omega = beta ** bank_id
        phase = (2.0 * math.pi * gate_ids) / max(2.0, scores.size / omega)
        vec_x = float(np.sum(scores * np.cos(phase)))
        vec_y = float(np.sum(scores * np.sin(phase)))
        activation = float(np.hypot(vec_x, vec_y) / total_energy)
        energy = float(total_energy / scores.size)
        banks.append(
            TemporalRadioChannel(
                bank_id=bank_id,
                omega=float(omega),
                activation=activation,
                energy=energy,
                dominant_fields=["uv"],
            )
        )
    return banks


def analyze_temporal_radio(
    arrays: dict[str, np.ndarray],
    *,
    dt: float,
    case_id: str = "",
    time_bins: int = 8,
    y_bins: int = 12,
    x_bins: int = 16,
) -> TemporalRadioReport:
    if "u" not in arrays or "v" not in arrays:
        raise KeyError("temporal-radio requires keys 'u' and 'v'")

    u = np.asarray(arrays["u"], dtype=np.float64)
    v = np.asarray(arrays["v"], dtype=np.float64)
    if u.shape != v.shape or u.ndim != 3:
        raise ValueError(f"u/v must have identical shape (T,H,W), got {u.shape} and {v.shape}")
    T_steps, H, W = u.shape

    h = np.asarray(arrays["h"], dtype=np.float64) if "h" in arrays else None
    bg_u = np.asarray(arrays["bg_u"], dtype=np.float64) if "bg_u" in arrays else None
    bg_v = np.asarray(arrays["bg_v"], dtype=np.float64) if "bg_v" in arrays else None
    obs_u = np.asarray(arrays["obs_u"], dtype=np.float64) if "obs_u" in arrays else None
    obs_v = np.asarray(arrays["obs_v"], dtype=np.float64) if "obs_v" in arrays else None

    refs = {
        "h": h is not None,
        "background_uv": bg_u is not None and bg_v is not None,
        "obs_uv": obs_u is not None and obs_v is not None,
    }

    rep_u = analyze_temporal_topology(u, dt=dt)
    rep_v = analyze_temporal_topology(v, dt=dt)

    speed = _norm2(u, v)
    speed_prev = _norm2(u[:-1], v[:-1])
    speed_next = _norm2(u[1:], v[1:])
    dot = u[:-1] * u[1:] + v[:-1] * v[1:]
    cross = u[:-1] * v[1:] - v[:-1] * u[1:]
    cosang = dot / (speed_prev * speed_next + 1e-9)
    sinang = cross / (speed_prev * speed_next + 1e-9)
    direction_drift = 0.5 * (1.0 - np.clip(cosang, -1.0, 1.0))
    growth = np.maximum(0.0, speed_next / (speed_prev + 1e-9) - 1.0)

    if refs["background_uv"] and refs["obs_uv"]:
        d_bg = _norm2(u - bg_u[None], v - bg_v[None])
        d_obs = _norm2(u - obs_u[None], v - obs_v[None])
        background_lock = d_obs / (d_bg + d_obs + 1e-9)
        obs_response = d_bg / (d_bg + d_obs + 1e-9)
        background_alignment = _alignment(u, v, bg_u, bg_v)
        obs_alignment = _alignment(u, v, obs_u, obs_v)
    else:
        background_lock = None
        obs_response = None
        background_alignment = None
        obs_alignment = None

    if refs["h"]:
        gx = np.empty_like(h)
        gy = np.empty_like(h)
        for t in range(T_steps):
            gx[t], gy[t] = _grad_xy(h[t])
        rotgx = -gy
        rotgy = gx
        geo = (rotgx * u + rotgy * v) / (_norm2(rotgx, rotgy) * speed + 1e-9)
        geostrophic_coherence = geo
        geostrophic_misalignment = 0.5 * (1.0 - np.clip(geo, -1.0, 1.0))
        grad_mag = _norm2(gx, gy)
        d_grad = np.maximum(0.0, grad_mag[1:] / (grad_mag[:-1] + 1e-9) - 1.0)
    else:
        geostrophic_coherence = None
        geostrophic_misalignment = None
        d_grad = None

    t_edges = _bin_edges(T_steps - 1, min(time_bins, max(1, T_steps - 1)))
    y_edges = _bin_edges(H, min(y_bins, H))
    x_edges = _bin_edges(W, min(x_bins, W))

    gate_scores = []
    gate_rows: list[dict[str, Any]] = []
    for ti, (ta, tb) in enumerate(t_edges):
        for yi, (ya, yb) in enumerate(y_edges):
            for xi, (xa, xb) in enumerate(x_edges):
                drift = float(direction_drift[ta:tb, ya:yb, xa:xb].mean())
                grow = float(growth[ta:tb, ya:yb, xa:xb].mean())
                lock = float(background_lock[ta:tb, ya:yb, xa:xb].mean()) if background_lock is not None else 0.0
                obs_mis = (
                    float(0.5 * (1.0 - np.clip(obs_alignment[ta:tb, ya:yb, xa:xb], -1.0, 1.0)).mean())
                    if obs_alignment is not None
                    else 0.0
                )
                geo_mis = (
                    float(geostrophic_misalignment[ta + 1 : tb + 1, ya:yb, xa:xb].mean())
                    if geostrophic_misalignment is not None else 0.0
                )
                coupling = (
                    float(np.abs(geostrophic_coherence[ta + 1 : tb + 1, ya:yb, xa:xb]).mean())
                    if geostrophic_coherence is not None
                    else 0.0
                )
                grad_growth = float(d_grad[ta:tb, ya:yb, xa:xb].mean()) if d_grad is not None else 0.0
                score = 0.28 * drift + 0.22 * grow + 0.18 * lock + 0.12 * geo_mis + 0.10 * obs_mis + 0.10 * grad_growth
                gate_id = _flatten_gate_id(ti, yi, xi, y_bins=len(y_edges), x_bins=len(x_edges))
                gate_scores.append(score)
                gate_rows.append(
                    {
                        "field": "uv",
                        "time_bin": ti,
                        "time_window": [ta, tb],
                        "y_bin": yi,
                        "x_bin": xi,
                        "score": float(score),
                        "direction_drift": drift,
                        "growth": grow,
                        "background_lock": lock if background_lock is not None else None,
                        "obs_misalignment": obs_mis if obs_alignment is not None else None,
                        "geostrophic_misalignment": geo_mis if geostrophic_misalignment is not None else None,
                        "geostrophic_coupling": coupling if geostrophic_coherence is not None else None,
                        "grad_growth": grad_growth if d_grad is not None else None,
                        "gate_id": gate_id,
                    }
                )
    gate_scores_arr = np.asarray(gate_scores, dtype=np.float64)

    top_idx = np.argsort(-gate_scores_arr)[:5]
    top_channels = [
        {
            "rank": rank + 1,
            "channel_id": f"uv:t{gate_rows[idx]['time_bin']}:y{gate_rows[idx]['y_bin']}:x{gate_rows[idx]['x_bin']}",
            "score": float(gate_rows[idx]["score"]),
        }
        for rank, idx in enumerate(top_idx)
    ]
    banks = _frequency_banks(gate_scores_arr)

    best = gate_rows[int(top_idx[0])]
    ta, tb = best["time_window"]
    ya, yb = y_edges[best["y_bin"]]
    xa, xb = x_edges[best["x_bin"]]
    lock_metrics = {
        "direction_drift": float(direction_drift[ta:tb, ya:yb, xa:xb].mean()),
        "background_lock": float(background_lock[ta:tb, ya:yb, xa:xb].mean()) if background_lock is not None else None,
        "obs_response": float(obs_response[ta:tb, ya:yb, xa:xb].mean()) if obs_response is not None else None,
        "background_alignment": (
            float(background_alignment[ta:tb, ya:yb, xa:xb].mean()) if background_alignment is not None else None
        ),
        "obs_alignment": float(obs_alignment[ta:tb, ya:yb, xa:xb].mean()) if obs_alignment is not None else None,
        "geostrophic_coherence": (
            float(geostrophic_coherence[ta + 1 : tb + 1, ya:yb, xa:xb].mean())
            if geostrophic_coherence is not None else None
        ),
        "grad_growth": float(d_grad[ta:tb, ya:yb, xa:xb].mean()) if d_grad is not None else None,
    }
    verdict_kind = "wind_direction_drift"
    if lock_metrics["background_lock"] is not None and lock_metrics["background_lock"] > 0.6:
        verdict_kind = "background_locked_wind_divergence"
    if (
        lock_metrics["geostrophic_coherence"] is not None
        and lock_metrics["geostrophic_coherence"] < -0.15
        and lock_metrics["direction_drift"] > 0.03
    ):
        verdict_kind = "geostrophic_wind_misalignment"
    lock = TemporalRadioLock(
        field="uv",
        time_bin=int(best["time_bin"]),
        y_bin=int(best["y_bin"]),
        x_bin=int(best["x_bin"]),
        score=float(best["score"]),
        gate_id=int(best["gate_id"]),
        time_window=[int(ta), int(tb)],
        spatial_window={"y_bin": int(best["y_bin"]), "x_bin": int(best["x_bin"])},
        direction_metrics=lock_metrics,
        crt_locator={
            "moduli": list(_CRT_MODULI),
            "residues": _crt_residues(int(best["gate_id"])),
            "recovered_gate_id": int(best["gate_id"]),
        },
        hotspots=_merge_hotspots(rep_u.hotspots, rep_v.hotspots, top_k=5),
        verdict={
            "kind": verdict_kind,
            "confidence": round(min(1.0, 0.5 + float(best["score"])), 4),
        },
    )

    field_scores = {
        "u": {
            "stable_score": round(max(0.0, 1.0 - rep_u.diverging_fraction), 6),
            "diverge_score": round(rep_u.diverging_fraction, 6),
            "rho_max": round(rep_u.rho_max, 6),
        },
        "v": {
            "stable_score": round(max(0.0, 1.0 - rep_v.diverging_fraction), 6),
            "diverge_score": round(rep_v.diverging_fraction, 6),
            "rho_max": round(rep_v.rho_max, 6),
        },
        "uv": {
            "stable_score": round(max(0.0, 1.0 - float(gate_scores_arr.mean())), 6),
            "diverge_score": round(float(gate_scores_arr.max()), 6),
        },
    }
    vector_scores = {
        "direction_drift_mean": round(float(direction_drift.mean()), 6),
        "growth_mean": round(float(growth.mean()), 6),
        "background_lock_mean": round(float(background_lock.mean()), 6) if background_lock is not None else None,
        "obs_response_mean": round(float(obs_response.mean()), 6) if obs_response is not None else None,
        "background_alignment_mean": round(float(background_alignment.mean()), 6) if background_alignment is not None else None,
        "obs_alignment_mean": round(float(obs_alignment.mean()), 6) if obs_alignment is not None else None,
        "geostrophic_coherence_mean": (
            round(float(geostrophic_coherence.mean()), 6) if geostrophic_coherence is not None else None
        ),
    }
    coupled_scores = {
        "u_v_hotspot_overlap": round(
            float(
                len({tuple(h["coord"]) for h in rep_u.hotspots} & {tuple(h["coord"]) for h in rep_v.hotspots})
                / max(1, len({tuple(h["coord"]) for h in rep_u.hotspots} | {tuple(h["coord"]) for h in rep_v.hotspots}))
            ),
            6,
        ),
        "h_uv_coupling_mean": round(float(np.abs(geostrophic_coherence).mean()), 6) if geostrophic_coherence is not None else None,
        "h_uv_anti_geo_fraction": round(float((geostrophic_coherence < 0.0).mean()), 6) if geostrophic_coherence is not None else None,
        "grad_growth_mean": round(float(d_grad.mean()), 6) if d_grad is not None else None,
    }

    return TemporalRadioReport(
        case_id=case_id or Path(str(arrays.get("__source__", "temporal-radio"))).stem,
        dt=float(dt),
        grid_shape=[H, W],
        time_steps=T_steps,
        time_bins=len(t_edges),
        y_bins=len(y_edges),
        x_bins=len(x_edges),
        field_scores=field_scores,
        vector_scores=vector_scores,
        coupled_scores=coupled_scores,
        frequency_banks=banks,
        top_channels=top_channels,
        lock=lock,
        references_available=refs,
        metadata={
            "available_fields": sorted(k for k in arrays.keys() if not k.startswith("__")),
        },
    )


def analyze_temporal_radio_file(
    path: str | Path,
    *,
    dt: float | None = None,
    case_id: str = "",
    time_bins: int = 8,
    y_bins: int = 12,
    x_bins: int = 16,
) -> TemporalRadioReport:
    arrays, dt_from_file = _load_radio_input(path)
    arrays["__source__"] = np.asarray(str(path))
    return analyze_temporal_radio(
        arrays,
        dt=dt if dt is not None else dt_from_file,
        case_id=case_id or Path(path).stem,
        time_bins=time_bins,
        y_bins=y_bins,
        x_bins=x_bins,
    )


def temporal_radio_report(report: TemporalRadioReport) -> str:
    lines = []
    lines.append("=" * 68)
    lines.append("Tensorearch Temporal Radio (scan -> lock)")
    lines.append("=" * 68)
    lines.append(
        f"case={report.case_id} T={report.time_steps} grid={report.grid_shape} dt={report.dt}"
    )
    lines.append("")
    lines.append("field scores")
    for name, score in report.field_scores.items():
        lines.append(
            f"  {name}: stable={score['stable_score']:.4f} diverge={score['diverge_score']:.4f}"
        )
    lines.append("")
    lines.append("vector scores")
    for key, value in report.vector_scores.items():
        lines.append(f"  {key}={value}")
    lines.append("")
    lines.append("coupled scores")
    for key, value in report.coupled_scores.items():
        lines.append(f"  {key}={value}")
    lines.append("")
    lines.append("top channels")
    for row in report.top_channels:
        lines.append(f"  #{row['rank']} {row['channel_id']} score={row['score']:.4f}")
    lines.append("")
    lines.append("lock")
    lines.append(
        f"  field={report.lock.field} time={report.lock.time_window} "
        f"bin=({report.lock.y_bin},{report.lock.x_bin}) score={report.lock.score:.4f}"
    )
    lines.append(f"  verdict={report.lock.verdict['kind']} conf={report.lock.verdict['confidence']:.4f}")
    return "\n".join(lines)


def temporal_radio_report_json(report: TemporalRadioReport) -> str:
    return json.dumps(report.to_dict(), indent=2)
