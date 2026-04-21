"""Temporal couple probe — focused h→uv coupling diagnosis.

Promoted from the `geostrophic_*` / `grad_growth` fields that were previously
inlined in `temporal_radio`. Same underlying math, but:

  - `temporal-radio` = broadcast scan / lock: "which (T, Y, X) cell has the
    strongest vector-field anomaly?" (uv-only, h is treated as an optional
    auxiliary reference).
  - `temporal-couple` = targeted coupling diagnosis: "how well does uv track
    ∇h (geostrophic balance) in each (T, Y, X) cell, and where is the
    decoupling worst?" (h and uv are peers; both are required).

Scope
-----
This first revision is **deliberately scoped to the single pair (h, uv)**.
Generalisation to arbitrary field pairs is possible but would dilute the
diagnostic — the geostrophic relationship ∇h × ẑ ∝ uv is the specific
instance that matters for shallow-water / TD-style rollouts, and getting
that one right is the first step.

Output version: ``trcouple.v1``.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Import the small utilities from temporal_radio so the two tools stay in
# lock-step on bin edges, CRT residues, and gate-id flattening. This is
# pure utility reuse — no behavioural coupling in the user-facing API.
from .temporal_radio import (
    _bin_edges,
    _crt_residues,
    _flatten_gate_id,
    _grad_xy,
    _norm2,
    _CRT_MODULI,
)


_TRCOUPLE_VERSION = "trcouple.v1"


# --------------------------------------------------------------------------
# Result dataclasses
# --------------------------------------------------------------------------

@dataclass
class CoupleChannel:
    """One (time_bin, y_bin, x_bin) cell in the coupling scan."""
    time_bin: int
    y_bin: int
    x_bin: int
    time_window: list[int]
    score: float
    gate_id: int
    coupling_metrics: dict[str, float | None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_bin": self.time_bin,
            "y_bin": self.y_bin,
            "x_bin": self.x_bin,
            "time_window": self.time_window,
            "score": self.score,
            "gate_id": self.gate_id,
            "coupling_metrics": self.coupling_metrics,
        }


@dataclass
class CoupleLock:
    pair: str
    time_bin: int
    y_bin: int
    x_bin: int
    score: float
    gate_id: int
    time_window: list[int]
    spatial_window: dict[str, int]
    direction_metrics: dict[str, float | None]
    coupling_metrics: dict[str, float | None]
    crt_locator: dict[str, Any]
    verdict: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair": self.pair,
            "time_bin": self.time_bin,
            "y_bin": self.y_bin,
            "x_bin": self.x_bin,
            "score": self.score,
            "gate_id": self.gate_id,
            "time_window": self.time_window,
            "spatial_window": self.spatial_window,
            "direction_metrics": self.direction_metrics,
            "coupling_metrics": self.coupling_metrics,
            "crt_locator": self.crt_locator,
            "verdict": self.verdict,
        }


@dataclass
class TemporalCoupleReport:
    case_id: str
    dt: float
    grid_shape: list[int]
    time_steps: int
    time_bins: int
    y_bins: int
    x_bins: int
    references_available: dict[str, bool]
    coupling_scores: dict[str, float | None]
    top_channels: list[CoupleChannel]
    lock: CoupleLock
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": _TRCOUPLE_VERSION,
            "meta": {
                "case_id": self.case_id,
                "dt": self.dt,
                "grid_shape": self.grid_shape,
                "time_steps": self.time_steps,
                "pair": ["h", "uv"],
            },
            "gate_indexing": {
                "pair": "h-uv",
                "time_bins": self.time_bins,
                "y_bins": self.y_bins,
                "x_bins": self.x_bins,
                "flatten_order": ["time_bin", "y_bin", "x_bin"],
            },
            "references_available": self.references_available,
            "coupling_scores": self.coupling_scores,
            "scan": {
                "top_channels": [c.to_dict() for c in self.top_channels],
            },
            "lock": self.lock.to_dict(),
            "metadata": self.metadata,
        }


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------

def _load_couple_input(path: str | Path) -> tuple[dict[str, np.ndarray], float]:
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
    raise ValueError("temporal-couple expects .npz or .json input")


# --------------------------------------------------------------------------
# Core diagnosis
# --------------------------------------------------------------------------

def _verdict(
    *,
    coupling_mean: float,
    anti_geo_fraction: float,
    grad_growth: float,
    coherence_mean_signed: float,
    refs_ok: bool,
) -> dict[str, Any]:
    """Verdict taxonomy for the h→uv coupling diagnosis.

    Ordering matters — anti-geostrophic is checked before coupled-geostrophic
    because a region with majority-negative coherence can still have high
    |coherence| magnitude; the sign matters for the fault signature.
    """
    if not refs_ok:
        kind = "no_reference"
        conf = 0.0
    elif anti_geo_fraction >= 0.55 and coupling_mean >= 0.10:
        # Most cells flow AGAINST the geostrophic direction. This is the
        # §11.24 wind-direction climo-lock fault signature: h gradient
        # is non-zero, but uv is pointing the wrong way.
        kind = "anti_geostrophic"
        conf = min(1.0, 0.5 + 0.5 * anti_geo_fraction)
    elif coherence_mean_signed >= 0.60:
        # Healthy geostrophic balance — uv ≈ ẑ × ∇h.
        kind = "coupled_geostrophic"
        conf = min(1.0, 0.5 + 0.5 * coherence_mean_signed)
    elif grad_growth >= 0.05 and coupling_mean <= 0.15:
        # h is evolving (∇h magnitude growing) but uv isn't following —
        # the two fields have decoupled. Typical of advection-dominant
        # regimes that the closure can't resolve.
        kind = "h_decoupled_drift"
        conf = min(1.0, 0.5 + grad_growth)
    else:
        # Neither strongly aligned nor strongly anti-aligned — generic
        # weak coupling. Often the default on synoptic-quiet rollouts.
        kind = "weak_coupling"
        conf = max(0.0, 0.5 - 0.5 * coupling_mean)
    return {"kind": kind, "confidence": round(float(conf), 4)}


def analyze_temporal_couple(
    arrays: dict[str, np.ndarray],
    *,
    dt: float,
    case_id: str = "",
    time_bins: int = 8,
    y_bins: int = 12,
    x_bins: int = 16,
) -> TemporalCoupleReport:
    """Analyse h→uv coupling at coarse (time, y, x) bins.

    Required keys: ``h``, ``u``, ``v`` (each shape ``(T, H, W)``).
    Optional: ``bg_u`` / ``bg_v`` / ``obs_u`` / ``obs_v`` (each shape ``(H, W)``).
    """
    for required in ("h", "u", "v"):
        if required not in arrays:
            raise KeyError(f"temporal-couple requires key '{required}'")

    h = np.asarray(arrays["h"], dtype=np.float64)
    u = np.asarray(arrays["u"], dtype=np.float64)
    v = np.asarray(arrays["v"], dtype=np.float64)
    if h.shape != u.shape or h.shape != v.shape or h.ndim != 3:
        raise ValueError(
            f"h / u / v must share shape (T, H, W); got {h.shape}, {u.shape}, {v.shape}"
        )
    T_steps, H, W = h.shape

    bg_u = np.asarray(arrays["bg_u"], dtype=np.float64) if "bg_u" in arrays else None
    bg_v = np.asarray(arrays["bg_v"], dtype=np.float64) if "bg_v" in arrays else None
    obs_u = np.asarray(arrays["obs_u"], dtype=np.float64) if "obs_u" in arrays else None
    obs_v = np.asarray(arrays["obs_v"], dtype=np.float64) if "obs_v" in arrays else None

    refs = {
        "h": True,                                   # required; always True here
        "background_uv": bg_u is not None and bg_v is not None,
        "obs_uv": obs_u is not None and obs_v is not None,
    }

    # ---- Geostrophic coherence field (T, H, W) --------------------------
    # rotated ∇h = (-∂h/∂y, ∂h/∂x) — the geostrophic wind direction in a
    # simple f-plane shallow-water model. Coherence is the cosine between
    # uv and rotated ∇h, in [-1, +1]. +1 = perfect geostrophy, -1 = fully
    # anti-geostrophic, 0 = decoupled.
    gx = np.empty_like(h)
    gy = np.empty_like(h)
    for t in range(T_steps):
        gx[t], gy[t] = _grad_xy(h[t])
    rot_gx = -gy
    rot_gy = gx
    speed = _norm2(u, v)
    geo_coh = (rot_gx * u + rot_gy * v) / (_norm2(rot_gx, rot_gy) * speed + 1e-9)
    geo_coh = np.clip(geo_coh, -1.0, 1.0)

    grad_mag = _norm2(gx, gy)
    grad_growth = np.maximum(0.0, grad_mag[1:] / (grad_mag[:-1] + 1e-9) - 1.0)  # (T-1, H, W)

    # ---- Background / obs lock fields -----------------------------------
    if refs["background_uv"] and refs["obs_uv"]:
        d_bg = _norm2(u - bg_u[None], v - bg_v[None])
        d_obs = _norm2(u - obs_u[None], v - obs_v[None])
        bg_lock = d_obs / (d_bg + d_obs + 1e-9)
        obs_resp = d_bg / (d_bg + d_obs + 1e-9)
    else:
        bg_lock = None
        obs_resp = None

    if refs["background_uv"]:
        bg_align = (u * bg_u[None] + v * bg_v[None]) / (
            _norm2(u, v) * _norm2(bg_u[None], bg_v[None]) + 1e-9
        )
    else:
        bg_align = None
    if refs["obs_uv"]:
        obs_align = (u * obs_u[None] + v * obs_v[None]) / (
            _norm2(u, v) * _norm2(obs_u[None], obs_v[None]) + 1e-9
        )
    else:
        obs_align = None

    # ---- Coarse-grained scan --------------------------------------------
    t_edges = _bin_edges(T_steps - 1, min(time_bins, max(1, T_steps - 1)))
    y_edges = _bin_edges(H, min(y_bins, H))
    x_edges = _bin_edges(W, min(x_bins, W))

    gate_rows: list[dict[str, Any]] = []
    for ti, (ta, tb) in enumerate(t_edges):
        for yi, (ya, yb) in enumerate(y_edges):
            for xi, (xa, xb) in enumerate(x_edges):
                # All per-bin means are computed on the same time slice
                # (ta..tb). grad_growth is indexed (ta..tb) with ta+1
                # offset; clip at tb-1 so we never exceed T-1.
                ga = ta
                gb = max(ga + 1, min(tb, T_steps - 1))
                coh_bin = geo_coh[ga:gb, ya:yb, xa:xb]
                coupling_mag = float(np.abs(coh_bin).mean())
                coh_signed = float(coh_bin.mean())
                anti_frac = float((coh_bin < 0.0).mean())
                gg_bin = grad_growth[ga:gb, ya:yb, xa:xb]
                grad_grow = float(gg_bin.mean())
                bg_lock_m = float(bg_lock[ga:gb, ya:yb, xa:xb].mean()) if bg_lock is not None else 0.0
                obs_resp_m = float(obs_resp[ga:gb, ya:yb, xa:xb].mean()) if obs_resp is not None else 0.0

                # Bin score weights: |coupling| magnitude and anti-geo
                # fraction dominate, with smaller contributions from
                # gradient growth, background lock, and obs response.
                # These weights focus the scan on *coupling health*
                # rather than generic anomaly strength.
                score = (
                    0.35 * coupling_mag
                    + 0.20 * anti_frac
                    + 0.15 * grad_grow
                    + 0.15 * bg_lock_m
                    + 0.15 * obs_resp_m
                )
                gate_id = _flatten_gate_id(ti, yi, xi, y_bins=len(y_edges), x_bins=len(x_edges))
                gate_rows.append(
                    {
                        "time_bin": ti,
                        "time_window": [int(ta), int(tb)],
                        "y_bin": yi,
                        "x_bin": xi,
                        "score": float(score),
                        "gate_id": gate_id,
                        "coupling_metrics": {
                            "h_uv_coupling_mean": coupling_mag,
                            "h_uv_anti_geo_fraction": anti_frac,
                            "geostrophic_coherence_mean": coh_signed,
                            "grad_growth_mean": grad_grow,
                            "background_lock_mean": bg_lock_m if bg_lock is not None else None,
                            "obs_response_mean": obs_resp_m if obs_resp is not None else None,
                        },
                    }
                )

    scores = np.asarray([r["score"] for r in gate_rows], dtype=np.float64)
    top_idx = np.argsort(-scores)[:5]
    top_channels = [
        CoupleChannel(
            time_bin=int(gate_rows[int(i)]["time_bin"]),
            y_bin=int(gate_rows[int(i)]["y_bin"]),
            x_bin=int(gate_rows[int(i)]["x_bin"]),
            time_window=list(gate_rows[int(i)]["time_window"]),
            score=float(gate_rows[int(i)]["score"]),
            gate_id=int(gate_rows[int(i)]["gate_id"]),
            coupling_metrics=gate_rows[int(i)]["coupling_metrics"],
        )
        for i in top_idx
    ]

    best = gate_rows[int(top_idx[0])]
    ta, tb = best["time_window"]
    ya, yb = y_edges[best["y_bin"]]
    xa, xb = x_edges[best["x_bin"]]
    ga = ta
    gb = max(ga + 1, min(tb, T_steps - 1))

    direction_metrics = {
        "background_alignment_mean": (
            float(bg_align[ga:gb, ya:yb, xa:xb].mean()) if bg_align is not None else None
        ),
        "obs_alignment_mean": (
            float(obs_align[ga:gb, ya:yb, xa:xb].mean()) if obs_align is not None else None
        ),
        "geostrophic_coherence_mean": float(geo_coh[ga:gb, ya:yb, xa:xb].mean()),
    }

    coupling_metrics_lock = best["coupling_metrics"]

    verdict = _verdict(
        coupling_mean=coupling_metrics_lock["h_uv_coupling_mean"],
        anti_geo_fraction=coupling_metrics_lock["h_uv_anti_geo_fraction"],
        grad_growth=coupling_metrics_lock["grad_growth_mean"],
        coherence_mean_signed=direction_metrics["geostrophic_coherence_mean"],
        refs_ok=True,
    )

    lock = CoupleLock(
        pair="h-uv",
        time_bin=int(best["time_bin"]),
        y_bin=int(best["y_bin"]),
        x_bin=int(best["x_bin"]),
        score=float(best["score"]),
        gate_id=int(best["gate_id"]),
        time_window=[int(ta), int(tb)],
        spatial_window={"y_bin": int(best["y_bin"]), "x_bin": int(best["x_bin"])},
        direction_metrics=direction_metrics,
        coupling_metrics=coupling_metrics_lock,
        crt_locator={
            "moduli": list(_CRT_MODULI),
            "residues": _crt_residues(int(best["gate_id"])),
            "recovered_gate_id": int(best["gate_id"]),
        },
        verdict=verdict,
    )

    # Global (domain-mean) summary scores — the single-scalar handles
    # that Codex wants on the CLI acceptance row.
    coupling_scores = {
        "h_uv_coupling_mean": round(float(np.abs(geo_coh).mean()), 6),
        "h_uv_anti_geo_fraction": round(float((geo_coh < 0.0).mean()), 6),
        "grad_growth_mean": round(float(grad_growth.mean()), 6),
        "background_lock_mean": round(float(bg_lock.mean()), 6) if bg_lock is not None else None,
        "obs_response_mean": round(float(obs_resp.mean()), 6) if obs_resp is not None else None,
        "background_alignment_mean": round(float(bg_align.mean()), 6) if bg_align is not None else None,
        "obs_alignment_mean": round(float(obs_align.mean()), 6) if obs_align is not None else None,
        "geostrophic_coherence_mean": round(float(geo_coh.mean()), 6),
    }

    return TemporalCoupleReport(
        case_id=case_id or Path(str(arrays.get("__source__", "temporal-couple"))).stem,
        dt=float(dt),
        grid_shape=[H, W],
        time_steps=T_steps,
        time_bins=len(t_edges),
        y_bins=len(y_edges),
        x_bins=len(x_edges),
        references_available=refs,
        coupling_scores=coupling_scores,
        top_channels=top_channels,
        lock=lock,
        metadata={
            "available_fields": sorted(k for k in arrays.keys() if not k.startswith("__")),
        },
    )


def analyze_temporal_couple_file(
    path: str | Path,
    *,
    dt: float | None = None,
    case_id: str = "",
    time_bins: int = 8,
    y_bins: int = 12,
    x_bins: int = 16,
) -> TemporalCoupleReport:
    arrays, dt_from_file = _load_couple_input(path)
    arrays["__source__"] = np.asarray(str(path))
    return analyze_temporal_couple(
        arrays,
        dt=dt if dt is not None else dt_from_file,
        case_id=case_id or Path(path).stem,
        time_bins=time_bins,
        y_bins=y_bins,
        x_bins=x_bins,
    )


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------

def temporal_couple_report(report: TemporalCoupleReport) -> str:
    lines = []
    lines.append("=" * 68)
    lines.append("Tensorearch Temporal Couple (h -> uv)")
    lines.append("=" * 68)
    lines.append(
        f"case={report.case_id} T={report.time_steps} grid={report.grid_shape} dt={report.dt}"
    )
    lines.append("")
    lines.append("coupling scores")
    for key, value in report.coupling_scores.items():
        lines.append(f"  {key}={value}")
    lines.append("")
    lines.append("top channels")
    for i, ch in enumerate(report.top_channels, start=1):
        lines.append(
            f"  #{i} h-uv:t{ch.time_bin}:y{ch.y_bin}:x{ch.x_bin} score={ch.score:.4f} "
            f"coupling={ch.coupling_metrics['h_uv_coupling_mean']:.4f} "
            f"anti={ch.coupling_metrics['h_uv_anti_geo_fraction']:.4f}"
        )
    lines.append("")
    lines.append("lock")
    lines.append(
        f"  pair={report.lock.pair} time={report.lock.time_window} "
        f"bin=({report.lock.y_bin},{report.lock.x_bin}) score={report.lock.score:.4f}"
    )
    lines.append(
        f"  verdict={report.lock.verdict['kind']} conf={report.lock.verdict['confidence']:.4f}"
    )
    return "\n".join(lines)


def temporal_couple_report_json(report: TemporalCoupleReport) -> str:
    return json.dumps(report.to_dict(), indent=2)
