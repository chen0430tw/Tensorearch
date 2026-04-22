"""Temporal-balance probe — generic potential / response / static-forcing diagnostic.

Abstraction (deliberately decoupled from any specific domain):

    potential_field   P(T, H, W)           — the "clean" scalar that balance is built on
    response_field    R_u, R_v (T, H, W)   — the vector the system actually produced
    static_forcing    list of (S_k(H, W) or (T, H, W), weight_k)
                                           — extra scalar fields folded in at runtime
    balance_operator  B[·]: scalar → vector
                                           — converts a scalar into its expected
                                             vector response (gradient or
                                             rotated_gradient for this MVP)

Three theoretical responses are compared against the observed response R:

    b^(p) = B[potential_only]        = B[P]
    b^(s) = B[static_only]           = B[Σ w_k S_k]
    b^(c) = B[combined]              = B[P + Σ w_k S_k]

Per cell:

    A(r, b) = (r · b) / (|r| |b| + eps)            # direction consistency
    M(r, b) = |ln(|r|+eps) - ln(|b|+eps)|          # magnitude gap
    G       = λ_a (1 - A) + λ_m M                   # consistency gap

Window aggregation computes per-(time, y, x) bin means of A, M, G and the
best matching theoretical mode. Verdict taxonomy is also per cell / global:

    potential_consistent
    weak_balance_coupling
    static_forcing_dominant
    static_forcing_overrides_potential_balance    ← KEY diagnostic
    response_magnitude_underpowered
    response_magnitude_overpowered

The key verdict ``static_forcing_overrides_potential_balance`` captures the
specific bug shape seen in the 2026-04-22 TD kernel audit: adding a static
term to the balance operator made the response LESS consistent with reality
than the potential-only balance. The rule is

    A^(p) > 0   AND   A^(c) < 0   AND   G^(c) > G^(p) + δ

and is how this tool generalises "runtime balance broke when I stirred in
a second forcing term" into a domain-agnostic test.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .temporal_radio import _bin_edges


_TRBALANCE_REPORT_VERSION = "trbalance_report.v1"
_TRBALANCE_INPUT_VERSION  = "trbalance.v1"
_EPS = 1.0e-9
_MIN_SPEED = 1.0e-6


# ---------------------------------------------------------------------------
# Spec dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StaticForcingSpec:
    key:    str
    weight: float = 1.0


@dataclass
class BalanceOperator:
    kind:  str   = "rotated_gradient"   # "gradient" | "rotated_gradient"
    scale: float = 1.0


@dataclass
class AnalysisSpec:
    time_bins:              int   = 8
    y_bins:                 int   = 12
    x_bins:                 int   = 16
    alignment_weight:       float = 0.7     # λ_a
    magnitude_weight:       float = 0.3     # λ_m
    static_override_delta:  float = 0.10    # δ threshold for override verdict
    alignment_threshold:    float = 0.7     # consistent / dominant trigger
    magnitude_log_threshold: float = 0.693  # ≈ ln(2) — 2× mis-scale


@dataclass
class BalanceSpec:
    case_id:         str = ""
    dt:              float = 1.0
    potential_key:   str = "h"
    response_u_key:  str = "u"
    response_v_key:  str = "v"
    static_forcings: list[StaticForcingSpec] = field(default_factory=list)
    operator:        BalanceOperator = field(default_factory=BalanceOperator)
    analysis:        AnalysisSpec = field(default_factory=AnalysisSpec)
    dx:              float = 1.0
    dy:              float = 1.0


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BalanceReport:
    version:     str
    meta:        dict[str, Any]
    global_:     dict[str, Any]
    time_series: list[dict[str, Any]]
    windows:     list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version":     self.version,
            "meta":        self.meta,
            "global":      self.global_,
            "time_series": self.time_series,
            "windows":     self.windows,
        }


# ---------------------------------------------------------------------------
# Core math helpers
# ---------------------------------------------------------------------------

def _apply_operator(
    scalar: np.ndarray,
    op: BalanceOperator,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply B[·] to a scalar field, returning a vector field (bx, by).

    Supported kinds:
      gradient          : (bx, by) = k * (∂/∂x, ∂/∂y)
      rotated_gradient  : (bx, by) = k * (-∂/∂y, +∂/∂x)   (rotate +90° CCW)

    The operator scale ``k`` is a pure multiplier on both components, so it
    cancels out in the direction-only alignment metric but sets the
    magnitude ratio that the magnitude gap metric sees. For geostrophic-
    wind-from-height in SI units, ``scale = g / f`` is the natural pick;
    for dimensionless tests ``scale = 1.0`` is fine.
    """
    if scalar.ndim == 2:
        gy, gx = np.gradient(scalar)
    elif scalar.ndim == 3:
        gy, gx = np.gradient(scalar, axis=(1, 2))
    else:
        raise ValueError(f"potential/static scalar must be 2D or 3D, got ndim={scalar.ndim}")
    gx = gx / float(dx)
    gy = gy / float(dy)
    k = float(op.scale)
    if op.kind == "gradient":
        return k * gx, k * gy
    if op.kind == "rotated_gradient":
        return -k * gy, k * gx
    raise ValueError(f"unknown balance operator kind: {op.kind!r}")


def _alignment(rx: np.ndarray, ry: np.ndarray, bx: np.ndarray, by: np.ndarray) -> np.ndarray:
    r_mag = np.sqrt(rx * rx + ry * ry) + _EPS
    b_mag = np.sqrt(bx * bx + by * by) + _EPS
    return np.clip((rx * bx + ry * by) / (r_mag * b_mag), -1.0, 1.0)


def _magnitude_gap(rx: np.ndarray, ry: np.ndarray, bx: np.ndarray, by: np.ndarray) -> np.ndarray:
    r_mag = np.sqrt(rx * rx + ry * ry)
    b_mag = np.sqrt(bx * bx + by * by)
    return np.abs(np.log(r_mag + _EPS) - np.log(b_mag + _EPS))


def _consistency_gap(A: np.ndarray, M: np.ndarray, lam_a: float, lam_m: float) -> np.ndarray:
    return lam_a * (1.0 - A) + lam_m * M


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------

def _classify(
    *,
    A_p: float, A_s: float, A_c: float,
    G_p: float, G_s: float, G_c: float,
    r_mag: float, b_p_mag: float, b_c_mag: float,
    analysis: AnalysisSpec,
) -> dict[str, Any]:
    """Choose a verdict + confidence for a single (global or window) cell."""
    delta   = analysis.static_override_delta
    thr_a   = analysis.alignment_threshold
    thr_mag = analysis.magnitude_log_threshold

    # PRIMARY: adding static forcing made consistency WORSE, AND flipped the
    # direction sign. This is the override bug shape.
    if A_p > 0.0 and A_c < 0.0 and G_c > G_p + delta:
        conf = min(1.0, 0.5 + 0.5 * min(1.0, abs(A_p - A_c)))
        return {"kind": "static_forcing_overrides_potential_balance",
                "confidence": round(conf, 4)}

    # Best theoretical mode by gap.
    gaps = {"potential_only": G_p, "static_only": G_s, "combined": G_c}
    best = min(gaps, key=gaps.get)

    # Magnitude mismatch check — only when direction broadly agrees with
    # potential_only (so it's a scale problem, not a direction problem).
    if A_p > 0.5 and r_mag > _MIN_SPEED and b_p_mag > _MIN_SPEED:
        log_ratio = math.log(max(r_mag, _MIN_SPEED) / max(b_p_mag, _MIN_SPEED))
        if log_ratio > thr_mag:
            return {"kind": "response_magnitude_overpowered",
                    "confidence": round(min(1.0, 0.5 + 0.1 * log_ratio), 4)}
        if log_ratio < -thr_mag:
            return {"kind": "response_magnitude_underpowered",
                    "confidence": round(min(1.0, 0.5 - 0.1 * log_ratio), 4)}

    if best == "static_only" and A_s > thr_a:
        return {"kind": "static_forcing_dominant",
                "confidence": round(float(min(1.0, A_s)), 4)}
    if best == "potential_only" and A_p > thr_a:
        return {"kind": "potential_consistent",
                "confidence": round(float(min(1.0, A_p)), 4)}

    # Fallback: nothing strongly matches.
    conf = max(0.0, 1.0 - float(gaps[best]))
    return {"kind": "weak_balance_coupling", "confidence": round(conf, 4)}


# ---------------------------------------------------------------------------
# Spec loading
# ---------------------------------------------------------------------------

def load_spec_from_dict(spec_dict: dict[str, Any]) -> BalanceSpec:
    """Parse a trbalance.v1 input JSON payload into a BalanceSpec."""
    if spec_dict.get("version") not in (_TRBALANCE_INPUT_VERSION, None):
        raise ValueError(f"unexpected spec version: {spec_dict.get('version')!r}")
    meta    = spec_dict.get("meta", {})
    fields_ = spec_dict.get("fields", {})
    op_d    = spec_dict.get("operator", {})
    ana_d   = spec_dict.get("analysis", {})

    static_list: list[StaticForcingSpec] = []
    static_raw = fields_.get("static", {}) or {}
    if isinstance(static_raw, dict):
        for key, meta_k in static_raw.items():
            if isinstance(meta_k, dict):
                w = float(meta_k.get("weight", 1.0))
            else:
                w = float(meta_k)
            static_list.append(StaticForcingSpec(key=key, weight=w))
    elif isinstance(static_raw, list):
        for row in static_raw:
            static_list.append(StaticForcingSpec(
                key=str(row["key"]), weight=float(row.get("weight", 1.0))
            ))

    operator = BalanceOperator(
        kind=str(op_d.get("kind", "rotated_gradient")),
        scale=float(op_d.get("scale", 1.0)),
    )
    analysis = AnalysisSpec(
        time_bins=int(ana_d.get("time_bins", 8)),
        y_bins=int(ana_d.get("y_bins", 12)),
        x_bins=int(ana_d.get("x_bins", 16)),
        alignment_weight=float(ana_d.get("alignment_weight", 0.7)),
        magnitude_weight=float(ana_d.get("magnitude_weight", 0.3)),
        static_override_delta=float(ana_d.get("static_override_delta", 0.10)),
        alignment_threshold=float(ana_d.get("alignment_threshold", 0.7)),
        magnitude_log_threshold=float(ana_d.get("magnitude_log_threshold", 0.693)),
    )
    return BalanceSpec(
        case_id=str(meta.get("case_id", "")),
        dt=float(meta.get("dt", 1.0)),
        potential_key=str(fields_.get("potential", "h")),
        response_u_key=str(fields_.get("response_u", "u")),
        response_v_key=str(fields_.get("response_v", "v")),
        static_forcings=static_list,
        operator=operator,
        analysis=analysis,
        dx=float(meta.get("dx", 1.0)),
        dy=float(meta.get("dy", 1.0)),
    )


def load_spec_from_file(path: str | Path) -> BalanceSpec:
    p = Path(path)
    return load_spec_from_dict(json.loads(p.read_text(encoding="utf-8")))


def _load_arrays(path: str | Path) -> dict[str, np.ndarray]:
    p = Path(path)
    if p.suffix.lower() == ".npz":
        with np.load(p) as npz:
            return {k: np.asarray(npz[k]) for k in npz.files}
    if p.suffix.lower() == ".json":
        payload = json.loads(p.read_text(encoding="utf-8"))
        return {k: np.asarray(v) for k, v in payload.items()}
    raise ValueError("temporal-balance expects .npz or .json input")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_temporal_balance(
    arrays: dict[str, np.ndarray],
    spec: BalanceSpec,
) -> BalanceReport:
    """Run the temporal-balance diagnostic on in-memory arrays."""
    if spec.potential_key not in arrays:
        raise KeyError(f"potential field {spec.potential_key!r} missing from arrays")
    if spec.response_u_key not in arrays:
        raise KeyError(f"response_u field {spec.response_u_key!r} missing from arrays")
    if spec.response_v_key not in arrays:
        raise KeyError(f"response_v field {spec.response_v_key!r} missing from arrays")

    u = np.asarray(arrays[spec.response_u_key], dtype=np.float64)
    v = np.asarray(arrays[spec.response_v_key], dtype=np.float64)
    if u.shape != v.shape or u.ndim != 3:
        raise ValueError(f"response u/v must share shape (T, H, W); got {u.shape} / {v.shape}")
    T, H, W = u.shape

    def _broadcast_to_TxHxW(arr: np.ndarray, name: str) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 3:
            if arr.shape != (T, H, W):
                raise ValueError(f"{name} shape {arr.shape} != response shape {(T, H, W)}")
            return arr
        if arr.ndim == 2:
            if arr.shape != (H, W):
                raise ValueError(f"{name} 2D shape {arr.shape} != spatial {(H, W)}")
            return np.broadcast_to(arr[None, :, :], (T, H, W)).copy()
        raise ValueError(f"{name} must be 2D or 3D, got ndim={arr.ndim}")

    potential = _broadcast_to_TxHxW(arrays[spec.potential_key], "potential")

    static_sum = np.zeros_like(potential)
    static_present = False
    for sf in spec.static_forcings:
        if sf.key not in arrays:
            raise KeyError(f"static forcing field {sf.key!r} missing from arrays")
        s_field = _broadcast_to_TxHxW(arrays[sf.key], f"static/{sf.key}")
        static_sum = static_sum + float(sf.weight) * s_field
        static_present = True

    combined = potential + static_sum if static_present else potential

    # Operator is applied to each theoretical scalar field.
    op = spec.operator
    bp_x, bp_y = _apply_operator(potential,  op, spec.dx, spec.dy)
    bs_x, bs_y = _apply_operator(static_sum, op, spec.dx, spec.dy)
    bc_x, bc_y = _apply_operator(combined,   op, spec.dx, spec.dy)

    # Per-cell metrics.
    A_p = _alignment(u, v, bp_x, bp_y)
    A_s = _alignment(u, v, bs_x, bs_y)
    A_c = _alignment(u, v, bc_x, bc_y)
    M_p = _magnitude_gap(u, v, bp_x, bp_y)
    M_s = _magnitude_gap(u, v, bs_x, bs_y)
    M_c = _magnitude_gap(u, v, bc_x, bc_y)
    lam_a = spec.analysis.alignment_weight
    lam_m = spec.analysis.magnitude_weight
    G_p = _consistency_gap(A_p, M_p, lam_a, lam_m)
    G_s = _consistency_gap(A_s, M_s, lam_a, lam_m)
    G_c = _consistency_gap(A_c, M_c, lam_a, lam_m)

    r_mag = np.sqrt(u * u + v * v)
    bp_mag = np.sqrt(bp_x * bp_x + bp_y * bp_y)
    bs_mag = np.sqrt(bs_x * bs_x + bs_y * bs_y)
    bc_mag = np.sqrt(bc_x * bc_x + bc_y * bc_y)

    # Window aggregation.
    t_edges = _bin_edges(T, min(spec.analysis.time_bins, max(1, T)))
    y_edges = _bin_edges(H, min(spec.analysis.y_bins,    H))
    x_edges = _bin_edges(W, min(spec.analysis.x_bins,    W))

    window_rows: list[dict[str, Any]] = []
    for ti, (ta, tb) in enumerate(t_edges):
        for yi, (ya, yb) in enumerate(y_edges):
            for xi, (xa, xb) in enumerate(x_edges):
                sl = (slice(ta, tb), slice(ya, yb), slice(xa, xb))
                row_A_p = float(A_p[sl].mean())
                row_A_s = float(A_s[sl].mean())
                row_A_c = float(A_c[sl].mean())
                row_M_p = float(M_p[sl].mean())
                row_M_s = float(M_s[sl].mean())
                row_M_c = float(M_c[sl].mean())
                row_G_p = float(G_p[sl].mean())
                row_G_s = float(G_s[sl].mean())
                row_G_c = float(G_c[sl].mean())
                r_w  = float(r_mag[sl].mean())
                bp_w = float(bp_mag[sl].mean())
                bc_w = float(bc_mag[sl].mean())
                verdict = _classify(
                    A_p=row_A_p, A_s=row_A_s, A_c=row_A_c,
                    G_p=row_G_p, G_s=row_G_s, G_c=row_G_c,
                    r_mag=r_w, b_p_mag=bp_w, b_c_mag=bc_w,
                    analysis=spec.analysis,
                )
                window_rows.append({
                    "time_bin": ti, "y_bin": yi, "x_bin": xi,
                    "time_window":     [int(ta), int(tb)],
                    "spatial_window":  {"y": [int(ya), int(yb)], "x": [int(xa), int(xb)]},
                    "alignment_potential": round(row_A_p, 6),
                    "alignment_static":    round(row_A_s, 6),
                    "alignment_combined":  round(row_A_c, 6),
                    "consistency_gap_potential": round(row_G_p, 6),
                    "consistency_gap_static":    round(row_G_s, 6),
                    "consistency_gap_combined":  round(row_G_c, 6),
                    "verdict":    verdict["kind"],
                    "confidence": verdict["confidence"],
                })

    # Time series — collapse over y, x.
    time_series: list[dict[str, Any]] = []
    for ti, (ta, tb) in enumerate(t_edges):
        sl = (slice(ta, tb), slice(None), slice(None))
        time_series.append({
            "time_bin": ti,
            "time_window": [int(ta), int(tb)],
            "alignment_potential": round(float(A_p[sl].mean()), 6),
            "alignment_static":    round(float(A_s[sl].mean()), 6),
            "alignment_combined":  round(float(A_c[sl].mean()), 6),
            "consistency_gap_potential": round(float(G_p[sl].mean()), 6),
            "consistency_gap_combined":  round(float(G_c[sl].mean()), 6),
        })

    # Global means.
    A_p_g = float(A_p.mean()); A_s_g = float(A_s.mean()); A_c_g = float(A_c.mean())
    G_p_g = float(G_p.mean()); G_s_g = float(G_s.mean()); G_c_g = float(G_c.mean())
    r_g   = float(r_mag.mean())
    bp_g  = float(bp_mag.mean()); bs_g = float(bs_mag.mean()); bc_g = float(bc_mag.mean())

    gaps_g = {"potential_only": G_p_g, "static_only": G_s_g, "combined": G_c_g}
    best_mode = min(gaps_g, key=gaps_g.get)
    v_global = _classify(
        A_p=A_p_g, A_s=A_s_g, A_c=A_c_g,
        G_p=G_p_g, G_s=G_s_g, G_c=G_c_g,
        r_mag=r_g, b_p_mag=bp_g, b_c_mag=bc_g,
        analysis=spec.analysis,
    )

    global_ = {
        "alignment_potential_only_mean": round(A_p_g, 6),
        "alignment_static_only_mean":    round(A_s_g, 6),
        "alignment_combined_mean":       round(A_c_g, 6),
        "speed_response_mean":           round(r_g,  6),
        "speed_potential_only_mean":     round(bp_g, 6),
        "speed_static_only_mean":        round(bs_g, 6),
        "speed_combined_mean":           round(bc_g, 6),
        "consistency_gap_potential_only_mean": round(G_p_g, 6),
        "consistency_gap_static_only_mean":    round(G_s_g, 6),
        "consistency_gap_combined_mean":       round(G_c_g, 6),
        "consistency_gap_mean":                round(G_c_g, 6),   # alias on combined
        "best_mode": best_mode,
        "verdict":    v_global["kind"],
        "confidence": v_global["confidence"],
    }

    # Top windows sorted by combined gap (worst first) — same diagnostic
    # aim as radio/couple's top_channels.
    top_windows = sorted(window_rows, key=lambda r: -r["consistency_gap_combined"])[:8]

    meta = {
        "case_id":    spec.case_id,
        "dt":         spec.dt,
        "grid_shape": [H, W],
        "time_steps": T,
        "dx":         spec.dx,
        "dy":         spec.dy,
        "operator":   {"kind": op.kind, "scale": op.scale},
        "fields": {
            "potential":  spec.potential_key,
            "response_u": spec.response_u_key,
            "response_v": spec.response_v_key,
            "static":     [{"key": s.key, "weight": s.weight} for s in spec.static_forcings],
        },
        "analysis": {
            "time_bins":              len(t_edges),
            "y_bins":                 len(y_edges),
            "x_bins":                 len(x_edges),
            "alignment_weight":       spec.analysis.alignment_weight,
            "magnitude_weight":       spec.analysis.magnitude_weight,
            "static_override_delta":  spec.analysis.static_override_delta,
            "alignment_threshold":    spec.analysis.alignment_threshold,
            "magnitude_log_threshold": spec.analysis.magnitude_log_threshold,
        },
    }

    return BalanceReport(
        version=_TRBALANCE_REPORT_VERSION,
        meta=meta,
        global_=global_,
        time_series=time_series,
        windows=top_windows,
    )


def analyze_temporal_balance_file(
    input_path: str | Path,
    spec: BalanceSpec,
) -> BalanceReport:
    arrays = _load_arrays(input_path)
    if not spec.case_id:
        spec.case_id = Path(input_path).stem
    return analyze_temporal_balance(arrays, spec)


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def temporal_balance_report(report: BalanceReport) -> str:
    lines: list[str] = []
    lines.append("=" * 68)
    lines.append("Tensorearch Temporal Balance (potential / response / static)")
    lines.append("=" * 68)
    m = report.meta
    lines.append(f"case={m['case_id']} T={m['time_steps']} grid={m['grid_shape']} "
                 f"op={m['operator']['kind']}  scale={m['operator']['scale']}")
    lines.append("")
    g = report.global_
    lines.append("global alignment (direction consistency)")
    lines.append(f"  potential_only = {g['alignment_potential_only_mean']:+.4f}")
    lines.append(f"  static_only    = {g['alignment_static_only_mean']:+.4f}")
    lines.append(f"  combined       = {g['alignment_combined_mean']:+.4f}")
    lines.append("")
    lines.append("global speed (m/s or whatever unit)")
    lines.append(f"  response       = {g['speed_response_mean']:.4f}")
    lines.append(f"  potential_only = {g['speed_potential_only_mean']:.4f}")
    lines.append(f"  static_only    = {g['speed_static_only_mean']:.4f}")
    lines.append(f"  combined       = {g['speed_combined_mean']:.4f}")
    lines.append("")
    lines.append(f"best_mode = {g['best_mode']}")
    lines.append(f"verdict   = {g['verdict']}  conf={g['confidence']:.4f}")
    lines.append("")
    lines.append("top-8 worst windows (combined gap)")
    for row in report.windows:
        lines.append(
            f"  t={row['time_bin']} y={row['y_bin']:>2} x={row['x_bin']:>2}  "
            f"A_p={row['alignment_potential']:+.3f} A_c={row['alignment_combined']:+.3f} "
            f"G_c={row['consistency_gap_combined']:.3f}  {row['verdict']}"
        )
    return "\n".join(lines)


def temporal_balance_report_json(report: BalanceReport) -> str:
    return json.dumps(report.to_dict(), indent=2)
