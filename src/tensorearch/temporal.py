"""Temporal-topology probe via symmetric cyclic numbers (对称循环数).

Tensorearch's original probes are single-snapshot spatial — built for LLM
forward passes where "time-domain evolution of a tensor" doesn't apply.
Numerical simulation traces have an additional axis: time. Stability bugs
(CFL violation, dispersion, checkerboard modes) manifest as **period-2Δt
sign flipping plus amplitude growth**, which is invisible to static
spatial-topology metrics.

Algebra — the symmetric cyclic number ring C = R² with:
    S_+(a, b) = (-b, a)     # rotation by +π/2, order 4
    S_-(a, b) = (b, -a)     # inverse
C is isomorphic to the complex numbers; S_+ is multiplication by i.

Detector — for each spatial cell with time series u_0 … u_{T-1}:
    z_k = u_k + i·u_{k+1}       (delay embedding in C)
    r_k = z_{k+1} / z_k          (per-step complex ratio)
    φ_k = arg(r_k)   ∈ (-π, π]   (phase increment; ωΔt per step)
    ρ_k = |r_k|                  (amplitude growth factor per step)

Signatures:
    2Δt checkerboard (CFL gravity-wave instability):  φ near ±π  AND  ρ > 1
    Quarter-period stable resonance (benign):         φ near ±π/2  AND  ρ ≈ 1
    Slow physical mode (expected):                    |φ| small  AND  ρ ≈ 1
    Diverging (any mechanism):                        ρ ≫ 1 sustained across cells

The detector returns fraction of cells caught by each signature, a per-step
time series of those fractions (so growth onset is visible), and a coarse
verdict. Works on any (T, *spatial) array — single scalar field, a velocity
component, or a stacked channel. For 2-component state (u, v), run once per
component and take the max signature fraction.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class TemporalReport:
    """Report from one analyze_temporal_topology() call."""
    shape_T: int
    shape_spatial: list[int]
    dt: float
    # Aggregate scores (fraction of cell-steps matching each signature)
    cfl_checkerboard_fraction: float
    quarter_resonance_fraction: float
    slow_mode_fraction: float
    diverging_fraction: float
    # Global growth stats — per-step |r| (noisy) and per-cell log-amp fit (robust)
    rho_median: float
    rho_p95: float
    rho_max: float
    growth_median: float
    growth_p95: float
    # Per-step trends (length T-2, listed for trend inspection)
    per_step_cfl_fraction: list[float]
    per_step_diverging_fraction: list[float]
    per_step_rho_mean: list[float]
    # Spatial hotspots — top-K cells ranked by cumulative instability score
    hotspots: list[dict[str, Any]]
    # Coarse verdict
    verdict: str
    verdict_reason: str
    # Thresholds used (for reproducibility)
    phase_threshold_rad: float
    growth_threshold: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": {"T": self.shape_T, "spatial": self.shape_spatial},
            "dt": self.dt,
            "cfl_checkerboard_fraction": self.cfl_checkerboard_fraction,
            "quarter_resonance_fraction": self.quarter_resonance_fraction,
            "slow_mode_fraction": self.slow_mode_fraction,
            "diverging_fraction": self.diverging_fraction,
            "rho_median": self.rho_median,
            "rho_p95": self.rho_p95,
            "rho_max": self.rho_max,
            "growth_median": self.growth_median,
            "growth_p95": self.growth_p95,
            "per_step_cfl_fraction": self.per_step_cfl_fraction,
            "per_step_diverging_fraction": self.per_step_diverging_fraction,
            "per_step_rho_mean": self.per_step_rho_mean,
            "hotspots": self.hotspots,
            "verdict": self.verdict,
            "verdict_reason": self.verdict_reason,
            "thresholds": {
                "phase_threshold_rad": self.phase_threshold_rad,
                "growth_threshold": self.growth_threshold,
            },
            "metadata": self.metadata,
        }


def analyze_temporal_topology(
    u: np.ndarray,
    dt: float = 1.0,
    growth_threshold: float = 1.01,
    phase_threshold_rad: float = 0.15,
    top_k_hotspots: int = 5,
    eps: float = 1e-12,
) -> TemporalReport:
    """Analyze a (T, *spatial) time series for CFL/dispersion instabilities.

    Parameters
    ----------
    u : np.ndarray
        Array of shape (T, *spatial). T ≥ 3 required.
    dt : float
        Physical time step (used only for reporting; algorithm is unit-agnostic).
    growth_threshold : float
        |r_k| > this counts as growing. 1.01 ≈ 1% per step — tight enough to
        separate numerical drift from real instability; loose enough to avoid
        false positives on round-off.
    phase_threshold_rad : float
        Phase proximity window for signature matching. Default 0.15 rad (~8.6°).
    top_k_hotspots : int
        Return top-K spatial cells by cumulative signature weight.
    eps : float
        Small number to regularize z_k ≈ 0 cells (near-zero fields).

    Returns
    -------
    TemporalReport
    """
    arr = np.asarray(u, dtype=np.float64)
    if arr.ndim < 1 or arr.shape[0] < 3:
        raise ValueError(f"need array of shape (T, *spatial) with T>=3, got {arr.shape}")

    T = int(arr.shape[0])
    spatial_shape = list(arr.shape[1:])
    flat_spatial = int(np.prod(spatial_shape)) if spatial_shape else 1

    # Delay embed: z_k = u_k + i u_{k+1}, shape (T-1, *spatial)
    z = arr[:-1] + 1j * arr[1:]
    # Per-step complex ratio r_k = z_{k+1} / z_k, shape (T-2, *spatial)
    denom = z[:-1]
    r = z[1:] / (denom + eps * (np.abs(denom) < eps))

    phase = np.angle(r)          # (-π, π]
    rho = np.abs(r)              # ≥ 0

    # Per-step |r| is noisy when ωΔt ≠ π/2 because the delay embedding z_k is
    # elliptical — |z_k| oscillates through each physical period even for a
    # pure-energy-conserving wave. Use per-cell log-amplitude regression on
    # |z| over time as the RESILIENT growth-rate estimator: fit log|z_k| ~
    # α·k, take g = exp(α). For truly stable oscillations, α → 0 → g → 1.
    z_mag = np.abs(z)   # (T-1, *spatial)
    log_mag = np.log(z_mag + eps)
    Tm1 = z_mag.shape[0]
    k_axis = np.arange(Tm1, dtype=np.float64)
    k_mean = k_axis.mean()
    k_var = float(np.var(k_axis))
    # least-squares slope of log|z| vs k, broadcast over spatial dims
    if spatial_shape:
        flat_log = log_mag.reshape(Tm1, -1)
        slope = ((k_axis[:, None] - k_mean) * (flat_log - flat_log.mean(axis=0, keepdims=True))
                 ).mean(axis=0) / max(k_var, eps)
        per_cell_growth = np.exp(slope)
    else:
        slope = float(np.mean((k_axis - k_mean) * (log_mag - log_mag.mean())) / max(k_var, eps))
        per_cell_growth = np.asarray([math.exp(slope)])

    # Signature masks (per cell-step). Use >= for growth so exact-threshold
    # synthetic cases (e.g. checkerboard mode with literal 1.01 factor) match.
    phase_to_pi = np.abs(np.abs(phase) - math.pi)         # close to ±π
    phase_to_half_pi = np.abs(np.abs(phase) - 0.5 * math.pi)  # close to ±π/2
    near_zero_phase = np.abs(phase)  # close to 0 (slow mode)

    mask_checkerboard = (phase_to_pi < phase_threshold_rad) & (rho >= growth_threshold)
    mask_quarter = (phase_to_half_pi < phase_threshold_rad) & (np.abs(rho - 1.0) < 0.1)
    mask_slow = (near_zero_phase < phase_threshold_rad) & (np.abs(rho - 1.0) < 0.1)
    mask_diverging = rho > 2.0  # independent of phase

    n_cell_steps = mask_checkerboard.size
    cfl_frac = float(mask_checkerboard.sum()) / n_cell_steps
    quarter_frac = float(mask_quarter.sum()) / n_cell_steps
    slow_frac = float(mask_slow.sum()) / n_cell_steps
    diverging_frac = float(mask_diverging.sum()) / n_cell_steps

    # ρ stats (kept for reporting; verdict uses per-cell-growth instead)
    rho_median = float(np.median(rho))
    rho_p95 = float(np.quantile(rho, 0.95))
    rho_max = float(rho.max())
    growth_median = float(np.median(per_cell_growth))
    growth_p95 = float(np.quantile(per_cell_growth, 0.95))

    # Per-step trends: fraction of spatial cells flagged at each step
    def _per_step_fraction(mask: np.ndarray) -> list[float]:
        if spatial_shape:
            m = mask.reshape(mask.shape[0], -1)
            return [float(x) for x in m.mean(axis=1)]
        return [float(x) for x in mask.astype(float)]

    per_step_cfl = _per_step_fraction(mask_checkerboard)
    per_step_div = _per_step_fraction(mask_diverging)

    if spatial_shape:
        rho_flat = rho.reshape(rho.shape[0], -1)
        per_step_rho_mean = [float(x) for x in rho_flat.mean(axis=1)]
    else:
        per_step_rho_mean = [float(x) for x in rho]

    # Spatial hotspots — rank cells by sum over time of (checkerboard + 0.5·diverging)
    if spatial_shape and flat_spatial > 1:
        cell_score = (mask_checkerboard.astype(np.float64) +
                      0.5 * mask_diverging.astype(np.float64))
        cell_total = cell_score.reshape(cell_score.shape[0], -1).sum(axis=0)
        k = min(top_k_hotspots, cell_total.size)
        if k > 0:
            idx = np.argpartition(-cell_total, k - 1)[:k]
            idx = idx[np.argsort(-cell_total[idx])]
            hotspots = []
            for ix in idx:
                coord = np.unravel_index(int(ix), tuple(spatial_shape))
                hotspots.append({
                    "coord": [int(c) for c in coord],
                    "cumulative_score": float(cell_total[ix]),
                    "flat_index": int(ix),
                })
        else:
            hotspots = []
    else:
        hotspots = []

    # Verdict logic — uses per-cell log-amplitude growth (robust), not
    # per-step |r| (contaminated by delay-embedding ellipticity at ωΔt ≠ π/2).
    if cfl_frac > 0.02:
        verdict = "CFL_violation"
        verdict_reason = (f"{cfl_frac:.1%} of cell-steps match 2Δt checkerboard + growth — "
                          f"gravity-wave CFL almost certainly violated; check √(g·h) term "
                          f"in compute_cfl_dt or equivalent time-step limiter")
    elif diverging_frac > 0.01 or rho_max > 10.0:
        verdict = "diverging"
        verdict_reason = (f"amplitude growth: {diverging_frac:.1%} cell-steps ρ>2, ρ_max={rho_max:.2e} — "
                          f"some non-checkerboard instability (boundary reflection, nonlinear "
                          f"blowup, or missing damping)")
    elif growth_p95 > 1.03:
        verdict = "mild_growth"
        verdict_reason = (f"per-cell growth p95={growth_p95:.4f}/step (median={growth_median:.4f}) — "
                          f"sub-critical but non-conservative; may be intentional forcing, or a "
                          f"nudging/forcing term without matching dissipation")
    else:
        verdict = "stable"
        verdict_reason = (f"per-cell growth median={growth_median:.4f}/step, CFL-flag={cfl_frac:.2%} — "
                          f"no CFL/dispersion signature, field evolves within expected bounds")

    return TemporalReport(
        shape_T=T,
        shape_spatial=spatial_shape,
        dt=dt,
        cfl_checkerboard_fraction=cfl_frac,
        quarter_resonance_fraction=quarter_frac,
        slow_mode_fraction=slow_frac,
        diverging_fraction=diverging_frac,
        rho_median=rho_median,
        rho_p95=rho_p95,
        rho_max=rho_max,
        growth_median=growth_median,
        growth_p95=growth_p95,
        per_step_cfl_fraction=per_step_cfl,
        per_step_diverging_fraction=per_step_div,
        per_step_rho_mean=per_step_rho_mean,
        hotspots=hotspots,
        verdict=verdict,
        verdict_reason=verdict_reason,
        phase_threshold_rad=phase_threshold_rad,
        growth_threshold=growth_threshold,
    )


def load_time_series(path: str | Path, key: str = "") -> tuple[np.ndarray, float]:
    """Load a time-series tensor from .npz, .npy, or .json.

    .npz: uses `key` if given, else the first array in the archive.
          If a 'dt' scalar key is present, returned as second element.
    .npy: returns array directly; dt defaults to 1.0.
    .json: expects {"u": [[...],...], "dt": float} or a raw nested list.
    """
    p = Path(path)
    suf = p.suffix.lower()
    dt = 1.0
    if suf == ".npz":
        with np.load(p) as npz:
            if key and key in npz.files:
                arr = npz[key]
            elif key:
                raise KeyError(f"key '{key}' not in {path}; available: {npz.files}")
            else:
                arr_keys = [k for k in npz.files if k != "dt"]
                if not arr_keys:
                    raise ValueError(f"{path} has no array keys")
                arr = npz[arr_keys[0]]
            if "dt" in npz.files:
                dt = float(np.asarray(npz["dt"]).item())
        return np.asarray(arr), dt
    if suf == ".npy":
        return np.load(p), dt
    if suf == ".json":
        payload = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            arr_key = key or "u"
            if arr_key not in payload:
                raise KeyError(f"key '{arr_key}' not in JSON payload; keys: {list(payload)}")
            arr = np.asarray(payload[arr_key], dtype=np.float64)
            dt = float(payload.get("dt", 1.0))
            return arr, dt
        return np.asarray(payload, dtype=np.float64), dt
    raise ValueError(f"unsupported extension: {suf} (use .npz, .npy, or .json)")


def analyze_time_series_file(
    path: str | Path,
    key: str = "",
    dt: float | None = None,
    growth_threshold: float = 1.01,
    phase_threshold_rad: float = 0.15,
) -> TemporalReport:
    """Load a file and run analyze_temporal_topology."""
    arr, dt_from_file = load_time_series(path, key=key)
    effective_dt = dt_from_file if dt is None else dt
    return analyze_temporal_topology(
        arr,
        dt=effective_dt,
        growth_threshold=growth_threshold,
        phase_threshold_rad=phase_threshold_rad,
    )


def temporal_report(report: TemporalReport) -> str:
    """Human-readable text report."""
    lines = []
    lines.append("=" * 68)
    lines.append("Tensorearch Temporal-Topology Probe (对称循环数)")
    lines.append("=" * 68)
    lines.append(f"Shape: T={report.shape_T}, spatial={report.shape_spatial}, dt={report.dt}")
    lines.append("")
    lines.append(f"VERDICT: {report.verdict.upper()}")
    lines.append(f"  {report.verdict_reason}")
    lines.append("")
    lines.append("Signature fractions (of all cell-steps):")
    lines.append(f"  CFL checkerboard (φ≈±π ∧ ρ>{report.growth_threshold}): "
                 f"{report.cfl_checkerboard_fraction:.3%}")
    lines.append(f"  Quarter-period resonance (φ≈±π/2 ∧ ρ≈1):              "
                 f"{report.quarter_resonance_fraction:.3%}")
    lines.append(f"  Slow physical mode (φ≈0 ∧ ρ≈1):                       "
                 f"{report.slow_mode_fraction:.3%}")
    lines.append(f"  Diverging (ρ>2):                                      "
                 f"{report.diverging_fraction:.3%}")
    lines.append("")
    lines.append("Per-cell log-amplitude growth rate (robust; use this for verdict):")
    lines.append(f"  median = {report.growth_median:.4f}")
    lines.append(f"  p95    = {report.growth_p95:.4f}")
    lines.append("Raw per-step |r| (noisy for ωΔt ≠ π/2):")
    lines.append(f"  median = {report.rho_median:.4f}  p95 = {report.rho_p95:.4f}  "
                 f"max = {report.rho_max:.4e}")
    lines.append("")
    T_steps = len(report.per_step_cfl_fraction)
    if T_steps:
        show = min(10, T_steps)
        idx_sample = [int(round(i * (T_steps - 1) / max(1, show - 1))) for i in range(show)]
        idx_sample = sorted(set(idx_sample))
        lines.append(f"Per-step CFL fraction (sampled {len(idx_sample)}/{T_steps} steps):")
        for i in idx_sample:
            lines.append(f"  step {i:4d}: CFL={report.per_step_cfl_fraction[i]:.2%}  "
                         f"ρ_mean={report.per_step_rho_mean[i]:.4f}  "
                         f"div={report.per_step_diverging_fraction[i]:.2%}")
    lines.append("")
    if report.hotspots:
        lines.append(f"Top-{len(report.hotspots)} spatial hotspots (cumulative instability):")
        for h in report.hotspots:
            coord_str = ",".join(str(c) for c in h["coord"])
            lines.append(f"  [{coord_str}]  score={h['cumulative_score']:.2f}")
        lines.append("")
    return "\n".join(lines)


def temporal_report_json(report: TemporalReport) -> str:
    """JSON-formatted report."""
    return json.dumps(report.to_dict(), indent=2)
