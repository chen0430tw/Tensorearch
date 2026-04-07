from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


BASELINE_DENSITY = {
    "residual": 0.3812296177637549,
    "attention": 0.11233611144671586,
    "ffn": 0.34391218143014296,
    "expert": 0.0,
    "propagation": 0.0,
    "kv": 0.16252208935938614,
}

# Pre-compiled per-dimension regexes — one pass per dimension instead of N passes
_RE_RESIDUAL = re.compile(
    r"\bresidual\b|\bx\s*\+\s*|\bskip\b|\bshortcut\b|\bln1\b|\bln2\b|\brmsnorm\b|\blayernorm\b",
    re.IGNORECASE,
)
_RE_ATTENTION = re.compile(
    r"\battention\b|\battn\b|\bq_proj\b|\bk_proj\b|\bv_proj\b|\bwo\b|\bwq\b|\bwk\b|\bwv\b"
    r"|\bsoftmax\b|\brope\b|\bmla\b|\bq_lora_rank\b|\bqk_",
    re.IGNORECASE,
)
_RE_FFN = re.compile(
    r"\bffn\b|\bmlp\b|\binter_dim\b|\bintermediate_size\b|\bup_proj\b|\bdown_proj\b"
    r"|\bgate_proj\b|\bsilu\b|\bgelu\b",
    re.IGNORECASE,
)
_RE_EXPERT = re.compile(
    r"\bexpert\b|\bmoe\b|\brouter\b|\bgate\b|\bn_routed_experts\b|\bn_shared_experts\b"
    r"|\bn_activated_experts\b|\bshared_experts\b|\btopk\b",
    re.IGNORECASE,
)
_RE_PROPAGATION = re.compile(
    r"\bprop\b|\bpropagation\b|\boscillat|\bstate[_ ]field\b|\bdynamics\b|\bphase\b|\bcoupl",
    re.IGNORECASE,
)
_RE_KV = re.compile(
    r"\bkv\b|\bkv_cache\b|\bcache\b|\bkey[_ ]value\b|\bwkv\b|\bkv_lora_rank\b|\blatent\b|\btransport\b",
    re.IGNORECASE,
)


@dataclass
class DensityVector:
    residual: float
    attention: float
    ffn: float
    expert: float
    propagation: float
    kv: float


def _keyword_hits(text: str, patterns: list[str]) -> int:
    total = 0
    for pattern in patterns:
        total += len(re.findall(pattern, text, flags=re.IGNORECASE))
    return total


def _normalize_density(density: DensityVector) -> dict[str, float]:
    data = asdict(density)
    total = sum(data.values())
    if total <= 0:
        return {k: 0.0 for k in data}
    return {k: v / total for k, v in data.items()}


def _delta_density(current: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {k: current[k] - baseline[k] for k in current}


def infer_density_from_source_text(text: str) -> DensityVector:
    return DensityVector(
        residual=1.0 + len(_RE_RESIDUAL.findall(text)) / 12.0,
        attention=len(_RE_ATTENTION.findall(text)) / 10.0,
        ffn=len(_RE_FFN.findall(text)) / 10.0,
        expert=len(_RE_EXPERT.findall(text)) / 10.0,
        propagation=len(_RE_PROPAGATION.findall(text)) / 8.0,
        kv=len(_RE_KV.findall(text)) / 8.0,
    )


def build_quadrupole_projection(normalized: dict[str, float]) -> dict[str, object]:
    delta = _delta_density(normalized, BASELINE_DENSITY)
    x = normalized["residual"]
    y = normalized["attention"] + normalized["kv"]
    z = normalized["kv"]
    w = normalized["propagation"]

    _axes = {"X": x, "Y": y, "Z": z, "W": w}
    dominant_axis = max(_axes, key=_axes.__getitem__)
    _classifications = {
        "X": "baseline residual",
        "Y": "latent-attention dominant",
        "Z": "kv-transport dominant",
        "W": "propagation dominant",
    }
    classification = _classifications[dominant_axis]

    return {
        "axes": {
            "X_residual": x,
            "Y_latent_attention": y,
            "Z_kv_transport": z,
            "W_propagation": w,
        },
        "extension": {
            "expert_extension": normalized["expert"],
            "ffn_extension": normalized["ffn"],
        },
        "delta_from_baseline": delta,
        "dominant_axis": dominant_axis,
        "classification": classification,
    }


def analyze_source_file(path: str | Path) -> dict[str, object]:
    source = Path(path)
    raw_bytes = source.read_bytes()
    text = None
    for encoding in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be"):
        try:
            text = raw_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = raw_bytes.decode("utf-8", errors="ignore")

    raw_density = infer_density_from_source_text(text)
    normalized = _normalize_density(raw_density)
    quad = build_quadrupole_projection(normalized)
    return {
        "mode": "source_file_auto",
        "source_file": str(source),
        "raw_density": asdict(raw_density),
        "normalized_density": normalized,
        "baseline_density": BASELINE_DENSITY,
        "quadrupole_projection": quad,
        "classification": quad["classification"],
    }


def space_report(payload: dict[str, object]) -> str:
    quad = payload["quadrupole_projection"]
    axes = quad["axes"]
    ext = quad["extension"]
    return (
        f"classification={payload['classification']}\n"
        f"dominant_axis={quad['dominant_axis']}\n"
        f"X_residual={axes['X_residual']:.4f}\n"
        f"Y_latent_attention={axes['Y_latent_attention']:.4f}\n"
        f"Z_kv_transport={axes['Z_kv_transport']:.4f}\n"
        f"W_propagation={axes['W_propagation']:.4f}\n"
        f"expert_extension={ext['expert_extension']:.4f}\n"
        f"ffn_extension={ext['ffn_extension']:.4f}"
    )


def space_report_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)
