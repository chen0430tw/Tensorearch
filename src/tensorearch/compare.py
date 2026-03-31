from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from .graph import ArchitectureGraph
from .metrics import (
    compliance_index,
    estimated_total_effects,
    freedom_index,
    global_coupling_efficiency,
    global_intelligence_score,
    global_obedience_score,
    propagated_costs,
    slice_costs,
)


@dataclass
class ComparisonSummary:
    left_name: str
    right_name: str
    left_predicted_bottleneck: str
    right_predicted_bottleneck: str
    left_obedience: float
    right_obedience: float
    left_intelligence: float
    right_intelligence: float
    left_coupling: float
    right_coupling: float


def _predicted_bottleneck(graph: ArchitectureGraph) -> str:
    costs = slice_costs(graph)
    propagated = propagated_costs(graph, costs, eta=0.8)
    return max(propagated.items(), key=lambda kv: kv[1])[0]


def compare_graphs(left: ArchitectureGraph, right: ArchitectureGraph) -> ComparisonSummary:
    left_costs = slice_costs(left)
    right_costs = slice_costs(right)
    left_prop = propagated_costs(left, left_costs, eta=0.8)
    right_prop = propagated_costs(right, right_costs, eta=0.8)
    left_ete = estimated_total_effects(left, left_prop)
    right_ete = estimated_total_effects(right, right_prop)
    return ComparisonSummary(
        left_name=left.system.name if left.system else "left",
        right_name=right.system.name if right.system else "right",
        left_predicted_bottleneck=_predicted_bottleneck(left),
        right_predicted_bottleneck=_predicted_bottleneck(right),
        left_obedience=global_obedience_score(left),
        right_obedience=global_obedience_score(right),
        left_intelligence=global_intelligence_score(left, left_ete),
        right_intelligence=global_intelligence_score(right, right_ete),
        left_coupling=global_coupling_efficiency(left),
        right_coupling=global_coupling_efficiency(right),
    )


def comparison_report(left: ArchitectureGraph, right: ArchitectureGraph) -> str:
    summary = compare_graphs(left, right)
    lines = [
        "Tensorearch comparison report",
        f"left={summary.left_name}",
        f"right={summary.right_name}",
        f"left_bottleneck={summary.left_predicted_bottleneck}",
        f"right_bottleneck={summary.right_predicted_bottleneck}",
        f"left_obedience={summary.left_obedience:.4f}",
        f"right_obedience={summary.right_obedience:.4f}",
        f"left_intelligence={summary.left_intelligence:.4f}",
        f"right_intelligence={summary.right_intelligence:.4f}",
        f"left_coupling={summary.left_coupling:.4f}",
        f"right_coupling={summary.right_coupling:.4f}",
        f"obedience_delta={(summary.left_obedience - summary.right_obedience):.4f}",
        f"intelligence_delta={(summary.left_intelligence - summary.right_intelligence):.4f}",
        f"coupling_delta={(summary.left_coupling - summary.right_coupling):.4f}",
    ]
    return "\n".join(lines)


def comparison_payload(left: ArchitectureGraph, right: ArchitectureGraph) -> dict:
    summary = compare_graphs(left, right)
    return asdict(summary) | {
        "obedience_delta": summary.left_obedience - summary.right_obedience,
        "intelligence_delta": summary.left_intelligence - summary.right_intelligence,
        "coupling_delta": summary.left_coupling - summary.right_coupling,
    }


def comparison_report_json(left: ArchitectureGraph, right: ArchitectureGraph) -> str:
    return json.dumps(comparison_payload(left, right), ensure_ascii=False, indent=2)
