from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .graph import ArchitectureGraph
from .metrics import (
    estimated_total_effects,
    global_intelligence_score,
    global_obedience_score,
    propagated_costs,
    slice_costs,
)


@dataclass
class AnalysisResult:
    name: str
    obedience: float
    intelligence: float
    predicted_bottleneck: str


def analyze_graph(graph: ArchitectureGraph) -> AnalysisResult:
    costs = slice_costs(graph)
    propagated = propagated_costs(graph, costs, eta=0.8)
    ete = estimated_total_effects(graph, propagated)
    bottleneck = max(propagated.items(), key=lambda kv: kv[1])[0]
    return AnalysisResult(
        name=graph.system.name if graph.system else "graph",
        obedience=global_obedience_score(graph),
        intelligence=global_intelligence_score(graph, ete),
        predicted_bottleneck=bottleneck,
    )


def analyze_graphs_parallel(
    graphs: list[ArchitectureGraph],
    max_workers: int = 4,
) -> list[AnalysisResult]:
    if not graphs:
        return []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(analyze_graph, graphs))
