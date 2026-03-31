import math

from .graph import ArchitectureGraph
from .propagation import normalize_weights, optimized_chain_weights
from .schema import SliceState


def base_cost(
    slice_state: SliceState,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    delta: float = 1.0,
) -> float:
    return (
        alpha * slice_state.flops
        + beta * slice_state.memory_bytes
        + gamma * slice_state.comm_bytes
        + delta * slice_state.sync_cost
    )


def slice_costs(
    graph: ArchitectureGraph,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    delta: float = 1.0,
) -> dict[str, float]:
    return {
        slice_state.slice_id: base_cost(slice_state, alpha, beta, gamma, delta)
        for slice_state in graph.slices
    }


def topological_congestion(graph: ArchitectureGraph, costs: dict[str, float]) -> dict[str, float]:
    chain = optimized_chain_weights(graph)
    return {
        slice_state.slice_id: sum(
            chain[(edge.src, edge.dst)] * costs.get(edge.dst, 0.0)
            for edge in graph.edges
            if edge.src == slice_state.slice_id
        )
        for slice_state in graph.slices
    }


def propagated_costs(
    graph: ArchitectureGraph,
    costs: dict[str, float],
    eta: float = 1.0,
) -> dict[str, float]:
    probs = normalize_weights(graph)
    propagated: dict[str, float] = {}
    for slice_state in graph.slices:
        downstream = 0.0
        for edge in graph.edges:
            if edge.src == slice_state.slice_id:
                downstream += probs[(edge.src, edge.dst)] * costs.get(edge.dst, 0.0)
        propagated[slice_state.slice_id] = costs[slice_state.slice_id] + eta * downstream
    return propagated


def slice_bottleneck_index(propagated: dict[str, float]) -> dict[str, float]:
    total = sum(propagated.values())
    if total == 0.0:
        return {key: 0.0 for key in propagated}
    return {key: value / total for key, value in propagated.items()}


def global_coupling_efficiency(graph: ArchitectureGraph) -> float:
    n = len(graph.slices)
    if n <= 1:
        return 0.0
    chain = optimized_chain_weights(graph)
    total = sum(chain[(edge.src, edge.dst)] for edge in graph.edges if edge.src != edge.dst)
    return total / (n * (n - 1))


def direction_of_interest(graph: ArchitectureGraph) -> dict[str, float]:
    return {slice_state.slice_id: slice_state.doi_alignment for slice_state in graph.slices}


def direct_effects(graph: ArchitectureGraph, costs: dict[str, float]) -> dict[str, float]:
    doi = direction_of_interest(graph)
    slice_map = graph.slice_map()
    return {
        sid: costs[sid] * doi[sid] * slice_map[sid].write_magnitude
        for sid in costs
    }


def estimated_total_effects(
    graph: ArchitectureGraph,
    propagated: dict[str, float],
) -> dict[str, float]:
    doi = direction_of_interest(graph)
    slice_map = graph.slice_map()
    return {
        sid: propagated[sid] * doi[sid] * slice_map[sid].write_magnitude
        for sid in propagated
    }


def edge_attributions(graph: ArchitectureGraph, costs: dict[str, float]) -> dict[tuple[str, str], float]:
    chain = optimized_chain_weights(graph)
    return {
        (edge.src, edge.dst): chain[(edge.src, edge.dst)] * (costs.get(edge.dst, 0.0) + edge.edge_bytes)
        for edge in graph.edges
    }


def _vec_norm(vec: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


def freedom_index(graph: ArchitectureGraph) -> dict[str, float]:
    probs = normalize_weights(graph)
    outgoing_entropy: dict[str, float] = {}
    for slice_state in graph.slices:
        sid = slice_state.slice_id
        outgoing = [probs[(edge.src, edge.dst)] for edge in graph.edges if edge.src == sid and probs[(edge.src, edge.dst)] > 0.0]
        if not outgoing:
            entropy = 0.0
        else:
            entropy = -sum(p * math.log(p + 1e-12) for p in outgoing)
        local_span = _vec_norm(slice_state.local_vector_space)
        routing_freedom = slice_state.write_magnitude * slice_state.read_sensitivity
        outgoing_entropy[sid] = entropy + 0.1 * local_span + 0.2 * routing_freedom
    return outgoing_entropy


def compliance_index(graph: ArchitectureGraph) -> dict[str, float]:
    fi = freedom_index(graph)
    compliance: dict[str, float] = {}
    for slice_state in graph.slices:
        sid = slice_state.slice_id
        observed = slice_state.doi_alignment
        target = slice_state.obedience_target
        deviation = abs(observed - target)
        compliance[sid] = target / (1.0 + fi[sid] + deviation)
    return compliance


def global_obedience_score(graph: ArchitectureGraph) -> float:
    comp = compliance_index(graph)
    if not comp:
        return 0.0
    return sum(comp.values()) / len(comp)


def _normalized_entropy(values: list[float]) -> float:
    positive = [max(v, 0.0) for v in values if v > 0.0]
    if len(positive) <= 1:
        return 0.0
    total = sum(positive)
    probs = [v / total for v in positive]
    ent = -sum(p * math.log(p + 1e-12) for p in probs)
    return ent / math.log(len(probs))


def routing_entropy(graph: ArchitectureGraph) -> dict[str, float]:
    probs = normalize_weights(graph)
    out: dict[str, float] = {}
    for slice_state in graph.slices:
        sid = slice_state.slice_id
        outgoing = [probs[(edge.src, edge.dst)] for edge in graph.edges if edge.src == sid]
        out[sid] = _normalized_entropy(outgoing)
    return out


def effect_entropy(graph: ArchitectureGraph, total_effects: dict[str, float]) -> dict[str, float]:
    probs = normalize_weights(graph)
    out: dict[str, float] = {}
    for slice_state in graph.slices:
        sid = slice_state.slice_id
        distributed = [
            probs[(edge.src, edge.dst)] * max(total_effects.get(edge.dst, 0.0), 0.0)
            for edge in graph.edges
            if edge.src == sid
        ]
        out[sid] = _normalized_entropy(distributed)
    return out


def compliance_entropy(graph: ArchitectureGraph) -> dict[str, float]:
    comp = compliance_index(graph)
    out: dict[str, float] = {}
    for slice_state in graph.slices:
        sid = slice_state.slice_id
        target = max(slice_state.obedience_target, 1e-6)
        ratio = max(min(comp[sid] / target, 1.0), 1e-6)
        anti = max(1.0 - ratio, 1e-6)
        out[sid] = -(ratio * math.log(ratio) + anti * math.log(anti)) / math.log(2.0)
    return out


def intelligence_index(
    graph: ArchitectureGraph,
    total_effects: dict[str, float],
) -> dict[str, float]:
    re = routing_entropy(graph)
    ee = effect_entropy(graph, total_effects)
    ce = compliance_entropy(graph)
    comp = compliance_index(graph)
    fi = freedom_index(graph)
    out: dict[str, float] = {}
    for slice_state in graph.slices:
        sid = slice_state.slice_id
        adaptive = 0.5 * re[sid] + 0.5 * ee[sid]
        disciplined = comp[sid] / (1.0 + fi[sid])
        out[sid] = adaptive * disciplined * (1.0 + ce[sid])
    return out


def global_intelligence_score(
    graph: ArchitectureGraph,
    total_effects: dict[str, float],
) -> float:
    ii = intelligence_index(graph, total_effects)
    if not ii:
        return 0.0
    return sum(ii.values()) / len(ii)
