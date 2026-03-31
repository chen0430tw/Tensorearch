import math

from .graph import ArchitectureGraph


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: list[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def local_similarity(src_vec: list[float], dst_vec: list[float]) -> float:
    if not src_vec or not dst_vec:
        return 1.0
    denom = _norm(src_vec) * _norm(dst_vec)
    if denom == 0.0:
        return 0.0
    cosine = _dot(src_vec, dst_vec) / denom
    return max(0.0, 0.5 * (cosine + 1.0))


def optimized_chain_weights(graph: ArchitectureGraph) -> dict[tuple[str, str], float]:
    slices = graph.slice_map()
    chain: dict[tuple[str, str], float] = {}
    for edge in graph.edges:
        src = slices[edge.src]
        dst = slices[edge.dst]
        geom = local_similarity(src.local_vector_space, dst.local_vector_space)
        chain[(edge.src, edge.dst)] = (
            max(edge.weight, 0.0)
            * max(edge.transport_scale, 0.0)
            * max(src.write_magnitude, 0.0)
            * max(dst.read_sensitivity, 0.0)
            * max(src.doi_alignment, 0.0)
            * geom
        )
    return chain


def normalize_weights(graph: ArchitectureGraph) -> dict[tuple[str, str], float]:
    chain = optimized_chain_weights(graph)
    outgoing: dict[str, float] = {}
    for edge in graph.edges:
        outgoing[edge.src] = outgoing.get(edge.src, 0.0) + chain[(edge.src, edge.dst)]

    probs: dict[tuple[str, str], float] = {}
    for edge in graph.edges:
        denom = outgoing.get(edge.src, 0.0)
        edge_w = chain[(edge.src, edge.dst)]
        probs[(edge.src, edge.dst)] = 0.0 if denom == 0.0 else edge_w / denom
    return probs


def propagate_state(
    graph: ArchitectureGraph,
    state: dict[str, float],
    lam: float = 0.3,
    steps: int = 3,
) -> dict[str, float]:
    probs = normalize_weights(graph)
    current = dict(state)
    for _ in range(steps):
        nxt: dict[str, float] = {}
        for slice_state in graph.slices:
            incoming = 0.0
            for edge in graph.edges:
                if edge.dst == slice_state.slice_id:
                    incoming += probs[(edge.src, edge.dst)] * current.get(edge.src, 0.0)
            nxt[slice_state.slice_id] = (1.0 - lam) * current.get(slice_state.slice_id, 0.0) + lam * incoming
        current = nxt
    return current
