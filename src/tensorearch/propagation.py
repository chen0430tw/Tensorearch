from .graph import ArchitectureGraph


def normalize_weights(graph: ArchitectureGraph) -> dict[tuple[str, str], float]:
    outgoing: dict[str, float] = {}
    for edge in graph.edges:
        outgoing[edge.src] = outgoing.get(edge.src, 0.0) + max(edge.weight, 0.0)

    probs: dict[tuple[str, str], float] = {}
    for edge in graph.edges:
        denom = outgoing.get(edge.src, 0.0)
        probs[(edge.src, edge.dst)] = 0.0 if denom == 0.0 else max(edge.weight, 0.0) / denom
    return probs
