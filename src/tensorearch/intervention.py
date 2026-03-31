from __future__ import annotations

from copy import deepcopy

from .graph import ArchitectureGraph
from .schema import Intervention, SliceEdge, SliceState


def clone_graph(graph: ArchitectureGraph) -> ArchitectureGraph:
    return deepcopy(graph)


def _remove_slice(graph: ArchitectureGraph, target: str) -> ArchitectureGraph:
    graph.slices = [s for s in graph.slices if s.slice_id != target]
    graph.edges = [e for e in graph.edges if e.src != target and e.dst != target]
    return graph


def _mask_edge(graph: ArchitectureGraph, target: str) -> ArchitectureGraph:
    src, dst = target.split("->", 1)
    for edge in graph.edges:
        if edge.src == src and edge.dst == dst:
            edge.weight = 0.0
            edge.transport_scale = 0.0
    return graph


def _scale_edge_bandwidth(graph: ArchitectureGraph, target: str, value: float) -> ArchitectureGraph:
    src, dst = target.split("->", 1)
    factor = max(value, 0.0)
    for edge in graph.edges:
        if edge.src == src and edge.dst == dst:
            edge.edge_bytes *= factor
            edge.transport_scale *= factor
    return graph


def _set_slice_field(graph: ArchitectureGraph, target: str, field_name: str, value: float) -> ArchitectureGraph:
    for slice_state in graph.slices:
        if slice_state.slice_id == target and hasattr(slice_state, field_name):
            setattr(slice_state, field_name, value)
    return graph


def _swap_topology(graph: ArchitectureGraph, value: float) -> ArchitectureGraph:
    mode = "local_window" if value <= 0.0 else "mixed_local"
    for edge in graph.edges:
        edge.metadata["topology_mode"] = mode
        if mode == "local_window":
            edge.weight = 1.0 if edge.kind in {"dataflow", "residual"} else edge.weight
        else:
            edge.weight *= 1.0 + min(value, 1.0) * 0.1
    return graph


def apply_intervention(graph: ArchitectureGraph, intervention: Intervention) -> ArchitectureGraph:
    out = clone_graph(graph)
    kind = intervention.kind
    if kind == "remove_slice":
        return _remove_slice(out, intervention.target)
    if kind == "mask_edge":
        return _mask_edge(out, intervention.target)
    if kind == "scale_edge_bandwidth":
        return _scale_edge_bandwidth(out, intervention.target, intervention.value)
    if kind == "set_write_magnitude":
        return _set_slice_field(out, intervention.target, "write_magnitude", intervention.value)
    if kind == "set_read_sensitivity":
        return _set_slice_field(out, intervention.target, "read_sensitivity", intervention.value)
    if kind == "set_doi_alignment":
        return _set_slice_field(out, intervention.target, "doi_alignment", intervention.value)
    if kind == "swap_topology":
        return _swap_topology(out, intervention.value)
    raise ValueError(f"unsupported intervention kind: {kind}")


def intervention_bundle(graph: ArchitectureGraph, interventions: list[Intervention]) -> ArchitectureGraph:
    out = clone_graph(graph)
    for intervention in interventions:
        out = apply_intervention(out, intervention)
    return out
