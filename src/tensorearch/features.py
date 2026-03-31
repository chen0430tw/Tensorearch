import math

from .graph import ArchitectureGraph
from .schema import SliceEdge, SliceState


def _safe_ratio(num: float, den: float) -> float:
    return 0.0 if den == 0.0 else num / den


def _log1p(x: float) -> float:
    return math.log1p(max(x, 0.0))


def infer_local_vector_space(slice_state: SliceState) -> list[float]:
    op = slice_state.op_type or slice_state.kind
    if op in {"attn", "attn_qkv", "attn_score", "attn_out", "attention"}:
        return [
            _safe_ratio(slice_state.flops, max(slice_state.tokens_in, 1)),
            _safe_ratio(slice_state.kv_bytes, max(slice_state.activation_bytes, 1.0)),
            _safe_ratio(slice_state.num_heads, max(slice_state.kv_heads, 1)),
            _safe_ratio(slice_state.sync_cost, max(slice_state.measured_latency_ms, 1.0)),
            _safe_ratio(slice_state.comm_bytes, max(slice_state.memory_bytes, 1.0)),
            _safe_ratio(slice_state.write_magnitude, max(slice_state.read_sensitivity, 1e-6)),
        ]
    if op in {"ffn", "ffn_up", "ffn_down", "mlp"}:
        return [
            _safe_ratio(slice_state.intermediate_size, max(slice_state.hidden_size, 1)),
            _safe_ratio(slice_state.flops, max(slice_state.tokens_in, 1)),
            _safe_ratio(slice_state.weight_bytes, max(slice_state.activation_bytes, 1.0)),
            _safe_ratio(slice_state.memory_bytes, max(slice_state.measured_latency_ms, 1.0)),
            _safe_ratio(slice_state.stall_ms, max(slice_state.measured_latency_ms, 1.0)),
            _safe_ratio(slice_state.write_magnitude, max(slice_state.read_sensitivity, 1e-6)),
        ]
    if op in {"allreduce", "allgather", "reduce_scatter", "tp_comm", "pp_comm", "kv_flow"}:
        return [
            _log1p(slice_state.comm_bytes),
            _safe_ratio(slice_state.comm_bytes, max(slice_state.tokens_out, 1)),
            _safe_ratio(slice_state.sync_cost, max(slice_state.measured_latency_ms, 1.0)),
            _safe_ratio(slice_state.stall_ms, max(slice_state.measured_latency_ms, 1.0)),
            _safe_ratio(slice_state.read_sensitivity, max(slice_state.write_magnitude, 1e-6)),
            1.0,
        ]
    return [
        _log1p(slice_state.flops),
        _log1p(slice_state.memory_bytes),
        _log1p(slice_state.comm_bytes),
        _safe_ratio(slice_state.sync_cost, max(slice_state.measured_latency_ms, 1.0)),
        slice_state.write_magnitude,
        slice_state.read_sensitivity,
    ]


def infer_obedience_target(slice_state: SliceState) -> float:
    op = slice_state.op_type or slice_state.kind
    if op in {"attn", "attn_qkv", "attn_score", "attn_out", "attention"}:
        base = 0.95
        penalty = 0.15 * _safe_ratio(slice_state.stall_ms, max(slice_state.measured_latency_ms, 1.0))
        return max(0.55, base - penalty)
    if op in {"ffn", "ffn_up", "ffn_down", "mlp"}:
        base = 0.90
        penalty = 0.10 * _safe_ratio(slice_state.memory_bytes, max(slice_state.weight_bytes, 1.0))
        return max(0.55, base - penalty)
    if op in {"allreduce", "allgather", "reduce_scatter", "tp_comm", "pp_comm", "kv_flow"}:
        return 0.98
    if op in {"lm_head", "readout"}:
        return 0.92
    return 0.85


def estimate_transport_scale(edge: SliceEdge) -> float:
    scale = 1.0
    scale *= 1.0 + 0.05 * _log1p(edge.edge_bytes)
    if edge.collective == "allreduce":
        scale *= 1.35
    elif edge.collective == "allgather":
        scale *= 1.25
    elif edge.collective == "reduce_scatter":
        scale *= 1.20
    if not edge.same_device:
        scale *= 1.30
    if not edge.same_stage:
        scale *= 1.15
    if edge.edge_type in {"tp_comm", "pp_comm", "kv_flow"}:
        scale *= 1.20
    return scale


def enrich_graph(graph: ArchitectureGraph) -> ArchitectureGraph:
    for slice_state in graph.slices:
        if not slice_state.local_vector_space:
            slice_state.local_vector_space = infer_local_vector_space(slice_state)
        if slice_state.obedience_target == 1.0:
            slice_state.obedience_target = infer_obedience_target(slice_state)
    for edge in graph.edges:
        if edge.transport_scale == 1.0 and (
            edge.edge_bytes > 0.0 or edge.collective != "none" or not edge.same_device or not edge.same_stage
        ):
            edge.transport_scale = estimate_transport_scale(edge)
    return graph
