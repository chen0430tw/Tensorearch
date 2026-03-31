from __future__ import annotations

from .features import enrich_graph
from .graph import ArchitectureGraph
from .schema import SliceEdge, SliceState, SystemTrace


def _base_system(name: str, model_arch: str, payload: dict) -> SystemTrace:
    return SystemTrace(
        name=name,
        model_arch=model_arch,
        num_layers=int(payload.get("num_layers", 0)),
        hidden_size=int(payload.get("hidden_size", 0)),
        intermediate_size=int(payload.get("intermediate_size", 0)),
        num_heads=int(payload.get("num_heads", 0)),
        kv_heads=int(payload.get("kv_heads", payload.get("num_heads", 0))),
        batch_size=int(payload.get("batch_size", 0)),
        seq_len=int(payload.get("seq_len", 0)),
        dtype=payload.get("dtype", "bf16"),
        measured_latency_ms=float(payload.get("measured_latency_ms", 0.0)),
        measured_tokens_per_sec=float(payload.get("measured_tokens_per_sec", 0.0)),
        device=payload.get("device", "unknown"),
        tp_degree=int(payload.get("tp_degree", 1)),
        pp_degree=int(payload.get("pp_degree", 1)),
        metadata=dict(payload.get("metadata", {})),
    )


def graph_from_transformer_trace(payload: dict) -> ArchitectureGraph:
    graph = ArchitectureGraph(system=_base_system(payload.get("name", "transformer-trace"), "transformer", payload))
    for layer in range(int(payload.get("num_layers", 0))):
        prefix = f"blk{layer}"
        graph.add_slice(
            SliceState(
                slice_id=f"{prefix}.attn",
                kind="attention",
                op_type="attn",
                layer_index=layer,
                hidden_size=int(payload.get("hidden_size", 0)),
                intermediate_size=int(payload.get("intermediate_size", 0)),
                num_heads=int(payload.get("num_heads", 0)),
                kv_heads=int(payload.get("kv_heads", payload.get("num_heads", 0))),
                tokens_in=int(payload.get("batch_size", 0)) * int(payload.get("seq_len", 0)),
                tokens_out=int(payload.get("batch_size", 0)) * int(payload.get("seq_len", 0)),
                flops=6.0 + layer,
                activation_bytes=5.0 + 0.2 * layer,
                memory_bytes=4.0 + 0.2 * layer,
                weight_bytes=2.8,
                kv_bytes=3.2,
                comm_bytes=1.0 + 0.2 * (payload.get("tp_degree", 1) - 1),
                sync_cost=0.8 + 0.1 * layer,
                kernel_time_ms=3.0 + 0.3 * layer,
                stall_ms=0.8 + 0.1 * layer,
                measured_latency_ms=4.0 + 0.5 * layer,
                write_magnitude=1.2 + 0.05 * layer,
                read_sensitivity=1.0 + 0.05 * layer,
                doi_alignment=1.0,
            )
        )
        graph.add_slice(
            SliceState(
                slice_id=f"{prefix}.ffn",
                kind="ffn",
                op_type="ffn",
                layer_index=layer,
                hidden_size=int(payload.get("hidden_size", 0)),
                intermediate_size=int(payload.get("intermediate_size", 0)),
                tokens_in=int(payload.get("batch_size", 0)) * int(payload.get("seq_len", 0)),
                tokens_out=int(payload.get("batch_size", 0)) * int(payload.get("seq_len", 0)),
                flops=5.0 + layer,
                activation_bytes=4.5 + 0.2 * layer,
                memory_bytes=3.5 + 0.2 * layer,
                weight_bytes=4.2,
                comm_bytes=0.4,
                sync_cost=0.3 + 0.05 * layer,
                kernel_time_ms=3.2 + 0.3 * layer,
                stall_ms=0.5 + 0.1 * layer,
                measured_latency_ms=4.0 + 0.4 * layer,
                write_magnitude=1.0 + 0.05 * layer,
                read_sensitivity=0.95 + 0.05 * layer,
                doi_alignment=0.95,
            )
        )
        graph.add_edge(SliceEdge(src=f"{prefix}.attn", dst=f"{prefix}.ffn", weight=0.9, edge_type="residual", edge_bytes=0.5))
        if layer + 1 < int(payload.get("num_layers", 0)):
            graph.add_edge(
                SliceEdge(
                    src=f"{prefix}.ffn",
                    dst=f"blk{layer+1}.attn",
                    weight=0.8,
                    edge_type="residual",
                    edge_bytes=0.5,
                )
            )
        if int(payload.get("tp_degree", 1)) > 1:
            graph.add_edge(
                SliceEdge(
                    src=f"{prefix}.attn",
                    dst=f"{prefix}.attn",
                    weight=0.4,
                    edge_type="tp_comm",
                    kind="comm",
                    edge_bytes=0.8,
                    collective="allreduce",
                    same_device=False,
                    same_stage=True,
                )
            )
    return enrich_graph(graph)


def graph_from_oscillator_trace(payload: dict) -> ArchitectureGraph:
    graph = ArchitectureGraph(system=_base_system(payload.get("name", "oscillator-trace"), "oscillator", payload))
    for layer in range(int(payload.get("num_layers", 0))):
        prefix = f"blk{layer}"
        graph.add_slice(
            SliceState(
                slice_id=f"{prefix}.phase",
                kind="phase",
                op_type="attn_qkv",
                layer_index=layer,
                hidden_size=int(payload.get("hidden_size", 0)),
                num_heads=int(payload.get("num_heads", 0)),
                kv_heads=int(payload.get("kv_heads", payload.get("num_heads", 0))),
                tokens_in=int(payload.get("batch_size", 0)) * int(payload.get("seq_len", 0)),
                tokens_out=int(payload.get("batch_size", 0)) * int(payload.get("seq_len", 0)),
                flops=4.0 + layer,
                activation_bytes=5.5 + 0.3 * layer,
                memory_bytes=4.8 + 0.2 * layer,
                kv_bytes=2.6,
                comm_bytes=1.2 + 0.3 * (payload.get("tp_degree", 1) - 1),
                sync_cost=1.0 + 0.2 * layer,
                kernel_time_ms=3.8 + 0.5 * layer,
                stall_ms=1.2 + 0.2 * layer,
                measured_latency_ms=5.0 + 0.6 * layer,
                write_magnitude=1.3 + 0.1 * layer,
                read_sensitivity=1.1 + 0.05 * layer,
                doi_alignment=0.9,
            )
        )
        graph.add_slice(
            SliceState(
                slice_id=f"{prefix}.prop",
                kind="propagation",
                op_type="attn_score",
                layer_index=layer,
                hidden_size=int(payload.get("hidden_size", 0)),
                num_heads=int(payload.get("num_heads", 0)),
                kv_heads=int(payload.get("kv_heads", payload.get("num_heads", 0))),
                tokens_in=int(payload.get("batch_size", 0)) * int(payload.get("seq_len", 0)),
                tokens_out=int(payload.get("batch_size", 0)) * int(payload.get("seq_len", 0)),
                flops=5.0 + layer,
                activation_bytes=6.0 + 0.4 * layer,
                memory_bytes=5.0 + 0.3 * layer,
                kv_bytes=2.8,
                comm_bytes=1.5 + 0.3 * (payload.get("tp_degree", 1) - 1),
                sync_cost=1.2 + 0.2 * layer,
                kernel_time_ms=4.3 + 0.6 * layer,
                stall_ms=1.5 + 0.2 * layer,
                measured_latency_ms=5.8 + 0.8 * layer,
                write_magnitude=1.4 + 0.1 * layer,
                read_sensitivity=1.15 + 0.05 * layer,
                doi_alignment=0.88,
            )
        )
        graph.add_slice(
            SliceState(
                slice_id=f"{prefix}.ffn",
                kind="ffn",
                op_type="ffn",
                layer_index=layer,
                hidden_size=int(payload.get("hidden_size", 0)),
                intermediate_size=int(payload.get("intermediate_size", 0)),
                tokens_in=int(payload.get("batch_size", 0)) * int(payload.get("seq_len", 0)),
                tokens_out=int(payload.get("batch_size", 0)) * int(payload.get("seq_len", 0)),
                flops=5.5 + layer,
                activation_bytes=4.8 + 0.2 * layer,
                memory_bytes=3.8 + 0.2 * layer,
                weight_bytes=4.2,
                comm_bytes=0.5,
                sync_cost=0.35 + 0.05 * layer,
                kernel_time_ms=3.4 + 0.4 * layer,
                stall_ms=0.7 + 0.1 * layer,
                measured_latency_ms=4.4 + 0.4 * layer,
                write_magnitude=1.05 + 0.05 * layer,
                read_sensitivity=0.95 + 0.05 * layer,
                doi_alignment=0.94,
            )
        )
        graph.add_edge(SliceEdge(src=f"{prefix}.phase", dst=f"{prefix}.prop", weight=0.85, edge_type="attention", edge_bytes=0.6))
        graph.add_edge(SliceEdge(src=f"{prefix}.prop", dst=f"{prefix}.ffn", weight=0.75, edge_type="residual", edge_bytes=0.6))
        if layer + 1 < int(payload.get("num_layers", 0)):
            graph.add_edge(
                SliceEdge(
                    src=f"{prefix}.ffn",
                    dst=f"blk{layer+1}.phase",
                    weight=0.7,
                    edge_type="residual",
                    edge_bytes=0.5,
                )
            )
        graph.add_edge(
            SliceEdge(
                src=f"{prefix}.prop",
                dst=f"{prefix}.prop",
                weight=0.5,
                edge_type="tp_comm",
                kind="comm",
                edge_bytes=1.0,
                collective="allreduce",
                same_device=False,
                same_stage=True,
            )
        )
    return enrich_graph(graph)
