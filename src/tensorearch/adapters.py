from __future__ import annotations

from pathlib import Path

from .features import enrich_graph
from .graph import ArchitectureGraph
from .schema import SliceEdge, SliceState, SystemTrace
from .space import analyze_source_file


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


# ================================================================
# Family-aware slice templates
# ================================================================

_FAMILY_TEMPLATES: dict[str, list[dict]] = {
    "diffusion_unet": [
        {"kind": "conv_in", "op_type": "conv", "flops_base": 3.0, "mem_base": 2.5},
        {"kind": "down_block", "op_type": "downsample", "flops_base": 5.0, "mem_base": 4.0},
        {"kind": "mid_block", "op_type": "attention", "flops_base": 6.0, "mem_base": 5.0},
        {"kind": "up_block", "op_type": "upsample", "flops_base": 5.0, "mem_base": 4.0},
        {"kind": "conv_out", "op_type": "conv", "flops_base": 2.0, "mem_base": 2.0},
    ],
    "latent_attention": [
        {"kind": "attention", "op_type": "attn", "flops_base": 6.0, "mem_base": 5.0},
        {"kind": "ffn", "op_type": "ffn", "flops_base": 5.0, "mem_base": 4.0},
    ],
    "adapterization": [
        {"kind": "base_layer", "op_type": "linear", "flops_base": 4.0, "mem_base": 3.5},
        {"kind": "lora_down", "op_type": "linear", "flops_base": 1.0, "mem_base": 0.5},
        {"kind": "lora_up", "op_type": "linear", "flops_base": 1.0, "mem_base": 0.5},
    ],
    "runtime_wrapper": [
        {"kind": "quantize", "op_type": "quant", "flops_base": 2.0, "mem_base": 1.5},
        {"kind": "kernel", "op_type": "compute", "flops_base": 5.0, "mem_base": 4.0},
        {"kind": "dequantize", "op_type": "dequant", "flops_base": 2.0, "mem_base": 1.5},
    ],
    "video_temporal": [
        {"kind": "spatial_conv", "op_type": "conv3d", "flops_base": 6.0, "mem_base": 5.5},
        {"kind": "temporal_attn", "op_type": "attn", "flops_base": 7.0, "mem_base": 6.0},
        {"kind": "motion_module", "op_type": "conv", "flops_base": 4.0, "mem_base": 3.5},
    ],
    "audio_spectral": [
        {"kind": "stft", "op_type": "fft", "flops_base": 3.0, "mem_base": 2.0},
        {"kind": "encoder", "op_type": "conv", "flops_base": 5.0, "mem_base": 4.0},
        {"kind": "decoder", "op_type": "conv", "flops_base": 5.0, "mem_base": 4.0},
        {"kind": "vocoder", "op_type": "conv", "flops_base": 4.0, "mem_base": 3.5},
    ],
    "threed_generative": [
        {"kind": "point_encoder", "op_type": "mlp", "flops_base": 4.0, "mem_base": 3.0},
        {"kind": "field_network", "op_type": "mlp", "flops_base": 6.0, "mem_base": 5.0},
        {"kind": "renderer", "op_type": "raycast", "flops_base": 8.0, "mem_base": 6.0},
    ],
    "speech_language": [
        {"kind": "audio_encoder", "op_type": "conv_attn", "flops_base": 5.0, "mem_base": 4.5},
        {"kind": "text_decoder", "op_type": "attn", "flops_base": 6.0, "mem_base": 5.0},
    ],
    "world_model": [
        {"kind": "encoder", "op_type": "conv", "flops_base": 4.0, "mem_base": 3.0},
        {"kind": "rnn_core", "op_type": "lstm", "flops_base": 5.0, "mem_base": 4.5},
        {"kind": "mixture_head", "op_type": "gmm", "flops_base": 3.0, "mem_base": 2.5},
    ],
    "multimodal_alignment": [
        {"kind": "vision_encoder", "op_type": "vit", "flops_base": 6.0, "mem_base": 5.0},
        {"kind": "qformer", "op_type": "cross_attn", "flops_base": 5.0, "mem_base": 4.5},
        {"kind": "language_model", "op_type": "attn", "flops_base": 6.0, "mem_base": 5.0},
    ],
    "graph_message_passing": [
        {"kind": "message", "op_type": "scatter", "flops_base": 3.0, "mem_base": 2.5},
        {"kind": "aggregate", "op_type": "reduce", "flops_base": 2.0, "mem_base": 2.0},
        {"kind": "update", "op_type": "mlp", "flops_base": 3.0, "mem_base": 2.5},
    ],
    "vision_detection": [
        {"kind": "backbone", "op_type": "conv", "flops_base": 7.0, "mem_base": 6.0},
        {"kind": "fpn", "op_type": "conv", "flops_base": 4.0, "mem_base": 3.0},
        {"kind": "rpn", "op_type": "conv", "flops_base": 3.0, "mem_base": 2.5},
        {"kind": "roi_head", "op_type": "pool_fc", "flops_base": 4.0, "mem_base": 3.0},
    ],
    "bio_sequence": [
        {"kind": "embedding", "op_type": "embed", "flops_base": 2.0, "mem_base": 2.0},
        {"kind": "evoformer", "op_type": "attn_pair", "flops_base": 8.0, "mem_base": 7.0},
        {"kind": "structure_head", "op_type": "mlp", "flops_base": 4.0, "mem_base": 3.0},
    ],
    "baseline_residual": [
        {"kind": "conv", "op_type": "conv", "flops_base": 4.0, "mem_base": 3.0},
        {"kind": "pool", "op_type": "pool", "flops_base": 1.0, "mem_base": 0.5},
        {"kind": "dense", "op_type": "fc", "flops_base": 3.0, "mem_base": 2.5},
    ],
    "propagation": [
        {"kind": "phase", "op_type": "phase", "flops_base": 4.0, "mem_base": 3.5},
        {"kind": "propagation", "op_type": "prop", "flops_base": 5.0, "mem_base": 4.5},
        {"kind": "ffn", "op_type": "ffn", "flops_base": 4.5, "mem_base": 3.5},
    ],
}


def graph_from_family(payload: dict, family: str) -> ArchitectureGraph:
    """从家族模板生成 ArchitectureGraph。"""
    template = _FAMILY_TEMPLATES.get(family, _FAMILY_TEMPLATES["baseline_residual"])
    graph = ArchitectureGraph(system=_base_system(
        payload.get("name", f"{family}-trace"), family, payload
    ))

    num_layers = max(int(payload.get("num_layers", 1)), 1)
    hidden = int(payload.get("hidden_size", 0))
    intermediate = int(payload.get("intermediate_size", hidden * 4 if hidden else 0))
    tokens = int(payload.get("batch_size", 1)) * int(payload.get("seq_len", 1))

    for layer in range(num_layers):
        prefix = f"blk{layer}"
        prev_slice_id = None

        for tmpl in template:
            slice_id = f"{prefix}.{tmpl['kind']}"
            graph.add_slice(SliceState(
                slice_id=slice_id,
                kind=tmpl["kind"],
                op_type=tmpl["op_type"],
                layer_index=layer,
                hidden_size=hidden,
                intermediate_size=intermediate,
                tokens_in=tokens,
                tokens_out=tokens,
                flops=tmpl["flops_base"] + 0.2 * layer,
                activation_bytes=tmpl["mem_base"] + 0.2 * layer,
                memory_bytes=tmpl["mem_base"] * 0.8 + 0.15 * layer,
                weight_bytes=tmpl["mem_base"] * 0.6,
                kernel_time_ms=tmpl["flops_base"] * 0.5 + 0.1 * layer,
                stall_ms=0.3 + 0.05 * layer,
                measured_latency_ms=tmpl["flops_base"] * 0.6 + 0.15 * layer,
                write_magnitude=1.0 + 0.05 * layer,
                read_sensitivity=0.95 + 0.05 * layer,
                doi_alignment=0.95,
            ))
            if prev_slice_id:
                graph.add_edge(SliceEdge(
                    src=prev_slice_id, dst=slice_id,
                    weight=0.85, edge_type="sequential", edge_bytes=0.5,
                ))
            prev_slice_id = slice_id

        if layer + 1 < num_layers and prev_slice_id:
            next_first = f"blk{layer + 1}.{template[0]['kind']}"
            graph.add_edge(SliceEdge(
                src=prev_slice_id, dst=next_first,
                weight=0.8, edge_type="residual", edge_bytes=0.5,
            ))

    return enrich_graph(graph)


def graph_from_source_file(source_path: str) -> ArchitectureGraph:
    """从源码自动推断家族并生成 trace graph。

    source file → space classify → family template → ArchitectureGraph
    """
    analysis = analyze_source_file(source_path)
    family = analysis["space_family_projection"]["dominant_family"]

    density = analysis.get("raw_density", {})
    total_density = sum(density.values())
    estimated_layers = max(1, min(12, int(total_density / 3)))

    payload = {
        "name": Path(source_path).stem,
        "num_layers": estimated_layers,
        "hidden_size": 0,
        "metadata": {
            "source_file": str(source_path),
            "classification": analysis["classification"],
            "family": family,
        },
    }

    return graph_from_family(payload, family)
