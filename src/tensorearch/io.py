import json
from pathlib import Path

from .features import enrich_graph
from .graph import ArchitectureGraph
from .schema import SliceEdge, SliceState, SystemTrace, TrainingStep, TrainingTrace


def load_graph_from_dict(payload: dict) -> ArchitectureGraph:
    system_d = payload.get("system", {})
    graph = ArchitectureGraph(
        system=SystemTrace(
            name=system_d.get("name", "unknown"),
            model_arch=system_d.get("model_arch", "unknown"),
            num_layers=int(system_d.get("num_layers", 0)),
            hidden_size=int(system_d.get("hidden_size", 0)),
            intermediate_size=int(system_d.get("intermediate_size", 0)),
            num_heads=int(system_d.get("num_heads", 0)),
            kv_heads=int(system_d.get("kv_heads", 0)),
            batch_size=int(system_d.get("batch_size", 0)),
            seq_len=int(system_d.get("seq_len", 0)),
            dtype=system_d.get("dtype", "unknown"),
            measured_latency_ms=float(system_d.get("measured_latency_ms", 0.0)),
            measured_tokens_per_sec=float(system_d.get("measured_tokens_per_sec", 0.0)),
            device=system_d.get("device", "unknown"),
            tp_degree=int(system_d.get("tp_degree", 1)),
            pp_degree=int(system_d.get("pp_degree", 1)),
            metadata=dict(system_d.get("metadata", {})),
        )
    )
    for item in payload.get("slices", []):
        graph.add_slice(
            SliceState(
                slice_id=item["slice_id"],
                kind=item["kind"],
                layer_index=int(item.get("layer_index", -1)),
                op_type=item.get("op_type", item["kind"]),
                hidden_size=int(item.get("hidden_size", 0)),
                intermediate_size=int(item.get("intermediate_size", 0)),
                num_heads=int(item.get("num_heads", 0)),
                kv_heads=int(item.get("kv_heads", 0)),
                tokens_in=int(item.get("tokens_in", 0)),
                tokens_out=int(item.get("tokens_out", 0)),
                flops=float(item.get("flops", 0.0)),
                activation_bytes=float(item.get("activation_bytes", 0.0)),
                memory_bytes=float(item.get("memory_bytes", 0.0)),
                weight_bytes=float(item.get("weight_bytes", 0.0)),
                kv_bytes=float(item.get("kv_bytes", 0.0)),
                comm_bytes=float(item.get("comm_bytes", 0.0)),
                sync_cost=float(item.get("sync_cost", 0.0)),
                kernel_time_ms=float(item.get("kernel_time_ms", 0.0)),
                stall_ms=float(item.get("stall_ms", 0.0)),
                measured_latency_ms=float(item.get("measured_latency_ms", 0.0)),
                write_magnitude=float(item.get("write_magnitude", 1.0)),
                read_sensitivity=float(item.get("read_sensitivity", 1.0)),
                doi_alignment=float(item.get("doi_alignment", 1.0)),
                obedience_target=float(item.get("obedience_target", 1.0)),
                local_vector_space=[float(x) for x in item.get("local_vector_space", [])],
                metadata=dict(item.get("metadata", {})),
            )
        )
    for item in payload.get("edges", []):
        graph.add_edge(
            SliceEdge(
                src=item["src"],
                dst=item["dst"],
                weight=float(item.get("weight", 0.0)),
                kind=item.get("kind", "dataflow"),
                edge_type=item.get("edge_type", item.get("kind", "dataflow")),
                edge_bytes=float(item.get("edge_bytes", 0.0)),
                transport_scale=float(item.get("transport_scale", 1.0)),
                collective=item.get("collective", "none"),
                same_device=bool(item.get("same_device", True)),
                same_stage=bool(item.get("same_stage", True)),
                metadata=dict(item.get("metadata", {})),
            )
        )
    return enrich_graph(graph)


def load_graph_from_json(path: str | Path) -> ArchitectureGraph:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return load_graph_from_dict(payload)


def load_training_trace_from_dict(payload: dict) -> TrainingTrace:
    return TrainingTrace(
        run_id=payload.get("run_id", "unknown_run"),
        checkpoint_path=payload.get("checkpoint_path", ""),
        target_metric=payload.get("target_metric", "val_metric"),
        steps=[
            TrainingStep(
                step=int(item["step"]),
                train_loss=float(item.get("train_loss", 0.0)),
                val_metric=float(item.get("val_metric", 0.0)),
                grad_norm=float(item.get("grad_norm", 0.0)),
                curvature=float(item.get("curvature", 0.0)),
                direction_consistency=float(item.get("direction_consistency", 0.0)),
                metadata=dict(item.get("metadata", {})),
            )
            for item in payload.get("steps", [])
        ],
        metadata=dict(payload.get("metadata", {})),
    )


def load_training_trace_from_json(path: str | Path) -> TrainingTrace:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return load_training_trace_from_dict(payload)
