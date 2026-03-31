import argparse
import json

from .graph import ArchitectureGraph
from .io import load_graph_from_json
from .metrics import (
    compliance_index,
    compliance_entropy,
    direct_effects,
    edge_attributions,
    estimated_total_effects,
    freedom_index,
    global_coupling_efficiency,
    global_intelligence_score,
    global_obedience_score,
    intelligence_index,
    routing_entropy,
    effect_entropy,
    propagated_costs,
    slice_bottleneck_index,
    slice_costs,
    topological_congestion,
)
from .propagation import propagate_state
from .schema import SliceEdge, SliceState


def build_sample_graph() -> ArchitectureGraph:
    graph = ArchitectureGraph()
    graph.add_slice(SliceState("embed", "embedding", flops=1.0, memory_bytes=2.0, comm_bytes=0.2, sync_cost=0.1, local_vector_space=[0.9, 0.2, 0.1], write_magnitude=0.8, read_sensitivity=0.8, doi_alignment=0.6))
    graph.add_slice(SliceState("blk0.attn", "attention", flops=6.0, memory_bytes=4.0, comm_bytes=1.2, sync_cost=0.8, local_vector_space=[0.8, 1.0, 0.3], write_magnitude=1.3, read_sensitivity=1.1, doi_alignment=1.1))
    graph.add_slice(SliceState("blk0.ffn", "ffn", flops=5.0, memory_bytes=3.5, comm_bytes=0.6, sync_cost=0.3, local_vector_space=[0.7, 0.9, 0.6], write_magnitude=1.1, read_sensitivity=1.0, doi_alignment=1.0))
    graph.add_slice(SliceState("blk1.attn", "attention", flops=7.0, memory_bytes=4.2, comm_bytes=1.5, sync_cost=1.0, local_vector_space=[0.6, 1.0, 0.8], write_magnitude=1.4, read_sensitivity=1.2, doi_alignment=1.2))
    graph.add_slice(SliceState("lm_head", "readout", flops=2.0, memory_bytes=1.0, comm_bytes=0.3, sync_cost=0.1, local_vector_space=[0.5, 0.6, 1.0], write_magnitude=0.9, read_sensitivity=1.1, doi_alignment=0.7))

    graph.add_edge(SliceEdge("embed", "blk0.attn", weight=0.9, transport_scale=1.0))
    graph.add_edge(SliceEdge("blk0.attn", "blk0.ffn", weight=0.8, transport_scale=1.1))
    graph.add_edge(SliceEdge("blk0.ffn", "blk1.attn", weight=0.7, transport_scale=1.0))
    graph.add_edge(SliceEdge("blk1.attn", "lm_head", weight=0.9, transport_scale=0.9))
    graph.add_edge(SliceEdge("blk0.attn", "blk1.attn", weight=0.4, kind="residual", transport_scale=1.2))
    return graph


def demo_report(graph: ArchitectureGraph | None = None) -> str:
    graph = graph or build_sample_graph()
    costs = slice_costs(graph)
    congestion = topological_congestion(graph, costs)
    propagated = propagated_costs(graph, costs, eta=0.8)
    sbi = slice_bottleneck_index(propagated)
    de = direct_effects(graph, costs)
    ete = estimated_total_effects(graph, propagated)
    edge_attr = edge_attributions(graph, costs)
    freedom = freedom_index(graph)
    compliance = compliance_index(graph)
    route_h = routing_entropy(graph)
    effect_h = effect_entropy(graph, ete)
    comp_h = compliance_entropy(graph)
    intel = intelligence_index(graph, ete)
    propagated_state = propagate_state(graph, costs, lam=0.3, steps=3)
    gce = global_coupling_efficiency(graph)
    obedience = global_obedience_score(graph)
    global_intel = global_intelligence_score(graph, ete)

    ranked = sorted(graph.slices, key=lambda s: propagated[s.slice_id], reverse=True)
    top_edges = sorted(edge_attr.items(), key=lambda kv: kv[1], reverse=True)[:3]
    lines = []
    lines.append("Tensorearch prototype report")
    if graph.system:
        lines.append(
            f"system name={graph.system.name} batch={graph.system.batch_size} seq={graph.system.seq_len} "
            f"latency_ms={graph.system.measured_latency_ms:.3f} tok/s={graph.system.measured_tokens_per_sec:.3f} device={graph.system.device}"
        )
    lines.append(f"global_coupling_efficiency={gce:.4f}")
    lines.append(f"global_obedience_score={obedience:.4f}")
    lines.append(f"global_intelligence_score={global_intel:.4f}")
    lines.append("")
    lines.append("slice metrics")
    for slice_state in ranked:
        sid = slice_state.slice_id
        lines.append(
            f"- {sid}: cost={costs[sid]:.3f} propagated={propagated[sid]:.3f} "
            f"sbi={sbi[sid]:.3f} congestion={congestion[sid]:.3f} "
            f"direct_effect={de[sid]:.3f} total_effect={ete[sid]:.3f} "
            f"freedom={freedom[sid]:.3f} compliance={compliance[sid]:.3f} "
            f"route_entropy={route_h[sid]:.3f} effect_entropy={effect_h[sid]:.3f} "
            f"compliance_entropy={comp_h[sid]:.3f} intelligence={intel[sid]:.3f} "
            f"state3={propagated_state[sid]:.3f}"
        )
    lines.append("")
    lines.append("top edge attributions")
    for (src, dst), score in top_edges:
        lines.append(f"- {src}->{dst}: edge_attr={score:.3f}")
    lines.append("")
    lines.append(f"predicted_bottleneck={ranked[0].slice_id}")
    return "\n".join(lines)


def demo_payload(graph: ArchitectureGraph | None = None) -> dict:
    graph = graph or build_sample_graph()
    costs = slice_costs(graph)
    congestion = topological_congestion(graph, costs)
    propagated = propagated_costs(graph, costs, eta=0.8)
    sbi = slice_bottleneck_index(propagated)
    de = direct_effects(graph, costs)
    ete = estimated_total_effects(graph, propagated)
    edge_attr = edge_attributions(graph, costs)
    freedom = freedom_index(graph)
    compliance = compliance_index(graph)
    route_h = routing_entropy(graph)
    effect_h = effect_entropy(graph, ete)
    comp_h = compliance_entropy(graph)
    intel = intelligence_index(graph, ete)
    propagated_state = propagate_state(graph, costs, lam=0.3, steps=3)
    gce = global_coupling_efficiency(graph)
    obedience = global_obedience_score(graph)
    global_intel = global_intelligence_score(graph, ete)
    ranked = sorted(graph.slices, key=lambda s: propagated[s.slice_id], reverse=True)
    top_edges = sorted(edge_attr.items(), key=lambda kv: kv[1], reverse=True)[:3]
    return {
        "system": {
            "name": graph.system.name if graph.system else "unknown",
            "batch_size": graph.system.batch_size if graph.system else 0,
            "seq_len": graph.system.seq_len if graph.system else 0,
            "latency_ms": graph.system.measured_latency_ms if graph.system else 0.0,
            "tokens_per_sec": graph.system.measured_tokens_per_sec if graph.system else 0.0,
            "device": graph.system.device if graph.system else "unknown",
        },
        "summary": {
            "global_coupling_efficiency": gce,
            "global_obedience_score": obedience,
            "global_intelligence_score": global_intel,
            "predicted_bottleneck": ranked[0].slice_id,
        },
        "slices": [
            {
                "slice_id": s.slice_id,
                "kind": s.kind,
                "op_type": s.op_type,
                "cost": costs[s.slice_id],
                "propagated_cost": propagated[s.slice_id],
                "slice_bottleneck_index": sbi[s.slice_id],
                "topological_congestion": congestion[s.slice_id],
                "direct_effect": de[s.slice_id],
                "estimated_total_effect": ete[s.slice_id],
                "freedom_index": freedom[s.slice_id],
                "compliance_index": compliance[s.slice_id],
                "route_entropy": route_h[s.slice_id],
                "effect_entropy": effect_h[s.slice_id],
                "compliance_entropy": comp_h[s.slice_id],
                "intelligence_index": intel[s.slice_id],
                "propagated_state": propagated_state[s.slice_id],
            }
            for s in ranked
        ],
        "top_edge_attributions": [
            {"src": src, "dst": dst, "edge_attribution": score}
            for (src, dst), score in top_edges
        ],
    }


def demo_report_json(graph: ArchitectureGraph | None = None) -> str:
    return json.dumps(demo_payload(graph), ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tensorearch prototype report")
    parser.add_argument("--input", type=str, default="", help="Path to a JSON trace file")
    args = parser.parse_args()
    graph = load_graph_from_json(args.input) if args.input else build_sample_graph()
    print(demo_report(graph))


if __name__ == "__main__":
    main()
