from pathlib import Path

from tensorearch.intervention import apply_intervention, intervention_bundle
from tensorearch.io import load_graph_from_json
from tensorearch.schema import Intervention


def _sample_graph():
    root = Path(__file__).resolve().parents[1]
    return load_graph_from_json(root / "examples" / "sample_trace.json")


def test_mask_edge_zeroes_weight():
    graph = _sample_graph()
    out = apply_intervention(graph, Intervention(kind="mask_edge", target="blk0.attn->blk1.attn"))
    target = [e for e in out.edges if e.src == "blk0.attn" and e.dst == "blk1.attn"][0]
    assert target.weight == 0.0


def test_bundle_remove_slice_prunes_edges():
    graph = _sample_graph()
    out = intervention_bundle(
        graph,
        [
            Intervention(kind="set_write_magnitude", target="blk0.attn", value=0.5),
            Intervention(kind="remove_slice", target="blk0.ffn"),
        ],
    )
    assert all(s.slice_id != "blk0.ffn" for s in out.slices)
    assert all(e.src != "blk0.ffn" and e.dst != "blk0.ffn" for e in out.edges)
