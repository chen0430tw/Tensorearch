from pathlib import Path

from tensorearch.demo import demo_report
from tensorearch.io import load_graph_from_json


def test_demo_report_contains_bottleneck():
    report = demo_report()
    assert "predicted_bottleneck=" in report
    assert "global_coupling_efficiency=" in report


def test_json_trace_loads_and_reports():
    root = Path(__file__).resolve().parents[1]
    graph = load_graph_from_json(root / "examples" / "sample_trace.json")
    report = demo_report(graph)
    assert "blk0.attn" in report
    assert "direct_effect=" in report
    assert "top edge attributions" in report
    assert "global_obedience_score=" in report
    assert "global_intelligence_score=" in report
    assert "freedom=" in report
    assert "compliance=" in report
    assert "route_entropy=" in report
    assert "effect_entropy=" in report
    assert "intelligence=" in report
