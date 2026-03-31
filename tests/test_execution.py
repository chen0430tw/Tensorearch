from pathlib import Path

from tensorearch.execution import analyze_graphs_parallel
from tensorearch.io import load_graph_from_json


def _sample_graph():
    root = Path(__file__).resolve().parents[1]
    return load_graph_from_json(root / "examples" / "sample_trace.json")


def test_parallel_analysis_returns_results():
    graphs = [_sample_graph(), _sample_graph()]
    results = analyze_graphs_parallel(graphs, max_workers=2)
    assert len(results) == 2
    assert all(result.predicted_bottleneck for result in results)
