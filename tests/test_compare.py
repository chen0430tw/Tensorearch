from pathlib import Path

from tensorearch.compare import comparison_report
from tensorearch.intervention import apply_intervention
from tensorearch.io import load_graph_from_json
from tensorearch.schema import Intervention


def _sample_graph():
    root = Path(__file__).resolve().parents[1]
    return load_graph_from_json(root / "examples" / "sample_trace.json")


def test_comparison_report_has_expected_fields():
    left = _sample_graph()
    right = apply_intervention(left, Intervention(kind="set_write_magnitude", target="blk0.attn", value=0.7))
    report = comparison_report(left, right)
    assert "Tensorearch comparison report" in report
    assert "left_bottleneck=" in report
    assert "intelligence_delta=" in report
