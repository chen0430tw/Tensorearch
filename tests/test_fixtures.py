"""Fixture-heavy tests covering branching traces, adapter inputs, and CLI subcommands."""

from pathlib import Path
import json
import subprocess
import sys

import pytest

from tensorearch.compare import comparison_report, comparison_report_json
from tensorearch.demo import demo_report, demo_report_json
from tensorearch.intervention import apply_intervention, intervention_bundle
from tensorearch.io import load_graph_from_json
from tensorearch.schema import Intervention


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def _env():
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    return env


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def branching_transformer():
    return load_graph_from_json(EXAMPLES / "branching_transformer_trace.json")


@pytest.fixture()
def branching_oscillator():
    return load_graph_from_json(EXAMPLES / "branching_oscillator_trace.json")


@pytest.fixture()
def sample():
    return load_graph_from_json(EXAMPLES / "sample_trace.json")


# ---------------------------------------------------------------------------
# Branching transformer trace
# ---------------------------------------------------------------------------

def test_branching_transformer_loads(branching_transformer):
    ids = [s.slice_id for s in branching_transformer.slices]
    assert "blk0.router" in ids
    assert "blk0.expert" in ids


def test_branching_transformer_report_has_bottleneck(branching_transformer):
    report = demo_report(branching_transformer)
    assert "predicted_bottleneck=" in report
    assert "global_coupling_efficiency=" in report


def test_branching_transformer_report_has_entropy(branching_transformer):
    report = demo_report(branching_transformer)
    assert "route_entropy=" in report
    assert "effect_entropy=" in report
    assert "intelligence=" in report


def test_branching_transformer_json_report(branching_transformer):
    payload = json.loads(demo_report_json(branching_transformer))
    assert "summary" in payload
    assert "predicted_bottleneck" in payload["summary"]
    slices = {s["slice_id"]: s for s in payload.get("slices", [])}
    assert "blk0.router" in slices
    assert "freedom_index" in slices["blk0.router"]
    assert "compliance_index" in slices["blk0.router"]


def test_branching_transformer_router_slice_metrics(branching_transformer):
    payload = json.loads(demo_report_json(branching_transformer))
    slices = {s["slice_id"]: s for s in payload["slices"]}
    router = slices["blk0.router"]
    assert router["freedom_index"] >= 0
    assert 0 <= router["compliance_index"] <= 2


def test_branching_transformer_has_tp_comm_edge(branching_transformer):
    tp_edges = [e for e in branching_transformer.edges if e.collective == "allreduce"]
    assert len(tp_edges) >= 1


# ---------------------------------------------------------------------------
# Branching oscillator trace
# ---------------------------------------------------------------------------

def test_branching_oscillator_loads(branching_oscillator):
    ids = [s.slice_id for s in branching_oscillator.slices]
    assert "blk0.phase" in ids
    assert "blk0.prop" in ids
    assert "blk0.local" in ids
    assert "blk0.mixed" in ids


def test_branching_oscillator_report_has_bottleneck(branching_oscillator):
    report = demo_report(branching_oscillator)
    assert "predicted_bottleneck=" in report
    assert "global_obedience_score=" in report


def test_branching_oscillator_json_report(branching_oscillator):
    payload = json.loads(demo_report_json(branching_oscillator))
    assert "summary" in payload
    slices = {s["slice_id"]: s for s in payload["slices"]}
    assert "blk0.prop" in slices
    assert "blk0.local" in slices


def test_branching_oscillator_has_branching_edges(branching_oscillator):
    # prop fans out to both local and ffn
    srcs = [(e.src, e.dst) for e in branching_oscillator.edges]
    assert ("blk0.prop", "blk0.local") in srcs
    assert ("blk0.prop", "blk0.ffn") in srcs


# ---------------------------------------------------------------------------
# Cross-architecture comparison
# ---------------------------------------------------------------------------

def test_compare_transformer_vs_oscillator(branching_transformer, branching_oscillator):
    report = comparison_report(branching_transformer, branching_oscillator)
    assert "Tensorearch comparison report" in report
    assert "left_bottleneck=" in report
    assert "intelligence_delta=" in report


def test_compare_json_transformer_vs_oscillator(branching_transformer, branching_oscillator):
    payload = json.loads(comparison_report_json(branching_transformer, branching_oscillator))
    assert "left_predicted_bottleneck" in payload
    assert "right_predicted_bottleneck" in payload
    assert "intelligence_delta" in payload


# ---------------------------------------------------------------------------
# Intervention on branching traces
# ---------------------------------------------------------------------------

def test_ablate_router_edge(branching_transformer):
    out = apply_intervention(
        branching_transformer,
        Intervention(kind="mask_edge", target="blk0.router->blk0.expert"),
    )
    masked = [e for e in out.edges if e.src == "blk0.router" and e.dst == "blk0.expert"]
    assert masked[0].weight == 0.0


def test_remove_expert_slice(branching_transformer):
    out = apply_intervention(
        branching_transformer,
        Intervention(kind="remove_slice", target="blk0.expert"),
    )
    assert all(s.slice_id != "blk0.expert" for s in out.slices)
    assert all(e.src != "blk0.expert" and e.dst != "blk0.expert" for e in out.edges)


def test_bundle_on_oscillator(branching_oscillator):
    out = intervention_bundle(
        branching_oscillator,
        [
            Intervention(kind="set_write_magnitude", target="blk0.prop", value=0.6),
            Intervention(kind="mask_edge", target="blk0.prop->blk0.local"),
        ],
    )
    prop = next(s for s in out.slices if s.slice_id == "blk0.prop")
    assert prop.write_magnitude == 0.6
    masked = [e for e in out.edges if e.src == "blk0.prop" and e.dst == "blk0.local"]
    assert masked[0].weight == 0.0


def test_ablate_then_compare_branching(branching_oscillator):
    altered = apply_intervention(
        branching_oscillator,
        Intervention(kind="set_write_magnitude", target="blk0.phase", value=0.5),
    )
    report = comparison_report(branching_oscillator, altered)
    assert "intelligence_delta=" in report


# ---------------------------------------------------------------------------
# CLI subcommands on branching traces
# ---------------------------------------------------------------------------

def test_cli_inspect_branching_transformer():
    result = subprocess.run(
        [sys.executable, "-m", "tensorearch", "inspect",
         "examples/branching_transformer_trace.json", "--json"],
        cwd=ROOT, env=_env(), capture_output=True, text=True, check=True,
    )
    payload = json.loads(result.stdout)
    assert "summary" in payload
    assert "slices" in payload


def test_cli_inspect_branching_oscillator():
    result = subprocess.run(
        [sys.executable, "-m", "tensorearch", "inspect",
         "examples/branching_oscillator_trace.json", "--json"],
        cwd=ROOT, env=_env(), capture_output=True, text=True, check=True,
    )
    payload = json.loads(result.stdout)
    assert "summary" in payload


def test_cli_compare_branching_traces():
    result = subprocess.run(
        [sys.executable, "-m", "tensorearch", "compare",
         "examples/branching_transformer_trace.json",
         "examples/branching_oscillator_trace.json",
         "--json"],
        cwd=ROOT, env=_env(), capture_output=True, text=True, check=True,
    )
    payload = json.loads(result.stdout)
    assert "left_predicted_bottleneck" in payload
    assert "right_predicted_bottleneck" in payload


def test_cli_ablate_branching_transformer():
    result = subprocess.run(
        [sys.executable, "-m", "tensorearch", "ablate",
         "examples/branching_transformer_trace.json",
         "--kind", "set_write_magnitude",
         "--target", "blk0.router",
         "--value", "0.5",
         "--json"],
        cwd=ROOT, env=_env(), capture_output=True, text=True, check=True,
    )
    payload = json.loads(result.stdout)
    assert "left_predicted_bottleneck" in payload
    assert "intelligence_delta" in payload


def test_cli_adapt_oscillator(tmp_path):
    out = tmp_path / "oscillator_adapted.json"
    subprocess.run(
        [sys.executable, "-m", "tensorearch", "adapt",
         "--adapter", "oscillator",
         "--input", "examples/oscillator_adapter_input.json",
         "--output", str(out)],
        cwd=ROOT, env=_env(), capture_output=True, text=True, check=True,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "predicted_bottleneck" in payload["summary"]


def test_cli_export_compare(tmp_path):
    out = tmp_path / "compare.json"
    subprocess.run(
        [sys.executable, "-m", "tensorearch", "export",
         "--mode", "compare",
         "--left", "examples/branching_transformer_trace.json",
         "--right", "examples/branching_oscillator_trace.json",
         "--output", str(out),
         "--json"],
        cwd=ROOT, env=_env(), capture_output=True, text=True, check=True,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "left_predicted_bottleneck" in payload


# ---------------------------------------------------------------------------
# JSON output field completeness
# ---------------------------------------------------------------------------

def test_inspect_json_slice_fields(sample):
    payload = json.loads(demo_report_json(sample))
    for s in payload["slices"]:
        assert "slice_id" in s
        assert "freedom_index" in s
        assert "compliance_index" in s
        assert "route_entropy" in s
        assert "effect_entropy" in s
        assert "intelligence_index" in s
        assert "cost" in s
        assert "propagated_cost" in s
        assert "slice_bottleneck_index" in s


def test_inspect_json_edge_fields(sample):
    payload = json.loads(demo_report_json(sample))
    for e in payload.get("edge_attributions", []):
        assert "src" in e
        assert "dst" in e
        assert "attribution" in e


def test_compare_json_all_delta_fields(sample):
    altered = apply_intervention(sample, Intervention(kind="mask_edge", target="blk0.attn->blk0.ffn"))
    payload = json.loads(comparison_report_json(sample, altered))
    for field in ["left_predicted_bottleneck", "right_predicted_bottleneck",
                  "left_intelligence", "right_intelligence", "intelligence_delta",
                  "left_obedience", "right_obedience", "obedience_delta",
                  "left_coupling", "right_coupling", "coupling_delta"]:
        assert field in payload, f"missing field: {field}"
